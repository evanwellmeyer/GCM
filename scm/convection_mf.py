# simplified mass-flux convection scheme inspired by Zhang-McFarlane.
#
# key improvements over the previous version:
#   - detrainment moistens the free troposphere at the level where
#     the plume loses buoyancy, depositing saturated air. this is the
#     main mechanism for the water vapor feedback under warming.
#   - uses dilute CAPE (entraining parcel) for the closure, which
#     gives a more realistic sensitivity to warming.
#   - the heating profile comes from compensating subsidence warming
#     plus latent heat release, not just local mixing.

import torch
from scm.thermo import (
    cp, Lv, g, Rd, Rv, eps,
    saturation_specific_humidity, virtual_temperature
)


def loaded_virtual_temperature(t, q_vapor, q_condensate):
    """Virtual temperature including condensate loading."""

    return t * (1.0 + (1.0 / eps - 1.0) * q_vapor.clamp(min=0.0) - q_condensate.clamp(min=0.0))


def dilute_cape(t, q, p, entrainment, condensate_retention=0.0, condensate_fallout=1.0):
    """CAPE computed with an entraining parcel. more realistic than
    undilute CAPE because it accounts for how environmental humidity
    affects buoyancy. returns (batch,) in J/kg."""

    batch = t.shape[0]
    nlevels = t.shape[1]

    t_parcel = t[:, -1].clone()
    q_parcel = q[:, -1].clone()
    p_parcel = p[:, -1].clone()
    qc_parcel = torch.zeros(batch, device=t.device, dtype=t.dtype)

    dcape = torch.zeros(batch, device=t.device)
    fallout_keep = 1.0 - float(torch.clamp(torch.as_tensor(condensate_fallout), min=0.0, max=1.0).item())
    cond_retain = float(torch.clamp(torch.as_tensor(condensate_retention), min=0.0, max=1.0).item())

    for k in range(nlevels - 2, -1, -1):
        p_target = p[:, k]
        dp_step = (p_parcel - p_target).abs()

        # entrain environmental air
        mix = 1.0 - torch.exp(-(entrainment * dp_step).clamp(min=0.0, max=5.0))
        t_parcel = (1.0 - mix) * t_parcel + mix * t[:, k]
        q_parcel = (1.0 - mix) * q_parcel + mix * q[:, k]
        qc_parcel = (1.0 - mix) * qc_parcel

        # adiabatic ascent
        qs_p = saturation_specific_humidity(t_parcel, p_target)
        saturated = (q_parcel >= qs_p).float()

        gamma_dry = Rd * t_parcel / (cp * p_target)
        num = (Rd * t_parcel / (cp * p_target)) * (1.0 + Lv * qs_p / (Rd * t_parcel))
        den = 1.0 + Lv * Lv * qs_p / (cp * Rv * t_parcel * t_parcel)
        gamma_moist = num / den
        gamma = (1.0 - saturated) * gamma_dry + saturated * gamma_moist

        dp_rise = p_target - p_parcel
        t_parcel = t_parcel + gamma * dp_rise
        p_parcel = p_target

        # condensation
        qs_new = saturation_specific_humidity(t_parcel, p_target)
        excess = torch.clamp(q_parcel - qs_new, min=0.0)
        q_parcel = q_parcel - excess
        t_parcel = t_parcel + Lv / cp * excess
        qc_parcel = fallout_keep * (qc_parcel + cond_retain * excess)

        # buoyancy contribution
        tv_parcel = loaded_virtual_temperature(t_parcel, q_parcel, qc_parcel)
        tv_env = virtual_temperature(t[:, k], q[:, k])
        buoyancy = torch.clamp((tv_parcel - tv_env) / tv_env, min=0.0)
        dlnp = torch.log(p[:, k + 1].clamp(min=1.0) / p[:, k].clamp(min=1.0))
        dcape = dcape + Rd * tv_env * buoyancy * dlnp

    return dcape


def _column_param(params, name, default, ref_tensor, batch):
    """Return a parameter as a (batch,) tensor."""

    value = params.get(name, default)
    if isinstance(value, torch.Tensor):
        value = value.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        if value.dim() == 0:
            return value.expand(batch)
        if value.dim() == 1:
            if value.shape[0] != batch:
                raise ValueError(f"{name} must have shape ({batch},), got {tuple(value.shape)}")
            return value
        raise ValueError(f"{name} must be scalar or 1D tensor, got ndim={value.dim()}")
    return torch.full((batch,), float(value), device=ref_tensor.device, dtype=ref_tensor.dtype)


def mass_flux_convection(state, grid, params):
    """simplified mass-flux scheme with detrainment moistening."""

    t = state['t']
    q = state['q']
    p = state['p']
    dp = state['dp']
    batch = t.shape[0]
    nlevels = t.shape[1]

    entrainment = params.get('entrainment_rate', 5.0e-6)  # per Pa
    tau_cape = _column_param(params, 'tau_cape', 3600.0, t, batch)
    precip_eff = params.get('precip_efficiency', 0.8)
    cape_threshold = params.get('cape_threshold', 50.0)
    detrain_rh = params.get('mf_detrain_rh', 0.7)
    mb_max = params.get('mf_mb_max', 0.05)
    bl_export_fraction = params.get('mf_bl_export_fraction', 0.02)
    max_dt_day = params.get('mf_max_dt_day', 10.0)
    max_dq_day = params.get('mf_max_dq_day', 5.0)
    cond_retain = params.get('mf_condensate_retention', 0.25)
    cond_fallout = params.get('mf_condensate_fallout', 0.45)
    enforce_mse = bool(params.get('mf_enforce_mse_conservation', True))

    # use dilute CAPE for the closure
    cape_val = dilute_cape(
        t, q, p, entrainment,
        condensate_retention=cond_retain,
        condensate_fallout=cond_fallout,
    )
    cape_excess = torch.clamp(cape_val - cape_threshold, min=0.0)

    tau_mode = str(params.get('mf_cape_timescale_mode', 'fixed'))
    tau_cape_eff = tau_cape
    if tau_mode == 'flow_dependent':
        sigma = grid['sigma_full'].to(device=t.device, dtype=t.dtype)
        ft_top_sigma = float(params.get('mf_tau_cape_ft_top_sigma', 0.30))
        ft_bottom_sigma = float(params.get('mf_tau_cape_ft_bottom_sigma', 0.80))
        ft_mask = ((sigma >= ft_top_sigma) & (sigma <= ft_bottom_sigma)).to(t.dtype).unsqueeze(0)
        ft_mass = torch.sum(ft_mask * dp / g, dim=1).clamp(min=1.0e-8)

        qs_env = saturation_specific_humidity(t, p)
        rh_env = (q / qs_env.clamp(min=1.0e-8)).clamp(min=0.0, max=1.5)
        rh_ft = torch.sum(rh_env * ft_mask * dp / g, dim=1) / ft_mass

        rh_ref = _column_param(params, 'mf_tau_cape_rh_ref', 0.55, t, batch)
        rh_sensitivity = _column_param(params, 'mf_tau_cape_rh_sensitivity', 1.0, t, batch)
        cape_ref = _column_param(params, 'mf_tau_cape_cape_ref', 500.0, t, batch).clamp(min=1.0)
        cape_sensitivity = _column_param(params, 'mf_tau_cape_cape_sensitivity', 1.0, t, batch).clamp(min=0.0)
        tau_min = _column_param(params, 'mf_tau_cape_min', 1800.0, t, batch)
        tau_max = _column_param(params, 'mf_tau_cape_max', 7200.0, t, batch)

        rh_factor = torch.exp(-rh_sensitivity * (rh_ft - rh_ref))
        cape_factor = torch.rsqrt(1.0 + cape_sensitivity * cape_excess / cape_ref)
        tau_cape_eff = tau_cape * rh_factor * cape_factor
        tau_cape_eff = torch.maximum(torch.minimum(tau_cape_eff, tau_max), tau_min)

    # march the plume upward
    t_plume = t[:, -1].clone()
    q_plume = q[:, -1].clone()
    qc_plume = torch.zeros(batch, device=t.device, dtype=t.dtype)
    fallout_keep = 1.0 - float(torch.clamp(torch.as_tensor(cond_fallout), min=0.0, max=1.0).item())
    cond_retain = float(torch.clamp(torch.as_tensor(cond_retain), min=0.0, max=1.0).item())

    dt_norm = torch.zeros_like(t)
    dq_norm = torch.zeros_like(q)
    # track the plume mass flux profile normalized by cloud-base mass flux.
    # it grows from entrainment and shrinks from detrainment.
    mf_profile = torch.ones(batch, device=t.device)

    if isinstance(detrain_rh, torch.Tensor) and detrain_rh.dim() == 1:
        detrain_rh_col = detrain_rh.unsqueeze(1)
    else:
        detrain_rh_col = detrain_rh

    for k in range(nlevels - 2, -1, -1):
        p_here = p[:, k]
        dp_layer = dp[:, k]
        dp_step = (p[:, k + 1] - p[:, k]).abs()

        # entrainment
        mix = 1.0 - torch.exp(-(entrainment * dp_step).clamp(min=0.0, max=5.0))
        t_plume = (1.0 - mix) * t_plume + mix * t[:, k]
        q_plume = (1.0 - mix) * q_plume + mix * q[:, k]
        qc_plume = (1.0 - mix) * qc_plume

        # mass flux increases from entrainment
        mf_profile = mf_profile * (1.0 + mix)

        # adiabatic cooling
        qs_p = saturation_specific_humidity(t_plume, p_here)
        saturated = (q_plume >= qs_p).float()
        gamma_dry = Rd * t_plume / (cp * p_here)
        num = (Rd * t_plume / (cp * p_here)) * (1.0 + Lv * qs_p / (Rd * t_plume))
        den = 1.0 + Lv * Lv * qs_p / (cp * Rv * t_plume * t_plume)
        gamma_moist = num / den
        gamma = (1.0 - saturated) * gamma_dry + saturated * gamma_moist

        dp_rise = p_here - p[:, k + 1]
        t_plume = t_plume + gamma * dp_rise

        # condense
        qs_p = saturation_specific_humidity(t_plume, p_here)
        condensate = torch.clamp(q_plume - qs_p, min=0.0)
        q_plume = q_plume - condensate
        t_plume = t_plume + Lv / cp * condensate
        qc_plume = fallout_keep * (qc_plume + cond_retain * condensate)

        # buoyancy
        tv_plume = loaded_virtual_temperature(t_plume, q_plume, qc_plume)
        tv_env = virtual_temperature(t[:, k], q[:, k])
        buoyant = torch.sigmoid((tv_plume - tv_env) * 5.0)

        # detrainment: where plume loses buoyancy, detrain mass.
        # detrainment rate increases where buoyancy decreases.
        detrain_frac = (1.0 - buoyant) * 0.15  # up to 15% per level (was 30%)
        mf_detrained = mf_profile * detrain_frac
        mf_profile = mf_profile * (1.0 - detrain_frac)

        # detrainment replaces a fraction of the layer with plume air.
        detrain_rate = mf_detrained * g / dp_layer  # 1/s per unit Mb

        # temperature tendency: warming from plume air mixing in
        dt_norm[:, k] = detrain_rate * (t_plume - t[:, k])

        # moisture tendency: detrain plume air, but cap its humidity to a
        # realistic anvil-layer RH target so the scheme cannot fill the free
        # troposphere to saturation. unlike the earlier formulation, this
        # can moisten or dry depending on the local environment.
        qs_env = saturation_specific_humidity(t[:, k], p_here)
        q_detrain = torch.minimum(q_plume, detrain_rh_col * qs_env)
        dq_norm[:, k] = detrain_rate * (q_detrain - q[:, k])

        # compensating subsidence is tied to actual mass-flux divergence,
        # not the entrainment coefficient alone.
        if k < nlevels - 2:
            subsidence_rate = mf_detrained * g / dp_layer
            dt_norm[:, k] = dt_norm[:, k] + subsidence_rate * (t[:, k + 1] - t[:, k])

        # kill plume where it's clearly not buoyant
        mf_profile = mf_profile * (0.3 + 0.7 * buoyant)

    # modest subcloud moisture export spread over the lowest few levels.
    # this avoids the previous behavior where the lowest model level was
    # dried aggressively enough to drive unrealistically large surface fluxes.
    export_levels = min(3, nlevels)
    for i in range(export_levels):
        k = nlevels - 1 - i
        dq_norm[:, k] = dq_norm[:, k] - (
            bl_export_fraction * g / dp[:, k] * q[:, k] / export_levels
        )

    # CAPE closure: only CAPE above threshold can force deep convection.
    col_heating = torch.sum(dt_norm.clamp(min=0.0) * dp / g, dim=1)
    col_mass = dp.sum(dim=1) / g
    col_heating_safe = col_heating.clamp(min=1e-8)
    mb = cape_excess * col_mass / (cp * col_heating_safe * tau_cape_eff)
    mb = mb.clamp(min=0.0)
    if isinstance(mb_max, torch.Tensor):
        mb = torch.minimum(mb, mb_max)
    else:
        mb = mb.clamp(max=mb_max)

    dt_tend = dt_norm * mb.unsqueeze(1)
    dq_tend = dq_norm * mb.unsqueeze(1)

    # limit tendencies
    max_dt = max_dt_day / 86400.0
    max_dq = max_dq_day * 1.0e-3 / 86400.0
    dt_tend = dt_tend.clamp(-max_dt, max_dt)
    dq_tend = dq_tend.clamp(-max_dq, max_dq)

    # Keep the capped heating and drying tendencies close to column
    # moist-enthalpy conserving so convection does not create energy
    # simply because the two profiles were limited independently.
    mse_residual = torch.sum((cp * dt_tend + Lv * dq_tend) * dp / g, dim=1)
    if enforce_mse:
        active_mask = ((dt_tend.abs() + dq_tend.abs()) > 0.0).to(t.dtype)
        active_mass = torch.sum(active_mask * dp / g, dim=1).clamp(min=1.0e-8)
        temp_correction = mse_residual / (cp * active_mass)
        dt_tend = dt_tend - temp_correction.unsqueeze(1) * active_mask
        dt_tend = dt_tend.clamp(-max_dt, max_dt)
        mse_residual = torch.sum((cp * dt_tend + Lv * dq_tend) * dp / g, dim=1)

    # precipitation follows the actual net convective drying tendency.
    precip = precip_eff * (-torch.sum(dq_tend * dp / g, dim=1)).clamp(min=0.0)
    precip = precip.clamp(max=50.0 / 86400.0)

    return {
        'dt': dt_tend,
        'dq': dq_tend,
        'precip': precip,
        'cape': cape_val,
        'tau_cape_eff': tau_cape_eff,
        'mse_residual': mse_residual,
    }
