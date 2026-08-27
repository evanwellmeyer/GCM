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
    saturation_specific_humidity, virtual_temperature, full_level_coordinate,
    half_level_coordinate,
)


def loaded_virtual_temperature(t, q_vapor, q_condensate):
    """Virtual temperature including condensate loading."""

    return t * (1.0 + (1.0 / eps - 1.0) * q_vapor.clamp(min=0.0) - q_condensate.clamp(min=0.0))


def _as_column_tensor(value, ref_tensor, batch, name):
    """Return a scalar or 1D value as a (batch,) tensor."""

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


def _column_param(params, name, default, ref_tensor, batch):
    """Return a parameter as a (batch,) tensor."""

    return _as_column_tensor(params.get(name, default), ref_tensor, batch, name)


def dilute_cape(
    t,
    q,
    p,
    entrainment,
    condensate_retention=0.0,
    condensate_fallout=1.0,
    max_pressure_step=2500.0,
):
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
    entrainment = _as_column_tensor(entrainment, t, batch, 'entrainment')
    fallout_keep = 1.0 - _as_column_tensor(
        condensate_fallout, t, batch, 'condensate_fallout'
    ).clamp(min=0.0, max=1.0)
    cond_retain = _as_column_tensor(
        condensate_retention, t, batch, 'condensate_retention'
    ).clamp(min=0.0, max=1.0)

    pressure_step = max(float(max_pressure_step), 100.0)
    for k in range(nlevels - 2, -1, -1):
        p_lower = p[:, k + 1]
        p_upper = p[:, k]
        layer_span = (p_lower - p_upper).abs()
        nsubsteps = max(1, int(torch.ceil(layer_span.max() / pressure_step).item()))

        for substep in range(1, nsubsteps + 1):
            fraction = substep / nsubsteps
            p_target = p_lower + fraction * (p_upper - p_lower)
            t_env = t[:, k + 1] + fraction * (t[:, k] - t[:, k + 1])
            q_env = q[:, k + 1] + fraction * (q[:, k] - q[:, k + 1])
            dp_step = (p_parcel - p_target).abs()

            # Mix at the parcel pressure before taking the next ascent step.
            mix = 1.0 - torch.exp(-(entrainment * dp_step).clamp(min=0.0, max=5.0))
            t_parcel = (1.0 - mix) * t_parcel + mix * t_env
            q_parcel = (1.0 - mix) * q_parcel + mix * q_env
            qc_parcel = (1.0 - mix) * qc_parcel

            qs_p = saturation_specific_humidity(t_parcel, p_target)
            saturated = (q_parcel >= qs_p).float()
            gamma_dry = Rd * t_parcel / (cp * p_target)
            num = (Rd * t_parcel / (cp * p_target)) * (
                1.0 + Lv * qs_p / (Rd * t_parcel)
            )
            den = 1.0 + Lv * Lv * qs_p / (cp * Rv * t_parcel * t_parcel)
            gamma_moist = num / den
            gamma = (1.0 - saturated) * gamma_dry + saturated * gamma_moist

            p_previous = p_parcel
            t_parcel = t_parcel + gamma * (p_target - p_parcel)
            p_parcel = p_target

            qs_new = saturation_specific_humidity(t_parcel, p_target)
            excess = torch.clamp(q_parcel - qs_new, min=0.0)
            q_parcel = q_parcel - excess
            t_parcel = t_parcel + Lv / cp * excess
            qc_parcel = fallout_keep * (qc_parcel + cond_retain * excess)

            tv_parcel = loaded_virtual_temperature(t_parcel, q_parcel, qc_parcel)
            tv_env = virtual_temperature(t_env, q_env)
            buoyancy = torch.clamp((tv_parcel - tv_env) / tv_env, min=0.0)
            dlnp = torch.log(p_previous.clamp(min=1.0) / p_target.clamp(min=1.0))
            dcape = dcape + Rd * tv_env * buoyancy * dlnp

    return dcape


def mass_flux_convection(state, grid, params):
    """simplified mass-flux scheme with detrainment moistening."""

    t = state['t']
    q = state['q']
    p = state['p']
    dp = state['dp']
    batch = t.shape[0]
    nlevels = t.shape[1]

    entrainment = _column_param(params, 'entrainment_rate', 5.0e-6, t, batch)  # per Pa
    detrainment = _column_param(params, 'mf_detrainment_rate', 3.0e-5, t, batch)
    plume_decay = _column_param(params, 'mf_plume_decay_rate', 1.5e-4, t, batch)
    tau_cape = _column_param(params, 'tau_cape', 3600.0, t, batch)
    precip_eff = _column_param(params, 'precip_efficiency', 0.8, t, batch)
    cape_threshold = _column_param(params, 'cape_threshold', 50.0, t, batch)
    detrain_rh = _column_param(params, 'mf_detrain_rh', 0.7, t, batch)
    mb_max = _column_param(params, 'mf_mb_max', 0.05, t, batch)
    bl_export_fraction = _column_param(params, 'mf_bl_export_fraction', 0.02, t, batch)
    max_dt_day = _column_param(params, 'mf_max_dt_day', 10.0, t, batch)
    max_dq_day = _column_param(params, 'mf_max_dq_day', 5.0, t, batch)
    cond_retain = _column_param(params, 'mf_condensate_retention', 0.25, t, batch)
    cond_fallout = _column_param(params, 'mf_condensate_fallout', 0.45, t, batch)
    enforce_mse = bool(params.get('mf_enforce_mse_conservation', True))

    # use dilute CAPE for the closure
    cape_val = dilute_cape(
        t, q, p, entrainment,
        condensate_retention=cond_retain,
        condensate_fallout=cond_fallout,
        max_pressure_step=params.get('mf_cape_max_pressure_step', 2500.0),
    )
    cape_excess = torch.clamp(cape_val - cape_threshold, min=0.0)

    tau_mode = str(params.get('mf_cape_timescale_mode', 'fixed'))
    tau_cape_eff = tau_cape
    if tau_mode == 'flow_dependent':
        sigma = full_level_coordinate(grid, state=state, device=t.device, dtype=t.dtype)
        ft_top_sigma = _column_param(params, 'mf_tau_cape_ft_top_sigma', 0.30, t, batch)
        ft_bottom_sigma = _column_param(params, 'mf_tau_cape_ft_bottom_sigma', 0.80, t, batch)
        ft_mask = (
            (sigma >= ft_top_sigma.unsqueeze(1))
            & (sigma <= ft_bottom_sigma.unsqueeze(1))
        ).to(t.dtype)
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
    fallout_keep = 1.0 - cond_fallout.clamp(min=0.0, max=1.0)
    cond_retain = cond_retain.clamp(min=0.0, max=1.0)

    dt_norm = torch.zeros_like(t)
    dq_norm = torch.zeros_like(q)
    # track the plume mass flux profile normalized by cloud-base mass flux.
    # it grows from entrainment and shrinks from detrainment.
    mf_profile = torch.ones(batch, device=t.device)

    for k in range(nlevels - 2, -1, -1):
        p_here = p[:, k]
        dp_layer = dp[:, k]
        dp_step = (p[:, k + 1] - p[:, k]).abs()

        # entrainment
        mix = 1.0 - torch.exp(-(entrainment * dp_step).clamp(min=0.0, max=5.0))
        t_plume = (1.0 - mix) * t_plume + mix * t[:, k]
        q_plume = (1.0 - mix) * q_plume + mix * q[:, k]
        qc_plume = (1.0 - mix) * qc_plume

        # Use the pressure-coordinate solution for plume-mass growth so that
        # splitting a layer does not change cumulative entrainment.
        mf_profile = mf_profile * torch.exp((entrainment * dp_step).clamp(max=5.0))

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

        # Detrainment and plume decay are rates per pascal. The previous
        # fixed fraction per model level changed when the same layer was
        # divided into two thinner layers.
        detrain_exponent = detrainment * (1.0 - buoyant) * dp_step
        detrain_frac = 1.0 - torch.exp(-detrain_exponent.clamp(min=0.0, max=5.0))
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
        q_detrain = torch.minimum(q_plume, detrain_rh * qs_env)
        dq_norm[:, k] = detrain_rate * (q_detrain - q[:, k])

        # compensating subsidence is tied to actual mass-flux divergence,
        # not the entrainment coefficient alone.
        if k < nlevels - 2:
            subsidence_rate = mf_detrained * g / dp_layer
            dt_norm[:, k] = dt_norm[:, k] + subsidence_rate * (t[:, k + 1] - t[:, k])

        decay_exponent = plume_decay * (1.0 - buoyant) * dp_step
        mf_profile = mf_profile * torch.exp(-decay_exponent.clamp(min=0.0, max=5.0))

    # Modest subcloud moisture export over a fixed sigma depth. Normalizing by
    # the selected layer mass keeps the column sink independent of level count.
    export_top_sigma = float(params.get('mf_bl_export_top_sigma', 0.96))
    sigma_half = half_level_coordinate(grid, state=state, device=t.device, dtype=t.dtype)
    sigma_span = (sigma_half[:, 1:] - sigma_half[:, :-1]).clamp(min=1.0e-8)
    export_overlap = (
        sigma_half[:, 1:] - torch.maximum(
            sigma_half[:, :-1],
            torch.as_tensor(export_top_sigma, device=t.device, dtype=t.dtype),
        )
    ).clamp(min=0.0)
    export_weights = (export_overlap / sigma_span).clamp(max=1.0)
    empty_export = export_weights.sum(dim=1) == 0
    export_weights[empty_export, -1] = 1.0
    layer_mass = dp / g
    export_mass = torch.sum(export_weights * layer_mass, dim=1).clamp(min=1.0e-8)
    export_q = torch.sum(export_weights * q * layer_mass, dim=1) / export_mass
    dq_norm = dq_norm - (
        bl_export_fraction * export_q / export_mass
    ).unsqueeze(1) * export_weights

    # CAPE closure: only CAPE above threshold can force deep convection.
    col_heating = torch.sum(dt_norm.clamp(min=0.0) * dp / g, dim=1)
    col_mass = dp.sum(dim=1) / g
    col_heating_safe = col_heating.clamp(min=1e-8)
    mb_unlimited = cape_excess * col_mass / (cp * col_heating_safe * tau_cape_eff)
    mb_unlimited = mb_unlimited.clamp(min=0.0)
    mb = torch.minimum(mb_unlimited, mb_max)

    dt_uncapped = dt_norm * mb.unsqueeze(1)
    dq_uncapped = dq_norm * mb.unsqueeze(1)

    # limit tendencies
    max_dt = (max_dt_day / 86400.0).unsqueeze(1)
    max_dq = (max_dq_day * 1.0e-3 / 86400.0).unsqueeze(1)
    dt_cap_active = dt_uncapped.abs() > max_dt
    dq_cap_active = dq_uncapped.abs() > max_dq
    dt_tend = torch.maximum(torch.minimum(dt_uncapped, max_dt), -max_dt)
    dq_tend = torch.maximum(torch.minimum(dq_uncapped, max_dq), -max_dq)

    # Keep the capped heating and drying tendencies close to column
    # moist-enthalpy conserving so convection does not create energy
    # simply because the two profiles were limited independently.
    mse_residual = torch.sum((cp * dt_tend + Lv * dq_tend) * dp / g, dim=1)
    if enforce_mse:
        active_mask = ((dt_tend.abs() + dq_tend.abs()) > 0.0).to(t.dtype)
        active_mass = torch.sum(active_mask * dp / g, dim=1).clamp(min=1.0e-8)
        temp_correction = mse_residual / (cp * active_mass)
        dt_tend = dt_tend - temp_correction.unsqueeze(1) * active_mask
        dt_tend = torch.maximum(torch.minimum(dt_tend, max_dt), -max_dt)
        mse_residual = torch.sum((cp * dt_tend + Lv * dq_tend) * dp / g, dim=1)

    # By default all net convective drying reaches the surface as rain. A
    # retained-condensate path is available for experiments, but it requires
    # separate long-run validation because anvil evaporation strongly changes
    # the thermal profile in this lightweight cloud scheme.
    column_drying = (-torch.sum(dq_tend * dp / g, dim=1)).clamp(min=0.0)
    precip_eff = precip_eff.clamp(min=0.0, max=1.0)
    if params.get('mf_retain_convective_condensate', False):
        precip = precip_eff * column_drying
        cloud_condensate = (1.0 - precip_eff) * column_drying
    else:
        precip = column_drying
        cloud_condensate = torch.zeros_like(column_drying)
    precip = precip.clamp(max=50.0 / 86400.0)

    return {
        'dt': dt_tend,
        'dq': dq_tend,
        'precip': precip,
        'cloud_condensate': cloud_condensate,
        'cape': cape_val,
        'tau_cape_eff': tau_cape_eff,
        'cloud_base_mass_flux': mb,
        'cloud_base_mass_flux_unlimited': mb_unlimited,
        'mass_flux_cap_active': mb_unlimited > mb_max,
        'temperature_cap_fraction': dt_cap_active.to(t.dtype).mean(dim=1),
        'moisture_cap_fraction': dq_cap_active.to(t.dtype).mean(dim=1),
        'mse_residual': mse_residual,
    }
