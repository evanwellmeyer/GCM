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
    half_level_coordinate, geopotential,
)
from scm.convective_transport import updraft, downdraft


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


def parcel_ascent(temperature, vapor, lower, upper):
    """Dry pressure work followed by enthalpy-conserving saturation adjustment."""
    dry = temperature * (upper / lower).clamp(min=1.0e-8) ** (Rd / cp)
    # Solve cp*T + Lv*q = cp*Tdry + Lv*qin with q <= qsat(T,p).
    low = torch.zeros_like(vapor)
    high = vapor.clamp(min=0.0)
    for iteration in range(28):
        condensed = (low + high) / 2
        warmed = dry + (Lv / cp) * condensed
        excess = vapor - condensed - saturation_specific_humidity(warmed, upper)
        low = torch.where(excess > 0, condensed, low)
        high = torch.where(excess > 0, high, condensed)
    condensed = (low + high) / 2
    condensed = torch.where(vapor > saturation_specific_humidity(dry, upper), condensed, torch.zeros_like(condensed))
    return dry + (Lv / cp) * condensed, vapor - condensed, condensed


def dilute_cape(
    t,
    q,
    p,
    entrainment,
    condensate_retention=0.0,
    condensate_fallout=1.0,
    max_pressure_step=1000.0,
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

            p_previous = p_parcel
            t_parcel, q_parcel, excess = parcel_ascent(t_parcel, q_parcel, p_parcel, p_target)
            p_parcel = p_target
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
    # The closure's CAPE parcel and the plume's own mixing are separate in CESM2: ZM
    # dilutes its CAPE parcel at about 1 per km while the plume carries its own
    # entrainment. Ours was one knob, so raising it to stop spurious deep convection in
    # BOMEX also moved the plume's detrainment and cooled the upper troposphere 6-7 K
    # (tests of 11-12 Sep 2026). Unset, this is the plume's rate and nothing changes.
    if 'mf_cape_entrainment_rate' in params:
        cape_entrainment = _column_param(params, 'mf_cape_entrainment_rate', 5.0e-6, t, batch)
    else:
        cape_entrainment = entrainment
    # Only the flux transport remains. The legacy transport, removed 10 Sep 2026,
    # lost about 120 W/m2 of column moist static energy and hid it with a uniform
    # correction.
    if params.get('mf_transport_form', 'flux') != 'flux':
        raise ValueError("mf_transport_form must be 'flux'; the legacy transport was removed")
    stop_at_lnb = bool(params.get('mf_plume_stop_at_neutral_buoyancy', False))
    height = geopotential(t, q, p, grid)
    detrainment = _column_param(params, 'mf_detrainment_rate', 3.0e-5, t, batch)
    plume_decay = _column_param(params, 'mf_plume_decay_rate', 1.5e-4, t, batch)
    tau_cape = _column_param(params, 'tau_cape', 3600.0, t, batch)
    precip_eff = _column_param(params, 'precip_efficiency', 0.8, t, batch)
    cape_threshold = _column_param(params, 'cape_threshold', 50.0, t, batch)
    mb_max = _column_param(params, 'mf_mb_max', 0.05, t, batch)
    max_dt_day = _column_param(params, 'mf_max_dt_day', 10.0, t, batch)
    max_dq_day = _column_param(params, 'mf_max_dq_day', 5.0, t, batch)
    cond_retain = _column_param(params, 'mf_condensate_retention', 0.25, t, batch)
    cond_fallout = _column_param(params, 'mf_condensate_fallout', 0.45, t, batch)
    if params.get('mf_retain_convective_condensate', False):
        raise ValueError('The flux updraft currently requires immediate condensate fallout')
    # Use the same pseudoadiabatic assumption in the CAPE parcel and the
    # transported plume; retained condensate is not implemented here yet.
    cond_retain = torch.zeros_like(cond_retain)
    cond_fallout = torch.ones_like(cond_fallout)
    buoyancy_detrainment = _column_param(
        params, 'mf_buoyancy_detrainment_weight', 1.0, t, batch
    ).clamp(min=0.0, max=1.0)
    if bool(torch.any(_column_param(params, 'mf_rain_evap_coefficient', 0., t, batch) > 0)):
        raise ValueError('Flux transport uses explicit downdraft evaporation; separate rain evaporation is not implemented')
    model_dt = float(params.get('dt', 900.0))
    fullsigma = full_level_coordinate(grid, state=state, device=t.device, dtype=t.dtype)

    # use dilute CAPE for the closure
    cape_val = dilute_cape(
        t, q, p, cape_entrainment,
        condensate_retention=cond_retain,
        condensate_fallout=cond_fallout,
        max_pressure_step=params.get('mf_cape_max_pressure_step', 1000.0),
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

    # A single bulk plume detrains all of its mass over a narrow depth, which
    # piles the detrained moisture into one or two layers. Real schemes avoid
    # that by carrying a spectrum: Zhang-McFarlane follows Arakawa-Schubert in
    # launching plumes across a range of entrainment rates, and the GFDL scheme
    # runs a shallow and a deep plume. Weakly entraining plumes stay buoyant
    # and detrain high, strongly entraining ones dilute and detrain low, so the
    # detrainment is spread through the depth of the cloud layer instead of
    # landing in a single layer.
    dt_norm = torch.zeros_like(t)
    dq_norm = torch.zeros_like(q)
    rainproduction = torch.zeros_like(q)
    plume_count = max(int(params.get("mf_plume_count", 1)), 1)
    entrainment_spread = float(params.get("mf_plume_entrainment_spread", 3.0))
    if plume_count > 1 and entrainment_spread > 1.0:
        plume_scales = torch.logspace(
            -1.0, 1.0, plume_count, base=entrainment_spread,
            device=t.device, dtype=t.dtype,
        )
    else:
        plume_scales = torch.ones(plume_count, device=t.device, dtype=t.dtype)
    plume_weight = 1.0 / plume_count
    entrainment_base = entrainment

    for plume_index in range(plume_count):
        entrainment = entrainment_base * plume_scales[plume_index]
        member = updraft(t, q, height, p, dp, entrainment, detrainment,
                         plume_decay, buoyancy_detrainment, floor=1e-7,
                         stop_at_neutral_buoyancy=stop_at_lnb)
        dt_norm = dt_norm + plume_weight * member['dt']
        dq_norm = dq_norm + plume_weight * member['dq']
        rainproduction = rainproduction + plume_weight * member['rain']

    entrainment = entrainment_base

    transportresidual = torch.sum((cp * dt_norm + Lv * dq_norm) * dp / g, dim=1)

    # Convective downdrafts. A downdraft starts in the mid troposphere, where
    # the air is cool and dry, and sinks. Compression warms it, entrainment
    # mixes in its surroundings, and evaporating rain cools it back toward
    # saturation. It arrives in the subcloud layer with much lower moist static
    # energy than the air already there, so detraining it cools and dries the
    # boundary layer. The updraft's rain feeds it; see `downdraft` in
    # convective_transport.py.
    downdraft_fraction = _column_param(params, 'mf_downdraft_fraction', 0.0, t, batch)
    draft = downdraft(t, q, height, p, dp, fullsigma, rainproduction,
                      downdraft_fraction, params)
    dt_norm = dt_norm + draft['dt']
    dq_norm = dq_norm + draft['dq']
    rainevaporation = draft['evaporation']

    draftresidual = torch.sum((cp * dt_norm + Lv * dq_norm) * dp / g, dim=1)

    sigma_half = half_level_coordinate(grid, state=state, device=t.device, dtype=t.dtype)
    sigma_span = (sigma_half[:, 1:] - sigma_half[:, :-1]).clamp(min=1.0e-8)
    layer_mass = dp / g

    closure_mode = str(params.get('mf_closure_mode', 'heating_proxy'))
    rawresidual = torch.sum((cp * dt_norm + Lv * dq_norm) * dp / g, dim=1)
    cape_response = torch.zeros_like(cape_val)
    closure_stabilizing = torch.ones_like(cape_val, dtype=torch.bool)

    if closure_mode == 'cape_response':
        trial_mass_flux = _column_param(
            params, 'mf_trial_mass_flux', 0.01, t, batch
        ).clamp(min=1.0e-6)
        trial_dt = dt_norm * trial_mass_flux.unsqueeze(1)
        trial_dq = dq_norm * trial_mass_flux.unsqueeze(1)

        trial_t = torch.clamp(t + model_dt * trial_dt, min=150.0, max=350.0)
        trial_q = torch.clamp(q + model_dt * trial_dq, min=1.0e-7, max=0.1)
        trial_cape = dilute_cape(
            trial_t,
            trial_q,
            p,
            cape_entrainment,
            condensate_retention=cond_retain,
            condensate_fallout=cond_fallout,
            max_pressure_step=params.get('mf_cape_max_pressure_step', 1000.0),
        )
        cape_response = (cape_val - trial_cape) / trial_mass_flux
        minimum_response = _column_param(
            params, 'mf_minimum_cape_response', 1.0, t, batch
        ).clamp(min=0.0)
        closure_stabilizing = cape_response > minimum_response
        target_reduction = cape_excess * (
            1.0 - torch.exp(-model_dt / tau_cape_eff.clamp(min=model_dt))
        )
        mb_unlimited = torch.where(
            closure_stabilizing,
            target_reduction / cape_response.clamp(min=1.0e-8),
            torch.zeros_like(cape_response),
        )

        source_top_sigma = float(params.get('mf_source_top_sigma', 0.90))
        source_overlap = (
            sigma_half[:, 1:] - torch.maximum(
                sigma_half[:, :-1],
                torch.as_tensor(source_top_sigma, device=t.device, dtype=t.dtype),
            )
        ).clamp(min=0.0)
        source_weights = (source_overlap / sigma_span).clamp(max=1.0)
        empty_source = source_weights.sum(dim=1) == 0
        source_weights[empty_source, -1] = 1.0
        source_mass = torch.sum(source_weights * layer_mass, dim=1)
        available_fraction = _column_param(
            params, 'mf_available_mass_fraction', 0.25, t, batch
        ).clamp(min=0.0, max=1.0)
        available_mass_limit = available_fraction * source_mass / model_dt
        mb_limit = torch.minimum(mb_max, available_mass_limit)
    else:
        col_heating = torch.sum(dt_norm.clamp(min=0.0) * dp / g, dim=1)
        col_mass = dp.sum(dim=1) / g
        col_heating_safe = col_heating.clamp(min=1e-8)
        mb_unlimited = cape_excess * col_mass / (cp * col_heating_safe * tau_cape_eff)
        mb_limit = mb_max

    mb_unlimited = mb_unlimited.clamp(min=0.0)
    mb = torch.minimum(mb_unlimited, mb_limit)

    dt_uncapped = dt_norm * mb.unsqueeze(1)
    dq_uncapped = dq_norm * mb.unsqueeze(1)

    # limit tendencies
    max_dt = (max_dt_day / 86400.0).unsqueeze(1)
    max_dq = (max_dq_day * 1.0e-3 / 86400.0).unsqueeze(1)
    dt_cap_active = dt_uncapped.abs() > max_dt
    dq_cap_active = dq_uncapped.abs() > max_dq
    dt_tend = torch.maximum(torch.minimum(dt_uncapped, max_dt), -max_dt)
    dq_tend = torch.maximum(torch.minimum(dq_uncapped, max_dq), -max_dq)
    limiter = torch.ones_like(mb)
    # A single scale preserves every paired heat/water/rain exchange.
    limiter = torch.minimum(limiter, (max_dt / dt_uncapped.abs().clamp(min=1e-30)).amin(dim=1))
    limiter = torch.minimum(limiter, (max_dq / dq_uncapped.abs().clamp(min=1e-30)).amin(dim=1))
    # Permit only floating-point roundoff around the host floor; otherwise
    # a vanishing upper-level cancellation can shut down the entire plume.
    tolerance = 16 * torch.finfo(q.dtype).eps * q.abs().clamp(min=1e-7)
    available = ((q - 1e-7).clamp(min=0) + tolerance) / (model_dt * (-dq_uncapped).clamp(min=1e-30))
    available = torch.where(dq_uncapped < 0, available, torch.ones_like(available))
    limiter = torch.minimum(limiter, available.amin(dim=1))
    cooling = (t - 150).clamp(min=0) / (model_dt * (-dt_uncapped).clamp(min=1e-30))
    warming = (350 - t).clamp(min=0) / (model_dt * dt_uncapped.clamp(min=1e-30))
    cooling = torch.where(dt_uncapped < 0, cooling, torch.ones_like(cooling))
    warming = torch.where(dt_uncapped > 0, warming, torch.ones_like(warming))
    limiter = torch.minimum(limiter, torch.minimum(cooling, warming).amin(dim=1))
    rainfall = (rainproduction.sum(dim=1) - rainevaporation.sum(dim=1)).clamp(min=0) * mb
    limiter = torch.minimum(limiter, (50. / 86400.) / rainfall.clamp(min=1e-30))
    dt_tend = dt_uncapped * limiter.unsqueeze(1)
    dq_tend = dq_uncapped * limiter.unsqueeze(1)

    # Column moist-enthalpy residual of the final tendencies, as a diagnostic.
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
    precip = (rainproduction.sum(dim=1) - rainevaporation.sum(dim=1)).clamp(min=0) * mb * limiter

    return {
        'dt': dt_tend,
        'dq': dq_tend,
        'precip': precip,
        'cloud_condensate': cloud_condensate,
        'cape': cape_val,
        'tau_cape_eff': tau_cape_eff,
        'cloud_base_mass_flux': mb,
        'cloud_base_mass_flux_unlimited': mb_unlimited,
        'cloud_base_mass_flux_limit': mb_limit,
        'cape_response_per_mass_flux': cape_response,
        'closure_stabilizing': closure_stabilizing,
        'mass_flux_cap_active': mb_unlimited > mb_limit,
        'temperature_cap_fraction': dt_cap_active.to(t.dtype).mean(dim=1),
        'moisture_cap_fraction': dq_cap_active.to(t.dtype).mean(dim=1),
        'mse_residual': mse_residual,
        'transport_limiter': limiter,
        'rain_production': rainproduction.sum(dim=1) * mb * limiter,
        'rain_evaporation': rainevaporation.sum(dim=1) * mb * limiter,
        'transport_mse_residual_per_mass_flux': transportresidual,
        'raw_mse_residual_per_mass_flux': rawresidual,
        'downdraft_mse_residual_per_mass_flux': draftresidual - transportresidual,
        'export_mse_residual_per_mass_flux': rawresidual - draftresidual,
    }
