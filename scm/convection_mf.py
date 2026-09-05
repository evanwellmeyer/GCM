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


def _conserve_mse(dt_tend, dq_tend, dp, correction_region=None):
    """Remove the column moist-energy residual from active temperature levels."""

    residual = torch.sum((cp * dt_tend + Lv * dq_tend) * dp / g, dim=1)
    active = ((dt_tend.abs() + dq_tend.abs()) > 0.0).to(dt_tend.dtype)
    if correction_region is not None:
        active = active * correction_region.to(dt_tend.dtype)
        empty = active.sum(dim=1) == 0
        active[empty] = ((dt_tend[empty].abs() + dq_tend[empty].abs()) > 0.0).to(dt_tend.dtype)
    active_mass = torch.sum(active * dp / g, dim=1).clamp(min=1.0e-8)
    correction = residual / (cp * active_mass)
    corrected = dt_tend - correction.unsqueeze(1) * active
    return corrected


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
    # Compensating-subsidence drying of the environment. On by default because
    # it is half of a matched pair with the subsidence warming below; the
    # switch exists so the two can be compared directly.
    subsidence_drying = bool(params.get('mf_subsidence_drying', True))
    max_dt_day = _column_param(params, 'mf_max_dt_day', 10.0, t, batch)
    max_dq_day = _column_param(params, 'mf_max_dq_day', 5.0, t, batch)
    cond_retain = _column_param(params, 'mf_condensate_retention', 0.25, t, batch)
    cond_fallout = _column_param(params, 'mf_condensate_fallout', 0.45, t, batch)
    buoyancy_detrainment = _column_param(
        params, 'mf_buoyancy_detrainment_weight', 1.0, t, batch
    ).clamp(min=0.0, max=1.0)
    enforce_mse = bool(params.get('mf_enforce_mse_conservation', True))
    model_dt = float(params.get('dt', 900.0))
    correction_top_sigma = float(params.get('mf_mse_correction_top_sigma', 0.0))
    fullsigma = full_level_coordinate(grid, state=state, device=t.device, dtype=t.dtype)
    correction_region = fullsigma >= correction_top_sigma

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
        dt_member = torch.zeros_like(t)
        dq_member = torch.zeros_like(q)
        # march the plume upward
        t_plume = t[:, -1].clone()
        q_plume = q[:, -1].clone()
        qc_plume = torch.zeros(batch, device=t.device, dtype=t.dtype)
        fallout_keep = 1.0 - cond_fallout.clamp(min=0.0, max=1.0)
        cond_retain = cond_retain.clamp(min=0.0, max=1.0)

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
            detrain_factor = (
                (1.0 - buoyancy_detrainment)
                + buoyancy_detrainment * (1.0 - buoyant)
            )
            detrain_exponent = detrainment * detrain_factor * dp_step
            detrain_frac = 1.0 - torch.exp(-detrain_exponent.clamp(min=0.0, max=5.0))
            mf_detrained = mf_profile * detrain_frac
            mf_profile = mf_profile * (1.0 - detrain_frac)

            # detrainment replaces a fraction of the layer with plume air.
            detrain_rate = mf_detrained * g / dp_layer  # 1/s per unit Mb

            # temperature tendency: warming from plume air mixing in
            dt_member[:, k] = detrain_rate * (t_plume - t[:, k])

            # moisture tendency: detrain plume air, but cap its humidity to a
            # realistic anvil-layer RH target so the scheme cannot fill the free
            # troposphere to saturation. unlike the earlier formulation, this
            # can moisten or dry depending on the local environment.
            qs_env = saturation_specific_humidity(t[:, k], p_here)
            q_detrain = torch.minimum(q_plume, detrain_rh * qs_env)
            dq_member[:, k] = detrain_rate * (q_detrain - q[:, k])

            # compensating subsidence is tied to actual mass-flux divergence,
            # not the entrainment coefficient alone.
            if k < nlevels - 2:
                subsidence_rate = mf_detrained * g / dp_layer
                dt_member[:, k] = dt_member[:, k] + subsidence_rate * (t[:, k + 1] - t[:, k])
                # The same descending environmental motion advects moisture as well
                # as heat. Water vapour falls off with height, so air arriving from
                # above is drier and the layer dries: dq/dt = -g * M * dq/dp. In
                # Zhang-McFarlane, and in the schemes CESM and GFDL run, the heat
                # and moisture tendencies are a matched pair driven by the same
                # mass flux; they differ in sign only because s and q have opposite
                # vertical gradients. Carrying the warming without the drying left
                # the free troposphere with no way to dry at all, which is what
                # kept 685-865 hPa pinned at saturation. Unlike the temperature
                # term above there is no compression to account for here, so this
                # is plain advection.
                if subsidence_drying:
                    # Written as a transport, not as a local gradient. A bare
                    # -g*M*dq/dp term integrates over the column to -M*(q_sfc -
                    # q_top), so it destroys water rather than moving it. Here the
                    # moisture the descending environment removes from this layer
                    # is deposited in the layer beneath it, upwind-differenced, so
                    # the pair is mass-weighted zero-sum and the column budget
                    # closes exactly. Free-tropospheric layers still dry, because
                    # the drier layer above sends down less than they give up.
                    moisture_transport = subsidence_rate * q[:, k]
                    dq_member[:, k] = dq_member[:, k] - moisture_transport
                    dq_member[:, k + 1] = dq_member[:, k + 1] + (
                        moisture_transport * dp_layer / dp[:, k + 1]
                    )

            decay_exponent = plume_decay * (1.0 - buoyant) * dp_step
            mf_profile = mf_profile * torch.exp(-decay_exponent.clamp(min=0.0, max=5.0))

        dt_norm = dt_norm + plume_weight * dt_member
        dq_norm = dq_norm + plume_weight * dq_member

    entrainment = entrainment_base

    # Convective downdrafts. A downdraft starts in the mid troposphere, where
    # the air is cool and dry, and sinks. Compression warms it, entrainment
    # mixes in its surroundings, and evaporating rain cools it back toward
    # saturation. It arrives in the subcloud layer with much lower moist static
    # energy than the air already there, so detraining it cools and dries the
    # boundary layer. This is the process the `mf_bl_export_fraction` term was
    # standing in for, and the reason a downdraft needs rain to evaporate into
    # it: without `mf_rain_evap_coefficient` above zero the draft warms
    # adiabatically on the way down and does almost nothing.
    downdraft_fraction = _column_param(params, 'mf_downdraft_fraction', 0.0, t, batch)
    if bool(torch.any(downdraft_fraction > 0.0)):
        start_sigma = float(params.get('mf_downdraft_start_sigma', 0.60))
        dd_entrain = float(params.get('mf_downdraft_entrainment', 0.05))
        dd_detrain = float(params.get('mf_downdraft_detrainment', 0.05))
        dd_release = float(params.get('mf_downdraft_release', 0.45))
        dd_release_sigma = float(params.get('mf_downdraft_release_sigma', 0.90))
        sigma_full_dd = full_level_coordinate(grid, state=state, device=t.device, dtype=t.dtype)
        start_level = int(torch.argmin((sigma_full_dd[0] - start_sigma).abs()).item())

        dd_rain_share = float(params.get('mf_downdraft_rain_share', 0.5))
        # rain generated above the starting level is what the draft can draw on
        rain_available = (
            (-dq_norm[:, :start_level + 1]).clamp(min=0.0) * dp[:, :start_level + 1] / g
        ).sum(dim=1)

        t_dd = t[:, start_level].clone()
        q_dd = q[:, start_level].clone()
        md = downdraft_fraction.clone()
        for k in range(start_level, nlevels - 1):
            # sink one layer: compress, then mix with the surroundings
            t_dd = t_dd * (p[:, k + 1] / p[:, k].clamp(min=1.0)) ** (Rd / cp)
            t_dd = (1.0 - dd_entrain) * t_dd + dd_entrain * t[:, k + 1]
            q_dd = (1.0 - dd_entrain) * q_dd + dd_entrain * q[:, k + 1]
            md = md * (1.0 + dd_entrain)

            # Evaporate rain into the draft. The uptake has to be drawn from
            # the rain that is actually falling, not conjured to force the
            # draft to saturation: a draft that is topped up without limit
            # arrives warm and moist and does the opposite of what a downdraft
            # is for. Its defining property is the low moist static energy it
            # keeps from the level it started at, so the entrainment that
            # dilutes that signature is also kept small.
            qs_dd = saturation_specific_humidity(t_dd, p[:, k + 1])
            demand = (qs_dd - q_dd).clamp(min=0.0)
            available = (rain_available * dd_rain_share / md.clamp(min=1.0e-8))
            uptake = torch.minimum(demand, available.clamp(min=0.0))
            q_dd = q_dd + uptake
            t_dd = t_dd - (Lv / cp) * uptake
            rain_available = (rain_available - uptake * md).clamp(min=0.0)

            # Detrain draft air into the layer it is passing through. Most of
            # a downdraft's mass is delivered below cloud base rather than
            # bled off on the way down, so detrainment stays weak until the
            # draft is under the cloud layer and then releases what is left.
            below_base = sigma_full_dd[0, k + 1] >= dd_release_sigma
            local_detrain = dd_release if bool(below_base) else dd_detrain
            rate = local_detrain * md * g / dp[:, k + 1]
            dt_norm[:, k + 1] = dt_norm[:, k + 1] + rate * (t_dd - t[:, k + 1])
            dq_norm[:, k + 1] = dq_norm[:, k + 1] + rate * (q_dd - q[:, k + 1])
            md = md * (1.0 - local_detrain)

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

    closure_mode = str(params.get('mf_closure_mode', 'heating_proxy'))
    cape_response = torch.zeros_like(cape_val)
    closure_stabilizing = torch.ones_like(cape_val, dtype=torch.bool)

    if closure_mode == 'cape_response':
        trial_mass_flux = _column_param(
            params, 'mf_trial_mass_flux', 0.01, t, batch
        ).clamp(min=1.0e-6)
        trial_dt = dt_norm * trial_mass_flux.unsqueeze(1)
        trial_dq = dq_norm * trial_mass_flux.unsqueeze(1)
        if enforce_mse:
            trial_dt = _conserve_mse(trial_dt, trial_dq, dp, correction_region)

        trial_t = torch.clamp(t + model_dt * trial_dt, min=150.0, max=350.0)
        trial_q = torch.clamp(q + model_dt * trial_dq, min=1.0e-7, max=0.1)
        trial_cape = dilute_cape(
            trial_t,
            trial_q,
            p,
            entrainment,
            condensate_retention=cond_retain,
            condensate_fallout=cond_fallout,
            max_pressure_step=params.get('mf_cape_max_pressure_step', 2500.0),
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

    # Keep the capped heating and drying tendencies close to column
    # moist-enthalpy conserving so convection does not create energy
    # simply because the two profiles were limited independently.
    mse_residual = torch.sum((cp * dt_tend + Lv * dq_tend) * dp / g, dim=1)
    if enforce_mse:
        dt_tend = _conserve_mse(dt_tend, dq_tend, dp, correction_region)
        dt_tend = torch.maximum(torch.minimum(dt_tend, max_dt), -max_dt)
        mse_residual = torch.sum((cp * dt_tend + Lv * dq_tend) * dp / g, dim=1)

    # Convective rain falls through the column rather than teleporting to the
    # surface. In unsaturated layers some of it evaporates, which moistens and
    # cools the air it passes through -- the main exchange between the plume's
    # precipitation and the environment in Zhang-McFarlane and in the GFDL
    # scheme, and the process that makes convective downdrafts possible. Both
    # budgets close by construction: what leaves the rain flux enters the
    # layer, and the cooling is exactly the latent heat of what evaporated, so
    # the column moist enthalpy is untouched.
    rain_evap = _column_param(params, 'mf_rain_evap_coefficient', 0.0, t, batch)
    evaporated = torch.zeros_like(dq_tend)
    if bool(torch.any(rain_evap > 0.0)):
        layer_mass_full = dp / g
        rain_flux = torch.zeros(batch, device=t.device, dtype=t.dtype)
        for k in range(nlevels):
            rain_flux = rain_flux + (-dq_tend[:, k]).clamp(min=0.0) * layer_mass_full[:, k]
            qs_here = saturation_specific_humidity(t[:, k], p[:, k])
            subsaturation = (1.0 - q[:, k] / qs_here.clamp(min=1.0e-12)).clamp(min=0.0, max=1.0)
            take = (rain_evap * subsaturation * rain_flux).clamp(min=0.0)
            # cannot evaporate more rain than is falling, nor more than the
            # layer can hold before it saturates.
            capacity = ((qs_here - q[:, k]).clamp(min=0.0) * layer_mass_full[:, k] / model_dt)
            take = torch.minimum(torch.minimum(take, rain_flux), capacity)
            evaporated[:, k] = take / layer_mass_full[:, k]
            rain_flux = rain_flux - take
        dq_tend = dq_tend + evaporated
        dt_tend = dt_tend - (Lv / cp) * evaporated

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
        'cloud_base_mass_flux_limit': mb_limit,
        'cape_response_per_mass_flux': cape_response,
        'closure_stabilizing': closure_stabilizing,
        'mass_flux_cap_active': mb_unlimited > mb_limit,
        'temperature_cap_fraction': dt_cap_active.to(t.dtype).mean(dim=1),
        'moisture_cap_fraction': dq_cap_active.to(t.dtype).mean(dim=1),
        'mse_residual': mse_residual,
    }
