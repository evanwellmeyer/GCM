# boundary layer mixing via implicit K-diffusion.
# backward euler with tridiagonal solve, unconditionally stable.

import torch
from scm.thermo import Lv, g, cp, Rd, p0, kappa, geopotential, virtual_temperature, full_level_coordinate


def boundary_layer_mixing(state, grid, params):
    """implicit vertical diffusion of temperature and moisture.

    Two modes are supported:
    - `constant`: legacy constant K-diffusion
    - `richardson`: bulk-Richardson-scaled diffusion
    """

    nlevels = grid['nlevels']
    k_diff = params.get('k_diff', 0.5)
    dt = params.get('dt', 900.0)
    scheme = params.get('boundary_layer_scheme', 'richardson')

    t = state['t']
    q = state['q']
    qc = state.get('qc', torch.zeros_like(q))
    p = state['p']
    dp = state['dp']

    batch = t.shape[0]
    mass = dp / g

    diagnose_depth = bool(params.get('bl_diagnose_depth', False))
    if diagnose_depth:
        mix_top = 0
    elif 'bl_top_sigma' in params:
        sigma = full_level_coordinate(grid, state=state, device=t.device, dtype=t.dtype)
        active = sigma[0] >= float(params['bl_top_sigma'])
        indices = torch.nonzero(active, as_tuple=False).flatten()
        mix_top = int(indices[0].item()) if indices.numel() else nlevels - 1
    else:
        mix_levels = int(params.get('bl_mix_levels', 8))
        mix_levels = max(1, min(mix_levels, nlevels))
        mix_top = max(0, nlevels - mix_levels)

    if scheme == 'constant':
        d = constant_diffusivity(state, grid, k_diff, mix_top)
    elif scheme in ['richardson', 'ri_diffusion']:
        d = richardson_diffusivity(state, grid, params, k_diff, mix_top)
    else:
        raise ValueError(f"unknown boundary layer scheme: {scheme}")

    # tridiagonal coefficients for implicit solve
    a = torch.zeros(batch, nlevels, device=t.device)
    b = torch.ones(batch, nlevels, device=t.device)
    c = torch.zeros(batch, nlevels, device=t.device)

    if mix_top < nlevels - 1:
        coeff_below = dt * g * d[:, mix_top:nlevels - 1] / mass[:, mix_top:nlevels - 1]
        c[:, mix_top:nlevels - 1] = -coeff_below
        b[:, mix_top:nlevels - 1] = b[:, mix_top:nlevels - 1] + coeff_below

        coeff_above = dt * g * d[:, mix_top:nlevels - 1] / mass[:, mix_top + 1:nlevels]
        a[:, mix_top + 1:nlevels] = -coeff_above
        b[:, mix_top + 1:nlevels] = b[:, mix_top + 1:nlevels] + coeff_above

    mix_total_water = bool(params.get('bl_mix_total_water', False))
    water = q + qc if mix_total_water else q
    water_mixed = tridiag_solve(a, b, c, water, mix_top, nlevels)
    water_rhs = water.clone()
    moisture_flux = params.get('_surface_moisture_flux', None)
    if moisture_flux is not None:
        moisture_flux = _as_batch_tensor(moisture_flux, batch, t.device, t.dtype)
        water_rhs[:, -1] = water_rhs[:, -1] + dt * moisture_flux / mass[:, -1]
    water_new = tridiag_solve(a, b, c, water_rhs, mix_top, nlevels)
    if mix_total_water:
        qc_mixed = tridiag_solve(a, b, c, qc, mix_top, nlevels)
        qc_new = qc_mixed
        q_mixed = torch.clamp(water_mixed - qc_mixed, min=0.0)
        q_new = torch.clamp(water_new - qc_new, min=0.0)
    else:
        q_mixed = water_mixed
        q_new = water_new
        qc_mixed = qc
        qc_new = qc

    if params.get('bl_mix_moist_static_energy', False):
        height = geopotential(t, q, p, grid)
        mse = cp * t + Lv * q + g * height
        mse_mixed = tridiag_solve(a, b, c, mse, mix_top, nlevels)
        mse_rhs = mse.clone()
        energy_flux = params.get('_surface_energy_flux', None)
        if energy_flux is not None:
            energy_flux = _as_batch_tensor(energy_flux, batch, t.device, t.dtype)
            mse_rhs[:, -1] = mse_rhs[:, -1] + dt * energy_flux / mass[:, -1]
        mse_new = tridiag_solve(a, b, c, mse_rhs, mix_top, nlevels)
        t_mixed = (mse_mixed - Lv * q_mixed - g * height) / cp
        t_new = (mse_new - Lv * q_new - g * height) / cp
    else:
        t_mixed = tridiag_solve(a, b, c, t, mix_top, nlevels)
        t_rhs = t.clone()
        sensible_flux = params.get('_surface_sensible_heat_flux', None)
        if sensible_flux is not None:
            sensible_flux = _as_batch_tensor(sensible_flux, batch, t.device, t.dtype)
            t_rhs[:, -1] = t_rhs[:, -1] + dt * sensible_flux / (cp * mass[:, -1])
        t_new = tridiag_solve(a, b, c, t_rhs, mix_top, nlevels)

    dt_tend = (t_new - t) / dt
    dq_tend = (q_new - q) / dt
    dqc_tend = (qc_new - qc) / dt
    surface_dt_tend = (t_new - t_mixed) / dt
    surface_dq_tend = (q_new - q_mixed) / dt
    surface_dqc_tend = (qc_new - qc_mixed) / dt

    boundary_depth = torch.zeros(batch, device=t.device, dtype=t.dtype)
    if diagnose_depth:
        wind_value = params.get(
            'relative_wind_speed_cell',
            params.get('relative_wind_speed', params.get('surface_wind_speed', params.get('wind_speed', 5.0))),
        )
        wind = _as_batch_tensor(wind_value, batch, t.device, t.dtype)
        shear_floor = _as_batch_tensor(params.get('bl_shear_floor', 1.0), batch, t.device, t.dtype)
        wind2 = wind * wind + shear_floor * shear_floor
        theta_v = virtual_temperature(t, q) * (p0 / p.clamp(min=1.0)) ** kappa
        height = geopotential(t, q, p, grid)
        ri_crit = _as_batch_tensor(params.get('ri_crit', 0.25), batch, t.device, t.dtype)
        boundary_depth = diagnose_boundary_layer_depth(height, theta_v, wind2, ri_crit, params)

    return {
        'dt': dt_tend,
        'dq': dq_tend,
        'dqc': dqc_tend,
        'mixing_dt': dt_tend - surface_dt_tend,
        'mixing_dq': dq_tend - surface_dq_tend,
        'mixing_dqc': dqc_tend - surface_dqc_tend,
        'surface_dt': surface_dt_tend,
        'surface_dq': surface_dq_tend,
        'surface_dqc': surface_dqc_tend,
        'boundary_layer_depth_m': boundary_depth,
    }


def _as_batch_tensor(x, batch, device, dtype):
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.dim() == 0:
        return t.expand(batch)
    if t.dim() == 2 and t.shape[1] == 1:
        t = t[:, 0]
    if t.dim() == 1:
        if t.shape[0] == 1:
            return t.expand(batch)
        if t.shape[0] == batch:
            return t
    raise ValueError(f"cannot broadcast BL parameter with shape {tuple(t.shape)} to batch={batch}")


def constant_diffusivity(state, grid, k_diff, mix_top):
    """Legacy constant-K interface coefficients."""

    t = state['t']
    p = state['p']
    batch, nlevels = t.shape
    d = torch.zeros(batch, nlevels, device=t.device, dtype=t.dtype)
    kd = _as_batch_tensor(k_diff, batch, t.device, t.dtype)

    if mix_top < nlevels - 1:
        p_upper = p[:, mix_top:nlevels - 1]
        p_lower = p[:, mix_top + 1:nlevels]
        dp_interface = (p_lower - p_upper).clamp(min=100.0)
        rho_ref = p_upper / (Rd * t[:, mix_top:nlevels - 1].clamp(min=150.0))
        d[:, mix_top:nlevels - 1] = kd.unsqueeze(1) * g * rho_ref * rho_ref / dp_interface

    return d


def richardson_diffusivity(state, grid, params, k_diff, mix_top):
    """Bulk-Richardson-scaled K-diffusion using the prescribed surface wind.

    The model has no momentum profile yet, so the shear term is represented
    by the prescribed near-surface wind plus a small floor. This is still a
    substantial improvement over a uniform, state-independent diffusivity:
    stable layers suppress mixing while unstable layers enhance it.
    """

    t = state['t']
    q = state['q']
    p = state['p']
    batch, nlevels = t.shape
    dtype = t.dtype
    device = t.device

    kd_base = _as_batch_tensor(k_diff, batch, device, dtype)
    kd_min = _as_batch_tensor(params.get('k_diff_min', 0.05), batch, device, dtype)
    kd_cap_factor = _as_batch_tensor(params.get('k_diff_cap_factor', 4.0), batch, device, dtype)
    ri_crit = _as_batch_tensor(params.get('ri_crit', 0.25), batch, device, dtype)
    unstable_boost = _as_batch_tensor(params.get('unstable_diffusion_boost', 4.0), batch, device, dtype)
    shear_floor = _as_batch_tensor(params.get('bl_shear_floor', 1.0), batch, device, dtype)
    # Coupled clients pass surface relative wind; standalone configurations
    # still use the historical prescribed wind_speed parameter.
    wind_value = params.get(
        'relative_wind_speed_cell',
        params.get('relative_wind_speed', params.get('surface_wind_speed', params.get('wind_speed', 5.0))),
    )
    wind = _as_batch_tensor(wind_value, batch, device, dtype)

    tv = virtual_temperature(t, q)
    theta_v = tv * (p0 / p.clamp(min=1.0)) ** kappa
    sigma_full = full_level_coordinate(grid, state=state, device=device, dtype=dtype)

    d = torch.zeros(batch, nlevels, device=device, dtype=dtype)
    wind2 = wind * wind + shear_floor * shear_floor

    if mix_top < nlevels - 1:
        p_upper = p[:, mix_top:nlevels - 1].clamp(min=1.0)
        p_lower = p[:, mix_top + 1:nlevels].clamp(min=1.0)
        dp_interface = (p_lower - p_upper).clamp(min=100.0)

        tv_mean = 0.5 * (tv[:, mix_top:nlevels - 1] + tv[:, mix_top + 1:nlevels]).clamp(min=150.0)
        dz = (Rd * tv_mean * torch.log((p_lower / p_upper).clamp(min=1.0 + 1.0e-6)) / g).clamp(min=1.0)

        theta_ref = 0.5 * (theta_v[:, mix_top:nlevels - 1] + theta_v[:, mix_top + 1:nlevels]).clamp(min=150.0)
        dtheta_v = theta_v[:, mix_top:nlevels - 1] - theta_v[:, mix_top + 1:nlevels]
        ri = g * dtheta_v * dz / (theta_ref * wind2.unsqueeze(1).clamp(min=1.0))

        stable_factor = 1.0 / (1.0 + torch.clamp(ri, min=0.0) / ri_crit.unsqueeze(1).clamp(min=1.0e-3))
        unstable_factor = 1.0 + unstable_boost.unsqueeze(1) * torch.clamp(-ri, min=0.0)
        stability_factor = torch.where(ri >= 0.0, stable_factor, unstable_factor)

        if params.get('bl_diagnose_depth', False):
            height = geopotential(t, q, p, grid)
            boundary_depth = diagnose_boundary_layer_depth(
                height, theta_v, wind2, ri_crit, params
            )
            interface_height = 0.5 * (
                height[:, mix_top:nlevels - 1] + height[:, mix_top + 1:nlevels]
            )
            depth_factor = (
                (boundary_depth.unsqueeze(1) - interface_height)
                / boundary_depth.unsqueeze(1).clamp(min=1.0)
            ).clamp(min=0.0, max=1.0) ** 2
        else:
            sigma_top = sigma_full[:, mix_top:mix_top + 1]
            sigma_interface = 0.5 * (
                sigma_full[:, mix_top:nlevels - 1] + sigma_full[:, mix_top + 1:nlevels]
            )
            depth_denominator = (1.0 - sigma_top).clamp(min=1.0e-3)
            depth_factor = ((sigma_interface - sigma_top) / depth_denominator).clamp(min=0.2, max=1.0)

        kd = kd_base.unsqueeze(1) * depth_factor * stability_factor
        kd = torch.maximum(kd, kd_min.unsqueeze(1) * depth_factor)
        kd = torch.minimum(kd, kd_base.unsqueeze(1) * kd_cap_factor.unsqueeze(1))

        rho_ref = p_upper / (Rd * t[:, mix_top:nlevels - 1].clamp(min=150.0))
        d[:, mix_top:nlevels - 1] = kd * g * rho_ref * rho_ref / dp_interface

    return d


def diagnose_boundary_layer_depth(height, theta_v, wind2, ri_crit, params):
    """Find the first bulk-Richardson crossing above the surface layer."""

    batch, nlevels = height.shape
    minimum_depth = _as_batch_tensor(
        params.get('bl_min_depth_m', 100.0), batch, height.device, height.dtype
    )
    maximum_depth = _as_batch_tensor(
        params.get('bl_max_depth_m', 1200.0), batch, height.device, height.dtype
    )
    surface_theta = theta_v[:, -1]
    bulk_ri = (
        g * height * (theta_v - surface_theta.unsqueeze(1))
        / (surface_theta.unsqueeze(1).clamp(min=150.0) * wind2.unsqueeze(1).clamp(min=1.0))
    )

    boundary_depth = maximum_depth.clone()
    unresolved = torch.ones(batch, device=height.device, dtype=torch.bool)
    for upper in range(nlevels - 2, -1, -1):
        lower = upper + 1
        crossing = unresolved & (bulk_ri[:, upper] >= ri_crit)
        ri_span = (bulk_ri[:, upper] - bulk_ri[:, lower]).clamp(min=1.0e-8)
        fraction = ((ri_crit - bulk_ri[:, lower]) / ri_span).clamp(min=0.0, max=1.0)
        crossing_height = height[:, lower] + fraction * (height[:, upper] - height[:, lower])
        boundary_depth = torch.where(crossing, crossing_height, boundary_depth)
        unresolved = unresolved & ~crossing

    lower_bound = torch.minimum(minimum_depth, maximum_depth)
    upper_bound = torch.maximum(minimum_depth, maximum_depth)
    return torch.maximum(torch.minimum(boundary_depth, upper_bound), lower_bound)


def tridiag_solve(a, b, c, rhs, k_start, k_end):
    """thomas algorithm for tridiagonal system."""

    result = rhs.clone()
    cc = c.clone()
    dd = result.clone()
    bb = b.clone()

    for k in range(k_start + 1, k_end):
        if bb[:, k - 1].abs().min() < 1e-30:
            continue
        w = a[:, k] / bb[:, k - 1]
        bb[:, k] = bb[:, k] - w * cc[:, k - 1]
        dd[:, k] = dd[:, k] - w * dd[:, k - 1]

    result[:, k_end - 1] = dd[:, k_end - 1] / bb[:, k_end - 1].clamp(min=1e-30)
    for k in range(k_end - 2, k_start - 1, -1):
        result[:, k] = (dd[:, k] - cc[:, k] * result[:, k + 1]) / bb[:, k].clamp(min=1e-30)

    return result
