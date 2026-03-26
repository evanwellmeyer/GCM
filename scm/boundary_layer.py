# boundary layer mixing via implicit K-diffusion.
# backward euler with tridiagonal solve, unconditionally stable.

import torch
from scm.thermo import g, cp, Rd, p0, kappa, virtual_temperature


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
    p = state['p']
    dp = state['dp']

    batch = t.shape[0]
    mass = dp / g

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

    for k in range(mix_top, nlevels):
        if k > mix_top:
            coeff_above = dt * g * d[:, k - 1] / mass[:, k]
            a[:, k] = -coeff_above
            b[:, k] = b[:, k] + coeff_above
        if k < nlevels - 1:
            coeff_below = dt * g * d[:, k] / mass[:, k]
            c[:, k] = -coeff_below
            b[:, k] = b[:, k] + coeff_below

    t_new = tridiag_solve(a, b, c, t, mix_top, nlevels)
    q_new = tridiag_solve(a, b, c, q, mix_top, nlevels)

    dt_tend = (t_new - t) / dt
    dq_tend = (q_new - q) / dt

    return {
        'dt': dt_tend,
        'dq': dq_tend,
    }


def _as_batch_tensor(x, batch, device, dtype):
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.dim() == 0:
        return t.expand(batch)
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

    for k in range(mix_top, nlevels - 1):
        dp_interface = (p[:, k + 1] - p[:, k]).clamp(min=100.0)
        rho_ref = p[:, k] / (Rd * t[:, k].clamp(min=150.0))
        d[:, k] = kd * g * rho_ref * rho_ref / dp_interface

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
    wind = _as_batch_tensor(params.get('wind_speed', 5.0), batch, device, dtype)

    tv = virtual_temperature(t, q)
    theta_v = tv * (p0 / p.clamp(min=1.0)) ** kappa
    sigma_full = grid['sigma_full'].to(device=device, dtype=dtype)
    sigma_top = sigma_full[mix_top]

    d = torch.zeros(batch, nlevels, device=device, dtype=dtype)
    wind2 = wind * wind + shear_floor * shear_floor

    for k in range(mix_top, nlevels - 1):
        p_upper = p[:, k].clamp(min=1.0)
        p_lower = p[:, k + 1].clamp(min=1.0)
        dp_interface = (p_lower - p_upper).clamp(min=100.0)

        tv_mean = 0.5 * (tv[:, k] + tv[:, k + 1]).clamp(min=150.0)
        dz = (Rd * tv_mean * torch.log((p_lower / p_upper).clamp(min=1.0 + 1.0e-6)) / g).clamp(min=1.0)

        theta_ref = 0.5 * (theta_v[:, k] + theta_v[:, k + 1]).clamp(min=150.0)
        dtheta_v = theta_v[:, k] - theta_v[:, k + 1]
        ri = g * dtheta_v * dz / (theta_ref * wind2.clamp(min=1.0))

        stable_factor = 1.0 / (1.0 + torch.clamp(ri, min=0.0) / ri_crit.clamp(min=1.0e-3))
        unstable_factor = 1.0 + unstable_boost * torch.clamp(-ri, min=0.0)
        stability_factor = torch.where(ri >= 0.0, stable_factor, unstable_factor)

        sigma_interface = 0.5 * (sigma_full[k] + sigma_full[k + 1])
        depth_factor = ((sigma_interface - sigma_top) / max(1.0 - float(sigma_top), 1.0e-3))
        depth_factor = torch.clamp(torch.as_tensor(depth_factor, device=device, dtype=dtype), min=0.2, max=1.0)

        kd = kd_base * depth_factor * stability_factor
        kd = torch.maximum(kd, kd_min * depth_factor)
        kd = torch.minimum(kd, kd_base * kd_cap_factor)

        rho_ref = p_upper / (Rd * t[:, k].clamp(min=150.0))
        d[:, k] = kd * g * rho_ref * rho_ref / dp_interface

    return d


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
