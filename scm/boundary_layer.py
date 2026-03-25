# boundary layer mixing via implicit K-diffusion.
# backward euler with tridiagonal solve, unconditionally stable.

import torch
from scm.thermo import g, cp


def boundary_layer_mixing(state, grid, params):
    """implicit vertical diffusion of temperature and moisture."""

    nlevels = grid['nlevels']
    k_diff = params.get('k_diff', 0.5)
    dt = params.get('dt', 900.0)

    t = state['t']
    q = state['q']
    p = state['p']
    dp = state['dp']

    batch = t.shape[0]
    mass = dp / g

    # mix bottom 8 levels
    mix_top = max(0, nlevels - 8)

    # diffusion coefficients at each interface
    d = torch.zeros(batch, nlevels, device=t.device)
    for k in range(mix_top, nlevels - 1):
        dp_interface = (p[:, k + 1] - p[:, k]).clamp(min=100.0)
        rho_ref = p[:, k] / (287.0 * t[:, k].clamp(min=150.0))

        if isinstance(k_diff, torch.Tensor):
            kd = k_diff
        else:
            kd = k_diff

        d[:, k] = kd * g * rho_ref * rho_ref / dp_interface

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
