# betts-miller convective adjustment.
# relaxes toward a moist adiabat for temperature and toward rhbm * qs
# for moisture, within the convective layer (where the adiabat is buoyant).
# convection dries the boundary layer and moistens the free troposphere,
# with net column drying producing precipitation.

import torch
from scm.thermo import (
    cp, Lv, g, cape, saturation_specific_humidity,
    moist_adiabat_profile
)


def betts_miller(state, grid, params):
    """betts-miller convective adjustment."""

    t = state['t']
    q = state['q']
    p = state['p']
    dp = state['dp']

    tau_bm = params.get('tau_bm', 7200.0)
    rhbm = params.get('rhbm', 0.7)
    cape_threshold = params.get('cape_threshold', 50.0)
    timestep = params.get('dt', 900.0)
    max_dt_day = params.get('bm_max_dt_day', 10.0)
    max_dq_day = params.get('bm_max_dq_day', 5.0)

    batch = t.shape[0]
    nlevels = t.shape[1]

    cape_val = cape(t, q, p, grid)
    cape_excess = torch.clamp(cape_val - cape_threshold, min=0.0)
    activation = (cape_excess > 0.0).to(t.dtype)

    if isinstance(tau_bm, torch.Tensor):
        relax = (timestep / tau_bm).clamp(max=0.8)
    else:
        relax = min(timestep / tau_bm, 0.8)

    # reference temperature: moist adiabat from boundary layer
    t_base = t[:, -1]
    p_base = p[:, -1]
    t_ref = moist_adiabat_profile(t_base, p_base, p)

    # convective layer mask: where the adiabat is warmer than environment
    buoyancy = t_ref - t
    conv_mask = (buoyancy > 0.0).to(t.dtype)

    # reference moisture at current temperature
    qs_current = saturation_specific_humidity(t, p)

    if isinstance(rhbm, torch.Tensor):
        rhbm_val = rhbm.unsqueeze(1) if rhbm.dim() == 1 else rhbm
    else:
        rhbm_val = rhbm

    q_ref = rhbm_val * qs_current

    # tendencies: convection dries the moist BL and moistens the dry
    # free troposphere, both toward rhbm * qs
    dt_raw = (t_ref - t) * conv_mask
    dq_raw = (q_ref - q) * conv_mask

    # ensure net column drying: cap moistening at 80% of drying
    moistening = torch.clamp(dq_raw, min=0.0) * dp / g
    drying = torch.clamp(dq_raw, max=0.0) * dp / g
    total_moist = moistening.sum(dim=1, keepdim=True).clamp(min=1e-10)
    total_dry = (-drying).sum(dim=1, keepdim=True).clamp(min=1e-10)
    scale = torch.clamp(0.8 * total_dry / total_moist, max=1.0)
    dq_raw = torch.where(dq_raw > 0, dq_raw * scale, dq_raw)

    # apply relaxation and activation
    if isinstance(relax, torch.Tensor):
        relax_broad = relax.unsqueeze(1)
    else:
        relax_broad = relax

    dt_tend = relax_broad * dt_raw * activation.unsqueeze(1)
    dq_tend = relax_broad * dq_raw * activation.unsqueeze(1)

    # limit tendencies
    max_dt = max_dt_day / 86400.0 * timestep
    max_dq = max_dq_day * 1.0e-3 / 86400.0 * timestep
    dt_tend = dt_tend.clamp(-max_dt, max_dt)
    dq_tend = dq_tend.clamp(-max_dq, max_dq)

    # precipitation: net moisture removal
    precip = (-torch.sum(dq_tend * dp / g, dim=1)).clamp(min=0.0) / timestep

    return {
        'dt': dt_tend / timestep,
        'dq': dq_tend / timestep,
        'precip': precip,
        'cape': cape_val,
    }
