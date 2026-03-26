import torch

from scm.thermo import Lv, cp, cape, relative_humidity


def shallow_convection(state, grid, params):
    """Conservative lower-tropospheric convective mixing.

    This is a simple shallow-convection closure, not a plume model. It moves
    moisture and moist static energy from the subcloud layer into the lower
    free troposphere while conserving column-integrated water and moist
    enthalpy across the shallow-convection layer. It does not precipitate.
    """

    t = state['t']
    q = state['q']
    p = state['p']
    dp = state['dp']
    batch = t.shape[0]
    dtype = t.dtype
    device = t.device
    zeros_t = torch.zeros_like(t)
    zeros_s = torch.zeros(batch, device=device, dtype=dtype)

    if not params.get('shallow_convection_enabled', False):
        return {'dt': zeros_t, 'dq': zeros_t, 'precip': zeros_s}

    sigma = grid['sigma_full'].to(device=device, dtype=dtype)
    top_sigma = float(params.get('shallow_top_sigma', 0.75))
    base_sigma = float(params.get('shallow_base_sigma', 0.90))
    low_mask_1d = (sigma >= base_sigma)
    up_mask_1d = (sigma >= top_sigma) & (sigma < base_sigma)

    if not low_mask_1d.any() or not up_mask_1d.any():
        return {'dt': zeros_t, 'dq': zeros_t, 'precip': zeros_s}

    low_mask = low_mask_1d.unsqueeze(0).to(dtype)
    up_mask = up_mask_1d.unsqueeze(0).to(dtype)
    mass = dp / 9.81
    low_mass = (low_mask * mass).sum(dim=1).clamp(min=1.0e-8)
    up_mass = (up_mask * mass).sum(dim=1).clamp(min=1.0e-8)

    h = cp * t + Lv * q
    rh = relative_humidity(q, t, p).clamp(min=0.0)

    q_low = (low_mask * q * mass).sum(dim=1) / low_mass
    h_low = (low_mask * h * mass).sum(dim=1) / low_mass
    q_up = (up_mask * q * mass).sum(dim=1) / up_mass
    h_up = (up_mask * h * mass).sum(dim=1) / up_mass
    rh_low = (low_mask * rh * mass).sum(dim=1) / low_mass

    rh_trigger = float(params.get('shallow_rh_trigger', 0.80))
    mse_scale = float(params.get('shallow_mse_scale', 4000.0))
    cape_suppress = float(params.get('shallow_cape_suppress', 500.0))
    tau = max(float(params.get('shallow_tau', 14400.0)), 1.0)
    dt = float(params.get('dt', 900.0))

    rh_factor = torch.clamp((rh_low - rh_trigger) / max(1.0 - rh_trigger, 1.0e-3), min=0.0, max=1.0)
    q_factor = torch.clamp((q_low - q_up) / q_low.clamp(min=1.0e-7), min=0.0, max=1.0)
    h_factor = torch.clamp((h_low - h_up) / max(mse_scale, 1.0), min=0.0, max=1.0)
    deep_cape = cape(t, q, p, grid)
    cape_factor = 1.0 / (1.0 + deep_cape / max(cape_suppress, 1.0))
    strength = rh_factor * q_factor * h_factor * cape_factor

    relax = min(dt / tau, 1.0)
    q_low_col = q_low.unsqueeze(1)
    h_low_col = h_low.unsqueeze(1)
    strength_col = strength.unsqueeze(1)

    dq_up_step = relax * strength_col * up_mask * (q_low_col - q).clamp(min=0.0)
    dh_up_step = relax * strength_col * up_mask * (h_low_col - h).clamp(min=0.0)

    added_q = (dq_up_step * mass).sum(dim=1)
    added_h = (dh_up_step * mass).sum(dim=1)
    dq_low_step = -(added_q / low_mass).unsqueeze(1) * low_mask
    dh_low_step = -(added_h / low_mass).unsqueeze(1) * low_mask

    dq_step = dq_up_step + dq_low_step
    dh_step = dh_up_step + dh_low_step

    # Keep the scheme weak and bounded; this is a supplement to BL mixing.
    max_dq = float(params.get('shallow_max_dq_day', 2.0)) * 1.0e-3 * dt / 86400.0
    max_dt = float(params.get('shallow_max_dt_day', 2.0)) * dt / 86400.0
    dq_step = dq_step.clamp(min=-max_dq, max=max_dq)
    dt_step = ((dh_step - Lv * dq_step) / cp).clamp(min=-max_dt, max=max_dt)

    return {
        'dt': dt_step / dt,
        'dq': dq_step / dt,
        'precip': zeros_s,
    }
