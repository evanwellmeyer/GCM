# semi-gray radiation scheme.
# longwave is split into a transparent window band and an absorbing band
# where optical depth depends on water vapor and CO2. shortwave is a single
# band attenuated by water vapor absorption.
#
# the key tuning targets for a tropical column at 300K surface temp:
#   OLR ~ 240 W/m2  (earth's global mean ~240, tropics ~250)
#   LW down at surface ~ 350-400 W/m2
#   2xCO2 forcing ~ 3-4 W/m2

import torch
from scm.thermo import sigma_sb, g, cp, Lv, saturation_specific_humidity


# diffusivity factor for converting vertical optical depth to effective
# path length through the atmosphere (accounts for non-vertical photon paths)
mu_diff = 1.66


def compute_longwave(t, q, ts, p, dp, params):
    """two-band longwave radiation. returns heating rates (batch, nlevels)
    in K/s, surface downwelling LW in W/m2, and OLR in W/m2."""

    batch = t.shape[0]
    nlevels = t.shape[1]

    f_win = params.get('f_window', 0.15)
    kappa_wv = params.get('kappa_wv', 0.15)
    co2 = params.get('co2', 400.0)
    co2_ref = params.get('co2_ref', 400.0)
    co2_log_factor = params.get('co2_log_factor', 0.14)
    co2_base_tau = params.get('co2_base_tau', 1.5)

    # ensure per-member params broadcast correctly to (batch, nlevels)
    def to_col(x):
        """reshape a scalar or (batch,) tensor to (batch, 1) for broadcasting."""
        if isinstance(x, torch.Tensor) and x.dim() == 1:
            return x.unsqueeze(1)
        return x

    kappa_wv = to_col(kappa_wv)
    co2_log_factor = to_col(co2_log_factor)
    co2_base_tau = to_col(co2_base_tau)
    f_win_col = to_col(f_win)

    # optical depth of each layer in the absorbing band
    tau_wv = kappa_wv * q * dp / g  # (batch, nlevels)

    # CO2 optical depth
    co2_ratio = co2 / co2_ref if not isinstance(co2_ref, torch.Tensor) else co2 / co2_ref
    co2_ratio_t = torch.as_tensor(co2_ratio, dtype=t.dtype, device=t.device)
    if co2_ratio_t.dim() == 1:
        co2_ratio_t = co2_ratio_t.unsqueeze(1)
    co2_total_tau = co2_base_tau + co2_log_factor * torch.log(co2_ratio_t.clamp(min=0.01))
    tau_co2 = co2_total_tau / nlevels

    dtau = tau_wv + tau_co2  # (batch, nlevels)
    transmissivity = torch.exp(-dtau * mu_diff)

    # planck emission in the absorbing band at each level
    f_abs = 1.0 - f_win_col
    b_level = f_abs * sigma_sb * t ** 4  # (batch, nlevels)

    # surface emission — need f_win as (batch,) not (batch, 1)
    f_win_sfc = f_win if not isinstance(f_win, torch.Tensor) else f_win
    f_abs_sfc = 1.0 - f_win_sfc
    b_surface = f_abs_sfc * sigma_sb * ts ** 4  # (batch,)

    # upwelling flux at interfaces, marching from surface upward.
    # interface nlevels is the surface, interface 0 is TOA.
    f_up = torch.zeros(batch, nlevels + 1, device=t.device)
    f_up[:, nlevels] = b_surface

    for k in range(nlevels - 1, -1, -1):
        f_up[:, k] = f_up[:, k + 1] * transmissivity[:, k] + b_level[:, k] * (1.0 - transmissivity[:, k])

    # downwelling flux, marching from TOA downward
    f_dn = torch.zeros(batch, nlevels + 1, device=t.device)
    f_dn[:, 0] = 0.0

    for k in range(nlevels):
        f_dn[:, k + 1] = f_dn[:, k] * transmissivity[:, k] + b_level[:, k] * (1.0 - transmissivity[:, k])

    f_net = f_up - f_dn

    heating = torch.zeros_like(t)
    for k in range(nlevels):
        heating[:, k] = -g / cp * (f_net[:, k] - f_net[:, k + 1]) / dp[:, k]

    olr_window = f_win_sfc * sigma_sb * ts ** 4
    olr = olr_window + f_up[:, 0]
    lw_down_sfc = f_dn[:, nlevels]

    return heating, lw_down_sfc, olr


def compute_shortwave(t, q, ts, p, dp, params):
    """single-band shortwave with water vapor absorption."""

    batch = t.shape[0]
    nlevels = t.shape[1]

    s0 = params.get('solar_constant', 1360.0)
    zenith_factor = params.get('zenith_factor', 0.25)
    albedo = params.get('albedo', 0.1)
    sw_kappa_wv = params.get('sw_kappa_wv', 0.01)

    toa_insolation = s0 * zenith_factor

    # sw_kappa_wv stays as (batch,) or scalar — it's used in a per-level
    # loop where q[:, k] and dp[:, k] are already (batch,)

    sw_down = torch.zeros(batch, nlevels + 1, device=t.device)
    if isinstance(toa_insolation, torch.Tensor):
        sw_down[:, 0] = toa_insolation.squeeze()
    else:
        sw_down[:, 0] = toa_insolation

    for k in range(nlevels):
        sw_tau = sw_kappa_wv * q[:, k] * dp[:, k] / g
        sw_down[:, k + 1] = sw_down[:, k] * torch.exp(-sw_tau)

    sw_absorbed_sfc = sw_down[:, nlevels] * (1.0 - albedo)

    heating = torch.zeros_like(t)
    for k in range(nlevels):
        absorbed_in_layer = sw_down[:, k] - sw_down[:, k + 1]
        heating[:, k] = g / cp * absorbed_in_layer / dp[:, k]

    return heating, sw_absorbed_sfc


def radiation(state, grid, params):
    """full radiation calculation."""

    p = state['p']
    dp = state['dp']

    lw_heating, lw_down_sfc, olr = compute_longwave(
        state['t'], state['q'], state['ts'], p, dp, params
    )
    sw_heating, sw_absorbed_sfc = compute_shortwave(
        state['t'], state['q'], state['ts'], p, dp, params
    )

    lw_up_sfc = sigma_sb * state['ts'] ** 4

    return {
        'dt': lw_heating + sw_heating,
        'dq': torch.zeros_like(state['q']),
        'lw_down_sfc': lw_down_sfc,
        'lw_up_sfc': lw_up_sfc,
        'sw_absorbed_sfc': sw_absorbed_sfc,
        'olr': olr,
    }
