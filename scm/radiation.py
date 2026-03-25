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


def _as_batch_tensor(x, batch, device, dtype):
    """broadcast scalar or batch-like input to shape (batch,)."""

    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.dim() == 0:
        return t.expand(batch)
    if t.dim() == 1:
        if t.shape[0] == 1:
            return t.expand(batch)
        if t.shape[0] == batch:
            return t
    raise ValueError(f"cannot broadcast value with shape {tuple(t.shape)} to batch={batch}")


def _trace_gases_enabled(params):
    mode = params.get('radiation_mode', 'semi_gray')
    return bool(params.get('trace_gases_enabled', False)) or mode == 'semi_gray_plus_trace_gases'


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
        return _as_batch_tensor(x, batch, t.device, t.dtype).unsqueeze(1)

    kappa_wv = to_col(kappa_wv)
    co2_log_factor = to_col(co2_log_factor)
    co2_base_tau = to_col(co2_base_tau)
    f_win_col = to_col(f_win)

    # optical depth of each layer in the absorbing band
    tau_wv = kappa_wv * q * dp / g  # (batch, nlevels)

    # CO2 optical depth
    co2_ratio = to_col(co2) / to_col(co2_ref).clamp(min=1e-6)
    co2_ratio_t = co2_ratio
    co2_total_tau = co2_base_tau + co2_log_factor * torch.log(co2_ratio_t.clamp(min=0.01))
    tau_co2 = co2_total_tau / nlevels

    tau_trace = torch.zeros_like(tau_co2)
    if _trace_gases_enabled(params):
        ch4_ratio = to_col(params.get('ch4', 1.8)) / to_col(params.get('ch4_ref', 1.8)).clamp(min=1e-6)
        n2o_ratio = to_col(params.get('n2o', 0.332)) / to_col(params.get('n2o_ref', 0.332)).clamp(min=1e-6)

        ch4_total_tau = (
            to_col(params.get('ch4_base_tau', 0.0))
            + to_col(params.get('ch4_log_factor', 0.0))
            * torch.log(ch4_ratio.clamp(min=0.01))
        )
        n2o_total_tau = (
            to_col(params.get('n2o_base_tau', 0.0))
            + to_col(params.get('n2o_log_factor', 0.0))
            * torch.log(n2o_ratio.clamp(min=0.01))
        )
        o3_lw_tau = to_col(params.get('o3_lw_tau', 0.0))
        other_ghg_tau = to_col(params.get('other_ghg_tau', 0.0))
        tau_trace = (ch4_total_tau + n2o_total_tau + o3_lw_tau + other_ghg_tau) / nlevels

    dtau = tau_wv + tau_co2 + tau_trace  # (batch, nlevels)
    transmissivity = torch.exp(-dtau * mu_diff)

    # planck emission in the absorbing band at each level
    f_abs = 1.0 - f_win_col
    b_level = f_abs * sigma_sb * t ** 4  # (batch, nlevels)

    # surface emission — use (batch,) not (batch, 1)
    f_win_sfc = _as_batch_tensor(f_win, batch, t.device, t.dtype)
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
    sw_kappa_wv = _as_batch_tensor(params.get('sw_kappa_wv', 0.01), batch, t.device, t.dtype)

    toa_insolation = s0 * zenith_factor
    o3_sw_layer_tau = torch.zeros(batch, device=t.device, dtype=t.dtype)
    if _trace_gases_enabled(params):
        o3_sw_layer_tau = _as_batch_tensor(
            params.get('o3_sw_tau', 0.0), batch, t.device, t.dtype
        ) / nlevels

    # sw_kappa_wv stays as (batch,) or scalar — it's used in a per-level
    # loop where q[:, k] and dp[:, k] are already (batch,)

    sw_down = torch.zeros(batch, nlevels + 1, device=t.device)
    if isinstance(toa_insolation, torch.Tensor):
        sw_down[:, 0] = toa_insolation.squeeze()
    else:
        sw_down[:, 0] = toa_insolation

    for k in range(nlevels):
        sw_tau = sw_kappa_wv * q[:, k] * dp[:, k] / g + o3_sw_layer_tau
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
