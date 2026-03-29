import torch

from scm.radiation_schemes.common import (
    as_batch_tensor,
    cloud_radiative_properties,
    cp,
    forward_flux_sweep,
    g,
    mu_diff,
    sigma_sb,
    trace_gases_enabled,
)


def compute_longwave(state, grid, params):
    """Two-band longwave radiation."""

    t = state["t"]
    q = state["q"]
    ts = state["ts"]
    p = state["p"]
    dp = state["dp"]
    batch = t.shape[0]
    nlevels = t.shape[1]

    f_win = params.get("f_window", 0.15)
    kappa_wv = params.get("kappa_wv", 0.15)
    co2 = params.get("co2", 400.0)
    co2_ref = params.get("co2_ref", 400.0)
    co2_log_factor = params.get("co2_log_factor", 0.14)
    co2_base_tau = params.get("co2_base_tau", 1.5)

    def to_col(x):
        return as_batch_tensor(x, batch, t.device, t.dtype).unsqueeze(1)

    kappa_wv = to_col(kappa_wv)
    co2_log_factor = to_col(co2_log_factor)
    co2_base_tau = to_col(co2_base_tau)
    f_win_col = to_col(f_win)

    tau_wv = kappa_wv * q * dp / g

    co2_ratio = to_col(co2) / to_col(co2_ref).clamp(min=1e-6)
    co2_total_tau = co2_base_tau + co2_log_factor * torch.log(co2_ratio.clamp(min=0.01))
    tau_co2 = co2_total_tau / nlevels

    tau_trace = torch.zeros_like(tau_co2)
    if trace_gases_enabled(params):
        ch4_ratio = to_col(params.get("ch4", 1.8)) / to_col(params.get("ch4_ref", 1.8)).clamp(min=1e-6)
        n2o_ratio = to_col(params.get("n2o", 0.332)) / to_col(params.get("n2o_ref", 0.332)).clamp(min=1e-6)

        ch4_total_tau = (
            to_col(params.get("ch4_base_tau", 0.0))
            + to_col(params.get("ch4_log_factor", 0.0))
            * torch.log(ch4_ratio.clamp(min=0.01))
        )
        n2o_total_tau = (
            to_col(params.get("n2o_base_tau", 0.0))
            + to_col(params.get("n2o_log_factor", 0.0))
            * torch.log(n2o_ratio.clamp(min=0.01))
        )
        o3_lw_tau = to_col(params.get("o3_lw_tau", 0.0))
        other_ghg_tau = to_col(params.get("other_ghg_tau", 0.0))
        tau_trace = (ch4_total_tau + n2o_total_tau + o3_lw_tau + other_ghg_tau) / nlevels

    _, _, tau_cloud = cloud_radiative_properties(state, grid, params, batch, t.dtype)

    dtau = tau_wv + tau_co2 + tau_trace + tau_cloud
    transmissivity = torch.exp(-dtau * mu_diff)

    f_abs = 1.0 - f_win_col
    b_level = f_abs * sigma_sb * t ** 4

    f_win_sfc = as_batch_tensor(f_win, batch, t.device, t.dtype)
    f_abs_sfc = 1.0 - f_win_sfc
    b_surface = f_abs_sfc * sigma_sb * ts ** 4

    emission = b_level * (1.0 - transmissivity)

    f_up = forward_flux_sweep(
        transmissivity.flip(1),
        emission.flip(1),
        b_surface,
    ).flip(1)

    f_dn = forward_flux_sweep(
        transmissivity,
        emission,
        torch.zeros(batch, device=t.device, dtype=t.dtype),
    )

    f_net = f_up - f_dn
    heating = -g / cp * (f_net[:, :-1] - f_net[:, 1:]) / dp

    olr_window = f_win_sfc * sigma_sb * ts ** 4
    olr = olr_window + f_up[:, 0]
    lw_down_sfc = f_dn[:, nlevels]

    return heating, lw_down_sfc, olr


def compute_shortwave(state, grid, params):
    """Single-band shortwave with water vapor absorption."""

    t = state["t"]
    q = state["q"]
    dp = state["dp"]
    batch = t.shape[0]
    nlevels = t.shape[1]

    s0 = params.get("solar_constant", 1360.0)
    zenith_factor = params.get("zenith_factor", 0.25)
    albedo = as_batch_tensor(params.get("albedo", 0.1), batch, t.device, t.dtype)
    sw_kappa_wv = as_batch_tensor(params.get("sw_kappa_wv", 0.01), batch, t.device, t.dtype)

    toa_insolation = as_batch_tensor(s0 * zenith_factor, batch, t.device, t.dtype)
    o3_sw_layer_tau = torch.zeros(batch, device=t.device, dtype=t.dtype)
    if trace_gases_enabled(params):
        o3_sw_layer_tau = as_batch_tensor(
            params.get("o3_sw_tau", 0.0), batch, t.device, t.dtype
        ) / nlevels

    cloud_reflectivity, cloud_sw_layer_tau, _ = cloud_radiative_properties(
        state, grid, params, batch, t.dtype
    )

    sw_top = toa_insolation * (1.0 - cloud_reflectivity)
    sw_reflected_cloud = toa_insolation * cloud_reflectivity
    sw_tau = (
        sw_kappa_wv.unsqueeze(1) * q * dp / g
        + o3_sw_layer_tau.unsqueeze(1)
        + cloud_sw_layer_tau
    )
    sw_trans = torch.exp(-sw_tau)
    one = torch.ones(batch, 1, device=t.device, dtype=t.dtype)
    down_prod = torch.cat([one, torch.cumprod(sw_trans, dim=1)], dim=1)
    sw_down = sw_top.unsqueeze(1) * down_prod

    sw_absorbed_sfc = sw_down[:, nlevels] * (1.0 - albedo)
    sw_up_sfc = sw_down[:, nlevels] * albedo
    up_prod = torch.cat([torch.cumprod(sw_trans.flip(1), dim=1).flip(1), one], dim=1)
    sw_up = sw_up_sfc.unsqueeze(1) * up_prod

    sw_reflected_toa = sw_reflected_cloud + sw_up[:, 0]
    asr = toa_insolation - sw_reflected_toa

    net_sw = sw_down - sw_up
    absorbed_in_layer = net_sw[:, :-1] - net_sw[:, 1:]
    heating = g / cp * absorbed_in_layer / dp

    return heating, sw_absorbed_sfc, asr, sw_reflected_toa, toa_insolation


def run_scheme(state, grid, params):
    lw_heating, lw_down_sfc, olr = compute_longwave(state, grid, params)
    sw_heating, sw_absorbed_sfc, asr, sw_reflected_toa, toa_insolation = compute_shortwave(
        state, grid, params
    )

    lw_up_sfc = sigma_sb * state["ts"] ** 4
    toa_net = asr - olr

    return {
        "dt": lw_heating + sw_heating,
        "dq": torch.zeros_like(state["q"]),
        "lw_down_sfc": lw_down_sfc,
        "lw_up_sfc": lw_up_sfc,
        "sw_absorbed_sfc": sw_absorbed_sfc,
        "sw_reflected_toa": sw_reflected_toa,
        "toa_insolation": toa_insolation,
        "asr": asr,
        "toa_net": toa_net,
        "olr": olr,
    }
