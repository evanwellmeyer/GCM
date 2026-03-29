import torch

from scm.cloud_optics import cloud_optical_properties
from scm.radiation_schemes.common import (
    as_batch_tensor,
    band_vector,
    cp,
    forward_flux_sweep,
    g,
    mu_diff,
    sigma_sb,
    trace_total_tau,
)


def ozone_layer_profile(grid, batch, device, dtype, params):
    sigma = grid["sigma_full"].to(device=device, dtype=dtype)
    peak = as_batch_tensor(params.get("o3_peak_sigma", 0.18), batch, device, dtype)
    width = as_batch_tensor(params.get("o3_width_sigma", 0.08), batch, device, dtype).clamp(min=0.03)
    sigma_2d = sigma.unsqueeze(0).expand(batch, -1)
    weights = torch.exp(-0.5 * ((sigma_2d - peak.unsqueeze(1)) / width.unsqueeze(1)) ** 2)
    return weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0e-8)


def compute_longwave_multiband(state, grid, params, force_clear_sky=False, ozone_profile=False):
    t = state["t"]
    q = state["q"]
    ts = state["ts"]
    dp = state["dp"]
    batch, nlevels = t.shape

    device = t.device
    dtype = t.dtype
    _, _, cloud_lw_tau = cloud_optical_properties(
        state, grid, params, batch, dtype, force_clear_sky=force_clear_sky
    )

    band_weights = band_vector(
        params.get("lw_band_weights"),
        [0.18, 0.32, 0.30, 0.20],
        device, dtype,
    )
    band_weights = band_weights / band_weights.sum().clamp(min=1.0e-8)
    band_wv_kappa = band_vector(
        params.get("lw_band_wv_kappa"),
        [0.0, 0.05, 0.12, 0.22],
        device, dtype,
    )
    band_co2_base = band_vector(
        params.get("lw_band_co2_base_tau"),
        [0.0, 0.10, 0.45, 0.25],
        device, dtype,
    )
    band_co2_log = band_vector(
        params.get("lw_band_co2_log_factor"),
        [0.0, 0.01, 0.09, 0.04],
        device, dtype,
    )
    band_trace_scale = band_vector(
        params.get("lw_band_trace_scale"),
        [0.0, 0.20, 0.60, 0.20],
        device, dtype,
    )
    band_o3_scale = band_vector(
        params.get("lw_band_o3_scale"),
        [0.0, 0.20, 0.60, 0.20],
        device, dtype,
    )
    band_o3_scale = band_o3_scale / band_o3_scale.sum().clamp(min=1.0e-8)

    co2 = as_batch_tensor(params.get("co2", 400.0), batch, device, dtype).unsqueeze(1)
    co2_ref = as_batch_tensor(params.get("co2_ref", 400.0), batch, device, dtype).unsqueeze(1)
    co2_ratio = co2 / co2_ref.clamp(min=1.0e-6)
    trace_tau = trace_total_tau(batch, device, dtype, params)
    o3_lw_tau = as_batch_tensor(params.get("o3_lw_tau", 0.0), batch, device, dtype).unsqueeze(1)
    o3_profile = ozone_layer_profile(grid, batch, device, dtype, params) if ozone_profile else None
    if ozone_profile:
        trace_tau = (trace_tau - o3_lw_tau).clamp(min=0.0)

    heating = torch.zeros_like(t)
    lw_down_sfc = torch.zeros(batch, device=device, dtype=dtype)
    olr = torch.zeros(batch, device=device, dtype=dtype)

    for band in range(band_weights.shape[0]):
        tau_wv = band_wv_kappa[band] * q * dp / g
        tau_co2 = (
            band_co2_base[band]
            + band_co2_log[band] * torch.log(co2_ratio.clamp(min=0.01))
        ) / nlevels
        tau_trace = band_trace_scale[band] * trace_tau / nlevels
        if ozone_profile:
            tau_trace = tau_trace + band_o3_scale[band] * o3_lw_tau * o3_profile
        dtau = tau_wv + tau_co2 + tau_trace + cloud_lw_tau
        transmissivity = torch.exp(-dtau * mu_diff)

        b_level = band_weights[band] * sigma_sb * t ** 4
        b_surface = band_weights[band] * sigma_sb * ts ** 4
        emission = b_level * (1.0 - transmissivity)

        f_up = forward_flux_sweep(
            transmissivity.flip(1), emission.flip(1), b_surface
        ).flip(1)
        f_dn = forward_flux_sweep(
            transmissivity, emission, torch.zeros(batch, device=device, dtype=dtype)
        )

        f_net = f_up - f_dn
        heating = heating + (-g / cp * (f_net[:, :-1] - f_net[:, 1:]) / dp)
        lw_down_sfc = lw_down_sfc + f_dn[:, nlevels]
        olr = olr + f_up[:, 0]

    return heating, lw_down_sfc, olr


def compute_shortwave_multiband(state, grid, params, force_clear_sky=False, ozone_profile=False):
    t = state["t"]
    q = state["q"]
    dp = state["dp"]
    batch, nlevels = t.shape
    device = t.device
    dtype = t.dtype

    s0 = params.get("solar_constant", 1360.0)
    zenith_factor = params.get("zenith_factor", 0.25)
    albedo = as_batch_tensor(params.get("albedo", 0.1), batch, device, dtype)
    toa_insolation = as_batch_tensor(s0 * zenith_factor, batch, device, dtype)
    cloud_reflectivity, cloud_sw_tau_layer, _ = cloud_optical_properties(
        state, grid, params, batch, dtype, force_clear_sky=force_clear_sky
    )

    band_weights = band_vector(
        params.get("sw_band_weights"),
        [0.55, 0.30, 0.15],
        device, dtype,
    )
    band_weights = band_weights / band_weights.sum().clamp(min=1.0e-8)
    band_wv_kappa = band_vector(
        params.get("sw_band_wv_kappa"),
        [0.0, 0.015, 0.0],
        device, dtype,
    )
    band_o3_tau = band_vector(
        params.get("sw_band_o3_tau"),
        [0.0, 0.02, 0.10],
        device, dtype,
    )
    band_cloud_abs_scale = band_vector(
        params.get("sw_band_cloud_abs_scale"),
        [0.10, 0.25, 0.10],
        device, dtype,
    )
    o3_profile = ozone_layer_profile(grid, batch, device, dtype, params) if ozone_profile else None

    heating = torch.zeros_like(t)
    sw_absorbed_sfc = torch.zeros(batch, device=device, dtype=dtype)
    asr = torch.zeros(batch, device=device, dtype=dtype)
    sw_reflected_toa = torch.zeros(batch, device=device, dtype=dtype)
    one = torch.ones(batch, 1, device=device, dtype=dtype)

    for band in range(band_weights.shape[0]):
        band_toa = toa_insolation * band_weights[band]
        band_top = band_toa * (1.0 - cloud_reflectivity)
        if ozone_profile:
            band_o3_layer_tau = band_o3_tau[band] * o3_profile
        else:
            band_o3_layer_tau = band_o3_tau[band] / nlevels
        band_tau = (
            band_wv_kappa[band] * q * dp / g
            + band_o3_layer_tau
            + band_cloud_abs_scale[band] * cloud_sw_tau_layer
        )
        band_trans = torch.exp(-band_tau)

        down_prod = torch.cat([one, torch.cumprod(band_trans, dim=1)], dim=1)
        sw_down = band_top.unsqueeze(1) * down_prod
        sw_abs_band = sw_down[:, nlevels] * (1.0 - albedo)
        sw_up_sfc = sw_down[:, nlevels] * albedo
        up_prod = torch.cat([torch.cumprod(band_trans.flip(1), dim=1).flip(1), one], dim=1)
        sw_up = sw_up_sfc.unsqueeze(1) * up_prod

        band_reflected = band_toa * cloud_reflectivity + sw_up[:, 0]
        net_sw = sw_down - sw_up
        absorbed_in_layer = net_sw[:, :-1] - net_sw[:, 1:]

        heating = heating + g / cp * absorbed_in_layer / dp
        sw_absorbed_sfc = sw_absorbed_sfc + sw_abs_band
        sw_reflected_toa = sw_reflected_toa + band_reflected
        asr = asr + (band_toa - band_reflected)

    return heating, sw_absorbed_sfc, asr, sw_reflected_toa, toa_insolation


def run_scheme(state, grid, params, force_clear_sky=False, ozone_profile=False):
    lw_heating, lw_down_sfc, olr = compute_longwave_multiband(
        state, grid, params, force_clear_sky=force_clear_sky, ozone_profile=ozone_profile
    )
    sw_heating, sw_absorbed_sfc, asr, sw_reflected_toa, toa_insolation = (
        compute_shortwave_multiband(
            state, grid, params, force_clear_sky=force_clear_sky, ozone_profile=ozone_profile
        )
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


def run_clear_sky_scheme(state, grid, params):
    return run_scheme(state, grid, params, force_clear_sky=True)


def run_ozone_profile_scheme(state, grid, params):
    return run_scheme(state, grid, params, ozone_profile=True)


def run_ozone_profile_clear_sky_scheme(state, grid, params):
    return run_scheme(state, grid, params, force_clear_sky=True, ozone_profile=True)
