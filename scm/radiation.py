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
from scm.thermo import sigma_sb, g, cp, saturation_specific_humidity


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


def _clouds_enabled(params):
    mode = params.get('radiation_mode', 'semi_gray')
    return bool(params.get('cloud_microphysics_enabled', False)) or bool(
        params.get('cloud_radiative_effects_enabled', False)
    ) or (
        mode == 'semi_gray_plus_clouds'
        or mode == 'semi_gray_plus_trace_gases_clouds'
    )


def _cloud_layer_weights(grid, batch, device, dtype, params):
    """return normalized cloud-layer weights on model full levels."""

    sigma = grid['sigma_full'].to(device=device, dtype=dtype)
    top = _as_batch_tensor(params.get('cloud_top_sigma', 0.65), batch, device, dtype)
    bottom = _as_batch_tensor(params.get('cloud_bottom_sigma', 0.95), batch, device, dtype)
    sigma_2d = sigma.unsqueeze(0).expand(batch, -1)
    mask = ((sigma_2d >= top.unsqueeze(1)) & (sigma_2d <= bottom.unsqueeze(1))).to(dtype)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return mask / counts


def _forward_flux_sweep(transmissivity, source, boundary):
    """Vectorized solution of y[k+1] = y[k] * transmissivity[k] + source[k]."""

    batch = transmissivity.shape[0]
    one = torch.ones(batch, 1, device=transmissivity.device, dtype=transmissivity.dtype)
    zero = torch.zeros(batch, 1, device=transmissivity.device, dtype=transmissivity.dtype)
    prefix = torch.cat([one, torch.cumprod(transmissivity, dim=1)], dim=1)
    scaled_source = source / prefix[:, 1:].clamp(min=1.0e-12)
    accum = torch.cumsum(scaled_source, dim=1)
    return prefix * (boundary.unsqueeze(1) + torch.cat([zero, accum], dim=1))


def _band_vector(values, default, device, dtype):
    v = torch.as_tensor(default if values is None else values, device=device, dtype=dtype)
    if v.dim() == 0:
        return v.unsqueeze(0)
    if v.dim() != 1:
        raise ValueError(f"band parameters must be 1D, got shape {tuple(v.shape)}")
    return v


def _cloud_radiative_properties(state, grid, params, batch, dtype):
    device = state['t'].device
    nlevels = state['t'].shape[1]

    if params.get('cloud_microphysics_enabled', False):
        cf = state.get('cloud_fraction', torch.zeros(batch, nlevels, device=device, dtype=dtype))
        sw_tau_layer = state.get('cloud_sw_tau_layer', torch.zeros(batch, nlevels, device=device, dtype=dtype))
        lw_tau_layer = state.get('cloud_lw_tau_layer', torch.zeros(batch, nlevels, device=device, dtype=dtype))
        cf = cf.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
        sw_tau_layer = sw_tau_layer.to(device=device, dtype=dtype)
        lw_tau_layer = lw_tau_layer.to(device=device, dtype=dtype)

        cloud_cover = 1.0 - torch.prod(1.0 - cf, dim=1)
        scatter_eff = _as_batch_tensor(
            params.get('cloud_sw_scattering_efficiency', 0.18),
            batch, device, dtype
        )
        reflectivity = cloud_cover * (1.0 - torch.exp(-scatter_eff * sw_tau_layer.sum(dim=1)))
        abs_frac = _as_batch_tensor(
            params.get('cloud_sw_absorption_fraction', 0.15),
            batch, device, dtype
        )
        return reflectivity.clamp(min=0.0, max=0.95), abs_frac.unsqueeze(1) * sw_tau_layer, lw_tau_layer

    if not _clouds_enabled(params):
        zeros_layer = torch.zeros(batch, nlevels, device=device, dtype=dtype)
        zeros_col = torch.zeros(batch, device=device, dtype=dtype)
        return zeros_col, zeros_layer, zeros_layer

    cloud_reflectivity = (
        _as_batch_tensor(params.get('cloud_fraction', 0.0), batch, device, dtype).clamp(min=0.0, max=1.0)
        * _as_batch_tensor(params.get('cloud_sw_reflectivity', 0.0), batch, device, dtype)
    ).clamp(min=0.0, max=0.95)
    cloud_sw_tau_total = (
        _as_batch_tensor(params.get('cloud_fraction', 0.0), batch, device, dtype).clamp(min=0.0, max=1.0)
        * _as_batch_tensor(params.get('cloud_sw_tau', 0.0), batch, device, dtype).clamp(min=0.0)
    )
    cloud_lw_tau_total = (
        _as_batch_tensor(params.get('cloud_fraction', 0.0), batch, device, dtype).clamp(min=0.0, max=1.0)
        * _as_batch_tensor(params.get('cloud_lw_tau', 0.0), batch, device, dtype).clamp(min=0.0)
    )
    weights = _cloud_layer_weights(grid, batch, device, dtype, params)
    return (
        cloud_reflectivity,
        cloud_sw_tau_total.unsqueeze(1) * weights,
        cloud_lw_tau_total.unsqueeze(1) * weights,
    )


def compute_longwave(state, grid, params):
    """two-band longwave radiation. returns heating rates (batch, nlevels)
    in K/s, surface downwelling LW in W/m2, and OLR in W/m2."""

    t = state['t']
    q = state['q']
    ts = state['ts']
    p = state['p']
    dp = state['dp']
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

    _, _, tau_cloud = _cloud_radiative_properties(state, grid, params, batch, t.dtype)

    dtau = tau_wv + tau_co2 + tau_trace + tau_cloud  # (batch, nlevels)
    transmissivity = torch.exp(-dtau * mu_diff)

    # planck emission in the absorbing band at each level
    f_abs = 1.0 - f_win_col
    b_level = f_abs * sigma_sb * t ** 4  # (batch, nlevels)

    # surface emission — use (batch,) not (batch, 1)
    f_win_sfc = _as_batch_tensor(f_win, batch, t.device, t.dtype)
    f_abs_sfc = 1.0 - f_win_sfc
    b_surface = f_abs_sfc * sigma_sb * ts ** 4  # (batch,)

    emission = b_level * (1.0 - transmissivity)

    # upwelling flux at interfaces. interface nlevels is the surface,
    # interface 0 is TOA.
    f_up = _forward_flux_sweep(
        transmissivity.flip(1),
        emission.flip(1),
        b_surface,
    ).flip(1)

    # downwelling flux, marching from TOA downward
    f_dn = _forward_flux_sweep(
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
    """single-band shortwave with water vapor absorption."""

    t = state['t']
    q = state['q']
    dp = state['dp']
    batch = t.shape[0]
    nlevels = t.shape[1]

    s0 = params.get('solar_constant', 1360.0)
    zenith_factor = params.get('zenith_factor', 0.25)
    albedo = _as_batch_tensor(params.get('albedo', 0.1), batch, t.device, t.dtype)
    sw_kappa_wv = _as_batch_tensor(params.get('sw_kappa_wv', 0.01), batch, t.device, t.dtype)

    toa_insolation = _as_batch_tensor(s0 * zenith_factor, batch, t.device, t.dtype)
    o3_sw_layer_tau = torch.zeros(batch, device=t.device, dtype=t.dtype)
    if _trace_gases_enabled(params):
        o3_sw_layer_tau = _as_batch_tensor(
            params.get('o3_sw_tau', 0.0), batch, t.device, t.dtype
        ) / nlevels

    cloud_reflectivity, cloud_sw_layer_tau, _ = _cloud_radiative_properties(
        state, grid, params, batch, t.dtype
    )

    # sw_kappa_wv stays as (batch,) or scalar — it's used in a per-level
    # loop where q[:, k] and dp[:, k] are already (batch,)

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


def _trace_total_tau(batch, device, dtype, params):
    if not _trace_gases_enabled(params):
        return torch.zeros(batch, 1, device=device, dtype=dtype)

    def to_col(x):
        return _as_batch_tensor(x, batch, device, dtype).unsqueeze(1)

    ch4_ratio = to_col(params.get('ch4', 1.8)) / to_col(params.get('ch4_ref', 1.8)).clamp(min=1.0e-6)
    n2o_ratio = to_col(params.get('n2o', 0.332)) / to_col(params.get('n2o_ref', 0.332)).clamp(min=1.0e-6)

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
    return ch4_total_tau + n2o_total_tau + o3_lw_tau + other_ghg_tau


def compute_longwave_multiband(state, grid, params):
    t = state['t']
    q = state['q']
    ts = state['ts']
    p = state['p']
    dp = state['dp']
    batch, nlevels = t.shape

    device = t.device
    dtype = t.dtype
    _, _, cloud_lw_tau = _cloud_radiative_properties(state, grid, params, batch, dtype)

    band_weights = _band_vector(
        params.get('lw_band_weights'),
        [0.18, 0.32, 0.30, 0.20],
        device, dtype,
    )
    band_weights = band_weights / band_weights.sum().clamp(min=1.0e-8)
    band_wv_kappa = _band_vector(
        params.get('lw_band_wv_kappa'),
        [0.0, 0.05, 0.12, 0.22],
        device, dtype,
    )
    band_co2_base = _band_vector(
        params.get('lw_band_co2_base_tau'),
        [0.0, 0.10, 0.45, 0.25],
        device, dtype,
    )
    band_co2_log = _band_vector(
        params.get('lw_band_co2_log_factor'),
        [0.0, 0.01, 0.09, 0.04],
        device, dtype,
    )
    band_trace_scale = _band_vector(
        params.get('lw_band_trace_scale'),
        [0.0, 0.20, 0.60, 0.20],
        device, dtype,
    )

    co2 = _as_batch_tensor(params.get('co2', 400.0), batch, device, dtype).unsqueeze(1)
    co2_ref = _as_batch_tensor(params.get('co2_ref', 400.0), batch, device, dtype).unsqueeze(1)
    co2_ratio = co2 / co2_ref.clamp(min=1.0e-6)
    trace_total_tau = _trace_total_tau(batch, device, dtype, params)

    heating = torch.zeros_like(t)
    lw_down_sfc = torch.zeros(batch, device=device, dtype=dtype)
    olr = torch.zeros(batch, device=device, dtype=dtype)

    for band in range(band_weights.shape[0]):
        tau_wv = band_wv_kappa[band] * q * dp / g
        tau_co2 = (
            band_co2_base[band]
            + band_co2_log[band] * torch.log(co2_ratio.clamp(min=0.01))
        ) / nlevels
        tau_trace = band_trace_scale[band] * trace_total_tau / nlevels
        dtau = tau_wv + tau_co2 + tau_trace + cloud_lw_tau
        transmissivity = torch.exp(-dtau * mu_diff)

        b_level = band_weights[band] * sigma_sb * t ** 4
        b_surface = band_weights[band] * sigma_sb * ts ** 4
        emission = b_level * (1.0 - transmissivity)

        f_up = _forward_flux_sweep(
            transmissivity.flip(1), emission.flip(1), b_surface
        ).flip(1)
        f_dn = _forward_flux_sweep(
            transmissivity, emission, torch.zeros(batch, device=device, dtype=dtype)
        )

        f_net = f_up - f_dn
        heating = heating + (-g / cp * (f_net[:, :-1] - f_net[:, 1:]) / dp)
        lw_down_sfc = lw_down_sfc + f_dn[:, nlevels]
        olr = olr + f_up[:, 0]

    return heating, lw_down_sfc, olr


def compute_shortwave_multiband(state, grid, params):
    t = state['t']
    q = state['q']
    dp = state['dp']
    batch, nlevels = t.shape
    device = t.device
    dtype = t.dtype

    s0 = params.get('solar_constant', 1360.0)
    zenith_factor = params.get('zenith_factor', 0.25)
    albedo = _as_batch_tensor(params.get('albedo', 0.1), batch, device, dtype)
    toa_insolation = _as_batch_tensor(s0 * zenith_factor, batch, device, dtype)
    cloud_reflectivity, cloud_sw_tau_layer, _ = _cloud_radiative_properties(
        state, grid, params, batch, dtype
    )

    band_weights = _band_vector(
        params.get('sw_band_weights'),
        [0.55, 0.30, 0.15],
        device, dtype,
    )
    band_weights = band_weights / band_weights.sum().clamp(min=1.0e-8)
    band_wv_kappa = _band_vector(
        params.get('sw_band_wv_kappa'),
        [0.0, 0.015, 0.0],
        device, dtype,
    )
    band_o3_tau = _band_vector(
        params.get('sw_band_o3_tau'),
        [0.0, 0.02, 0.10],
        device, dtype,
    )
    band_cloud_abs_scale = _band_vector(
        params.get('sw_band_cloud_abs_scale'),
        [0.10, 0.25, 0.10],
        device, dtype,
    )

    heating = torch.zeros_like(t)
    sw_absorbed_sfc = torch.zeros(batch, device=device, dtype=dtype)
    asr = torch.zeros(batch, device=device, dtype=dtype)
    sw_reflected_toa = torch.zeros(batch, device=device, dtype=dtype)
    one = torch.ones(batch, 1, device=device, dtype=dtype)

    for band in range(band_weights.shape[0]):
        band_toa = toa_insolation * band_weights[band]
        band_top = band_toa * (1.0 - cloud_reflectivity)
        band_tau = (
            band_wv_kappa[band] * q * dp / g
            + band_o3_tau[band] / nlevels
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


def semi_gray_radiation(state, grid, params):
    """semi-gray radiation with optional trace-gas and cloud extensions."""

    lw_heating, lw_down_sfc, olr = compute_longwave(state, grid, params)
    sw_heating, sw_absorbed_sfc, asr, sw_reflected_toa, toa_insolation = compute_shortwave(
        state, grid, params
    )

    lw_up_sfc = sigma_sb * state['ts'] ** 4
    toa_net = asr - olr

    return {
        'dt': lw_heating + sw_heating,
        'dq': torch.zeros_like(state['q']),
        'lw_down_sfc': lw_down_sfc,
        'lw_up_sfc': lw_up_sfc,
        'sw_absorbed_sfc': sw_absorbed_sfc,
        'sw_reflected_toa': sw_reflected_toa,
        'toa_insolation': toa_insolation,
        'asr': asr,
        'toa_net': toa_net,
        'olr': olr,
    }


def multiband_radiation(state, grid, params):
    """Multi-band grey-gas radiation with optional trace gases and cloud coupling."""

    lw_heating, lw_down_sfc, olr = compute_longwave_multiband(state, grid, params)
    sw_heating, sw_absorbed_sfc, asr, sw_reflected_toa, toa_insolation = (
        compute_shortwave_multiband(state, grid, params)
    )

    lw_up_sfc = sigma_sb * state['ts'] ** 4
    toa_net = asr - olr

    return {
        'dt': lw_heating + sw_heating,
        'dq': torch.zeros_like(state['q']),
        'lw_down_sfc': lw_down_sfc,
        'lw_up_sfc': lw_up_sfc,
        'sw_absorbed_sfc': sw_absorbed_sfc,
        'sw_reflected_toa': sw_reflected_toa,
        'toa_insolation': toa_insolation,
        'asr': asr,
        'toa_net': toa_net,
        'olr': olr,
    }


def radiation(state, grid, params):
    """full radiation calculation with scheme dispatch."""

    scheme = params.get('radiation_scheme', 'semi_gray')
    if scheme == 'semi_gray':
        return semi_gray_radiation(state, grid, params)
    if scheme == 'multiband':
        return multiband_radiation(state, grid, params)
    raise ValueError(f"unknown radiation scheme: {scheme}")
