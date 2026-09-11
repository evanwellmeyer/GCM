import torch

from scm.cloud_optics import cloud_optical_properties
from scm.thermo import eps, full_level_coordinate, p0
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


def planck_band_fractions(temperature, edges, samples=32):
    """Fraction of terrestrial blackbody emission in wavenumber bands."""
    edges = torch.as_tensor(edges, device=temperature.device, dtype=temperature.dtype)
    if edges.ndim != 1 or edges.numel() < 2 or torch.any(edges[1:] <= edges[:-1]):
        raise ValueError('lw_band_edges_cm1 must be a strictly increasing vector')
    coordinate = torch.linspace(0, 1, samples, device=temperature.device,
                                dtype=temperature.dtype)
    wavenumber = edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * coordinate
    exponent = 1.438776877 * wavenumber / temperature[..., None, None].clamp(min=100)
    density = wavenumber.pow(3) / torch.expm1(exponent).clamp(min=1e-30)
    integral = torch.trapezoid(density, wavenumber, dim=-1)
    return integral / integral.sum(dim=-1, keepdim=True).clamp(min=1e-30)


def ozone_layer_profile(grid, batch, device, dtype, params):
    sigma = full_level_coordinate(grid, batch=batch, device=device, dtype=dtype)
    peak = as_batch_tensor(params.get("o3_peak_sigma", 0.18), batch, device, dtype)
    width = as_batch_tensor(params.get("o3_width_sigma", 0.08), batch, device, dtype).clamp(min=0.03)
    weights = torch.exp(-0.5 * ((sigma - peak.unsqueeze(1)) / width.unsqueeze(1)) ** 2)
    return weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0e-8)


def compute_longwave_multiband(state, grid, params, force_clear_sky=False, ozone_profile=False):
    t = state["t"]
    q = state["q"]
    ts = state["ts"]
    dp = state["dp"]
    p_full = state["p"]
    batch, nlevels = t.shape

    device = t.device
    dtype = t.dtype
    _, _, cloud_lw_tau = cloud_optical_properties(
        state, grid, params, batch, dtype, force_clear_sky=force_clear_sky
    )

    spectral_edges = params.get('lw_band_edges_cm1')
    if spectral_edges is None:
        band_weights = band_vector(
            params.get("lw_band_weights"),
            [0.18, 0.32, 0.30, 0.20],
            device, dtype,
        )
        band_weights = band_weights / band_weights.sum().clamp(min=1.0e-8)
        level_weights = band_weights.view(1, 1, -1).expand(batch, nlevels, -1)
        surface_weights = band_weights.view(1, -1).expand(batch, -1)
    else:
        level_weights = planck_band_fractions(t, spectral_edges)
        surface_weights = planck_band_fractions(ts, spectral_edges)
        band_weights = torch.ones(
            level_weights.shape[-1], device=device, dtype=dtype)
    band_wv_kappa = band_vector(
        params.get("lw_band_wv_kappa"),
        [0.0, 0.05, 0.12, 0.22],
        device, dtype,
    )
    # Water-vapour continuum. Line absorption above is linear in specific
    # humidity, but a real atmosphere also absorbs through the continuum, whose
    # self-broadened part scales with the product of vapour amount and vapour
    # pressure -- so it grows roughly as the square of humidity and is what
    # closes the 8-12 micron window in moist air. RRTMG, used by both CESM and
    # GFDL, carries this as MT_CKD. Without it the greenhouse effect is far too
    # weak a function of humidity: a column that dries loses its trapping
    # almost in proportion, with nothing to arrest a cold-and-dry descent.
    band_wv_continuum = band_vector(
        params.get("lw_band_wv_continuum"),
        [0.0, 0.0, 0.0, 0.0],
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

    # k-distribution. Each band can be split into g-points: sub-bands that share
    # the band's Planck emission in the given fractions but carry their own
    # water-vapour and CO2 absorption strengths. Strong line centres then stay
    # opaque in dry air while weak wings stay transparent, which is how RRTMG
    # (CESM2's longwave) represents line saturation and one grey coefficient per
    # band cannot. Without `lw_gpoint_fractions` every band is a single g-point,
    # exactly as before. With it, `lw_band_wv_kappa` and `lw_band_co2_base_tau`
    # hold bands x g-points values, band by band.
    # The shares can be one set used by every band, or one set per band (bands x
    # g-points, band by band); a window band and a strong-line band need different
    # shares.
    gpoint_fractions = params.get("lw_gpoint_fractions")
    if gpoint_fractions is None:
        gpoints = 1
    else:
        nbands = band_weights.shape[0]
        gpoint_fractions = band_vector(gpoint_fractions, [1.0], device, dtype)
        gpoints = band_wv_kappa.shape[0] // nbands
        if gpoint_fractions.shape[0] == gpoints:
            gpoint_fractions = gpoint_fractions.repeat(nbands)
        if gpoints < 1 or gpoint_fractions.shape[0] != nbands * gpoints:
            raise ValueError(
                f"lw_gpoint_fractions needs {max(gpoints, 1)} or {nbands * max(gpoints, 1)} "
                f"values, got {gpoint_fractions.shape[0]}"
            )
        gpoint_fractions = gpoint_fractions.view(nbands, gpoints)
        gpoint_fractions = gpoint_fractions / gpoint_fractions.sum(dim=1, keepdim=True).clamp(min=1.0e-8)
        for name, values in (("lw_band_wv_kappa", band_wv_kappa), ("lw_band_co2_base_tau", band_co2_base)):
            if values.shape[0] != nbands * gpoints:
                raise ValueError(
                    f"{name} needs {nbands * gpoints} values "
                    f"(bands x g-points), got {values.shape[0]}"
                )

    # Pressure dependence of the water-vapour absorption strength, per g-point:
    # tau is multiplied by (p / p0) ** n. Line centres grow stronger at low
    # pressure (n < 0) and line wings weaker (n > 0); RRTMG's k-distribution
    # varies with pressure in this way. Unset, absorption does not depend on
    # pressure, exactly as before.
    wv_pressure_exponent = params.get("lw_band_wv_pressure_exponent")
    if wv_pressure_exponent is not None:
        wv_pressure_exponent = band_vector(wv_pressure_exponent, [0.0], device, dtype)
        if wv_pressure_exponent.shape[0] != band_wv_kappa.shape[0]:
            raise ValueError(
                "lw_band_wv_pressure_exponent needs one value per water-vapour "
                f"strength ({band_wv_kappa.shape[0]}), got {wv_pressure_exponent.shape[0]}"
            )

    co2 = as_batch_tensor(params.get("co2", 400.0), batch, device, dtype).unsqueeze(1)
    co2_ref = as_batch_tensor(params.get("co2_ref", 400.0), batch, device, dtype).unsqueeze(1)
    co2_ratio = co2 / co2_ref.clamp(min=1.0e-6)
    trace_tau = trace_total_tau(batch, device, dtype, params)
    o3_lw_tau = as_batch_tensor(params.get("o3_lw_tau", 0.0), batch, device, dtype).unsqueeze(1)
    o3_profile = ozone_layer_profile(grid, batch, device, dtype, params) if ozone_profile else None
    if ozone_profile:
        trace_tau = (trace_tau - o3_lw_tau).clamp(min=0.0)

    # CO2 and the trace gases are well mixed, so their optical depth belongs to
    # the mass of each layer rather than to the layer count. Dividing by nlevels
    # handed a thin near-surface layer the same absorption as a thick
    # mid-tropospheric one -- on the standard 20-level grid the bottom 5 hPa
    # layer took ten times its share -- which made the radiative answer depend
    # on how the levels happen to be distributed. Weighting by mass keeps the
    # column total identical, so the tuned band coefficients still mean the same
    # thing; only the vertical distribution changes. Ozone is deliberately left
    # out of this: it is not well mixed, and `ozone_profile` is the path that
    # gives it a proper vertical structure.
    mass_fraction = dp / dp.sum(dim=1, keepdim=True).clamp(min=1.0e-8)

    heating = torch.zeros_like(t)
    lw_down_sfc = torch.zeros(batch, device=device, dtype=dtype)
    olr = torch.zeros(batch, device=device, dtype=dtype)

    for band in range(band_weights.shape[0]):
        for point in range(gpoints):
            index = band * gpoints + point
            tau_wv = band_wv_kappa[index] * q * dp / g
            if wv_pressure_exponent is not None:
                tau_wv = tau_wv * (p_full / p0).clamp(min=1.0e-6) ** wv_pressure_exponent[index]
            # vapour pressure e = q * p / (eps + (1 - eps) q); the continuum path is
            # proportional to q * e, hence quadratic in humidity.
            vapour_pressure = q * p_full / (eps + (1.0 - eps) * q.clamp(min=0.0))
            tau_wv = tau_wv + (
                band_wv_continuum[band] * q * (vapour_pressure / p0) * dp / g
            )
            tau_co2 = (
                band_co2_base[index]
                + band_co2_log[band] * torch.log(co2_ratio.clamp(min=0.01))
            ) * mass_fraction
            tau_trace = band_trace_scale[band] * trace_tau * mass_fraction
            if ozone_profile:
                tau_trace = tau_trace + band_o3_scale[band] * o3_lw_tau * o3_profile
            dtau = tau_wv + tau_co2 + tau_trace + cloud_lw_tau
            transmissivity = torch.exp(-dtau * mu_diff)

            b_level = level_weights[:, :, band] * sigma_sb * t ** 4
            b_surface = surface_weights[:, band] * sigma_sb * ts ** 4
            if gpoints > 1:
                b_level = b_level * gpoint_fractions[band, point]
                b_surface = b_surface * gpoint_fractions[band, point]
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
