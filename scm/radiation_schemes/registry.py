from . import multiband, semi_gray


RADIATION_SCHEME_REGISTRY = {
    "semi_gray": semi_gray.run_scheme,
    "semi_gray_all_sky": semi_gray.run_scheme,
    "semi_gray_clear_sky": semi_gray.run_clear_sky_scheme,
    "multiband": multiband.run_scheme,
    "multiband_all_sky": multiband.run_scheme,
    "multiband_clear_sky": multiband.run_clear_sky_scheme,
    "multiband_ozone_profile": multiband.run_ozone_profile_scheme,
    "multiband_ozone_profile_all_sky": multiband.run_ozone_profile_scheme,
    "multiband_ozone_profile_clear_sky": multiband.run_ozone_profile_clear_sky_scheme,
}

RADIATION_SCHEME_DESCRIPTIONS = {
    "semi_gray": "Two-band semi-gray radiation with optional trace-gas and cloud extensions.",
    "semi_gray_all_sky": "Explicit all-sky alias for the semi-gray scheme.",
    "semi_gray_clear_sky": "Semi-gray clear-sky radiation with cloud optics disabled.",
    "multiband": "Multi-band grey-gas radiation with optional trace gases and cloud coupling.",
    "multiband_all_sky": "Explicit all-sky alias for the multi-band grey-gas scheme.",
    "multiband_clear_sky": "Multi-band grey-gas radiation with cloud optics disabled.",
    "multiband_ozone_profile": "Multi-band grey-gas radiation with ozone concentrated in a stratospheric profile.",
    "multiband_ozone_profile_all_sky": "Explicit all-sky alias for the profiled-ozone multi-band scheme.",
    "multiband_ozone_profile_clear_sky": "Profiled-ozone multi-band radiation with cloud optics disabled.",
}


def available_radiation_schemes():
    return sorted(RADIATION_SCHEME_REGISTRY.keys())


def radiation_scheme_descriptions():
    return dict(RADIATION_SCHEME_DESCRIPTIONS)


def clear_sky_partner_scheme(scheme):
    if scheme.startswith("semi_gray"):
        return "semi_gray_clear_sky"
    if scheme.startswith("multiband_ozone_profile"):
        return "multiband_ozone_profile_clear_sky"
    if scheme.startswith("multiband"):
        return "multiband_clear_sky"
    valid = ", ".join(available_radiation_schemes())
    raise ValueError(f"unknown radiation scheme: {scheme} (valid: {valid})")


def run_radiation_scheme(state, grid, params):
    scheme = params.get("radiation_scheme", "semi_gray")
    try:
        scheme_fn = RADIATION_SCHEME_REGISTRY[scheme]
    except KeyError as exc:
        valid = ", ".join(available_radiation_schemes())
        raise ValueError(f"unknown radiation scheme: {scheme} (valid: {valid})") from exc
    result = scheme_fn(state, grid, params)

    if params.get("radiation_clear_sky_diagnostics", False):
        clear_scheme = clear_sky_partner_scheme(scheme)
        if clear_scheme == scheme:
            clear_result = result
        else:
            clear_result = RADIATION_SCHEME_REGISTRY[clear_scheme](state, grid, params)
        result = dict(result)
        result.update({
            "clear_sky_olr": clear_result["olr"],
            "clear_sky_asr": clear_result["asr"],
            "clear_sky_toa_net": clear_result["toa_net"],
            "cloud_lw_cre": clear_result["olr"] - result["olr"],
            "cloud_sw_cre": result["asr"] - clear_result["asr"],
            "cloud_toa_cre": result["toa_net"] - clear_result["toa_net"],
        })

    return result
