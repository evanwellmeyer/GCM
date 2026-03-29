from . import multiband, semi_gray


RADIATION_SCHEME_REGISTRY = {
    "semi_gray": semi_gray.run_scheme,
    "multiband": multiband.run_scheme,
}

RADIATION_SCHEME_DESCRIPTIONS = {
    "semi_gray": "Two-band semi-gray radiation with optional trace-gas and cloud extensions.",
    "multiband": "Multi-band grey-gas radiation with optional trace gases and cloud coupling.",
}


def available_radiation_schemes():
    return sorted(RADIATION_SCHEME_REGISTRY.keys())


def radiation_scheme_descriptions():
    return dict(RADIATION_SCHEME_DESCRIPTIONS)


def run_radiation_scheme(state, grid, params):
    scheme = params.get("radiation_scheme", "semi_gray")
    try:
        scheme_fn = RADIATION_SCHEME_REGISTRY[scheme]
    except KeyError as exc:
        valid = ", ".join(available_radiation_schemes())
        raise ValueError(f"unknown radiation scheme: {scheme} (valid: {valid})") from exc
    return scheme_fn(state, grid, params)
