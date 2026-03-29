"""Radiation compatibility wrapper with modular scheme dispatch."""

from scm.radiation_schemes.multiband import (
    compute_longwave_multiband,
    compute_shortwave_multiband,
    run_scheme as multiband_radiation,
)
from scm.radiation_schemes.registry import (
    RADIATION_SCHEME_REGISTRY,
    available_radiation_schemes,
    radiation_scheme_descriptions,
    run_radiation_scheme,
)
from scm.radiation_schemes.semi_gray import (
    compute_longwave,
    compute_shortwave,
    run_scheme as semi_gray_radiation,
)


def radiation(state, grid, params):
    """Full radiation calculation with modular scheme dispatch."""

    return run_radiation_scheme(state, grid, params)


__all__ = [
    "RADIATION_SCHEME_REGISTRY",
    "available_radiation_schemes",
    "compute_longwave",
    "compute_longwave_multiband",
    "compute_shortwave",
    "compute_shortwave_multiband",
    "multiband_radiation",
    "radiation",
    "radiation_scheme_descriptions",
    "semi_gray_radiation",
]
