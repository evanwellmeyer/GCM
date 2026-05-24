"""Registry for SCM physics schemes and named suites.

The column step still runs physics in its established order. This module keeps
scheme selection in one place so new parameterizations can be added without
turning the timestep loop into a long list of special cases.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from scm.boundary_layer import boundary_layer_mixing
from scm.condensation import condensation
from scm.convection_bm import betts_miller
from scm.convection_mf import mass_flux_convection
from scm.convection_shallow import shallow_convection
from scm.radiation import radiation
from scm.radiation_schemes.registry import available_radiation_schemes
from scm.surface import surface_fluxes


@dataclass(frozen=True)
class PhysicsScheme:
    category: str
    name: str
    runner: Callable
    description: str
    aliases: tuple = ()


@dataclass(frozen=True)
class PhysicsSuite:
    name: str
    defaults: dict
    description: str


_SCHEMES = {}
_ALIASES = {}
_SUITES = {}


def _zero_column(state):
    return torch.zeros_like(state["t"])


def _zero_surface(state):
    return torch.zeros(state["t"].shape[0], device=state["t"].device, dtype=state["t"].dtype)


def _copy_with(params, **updates):
    out = dict(params)
    out.update(updates)
    return out


def _radiation_runner(name):
    return lambda state, grid, params: radiation(state, grid, _copy_with(params, radiation_scheme=name))


def _boundary_layer_runner(name):
    return lambda state, grid, params: boundary_layer_mixing(
        state, grid, _copy_with(params, boundary_layer_scheme=name)
    )


def _zero_shallow(state, grid, params):
    del grid, params
    return {"dt": _zero_column(state), "dq": _zero_column(state), "precip": _zero_surface(state)}


def _zero_convection(state, grid, params):
    del grid, params
    return {"dt": _zero_column(state), "dq": _zero_column(state), "precip": _zero_surface(state)}


def _zero_condensation(state, grid, params):
    del grid, params
    zeros = _zero_column(state)
    return {"dt": zeros, "dq": zeros, "precip": _zero_surface(state), "cloud_source": zeros}


def register_physics_scheme(category, name, runner, description="", aliases=()):
    scheme = PhysicsScheme(category, name, runner, description, tuple(aliases))
    _SCHEMES.setdefault(category, {})[name] = scheme
    for alias in aliases:
        _ALIASES[(category, alias)] = name
    return scheme


def register_physics_suite(name, defaults, description=""):
    suite = PhysicsSuite(name, dict(defaults), description)
    _SUITES[name] = suite
    return suite


def available_physics_schemes(category=None):
    if category is not None:
        return sorted(_SCHEMES.get(category, {}))
    return {name: sorted(schemes) for name, schemes in sorted(_SCHEMES.items())}


def available_physics_suites():
    return sorted(_SUITES)


def physics_suite_components(name):
    try:
        return dict(_SUITES[name].defaults)
    except KeyError as exc:
        valid = ", ".join(available_physics_suites())
        raise ValueError(f"unknown physics suite: {name} (valid: {valid})") from exc


def apply_physics_suite_defaults(params):
    suite_name = params.get("physics_suite", "mass_flux_default")
    defaults = physics_suite_components(suite_name)
    out = dict(defaults)
    out.update(params)
    return out


def run_physics_scheme(category, name, state, grid, params):
    canonical = _ALIASES.get((category, name), name)
    try:
        scheme = _SCHEMES[category][canonical]
    except KeyError as exc:
        valid = ", ".join(available_physics_schemes(category))
        raise ValueError(f"unknown {category} scheme: {name} (valid: {valid})") from exc
    return scheme.runner(state, grid, params)


for _name in available_radiation_schemes():
    register_physics_scheme(
        "radiation",
        _name,
        _radiation_runner(_name),
        "Radiation scheme from scm.radiation_schemes.",
    )

register_physics_scheme("surface", "bulk_aero", surface_fluxes, "Bulk aerodynamic surface fluxes.", aliases=("bulk",))

register_physics_scheme(
    "boundary_layer",
    "richardson",
    _boundary_layer_runner("richardson"),
    "Bulk-Richardson-scaled implicit K diffusion.",
    aliases=("ri_diffusion",),
)
register_physics_scheme(
    "boundary_layer",
    "constant",
    _boundary_layer_runner("constant"),
    "Legacy constant-K implicit diffusion.",
)

register_physics_scheme("shallow_convection", "simple", shallow_convection, "Simple shallow convective adjustment.")
register_physics_scheme("shallow_convection", "none", _zero_shallow, "Disable shallow convection.")

register_physics_scheme(
    "convection",
    "mass_flux",
    mass_flux_convection,
    "Mass-flux deep convection.",
    aliases=("mf",),
)
register_physics_scheme(
    "convection",
    "betts_miller",
    betts_miller,
    "Betts-Miller relaxation convection.",
    aliases=("bm", "betts-miller"),
)
register_physics_scheme("convection", "none", _zero_convection, "Disable deep convection.")

register_physics_scheme("condensation", "large_scale", condensation, "Large-scale saturation adjustment.")
register_physics_scheme("condensation", "none", _zero_condensation, "Disable large-scale condensation.")

register_physics_suite(
    "mass_flux_default",
    {
        "radiation_scheme": "semi_gray",
        "surface_scheme": "bulk_aero",
        "boundary_layer_scheme": "richardson",
        "shallow_convection_scheme": "simple",
        "convection_scheme": "mass_flux",
        "condensation_scheme": "large_scale",
    },
    "Default forward-development suite using mass-flux deep convection.",
)
register_physics_suite(
    "legacy_betts_miller",
    {
        "radiation_scheme": "semi_gray",
        "surface_scheme": "bulk_aero",
        "boundary_layer_scheme": "richardson",
        "shallow_convection_scheme": "simple",
        "convection_scheme": "betts_miller",
        "condensation_scheme": "large_scale",
    },
    "Legacy/control suite preserving Betts-Miller convection.",
)
register_physics_suite(
    "clear_sky_mass_flux",
    {
        "radiation_scheme": "semi_gray_clear_sky",
        "surface_scheme": "bulk_aero",
        "boundary_layer_scheme": "richardson",
        "shallow_convection_scheme": "simple",
        "convection_scheme": "mass_flux",
        "condensation_scheme": "large_scale",
    },
    "Mass-flux suite with clear-sky radiation diagnostics path.",
)
