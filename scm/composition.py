# Composition and chemistry-coupling contract helpers.

import torch

from scm.surface_context import batch_param


COMPOSITION_STATE_FIELDS = (
    "co2",
    "ch4",
    "n2o",
    "o3",
    "o3_lw_tau",
    "o3_sw_tau",
    "aerosol_optical_depth",
)

CHEMISTRY_EXCHANGE_FIELDS = (
    "surface_emissions",
    "dry_deposition_velocity",
    "wet_deposition_rate",
)


def apply_composition_param_aliases(params):
    """Return params with chemistry-friendly aliases mapped to radiation names."""

    out = dict(params)
    aliases = {
        "co2_ppm": "co2",
        "ch4_ppm": "ch4",
        "n2o_ppm": "n2o",
        "ozone_lw_tau": "o3_lw_tau",
        "ozone_sw_tau": "o3_sw_tau",
    }
    for alias, canonical in aliases.items():
        if alias in out:
            out[canonical] = out[alias]

    trace_keys = ("ch4", "n2o", "o3_lw_tau", "o3_sw_tau", "other_ghg_tau")
    if "trace_gases_enabled" not in out and any(key in out for key in trace_keys):
        out["trace_gases_enabled"] = True
    return out


def _optional_numeric(params, name, like):
    if name not in params:
        return None
    value = params[name]
    if isinstance(value, str):
        return None
    try:
        return batch_param(name, value, like)
    except (TypeError, ValueError):
        return None


def composition_diagnostics(params, like):
    """Diagnostics for composition fields consumed or reserved by the SCM."""

    diag = {}
    for name in COMPOSITION_STATE_FIELDS:
        value = _optional_numeric(params, name, like)
        if value is not None:
            diag[f"composition_{name}"] = value

    for name in CHEMISTRY_EXCHANGE_FIELDS:
        value = _optional_numeric(params, name, like)
        if value is not None:
            diag[f"chemistry_{name}"] = value

    return diag
