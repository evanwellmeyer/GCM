# Surface-context contract helpers for host/dycore coupling.

import torch


SURFACE_STATIC_FIELDS = (
    "surface_type",
    "land_fraction",
    "ocean_fraction",
    "sea_ice_fraction",
    "glacier_fraction",
    "land_use_type",
    "soil_type",
    "topography",
)

SURFACE_STATE_FIELDS = (
    "surface_temperature",
    "soil_moisture",
    "soil_temperature",
    "snow_water_equivalent",
    "sea_ice_thickness",
)

SURFACE_EXCHANGE_FIELDS = (
    "albedo",
    "surface_albedo",
    "roughness_length",
    "exchange_coefficient_heat",
    "exchange_coefficient_moisture",
    "surface_emissions",
)

SURFACE_OUTPUT_FIELDS = (
    "shf",
    "lhf",
    "precip_total",
    "evaporation_mass_flux",
    "runoff_rate",
    "drainage_rate",
    "sw_absorbed_sfc",
    "lw_down_sfc",
    "lw_up_sfc",
)


def batch_param(name, value, like):
    tensor = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    batch = like.shape[0]
    if tensor.dim() == 0:
        return tensor.expand(batch)
    tensor = tensor.reshape(-1)
    if tensor.numel() == 1:
        return tensor.expand(batch)
    if tensor.numel() == batch:
        return tensor
    raise ValueError(f"{name} must be scalar or length batch={batch}, got {tuple(tensor.shape)}")


def first_present(mapping, names, default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def apply_surface_param_aliases(params):
    """Return params with canonical aliases filled for radiation/surface code."""

    out = dict(params)
    if "surface_albedo" in out:
        out["albedo"] = out["surface_albedo"]
    if "surface_albedo" not in out and "albedo" in out:
        out["surface_albedo"] = out["albedo"]
    return out


def apply_surface_state_inputs(state, params):
    """Copy host-provided surface state fields into the SCM state dictionary."""

    ref = torch.as_tensor(state["ts"])
    for name in SURFACE_STATE_FIELDS:
        if name not in params:
            continue
        value = batch_param(name, params[name], ref)
        if name == "surface_temperature":
            state["ts"] = value.to(dtype=state["ts"].dtype)
        else:
            state[name] = value
    return state


def surface_temperature(state, params):
    """Surface temperature boundary condition used by radiation and fluxes."""

    ref = torch.as_tensor(state["ts"])
    value = first_present(params, ("surface_temperature", "skin_temperature"), None)
    if value is not None:
        return batch_param("surface_temperature", value, ref)
    if "surface_temperature" in state:
        return batch_param("surface_temperature", state["surface_temperature"], ref)
    return ref


def surface_fractions(params, like):
    """Return bounded surface fractions for land/ocean/ice/glacier categories."""

    land = batch_param("land_fraction", params.get("land_fraction", 0.0), like).clamp(0.0, 1.0)
    sea_ice = batch_param("sea_ice_fraction", params.get("sea_ice_fraction", 0.0), like).clamp(0.0, 1.0)
    glacier = batch_param("glacier_fraction", params.get("glacier_fraction", 0.0), like).clamp(0.0, 1.0)
    if "ocean_fraction" in params:
        ocean = batch_param("ocean_fraction", params["ocean_fraction"], like).clamp(0.0, 1.0)
    else:
        ocean = (1.0 - land - sea_ice - glacier).clamp(0.0, 1.0)

    total = land + ocean + sea_ice + glacier
    scale = torch.where(total > 1.0, total.clamp_min(1.0e-8), torch.ones_like(total))
    return {
        "land_fraction": land / scale,
        "ocean_fraction": ocean / scale,
        "sea_ice_fraction": sea_ice / scale,
        "glacier_fraction": glacier / scale,
        "surface_fraction_sum": total,
    }


def exchange_coefficients(params, like):
    """Heat and moisture exchange coefficients with roughness fallback."""

    cd_default = batch_param("cd", params.get("cd", 1.2e-3), like).clamp_min(0.0)
    if "roughness_length" in params and (
        "exchange_coefficient_heat" not in params
        and "exchange_coefficient_moisture" not in params
    ):
        z0 = batch_param("roughness_length", params["roughness_length"], like).clamp_min(1.0e-6)
        z_ref = batch_param("surface_reference_height", params.get("surface_reference_height", 10.0), like)
        cd_default = (0.40 / torch.log((z_ref / z0).clamp_min(1.01))).square().clamp(0.0, 0.02)

    cd_heat = batch_param(
        "exchange_coefficient_heat",
        params.get("exchange_coefficient_heat", cd_default),
        like,
    ).clamp_min(0.0)
    cd_moist = batch_param(
        "exchange_coefficient_moisture",
        params.get("exchange_coefficient_moisture", params.get("exchange_coefficient_water", cd_heat)),
        like,
    ).clamp_min(0.0)
    return cd_heat, cd_moist


def optional_numeric_param(params, name, like):
    if name not in params:
        return None
    value = params[name]
    if isinstance(value, str):
        return None
    try:
        return batch_param(name, value, like)
    except (TypeError, ValueError):
        return None


def surface_context_diagnostics(state, params, like):
    diag = surface_fractions(params, like)
    cd_heat, cd_moist = exchange_coefficients(params, like)
    diag["exchange_coefficient_heat"] = cd_heat
    diag["exchange_coefficient_moisture"] = cd_moist

    albedo = first_present(params, ("surface_albedo", "albedo"), None)
    if albedo is not None:
        diag["surface_albedo"] = batch_param("surface_albedo", albedo, like).clamp(0.0, 1.0)

    for name in (
        "roughness_length",
        "surface_type",
        "land_use_type",
        "soil_type",
        "topography",
        "soil_temperature",
        "snow_water_equivalent",
        "sea_ice_thickness",
    ):
        value = optional_numeric_param(params, name, like)
        if value is None and name in state:
            value = batch_param(name, state[name], like)
        if value is not None:
            diag[name] = value

    return diag
