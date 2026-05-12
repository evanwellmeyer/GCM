# Minimal land-surface water bucket for column coupling.

import torch

from scm.thermo import Lv, rho_water
from scm.surface_context import batch_param, surface_fractions


def _batch_param(name, value, like):
    return batch_param(name, value, like)


def land_fraction(params, like):
    """Return grid-cell land fraction as a batch vector."""

    return surface_fractions(params, like)["land_fraction"]


def soil_water_capacity(params, like):
    """Maximum one-layer bucket water storage in meters of liquid water."""

    return _batch_param("soil_water_capacity", params.get("soil_water_capacity", 0.15), like).clamp_min(1.0e-8)


def initialize_land_state(batch, params, device="cpu", dtype=None):
    """Create restartable land-state fields.

    ``soil_moisture`` is stored as meters of liquid-water equivalent over the
    land part of a grid cell. It is inert when ``land_fraction`` is zero.
    """

    ref = torch.empty(int(batch), device=device, dtype=dtype or torch.get_default_dtype())
    capacity = soil_water_capacity(params, ref)
    initial_fraction = _batch_param(
        "soil_moisture_initial_fraction",
        params.get("soil_moisture_initial_fraction", 0.75),
        ref,
    ).clamp(0.0, 1.0)
    return {"soil_moisture": initial_fraction * capacity}


def soil_evaporation_beta(state, params, like):
    """Moisture-availability factor that limits land latent heat flux."""

    capacity = soil_water_capacity(params, like)
    default_soil = (
        _batch_param("soil_moisture_initial_fraction", params.get("soil_moisture_initial_fraction", 0.75), like)
        .clamp(0.0, 1.0)
        * capacity
    )
    soil = torch.as_tensor(
        state.get("soil_moisture", default_soil),
        device=like.device,
        dtype=like.dtype,
    ).reshape(-1)
    wilting = _batch_param("soil_wilting_fraction", params.get("soil_wilting_fraction", 0.10), like)
    critical = _batch_param("soil_evap_critical_fraction", params.get("soil_evap_critical_fraction", 0.50), like)
    rel = (soil / capacity).clamp(0.0, 1.0)
    denom = (critical - wilting).clamp_min(1.0e-6)
    return ((rel - wilting) / denom).clamp(0.0, 1.0)


def land_latent_heat_cap(state, params, like):
    """Maximum land latent heat flux allowed by current bucket water."""

    dt = _batch_param("dt", params.get("dt", 900.0), like).clamp_min(1.0e-6)
    capacity = soil_water_capacity(params, like)
    default_soil = (
        _batch_param("soil_moisture_initial_fraction", params.get("soil_moisture_initial_fraction", 0.75), like)
        .clamp(0.0, 1.0)
        * capacity
    )
    soil = torch.as_tensor(
        state.get("soil_moisture", default_soil),
        device=like.device,
        dtype=like.dtype,
    ).reshape(-1)
    return soil.clamp_min(0.0) * rho_water * Lv / dt


def update_soil_bucket(state, params, precip_rate, land_lhf, dt):
    """Update one-layer soil moisture after atmospheric precipitation is known.

    ``precip_rate`` is kg m-2 s-1 over the atmospheric column. ``land_lhf`` is
    the land-area latent heat flux in W m-2 before land-fraction weighting.
    Diagnostics returned here are grid-cell mean rates, so they are weighted by
    ``land_fraction``.
    """

    ref = torch.as_tensor(state["ts"])
    precip_rate = torch.as_tensor(precip_rate, device=ref.device, dtype=ref.dtype).reshape(-1).clamp_min(0.0)
    land_lhf = torch.as_tensor(land_lhf, device=ref.device, dtype=ref.dtype).reshape(-1).clamp_min(0.0)
    dt_tensor = torch.as_tensor(dt, device=ref.device, dtype=ref.dtype).clamp_min(1.0e-6)
    frac = land_fraction(params, ref)
    capacity = soil_water_capacity(params, ref)
    field_capacity = (
        _batch_param("soil_field_capacity_fraction", params.get("soil_field_capacity_fraction", 0.85), ref)
        .clamp(0.0, 1.0)
        * capacity
    )

    if "soil_moisture" in params:
        state["soil_moisture"] = _batch_param("soil_moisture", params["soil_moisture"], ref)
    elif "soil_moisture" not in state:
        state.update(initialize_land_state(ref.shape[0], params, device=ref.device, dtype=ref.dtype))

    soil = torch.as_tensor(state["soil_moisture"], device=ref.device, dtype=ref.dtype).reshape(-1)
    precip_depth = precip_rate * dt_tensor / rho_water
    evap_depth = land_lhf * dt_tensor / (Lv * rho_water)
    evap_depth = torch.minimum(evap_depth, (soil + precip_depth).clamp_min(0.0))

    raw_soil = soil + precip_depth - evap_depth
    runoff_depth = (raw_soil - capacity).clamp_min(0.0)
    after_runoff = torch.minimum(raw_soil, capacity)

    drainage_timescale = _batch_param(
        "soil_drainage_timescale",
        params.get("soil_drainage_timescale", 0.0),
        ref,
    )
    drainage_factor = torch.where(
        drainage_timescale > 0.0,
        torch.clamp(dt_tensor / drainage_timescale.clamp_min(1.0e-6), max=1.0),
        torch.zeros_like(drainage_timescale),
    )
    drainage_depth = torch.minimum((after_runoff - field_capacity).clamp_min(0.0) * drainage_factor, after_runoff)

    updated = (after_runoff - drainage_depth).clamp_min(0.0)
    land_mask = frac > 0.0
    update_enabled = params.get("soil_moisture_update_enabled", "soil_moisture" not in params)
    if update_enabled:
        state["soil_moisture"] = torch.where(land_mask, updated, soil).to(dtype=ref.dtype)

    return {
        "land_fraction": frac,
        "soil_moisture": state["soil_moisture"],
        "soil_moisture_fraction": (state["soil_moisture"] / capacity).clamp(0.0, 1.0),
        "soil_precipitation_rate": precip_rate * frac,
        "soil_evaporation_rate": land_lhf / Lv * frac,
        "runoff_rate": runoff_depth * rho_water / dt_tensor * frac,
        "drainage_rate": drainage_depth * rho_water / dt_tensor * frac,
    }
