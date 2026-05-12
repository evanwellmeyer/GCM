# Surface System Plan

This repository currently treats the SCM as the atmospheric column physics
package. Land, sea ice, and ocean state should grow as coupled surface
components around that column, not as hidden internals of convection,
radiation, or the dycore.

## Ownership

The host model or dycore owns grid geometry and static surface identity:

- land fraction
- ocean fraction
- sea-ice fraction
- land-use type
- soil type
- topography
- bathymetry

The surface physics owns what those fields do physically:

- albedo effects
- aerodynamic exchange
- evapotranspiration limits
- runoff and drainage
- snow/ice insulation
- freshwater and salt fluxes
- surface chemistry and emissions later

Restartable surface state belongs with the physics state:

- soil moisture
- soil temperature
- canopy water
- snow water equivalent
- sea-ice thickness
- sea-ice surface temperature

## Component Organization

```text
Land model
  soil moisture
  runoff/drainage
  soil temperature
  vegetation/land use
  snow on land

Sea-ice model
  ice fraction
  ice thickness
  surface temperature
  albedo
  insulation between ocean and atmosphere
  brine/salt/freshwater effects later

Ocean model
  mixed-layer/full ocean state
  receives heat, freshwater, momentum from atmosphere/ice

Surface coupler
  decides what fraction of each column is ocean/land/ice
  blends fluxes back to atmosphere/ocean/land/ice
```

## Current First Step

`scm/land_surface.py` now implements a minimal one-layer bucket:

- `soil_moisture` is meters of liquid-water equivalent over land.
- `land_fraction` controls how much of a grid cell uses land-surface logic.
- dry land throttles latent heat flux through `soil_evap_beta`.
- precipitation refills the bucket after the atmospheric step.
- excess water becomes `runoff_rate`.
- optional drainage can be enabled with `soil_drainage_timescale`.

The default remains ocean-like because `land_fraction = 0.0` unless the caller
sets it. A standalone all-land column can set:

```toml
[land_surface]
land_fraction = 1.0
soil_water_capacity = 0.15
soil_moisture_initial_fraction = 0.75
soil_wilting_fraction = 0.10
soil_evap_critical_fraction = 0.50
soil_field_capacity_fraction = 0.85
soil_drainage_timescale = 2592000.0
```

## Near-Term Steps

1. Add a standalone land-column config that sets `land_fraction = 1.0` and
   verifies soil moisture dries down under net evaporation.
2. Add land-use parameter lookup for albedo, roughness, rooting depth, and
   evapotranspiration resistance.
3. Split the current surface temperature reservoir into ocean and land thermal
   reservoirs, so land heat capacity is not represented by the slab-ocean depth.
4. Add snow-on-land as a land-surface extension with snow water equivalent,
   snow albedo, melt, and refreeze.
5. Add a separate sea-ice component for atmosphere/ocean coupling over icy
   ocean cells.
6. Route dycore-provided surface identity fields through the VFS adapter:
   `land_fraction`, `ocean_fraction`, `sea_ice_fraction`, `land_use_type`,
   `soil_type`, and topographic fields.
