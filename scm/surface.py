# surface fluxes and slab ocean.

import torch
from scm.thermo import (
    cp, Lv, g, rho_water, c_water, saturation_specific_humidity,
    half_level_coordinate,
)
from scm.land_surface import land_fraction, land_latent_heat_cap, soil_evaporation_beta
from scm.surface_context import batch_param, exchange_coefficients, surface_fractions, surface_temperature

rho_air = 1.2

def slab_heat_capacity(params):
    depth = params.get('ocean_depth', 50.0)
    if torch.is_tensor(depth):
        dtype = depth.dtype if depth.device.type == 'mps' else torch.float64
        return (rho_water * c_water * depth).to(dtype=dtype)
    return rho_water * c_water * depth


def surface_fluxes(state, grid, params):
    """bulk aerodynamic surface fluxes."""

    t_lowest = state['t'][:, -1]
    q_lowest = state['q'][:, -1]
    ts = surface_temperature(state, params)
    p_lowest = state['p'][:, -1]
    cd_heat, cd_moist = exchange_coefficients(params, ts)
    wind_value = params.get(
        'relative_wind_speed_cell',
        params.get('relative_wind_speed', params.get('surface_wind_speed', params.get('wind_speed', 5.0))),
    )
    # Coupled runs supply air-ocean relative wind per column; standalone SCM
    # configs keep the legacy prescribed wind_speed path.
    # Use ts as the dtype reference so the legacy scalar path keeps its
    # historical promotion behavior when standalone slab temperatures are fp64.
    wind = batch_param('wind_speed', wind_value, ts).clamp(min=0.0)

    qs_sfc = saturation_specific_humidity(ts, p_lowest)

    shf = rho_air * cp * cd_heat * wind * (ts - t_lowest)
    potential_lhf = rho_air * Lv * cd_moist * wind * (qs_sfc - q_lowest)
    potential_lhf = torch.clamp(potential_lhf, min=0.0)

    fractions = surface_fractions(params, ts)
    frac_land = land_fraction(params, ts)
    beta_land = soil_evaporation_beta(state, params, ts)
    land_lhf = torch.minimum(potential_lhf * beta_land, land_latent_heat_cap(state, params, ts))
    ocean_lhf = potential_lhf
    lhf = (1.0 - frac_land) * ocean_lhf + frac_land * land_lhf

    # By default the sensible-heat flux is distributed over the lower
    # boundary layer and the latent-heat moisture source over the shallowest
    # surface layers. The layer counts are configurable because the richer
    # radiation/cloud path can be sensitive to how strongly the lowest model
    # level is coupled to the slab surface.
    nlevels = state['t'].shape[1]
    sigma_half = half_level_coordinate(grid, state=state)

    def surface_layer_weights(depth_name, levels_name, default_levels):
        if depth_name in params:
            depth = max(0.0, min(float(params[depth_name]), 1.0))
            cutoff = 1.0 - depth
            span = (sigma_half[:, 1:] - sigma_half[:, :-1]).clamp(min=1.0e-8)
            overlap = (
                sigma_half[:, 1:] - torch.maximum(
                    sigma_half[:, :-1], torch.as_tensor(cutoff, device=t_lowest.device)
                )
            ).clamp(min=0.0)
            return (overlap / span).clamp(max=1.0)

        levels = int(params.get(levels_name, default_levels))
        levels = max(1, min(levels, nlevels))
        weights = torch.zeros_like(state['t'])
        weights[:, -levels:] = 1.0
        return weights

    heat_weights = surface_layer_weights(
        'surface_heat_sigma_depth', 'surface_heat_levels', 8
    )
    moist_weights = surface_layer_weights(
        'surface_moisture_sigma_depth', 'surface_moisture_levels', 3
    )
    layer_mass = state['dp'] / g
    heat_mass = (heat_weights * layer_mass).sum(dim=1).clamp(min=1.0e-8)
    moist_mass = (moist_weights * layer_mass).sum(dim=1).clamp(min=1.0e-8)

    dt = heat_weights * (shf / (heat_mass * cp)).unsqueeze(1)
    dq = moist_weights * (lhf / (moist_mass * Lv)).unsqueeze(1)

    return {
        'dt': dt,
        'dq': dq,
        'shf': shf,
        'lhf': lhf,
        'lhf_potential': potential_lhf,
        'land_lhf': land_lhf,
        'ocean_lhf': ocean_lhf,
        'land_fraction': frac_land,
        'ocean_fraction': fractions['ocean_fraction'],
        'sea_ice_fraction': fractions['sea_ice_fraction'],
        'glacier_fraction': fractions['glacier_fraction'],
        'exchange_coefficient_heat': cd_heat,
        'exchange_coefficient_moisture': cd_moist,
        'soil_evap_beta': beta_land,
    }


def slab_ocean_tendency(state, rad_output, sfc_output, params, precip_heat_flux=None):
    heat_capacity = slab_heat_capacity(params)
    if precip_heat_flux is None:
        precip_heat_flux = torch.zeros_like(state['ts'])

    net_flux = (
        rad_output['sw_absorbed_sfc']
        + rad_output['lw_down_sfc']
        - rad_output['lw_up_sfc']
        - sfc_output['shf']
        - sfc_output['lhf']
        + precip_heat_flux
    )

    return net_flux / heat_capacity
