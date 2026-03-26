# surface fluxes and slab ocean.

import torch
from scm.thermo import cp, Lv, g, rho_water, c_water, saturation_specific_humidity

rho_air = 1.2


def slab_heat_capacity(params):
    depth = params.get('ocean_depth', 50.0)
    return rho_water * c_water * depth


def surface_fluxes(state, grid, params):
    """bulk aerodynamic surface fluxes."""

    cd = params.get('cd', 1.2e-3)
    wind = params.get('wind_speed', 5.0)

    t_lowest = state['t'][:, -1]
    q_lowest = state['q'][:, -1]
    ts = state['ts']
    p_lowest = state['p'][:, -1]

    qs_sfc = saturation_specific_humidity(ts, p_lowest)

    shf = rho_air * cp * cd * wind * (ts - t_lowest)
    lhf = rho_air * Lv * cd * wind * (qs_sfc - q_lowest)
    lhf = torch.clamp(lhf, min=0.0)

    # By default the sensible-heat flux is distributed over the lower
    # boundary layer and the latent-heat moisture source over the shallowest
    # surface layers. The layer counts are configurable because the richer
    # radiation/cloud path can be sensitive to how strongly the lowest model
    # level is coupled to the slab surface.
    nlevels = state['t'].shape[1]
    heat_levels = int(params.get('surface_heat_levels', 8))
    moist_levels = int(params.get('surface_moisture_levels', 3))
    heat_levels = max(1, min(heat_levels, nlevels))
    moist_levels = max(1, min(moist_levels, nlevels))

    total_mass = torch.zeros_like(t_lowest)
    for i in range(heat_levels):
        k = nlevels - 1 - i
        total_mass = total_mass + state['dp'][:, k] / g

    moist_mass = torch.zeros_like(t_lowest)
    for i in range(moist_levels):
        k = nlevels - 1 - i
        moist_mass = moist_mass + state['dp'][:, k] / g

    dt_uniform = shf / (total_mass * cp)
    dq_uniform = lhf / (moist_mass * Lv)

    dt = torch.zeros_like(state['t'])
    dq = torch.zeros_like(state['q'])
    for i in range(heat_levels):
        k = nlevels - 1 - i
        dt[:, k] = dt_uniform
    for i in range(moist_levels):
        k = nlevels - 1 - i
        dq[:, k] = dq_uniform

    return {
        'dt': dt,
        'dq': dq,
        'shf': shf,
        'lhf': lhf,
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
