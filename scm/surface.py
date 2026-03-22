# surface fluxes and slab ocean.

import torch
from scm.thermo import cp, Lv, g, rho_water, c_water, saturation_specific_humidity

rho_air = 1.2


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

    # heat spread across 8 levels, moisture across bottom 3
    nlevels = state['t'].shape[1]

    total_mass = torch.zeros_like(t_lowest)
    for i in range(8):
        k = nlevels - 1 - i
        total_mass = total_mass + state['dp'][:, k] / g

    moist_mass = torch.zeros_like(t_lowest)
    for i in range(3):
        k = nlevels - 1 - i
        moist_mass = moist_mass + state['dp'][:, k] / g

    dt_uniform = shf / (total_mass * cp)
    dq_uniform = lhf / (moist_mass * Lv)

    dt = torch.zeros_like(state['t'])
    dq = torch.zeros_like(state['q'])
    for i in range(8):
        k = nlevels - 1 - i
        dt[:, k] = dt_uniform
    for i in range(3):
        k = nlevels - 1 - i
        dq[:, k] = dq_uniform

    return {
        'dt': dt,
        'dq': dq,
        'shf': shf,
        'lhf': lhf,
    }


def slab_ocean_tendency(state, rad_output, sfc_output, params):
    depth = params.get('ocean_depth', 50.0)
    heat_capacity = rho_water * c_water * depth

    net_flux = (
        rad_output['sw_absorbed_sfc']
        + rad_output['lw_down_sfc']
        - rad_output['lw_up_sfc']
        - sfc_output['shf']
        - sfc_output['lhf']
    )

    return net_flux / heat_capacity
