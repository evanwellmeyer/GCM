import torch

from scm.boundary_layer import boundary_layer_mixing
from scm.boundary_layer_tke_v2 import tke_boundary_layer
from scm.boundary_layer_uw import uw_moist_turbulence
from scm.boundary_layer_edmf_v3 import edmf_boundary_layer
from scm.column_model import initial_state, update_derived
from scm.convection_shallow import shallow_convection
from scm.convection_uw import uw_shallow_convection
from scm.shallow_plume_v2 import shallow_plume, partition_plume
from scm.ensemble import default_params
from scm.thermo import Lv, Rd, cp, g, geopotential, kappa, p0, relative_humidity, saturation_specific_humidity


def linear_profile(height, points, values):
    result = torch.full_like(height, float(values[0]))
    for lower in range(len(points) - 1):
        upper = lower + 1
        span = float(points[upper] - points[lower])
        fraction = ((height - points[lower]) / span).clamp(min=0.0, max=1.0)
        segment = values[lower] + fraction * (values[upper] - values[lower])
        active = (height >= points[lower]) & (height <= points[upper])
        result = torch.where(active, segment, result)
    result = torch.where(height > points[-1], torch.as_tensor(values[-1], device=height.device), result)
    return result


def benchmark_state(grid, surface_pressure=101500.0):
    params = default_params()
    params.update({'ps0': surface_pressure, 'use_slab_ocean': False})
    state = initial_state(1, grid, params)
    state['ps'][:] = surface_pressure
    state = update_derived(state, grid)
    return state, params


def model_height(state, grid):
    return geopotential(state['t'], state['q'], state['p'], grid)


def vertical_gradient(field, height):
    gradient = torch.zeros_like(field)
    gradient[:, 1:-1] = (
        (field[:, :-2] - field[:, 2:])
        / (height[:, :-2] - height[:, 2:]).clamp(min=1.0)
    )
    gradient[:, 0] = (field[:, 0] - field[:, 1]) / (height[:, 0] - height[:, 1]).clamp(min=1.0)
    gradient[:, -1] = (field[:, -2] - field[:, -1]) / (height[:, -2] - height[:, -1]).clamp(min=1.0)
    return gradient


def initialize_dry_mixed_layer(grid):
    state, params = benchmark_state(grid, 100000.0)
    height = -8000.0 * torch.log(state['p'] / state['ps'].unsqueeze(1))
    theta = torch.where(
        height <= 1000.0,
        torch.full_like(height, 300.0),
        302.0 + 0.003 * (height - 1000.0),
    )
    state['t'] = theta * (state['p'] / p0) ** kappa
    state['q'][:] = 0.005
    state['qc'].zero_()
    return update_derived(state, grid), params


def initialize_bomex(grid):
    """Siebesma et al. (2003), Table B1, in specific-water units.

    Select a surface-referenced height datum on this benchmark grid. Above the
    published 3 km sounding, append a subsaturated free troposphere for the
    full-column grid; that extension is not part of the scored case.
    """
    grid['height_surface_pressure_pa'] = 101500.0
    state, params = benchmark_state(grid, 101500.0)
    for _ in range(12):
        height = model_height(state, grid)
        theta = linear_profile(
            height,
            [0.0, 520.0, 1480.0, 2000.0, 3000.0],
            [298.7, 298.7, 302.4, 308.2, 311.85],
        )
        total_water = linear_profile(
            height,
            [0.0, 520.0, 1480.0, 2000.0, 3000.0],
            [0.0170, 0.0163, 0.0107, 0.0042, 0.0030],
        )
        wind = linear_profile(
            height,
            [0.0, 700.0, 3000.0],
            [-8.75, -8.75, -4.61],
        )
        state['t'], state['q'], state['qc'] = partition_plume(theta, total_water, state['p'])
        above = height > 3000.0
        temperature = ((311.85 + .00365 * (height - 3000.0)) * (state['p'] / p0) ** kappa).clamp(min=195.0)
        # Match RH at the sounding top and keep the extension subsaturated.
        index = int(torch.nonzero(height[0] <= 3000.).flatten()[0])
        upper = max(0, index - 1)
        share = ((3000. - height[0, index]) / (height[0, upper] - height[0, index]).clamp(min=1.))
        pressure = state['p'][0, index] + share * (state['p'][0, upper] - state['p'][0, index])
        relative = .003 / saturation_specific_humidity(311.85 * (pressure / p0) ** kappa, pressure)
        humidity = torch.minimum(torch.full_like(temperature, .003), relative.clamp(max=1.) * saturation_specific_humidity(temperature, state['p']))
        state['t'] = torch.where(above, temperature, state['t'])
        state['q'] = torch.where(above, humidity, state['q'])
        state['qc'] = torch.where(above, torch.zeros_like(state['qc']), state['qc'])
        state['u'] = wind
        state['v'].zero_()
        state = update_derived(state, grid)
    return state, params


def apply_boundary_layer(state, grid, params, sensible_flux, moisture_flux, scheme='richardson'):
    local = dict(params)
    local['_surface_sensible_heat_flux'] = torch.as_tensor(
        [sensible_flux], device=state['t'].device, dtype=state['t'].dtype
    )
    local['_surface_moisture_flux'] = torch.as_tensor(
        [moisture_flux], device=state['t'].device, dtype=state['t'].dtype
    )
    local['_surface_energy_flux'] = local['_surface_sensible_heat_flux'] + Lv * local['_surface_moisture_flux']
    if scheme == 'tke':
        output = tke_boundary_layer(state, grid, local)
        state['tke'] = output['tke']
    elif scheme == 'edmf':
        output = edmf_boundary_layer(state, grid, local)
        state['tke'] = output['tke']
    elif scheme == 'uw':
        output = uw_moist_turbulence(state, grid, local)
        state['tke'] = output['tke']
    else:
        output = boundary_layer_mixing(state, grid, local)
    timestep = float(local['dt'])
    if 'tke_interfaces' in output:
        state['tke_interfaces'] = output['tke_interfaces']
    else:
        state.pop('tke_interfaces', None)
    state['t'] = state['t'] + output['dt'] * timestep
    state['q'] = state['q'] + output['dq'] * timestep
    state['qc'] = torch.clamp(state['qc'] + output['dqc'] * timestep, min=0.0)
    state['u'] = state['u'] + output.get('du', torch.zeros_like(state['u'])) * timestep
    state['v'] = state['v'] + output.get('dv', torch.zeros_like(state['v'])) * timestep
    if 'cloud_fraction' in output:
        state['cloud_fraction'] = output['cloud_fraction']
    if 'boundary_layer_depth_m' in output:
        state['boundary_layer_depth_m'] = output['boundary_layer_depth_m']
    return update_derived(state, grid), output


def run_dry_mixed_layer(grid, hours=6.0, timestep=60.0, scheme='richardson'):
    state, params = initialize_dry_mixed_layer(grid)
    params.update({
        'dt': timestep,
        'boundary_layer_scheme': 'richardson',
        'bl_diagnose_depth': True,
        'bl_min_depth_m': 50.0,
        'bl_max_depth_m': 2500.0,
        'bl_mix_moist_static_energy': True,
        'bl_mix_total_water': True,
        'wind_speed': 1.0,
    })
    initial_theta = state['t'] * (p0 / state['p']) ** kappa
    initial_energy = torch.sum((cp * state['t'] + Lv * state['q']) * state['dp'] / g, dim=1)
    steps = round(hours * 3600.0 / timestep)
    depth = torch.zeros(1)
    for _ in range(steps):
        state, output = apply_boundary_layer(state, grid, params, 100.0, 0.0, scheme=scheme)
        depth = output['boundary_layer_depth_m']

    theta = state['t'] * (p0 / state['p']) ** kappa
    height = model_height(state, grid)
    mixed = height <= depth.unsqueeze(1)
    mixed_values = theta[mixed]
    final_energy = torch.sum((cp * state['t'] + Lv * state['q']) * state['dp'] / g, dim=1)
    expected_energy = 100.0 * hours * 3600.0
    return {
        'boundary_layer_depth_m': float(depth[0]),
        'mixed_layer_theta_spread_k': float(mixed_values.max() - mixed_values.min()),
        'surface_theta_change_k': float(theta[0, -1] - initial_theta[0, -1]),
        'energy_error_wm2': float((final_energy - initial_energy - expected_energy)[0] / (hours * 3600.0)),
    }


def bomex_forcing(state, grid):
    """Return liquid-potential-temperature and total-specific-water tendencies."""
    height = model_height(state, grid)
    theta = (state['t'] - Lv * state['qc'] / cp) * (p0 / state['p']) ** kappa
    total_water = state['q'] + state['qc']
    subsidence = linear_profile(
        height,
        [0.0, 1500.0, 2100.0, 3500.0],
        [0.0, -0.0065, 0.0, 0.0],
    )
    radiative = linear_profile(
        height,
        [0.0, 1500.0, 3000.0, 3500.0],
        [-2.0 / 86400.0, -2.0 / 86400.0, 0.0, 0.0],
    )
    moisture_advection = linear_profile(
        height,
        [0.0, 300.0, 500.0, 3500.0],
        [-1.2e-8, -1.2e-8, 0.0, 0.0],
    )

    theta_gradient = vertical_gradient(theta, height)
    water_gradient = vertical_gradient(total_water, height)
    return radiative - subsidence * theta_gradient, moisture_advection - subsidence * water_gradient


def bomex_momentum_forcing(state, grid):
    """Prescribed subsidence, geostrophic acceleration and surface stress."""
    height = model_height(state, grid)
    subsidence = linear_profile(height, [0., 1500., 2100., 3500.], [0., -.0065, 0., 0.])
    geostrophic = -10.0 + .0018 * height
    zonal = -subsidence * vertical_gradient(state['u'], height) + .376e-4 * state['v']
    meridional = -subsidence * vertical_gradient(state['v'], height) - .376e-4 * (state['u'] - geostrophic)
    density = state['p'][:, -1] / (Rd * state['t'][:, -1])
    speed = torch.sqrt(state['u'][:, -1].square() + state['v'][:, -1].square()).clamp(min=1e-8)
    stress = density * .28**2 / (state['dp'][:, -1] / g)
    zonal[:, -1] = zonal[:, -1] - stress * state['u'][:, -1] / speed
    meridional[:, -1] = meridional[:, -1] - stress * state['v'][:, -1] / speed
    return zonal, meridional


def run_bomex(
    grid, hours=6.0, timestep=60.0, use_shallow=True,
    scheme='richardson', shallow_scheme='legacy', parameter_updates=None,
):
    state, params = initialize_bomex(grid)
    params.update({
        'dt': timestep,
        'boundary_layer_scheme': 'richardson',
        'bl_diagnose_depth': True,
        'bl_min_depth_m': 50.0,
        'bl_max_depth_m': 2500.0,
        'bl_mix_moist_static_energy': True,
        'bl_mix_total_water': True,
        'wind_speed': 8.75,
        'shallow_convection_scheme': 'simple',
        'shallow_convection_enabled': use_shallow,
        'shallow_tau': 14400.0,
        'shallow_top_sigma': 0.72,
        'shallow_base_sigma': 0.90,
        'shallow_rh_trigger': 0.78,
        'shallow_detrain_rh': 0.85,
        'shallow_cape_suppress': 600.0,
        'shallow_mse_scale': 3000.0,
        'shallow_max_dt_day': 1.5,
        'shallow_max_dq_day': 1.5,
        'shallow_enforce_mse_conservation': True,
    })
    if parameter_updates:
        params.update(parameter_updates)
    steps = round(hours * 3600.0 / timestep)
    depth = torch.zeros(1)
    shallow_mass_flux = torch.zeros(1)
    for _ in range(steps):
        theta_tendency, water_tendency = bomex_forcing(state, grid)
        zonal, meridional = bomex_momentum_forcing(state, grid)
        state['u'] = state['u'] + zonal * timestep
        state['v'] = state['v'] + meridional * timestep
        state['t'] = state['t'] + theta_tendency * (state['p'] / p0) ** kappa * timestep
        state['q'] = torch.clamp(state['q'] + water_tendency * timestep, min=1.0e-7)
        state = update_derived(state, grid)
        density = state['p'][:, -1] / (Rd * state['t'][:, -1])
        sensible_flux = float((density * cp * (state['ps'] / p0) ** kappa * 8.0e-3)[0])
        moisture_flux = float((density * 5.2e-5)[0])
        state, output = apply_boundary_layer(
            state, grid, params, sensible_flux, moisture_flux, scheme=scheme
        )
        depth = output['boundary_layer_depth_m']
        shallow_mass_flux = output.get('cloud_base_mass_flux', shallow_mass_flux)
        if use_shallow:
            params['_surface_sensible_heat_flux'] = torch.as_tensor(
                [sensible_flux], device=state['t'].device, dtype=state['t'].dtype
            )
            params['_surface_moisture_flux'] = torch.as_tensor(
                [moisture_flux], device=state['t'].device, dtype=state['t'].dtype
            )
            if shallow_scheme == 'plume':
                shallow = shallow_plume(state, grid, params)
            elif shallow_scheme == 'uw':
                shallow = uw_shallow_convection(state, grid, params)
            else:
                shallow = shallow_convection(state, grid, params)
            state['t'] = state['t'] + shallow['dt'] * timestep
            state['q'] = torch.clamp(state['q'] + shallow['dq'] * timestep, min=1.0e-7)
            state['qc'] = torch.clamp(
                state['qc'] + shallow.get('dqc', torch.zeros_like(state['qc'])) * timestep,
                min=0.0,
            )
            if 'cloud_fraction' in shallow:
                state['cloud_fraction'] = shallow['cloud_fraction']
            shallow_mass_flux = shallow.get('cloud_base_mass_flux', shallow_mass_flux)
            state = update_derived(state, grid)

    height = model_height(state, grid)[0]
    rh = relative_humidity(state['q'], state['t'], state['p'])[0]
    cloud_layer = (height >= 500.0) & (height <= 2000.0)
    subcloud = height <= 500.0
    theta = state['t'] * (p0 / state['p']) ** kappa
    return {
        'boundary_layer_depth_m': float(depth[0]),
        'subcloud_theta_spread_k': float(theta[0, subcloud].max() - theta[0, subcloud].min()),
        'cloud_layer_max_rh': float(rh[cloud_layer].max()),
        'cloud_layer_mean_rh': float(rh[cloud_layer].mean()),
        'levels_below_2000m': int(torch.count_nonzero(height <= 2000.0)),
        'cloud_water_path_kgm2': float(torch.sum(state['qc'][0] * state['dp'][0] / g)),
        'maximum_cloud_fraction': float(torch.max(state['cloud_fraction'][0])),
        'shallow_mass_flux_kgm2s': float(shallow_mass_flux[0]),
    }
