import torch

from scm.boundary_layer_tke_v2 import tke_boundary_layer
from scm.shallow_plume_v2 import shallow_plume
from scm.thermo import Lv, cp, g, geopotential


def edmf_boundary_layer(state, grid, params):
    """Partition turbulent transport between local diffusion and a shallow plume."""

    timestep = float(params.get('dt', 60.0))
    plume_fraction = float(params.get('edmf_plume_fraction', 0.25))
    local_fraction = 1.0 - plume_fraction

    local_params = dict(params)
    local_params['tke_diffusivity_constant'] = (
        local_fraction * float(params.get('tke_diffusivity_constant', 0.18))
    )
    local = tke_boundary_layer(state, grid, local_params)

    intermediate = dict(state)
    intermediate['t'] = state['t'] + timestep * local['dt']
    intermediate['q'] = state['q'] + timestep * local['dq']
    intermediate['qc'] = state.get('qc', torch.zeros_like(state['q'])) + timestep * local['dqc']
    intermediate['tke'] = local['tke']

    plume_params = dict(params)
    plume_params['shallow_plume_surface_flux_fraction'] = plume_fraction
    plume = shallow_plume(intermediate, grid, plume_params)

    final_t = intermediate['t'] + timestep * plume['dt']
    final_q = intermediate['q'] + timestep * plume['dq']
    final_qc = intermediate['qc'] + timestep * plume['dqc']
    mass = state['dp'] / g
    height = geopotential(state['t'], state['q'], state['p'], grid)
    initial_water = torch.sum(
        (state['q'] + state.get('qc', torch.zeros_like(state['q']))) * mass,
        dim=1,
    )
    final_water = torch.sum((final_q + final_qc) * mass, dim=1)
    moisture_flux = params.get('_surface_moisture_flux', 0.0)
    moisture_flux = torch.as_tensor(
        moisture_flux, device=state['t'].device, dtype=state['t'].dtype
    ).reshape(-1)
    expected_water = initial_water + timestep * moisture_flux
    water_error = final_water - expected_water
    final_q = final_q.clone()
    final_q[:, -1] = final_q[:, -1] - water_error / mass[:, -1].clamp(min=1.0e-8)

    initial_mse = torch.sum(
        (cp * state['t'] + Lv * state['q'] + g * height) * mass,
        dim=1,
    )
    final_mse = torch.sum(
        (cp * final_t + Lv * final_q + g * height) * mass,
        dim=1,
    )
    sensible_flux = params.get('_surface_sensible_heat_flux', 0.0)
    sensible_flux = torch.as_tensor(
        sensible_flux, device=state['t'].device, dtype=state['t'].dtype
    ).reshape(-1)
    expected_mse = initial_mse + timestep * (sensible_flux + Lv * moisture_flux)
    final_mse = torch.sum(
        (cp * final_t + Lv * final_q + g * height) * mass,
        dim=1,
    )
    energy_error = final_mse - expected_mse
    final_t = final_t.clone()
    final_t[:, -1] = final_t[:, -1] - energy_error / (
        cp * mass[:, -1].clamp(min=1.0e-8)
    )
    final_water = torch.sum((final_q + final_qc) * mass, dim=1)
    final_mse = torch.sum(
        (cp * final_t + Lv * final_q + g * height) * mass,
        dim=1,
    )

    return {
        'dt': (final_t - state['t']) / timestep,
        'dq': (final_q - state['q']) / timestep,
        'dqc': (final_qc - state.get('qc', torch.zeros_like(state['q']))) / timestep,
        'tke': local['tke'],
        'diffusivity': local['diffusivity'],
        'boundary_layer_depth_m': local['boundary_layer_depth_m'],
        'cloud_base_mass_flux': plume['cloud_base_mass_flux'],
        'cloud_fraction': plume['cloud_fraction'],
        'plume_condensate': plume['plume_condensate'],
        'plume_mass_flux_profile': plume['plume_mass_flux_profile'],
        'condensate_detrainment': plume['condensate_detrainment'],
        'water_residual': (final_water - expected_water) / timestep,
        'energy_residual': (final_mse - expected_mse) / timestep,
    }
