"""Isolate dry boundary-layer transport and prescribed surface heating."""
import json
import argparse
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from scm.boundary_layer import boundary_layer_mixing, richardson_diffusivity
from scm.boundary_layer_tke_v2 import tke_boundary_layer
from scm.boundary_layer_uw import uw_moist_turbulence
from scm.configuration import load_run_config, extract_param_overrides
from scm.column_model import initial_state, update_derived
from scm.ensemble import default_params
from scm.thermo import make_grid, cp, g, Rd, kappa, p0, geopotential

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--uw-layer-closure', action='store_true')
args = parser.parse_args()


def experiment(transport, flux, timestep=60.0, overrides=None):
    grid = make_grid(20, dtype=torch.float64)
    params = default_params()
    params.update(extract_param_overrides(load_run_config(root / 'scm/configs/atm407.toml')))
    params.update(dt=timestep, bl_mix_moist_static_energy=transport == 'mse',
                  _surface_sensible_heat_flux=flux, _surface_energy_flux=flux)
    params.update(overrides or {})
    if args.uw_layer_closure and transport == 'uw':
        params['uw_layer_closure'] = True
    state = initial_state(1, grid, params)
    state = {key: value.double() if torch.is_tensor(value) else value
             for key, value in state.items()}
    update_derived(state, grid)
    pressure = state['p']
    # Dry neutral below 850 hPa, with a stable free atmosphere above it.
    theta = 300 + 35 * ((85000 - pressure) / 85000).clamp(min=0)
    state['t'] = theta * (pressure / p0) ** kappa
    state['q'].zero_()
    state['qc'].zero_()
    state['u'].zero_()
    state['v'].zero_()
    update_derived(state, grid)
    initial = state['t'].clone()
    mass = state['dp'] / g
    history = []
    steps = round(21600 / timestep)
    hourly = round(3600 / timestep)
    for step in range(steps):
        scheme = {'tke': tke_boundary_layer, 'uw': uw_moist_turbulence}.get(transport, boundary_layer_mixing)
        output = scheme(state, grid, params)
        for name in ('t', 'q', 'qc', 'u', 'v'):
            if 'd' + name in output:
                state[name] += params['dt'] * output['d' + name]
        if 'tke' in output:
            state['tke'] = output['tke']
        update_derived(state, grid)
        if (step + 1) % hourly == 0:
            height = geopotential(state['t'], state['q'], state['p'], grid)
            potential = state['t'] / (pressure / p0) ** kappa
            difference = potential[:, 1:] - potential[:, :-1]
            gradient = difference / (height[:, :-1] - height[:, 1:]) * 1000
            history.append({'hour': (step + 1) / hourly,
                            'max_unstable_theta_gradient_kkm': gradient.max().item(),
                            'depth_m': output['boundary_layer_depth_m'].item()})
    conductance = richardson_diffusivity(state, grid, params, params.get('k_diff', 0.5), 0)
    height = geopotential(state['t'], state['q'], pressure, grid)
    density = pressure[:, :-1] / (Rd * state['t'][:, :-1])
    spacing = (pressure[:, 1:] - pressure[:, :-1]).clamp(min=100)
    diffusivity = conductance[:, :-1] * spacing / (g * density ** 2)
    thermal = cp * state['t']
    if transport == 'mse':
        thermal = thermal + g * height
    heatflux = conductance[:, :-1] * (thermal[:, 1:] - thermal[:, :-1])
    if transport in ('tke', 'uw'):
        diffusivity = output.get('heat_diffusivity', output.get('diffusivity'))
        # Infer interface heat flux from finite-volume storage, with zero top flux.
        heatflux = torch.cumsum(cp * output['dt'] * mass, dim=1)[:, :-1]
    storage = ((state['t'] - initial) * mass * cp).sum().item()
    return {'transport': transport, 'surface_flux_wm2': flux,
            'overrides': overrides or {},
            'hours': 6, 'dt_s': timestep, 'levels': 20,
            'energy_residual_wm2': storage / 21600 - flux,
            'max_temperature_change_k': (state['t'] - initial).abs().max().item(),
            'maximum_diffusivity_m2s': diffusivity.max().item(),
            'upward_heat_flux_wm2': heatflux[0].tolist(),
            'pressure_hpa': (pressure[0] / 100).tolist(),
            'potential_temperature_k': (state['t'][0] / (pressure[0] / p0) ** kappa).tolist(),
            'hourly': history}


results = [experiment(transport, flux) for transport in ('temperature', 'mse', 'tke', 'uw')
           for flux in (0.0, 100.0)]
results += [experiment(transport, 100.0, 30.0) for transport in ('temperature', 'mse', 'tke', 'uw')]
results += [experiment('uw', 100.0, overrides=overrides) for overrides in (
    {'uw_diffusivity_max_m2s': 400.0}, {'bl_max_depth_m': 3000.0})]
label = 'dry_surface_connected_layers_6hour' if args.uw_layer_closure else 'dry_surface_candidates_6hour'
path = root / ('outputs/column/diagnostics/' + label + '.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(results, indent=2) + '\n')
for result in results:
    print(result['transport'], result['surface_flux_wm2'],
          'temperature change', result['max_temperature_change_k'],
          'energy residual', result['energy_residual_wm2'],
          'Kmax', result['maximum_diffusivity_m2s'], result['hourly'][-1])
print(path)
