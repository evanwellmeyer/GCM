from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scm.column_model import initial_state, run, update_derived
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params
from scm.thermo import g, make_grid, relative_humidity
from scm.boundary_layer_tke_v2 import tke_diffusivity, tke_mixing_length
from scm.thermo import geopotential


parser = argparse.ArgumentParser()
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--days', type=int, default=10)
parser.add_argument('--output', type=Path)
args = parser.parse_args()

device = torch.device('cpu')
reference = np.load(args.reference)
levels = len(reference['sigma_full'])
grid = make_grid(levels, device=device)
config = load_run_config(args.config)
params = default_params(device=device)
params.update(extract_param_overrides(config))
params.update({
    'dt': 900.0,
    'ps0': 100000.0,
    'solar_constant': 1360.0,
    'zenith_factor': 0.25,
    'ocean_depth': 50.0,
    'wind_speed': 5.0,
    'convection_scheme': 'mass_flux',
    'use_slab_ocean': True,
    'profile_diagnostics': True,
})

state = initial_state(1, grid, params, device=device)
for name in ('t', 'q', 'qc', 'cloud_fraction'):
    state[name][0] = torch.as_tensor(reference[name], dtype=state[name].dtype)
if 'tke' in reference.files:
    state['tke'] = torch.as_tensor(
        reference['tke'], dtype=state['t'].dtype
    ).unsqueeze(0)
state['ts'][0] = float(reference['ts'])
state['ps'][0] = float(reference['ps'])
state['slab_ts_ref'] = state['ts'].clone()
state['slab_energy'].zero_()
update_derived(state, grid)

stepsperday = round(86400 / params['dt'])
state, history = run(
    state,
    grid,
    params,
    args.days * stepsperday,
    rad_interval=8,
    diag_interval=1,
)

processes = [
    'radiation',
    'surface',
    'boundary_layer',
    'shallow',
    'deep',
    'condensation',
    'cloud',
]


def meanprofile(name):
    return torch.stack([item[name][0] for item in history]).mean(dim=0)


def meanvalue(name):
    return torch.stack([item[name][0] for item in history]).mean()


temperature = {}
moisture = {}
for process in processes:
    temperature[process] = meanprofile(f'{process}_temperature_tendency') * 86400
    moisture[process] = meanprofile(f'{process}_moisture_tendency') * 86400 * 1000

pressure = state['p'][0] / 100
rh = relative_humidity(state['q'], state['t'], state['p'])[0] * 100
mass = state['dp'][0] / g
rh95mass = torch.sum((rh >= 95.0) * mass) / torch.sum(mass)
height = geopotential(state['t'], state['q'], state['p'], grid)
mixinglength = tke_mixing_length(height, params)
diffusivity, _ = tke_diffusivity(
    state['t'], state['q'], state['u'], state['v'], state['p'],
    height, state['tke'], mixinglength, params,
)
columnmoisture = {
    process: torch.sum(moisture[process] / 1000 * mass).item()
    for process in processes
}

result = {
    'reference': str(args.reference),
    'configuration_label': config['run']['label'],
    'averaging_days': args.days,
    'pressure_hpa': pressure.tolist(),
    'temperature_k': state['t'][0].tolist(),
    'relative_humidity_percent': rh.tolist(),
    'cloud_condensate_gkg': (state['qc'][0] * 1000).tolist(),
    'tke_m2s2': state['tke'][0].tolist(),
    'diffusivity_m2s': diffusivity[0].tolist(),
    'tke_production_m2s3': meanprofile('tke_production').tolist(),
    'tke_dissipation_m2s3': meanprofile('tke_dissipation').tolist(),
    'tke_transport_m2s3': meanprofile('tke_transport').tolist(),
    'temperature_tendency_kday': {
        process: temperature[process].tolist() for process in processes
    },
    'moisture_tendency_gkgday': {
        process: moisture[process].tolist() for process in processes
    },
    'column_vapor_tendency_kgm2day': columnmoisture,
    'summary': {
        'surface_temperature_k': meanvalue('ts').item(),
        'surface_temperature_drift_k': (history[-1]['ts'][0] - history[0]['ts'][0]).item(),
        'toa_net_wm2': meanvalue('toa_net').item(),
        'clear_sky_toa_net_wm2': meanvalue('clear_sky_toa_net').item(),
        'cloud_toa_effect_wm2': (
            meanvalue('toa_net') - meanvalue('clear_sky_toa_net')
        ).item(),
        'cloud_shortwave_effect_wm2': meanvalue('cloud_sw_cre').item(),
        'cloud_longwave_effect_wm2': meanvalue('cloud_lw_cre').item(),
        'clear_sky_asr_wm2': meanvalue('clear_sky_asr').item(),
        'clear_sky_olr_wm2': meanvalue('clear_sky_olr').item(),
        'surface_total_flux_wm2': meanvalue('surface_total_flux').item(),
        'cape_jkg': meanvalue('cape').item(),
        'rh95_mass_fraction': rh95mass.item(),
        'boundary_layer_depth_m': meanvalue('boundary_layer_depth_m').item(),
        'surface_buoyancy_flux_m2s3': meanvalue('surface_buoyancy_flux_m2s3').item(),
        'maximum_tke_m2s2': torch.max(state['tke']).item(),
        'maximum_diffusivity_m2s': torch.max(diffusivity).item(),
        'deep_precipitation_mmday': meanvalue('precip_conv').item() * 86400,
        'large_scale_precipitation_mmday': meanvalue('precip_ls').item() * 86400,
        'cloud_precipitation_mmday': meanvalue('precip_cloud').item() * 86400,
        'mass_flux_cap_fraction': meanvalue('mass_flux_cap_active').item(),
        'temperature_cap_fraction': meanvalue('temperature_cap_fraction').item(),
        'moisture_cap_fraction': meanvalue('moisture_cap_fraction').item(),
        'column_energy_residual_wm2': meanvalue('column_energy_residual').item(),
        'column_mse_residual_wm2': meanvalue('column_mse_residual').item(),
        'column_water_residual_kgm2s': meanvalue('column_water_residual').item(),
    },
    'cloud_condensate_tendency_gkgday': (
        meanprofile('cloud_condensate_tendency') * 86400 * 1000
    ).tolist(),
    'cloud_total_water_tendency_gkgday': (
        meanprofile('cloud_total_water_tendency') * 86400 * 1000
    ).tolist(),
}

output = args.output
if output is None:
    output = root / 'outputs' / 'column' / 'diagnostics' / 'atm407_budget.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + '\n')

print(output)
print(json.dumps(result['summary'], indent=2))
print('column vapor tendencies (kg m-2 day-1)')
for process in processes:
    print(f'{process:16s} {columnmoisture[process]:9.4f}')
print('pressure  rh    surface      bl shallow    deep condensation cloud')
for level in range(grid['nlevels']):
    values = ' '.join(f'{moisture[process][level].item():8.3f}' for process in processes[1:])
    print(f'{pressure[level].item():7.1f} {rh[level].item():5.1f} {values}')
