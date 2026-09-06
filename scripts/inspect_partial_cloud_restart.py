"""Sample a ten-day continuation without replacing the candidate checkpoint."""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from scm.column_model import initial_state, physics_step, update_derived
from scm.configuration import load_run_config, extract_param_overrides
from scm.ensemble import default_params
from scm.thermo import make_grid, relative_humidity, g

path = root / 'notebooks/data/atm407_equilibrium_20level_partial_cloud_rh095_5m_radswitch.npz'
grid = make_grid(20)
params = default_params()
params.update(extract_param_overrides(load_run_config(root / 'scm/configs/atm407.toml')))
params.update(dt=900.0, ocean_depth=5.0, condensation_rh_crit=0.95, use_slab_ocean=True)
parser = argparse.ArgumentParser()
parser.add_argument('--conserved-mixing', action='store_true')
parser.add_argument('--output-label', default='corrected_diffusion')
args = parser.parse_args()
if args.conserved_mixing:
    params.update(bl_mix_moist_static_energy=True, bl_mix_total_water=True)
state = initial_state(1, grid, params)
with np.load(path) as reference:
    for name in ('t', 'q', 'qc', 'cloud_fraction'):
        state[name][0] = torch.as_tensor(reference[name])
    state['ts'][0] = float(reference['ts'])
    state['ps'][0] = float(reference['ps'])
    if 'tke' in reference.files:
        state['tke'] = torch.as_tensor(reference['tke']).unsqueeze(0)
state['slab_ts_ref'] = state['ts'].clone()
state['slab_energy'].zero_()
update_derived(state, grid)


def profile():
    return {
        'pressure_hpa': (state['p'][0] / 100).tolist(),
        'temperature_k': state['t'][0].tolist(),
        'rh_percent': (relative_humidity(state['q'], state['t'], state['p'])[0] * 100).tolist(),
        'condensate_gkg': (state['qc'][0] * 1000).tolist(),
    }


initial = profile()
start = float(state['ts'][0])
rows = []
for step in range(960):
    state, diagnostic, _ = physics_step(state, grid, params)
    row = {'day': (step + 1) / 96, 'ts': float(state['ts'][0])}
    for name in ('toa_net', 'surface_total_flux', 'cape', 'precip_conv', 'precip_ls',
                 'precip_cloud', 'precip_total', 'column_water_residual', 'column_energy_residual'):
        row[name] = float(diagnostic[name][0])
    row['cwp'] = float(torch.sum(state['qc'] * state['dp'] / g))
    rows.append(row)
daily = []
for day in range(10):
    block = rows[day * 96:(day + 1) * 96]
    daily.append({name: float(np.mean([row[name] for row in block])) for name in block[0]})
result = {'reference': str(path), 'conserved_mixing': args.conserved_mixing,
          'output_label': args.output_label, 'initial_profile': initial, 'final_profile': profile(),
          'temperature_change_k': float(state['ts'][0]) - start, 'daily_means': daily,
          'quarter_hourly': rows}
suffix = '_conserved_mixing' if args.conserved_mixing else ''
suffix += '_' + args.output_label
output = root / f'outputs/column/diagnostics/partial_cloud_rh095_restart_10day{suffix}.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + '\n')
print('temperature change', result['temperature_change_k'])
for row in daily:
    print({key: round(value, 5) for key, value in row.items()})
print('initial profile', initial)
print('final profile', result['final_profile'])
