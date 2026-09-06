"""Compare the first day of partial-cloud adjustment against its control."""
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
from scm.thermo import make_grid, g, cp, Lv

reference = np.load(root / 'notebooks/data/atm407_equilibrium_20level.npz')
parser = argparse.ArgumentParser()
parser.add_argument('--clear-sky', action='store_true')
args = parser.parse_args()
results = {}
for critical in (1.0, 0.95, 0.9):
    grid = make_grid(20)
    params = default_params()
    params.update(extract_param_overrides(load_run_config(root / 'scm/configs/atm407.toml')))
    params.update(dt=900.0, ocean_depth=5.0, profile_diagnostics=True,
                  condensation_rh_crit=critical, use_slab_ocean=True)
    if args.clear_sky:
        params['cloud_optics_scheme'] = 'clear_sky'
    state = initial_state(1, grid, params)
    for name in ('t', 'q', 'qc', 'cloud_fraction'):
        state[name][0] = torch.as_tensor(reference[name])
    state['ts'][0] = float(reference['ts'])
    state['ps'][0] = float(reference['ps'])
    state['slab_ts_ref'] = state['ts'].clone()
    state['slab_energy'].zero_()
    update_derived(state, grid)
    history = []
    cache = None
    for step in range(96):
        interval = min(8, max(1, int(params.get('rad_interval_microphysics_steps', 1))))
        if step % interval == 0:
            cache = None
        state, diagnostic, cache = physics_step(state, grid, params, rad_cache=cache)
        mass = state['dp'][0] / g
        row = {'hour': (step + 1) / 4, 'ts': float(state['ts'][0])}
        for name in ('toa_net', 'asr', 'olr', 'surface_total_flux',
                     'lhf', 'shf', 'column_energy_residual', 'column_water_residual'):
            row[name] = float(diagnostic[name][0])
        row['water'] = float(torch.sum(state['q'][0] * mass))
        row['condensate'] = float(torch.sum(state['qc'][0] * mass))
        row['processes'] = {}
        for process in ('radiation', 'surface', 'boundary_layer', 'shallow', 'deep', 'condensation', 'cloud'):
            heat = diagnostic[f'{process}_temperature_tendency'][0]
            vapor = diagnostic[f'{process}_moisture_tendency'][0]
            row['processes'][process] = {
                'heating_wm2': float(torch.sum(cp * heat * mass)),
                'vapor_kgm2day': float(torch.sum(vapor * mass) * 86400),
                'temperature_kday': (heat * 86400).tolist(),
            }
        history.append(row)
    results[str(critical)] = history
suffix = '_clear_sky' if args.clear_sky else ''
output = root / f'outputs/column/diagnostics/partial_cloud_first_day{suffix}.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + '\n')
for critical, history in results.items():
    print(critical)
    for index in (0, 8, 24, 95):
        row = history[index]
        print({key: round(value, 4) for key, value in row.items() if key != 'processes'})
    print('first step processes', {name: {key: round(value, 4) for key, value in data.items() if key != 'temperature_kday'} for name, data in history[0]['processes'].items()})
