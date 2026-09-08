"""Attribute changes in dry stability to each physics stage."""
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import scm.column_model as model
from scm.configuration import load_run_config, extract_param_overrides
from scm.ensemble import default_params
from scm.thermo import make_grid, Rd, g, kappa, p0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, default=root / 'scm/configs/atm407.toml')
parser.add_argument('--reference', type=Path, default=root / 'notebooks/data/atm407_equilibrium_20level.npz')
parser.add_argument('--days', type=int, default=2)
parser.add_argument('--ocean-depth', type=float, default=5.0)
parser.add_argument('--output', type=Path, default=root / 'outputs/column/diagnostics/dry_adjustment_stage_trace.json')
args = parser.parse_args()
config = load_run_config(args.config)
params = default_params()
params.update(extract_param_overrides(config))
params.update(dt=config['numerics']['dt'], ocean_depth=args.ocean_depth,
              use_slab_ocean=True, profile_diagnostics=True)
grid = make_grid(config['numerics']['nlevels'])
state = model.initial_state(1, grid, params)
path = args.reference
with np.load(path) as reference:
    for name in ('t', 'q', 'qc', 'cloud_fraction'):
        state[name][0] = torch.as_tensor(reference[name])
    state['ts'][0] = float(reference['ts'])
    state['ps'][0] = float(reference['ps'])
state['slab_ts_ref'] = state['ts'].clone()
state['slab_energy'].zero_()
model.update_derived(state, grid)
samples = {}


def record(name, state):
    t, q, qc, p = (state[key] for key in ('t', 'q', 'qc', 'p'))
    theta = t / (p / p0) ** kappa * (1 + 0.608 * q - qc)
    mean = (t[:, :-1] + t[:, 1:]) / 2
    thickness = Rd * mean / g * torch.log(p[:, 1:] / p[:, :-1])
    excess = (theta[:, 1:] - theta[:, :-1]) / thickness * mean / (
        (theta[:, 1:] + theta[:, :-1]) / 2) * 1000
    samples.setdefault(name, []).append(excess[0].tolist())


def wrap(function, name):
    def call(state, *args, **kwargs):
        record(name, state)
        return function(state, *args, **kwargs)
    return call


for name in ('radiation', 'surface_fluxes', 'dry_adjustment', 'dispatch_convection', 'cloud_microphysics_step'):
    setattr(model, name, wrap(getattr(model, name), name))
dispatch = model.run_physics_scheme


def scheme(category, name, state, *args, **kwargs):
    record(category, state)
    return dispatch(category, name, state, *args, **kwargs)


model.run_physics_scheme = scheme
for step in range(round(args.days * 86400 / params['dt'])):
    state, diagnostic, _ = model.physics_step(state, grid, params)
    record('end', state)
result = {'reference': str(path), 'config': str(args.config), 'days': args.days,
          'ocean_depth_m': args.ocean_depth,
          'bl_diagnose_depth': params.get('bl_diagnose_depth', False),
          'interface_pressure_hpa': ((state['p'][0, :-1] + state['p'][0, 1:]) / 200).tolist(),
          'mean_excess_kkm_before_stage': {name: np.mean(values, axis=0).tolist() for name, values in samples.items()},
          'trigger_fraction_before_stage': {name: np.mean(np.array(values) > 3, axis=0).tolist() for name, values in samples.items()},
          'samples_excess_kkm': samples}
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + '\n')
for name, values in result['mean_excess_kkm_before_stage'].items():
    print(name, np.round(values[10:], 3))
print('dry trigger', np.round(result['trigger_fraction_before_stage']['dry_adjustment'][10:], 3))
