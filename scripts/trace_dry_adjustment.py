"""Attribute changes in dry stability to each physics stage."""
import json
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

params = default_params()
params.update(extract_param_overrides(load_run_config(None)))
params.update(dt=900.0, ocean_depth=5.0, condensation_rh_crit=0.95,
              use_slab_ocean=True, profile_diagnostics=True)
grid = make_grid(20)
state = model.initial_state(1, grid, params)
path = root / 'notebooks/data/atm407_equilibrium_20level_partial_cloud_rh095_diffusionfix_5m.npz'
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
for step in range(192):
    state, diagnostic, _ = model.physics_step(state, grid, params)
    record('end', state)
result = {'reference': str(path), 'days': 2,
          'interface_pressure_hpa': ((state['p'][0, :-1] + state['p'][0, 1:]) / 200).tolist(),
          'mean_excess_kkm_before_stage': {name: np.mean(values, axis=0).tolist() for name, values in samples.items()},
          'trigger_fraction_before_stage': {name: np.mean(np.array(values) > 3, axis=0).tolist() for name, values in samples.items()},
          'samples_excess_kkm': samples}
output = root / 'outputs/column/diagnostics/dry_adjustment_stage_trace.json'
output.write_text(json.dumps(result, indent=2) + '\n')
for name, values in result['mean_excess_kkm_before_stage'].items():
    print(name, np.round(values[10:], 3))
print('dry trigger', np.round(result['trigger_fraction_before_stage']['dry_adjustment'][10:], 3))
