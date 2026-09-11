"""Attribute the near-surface static instability to individual physics schemes.

The accepted checkpoint is statically unstable through its lowest ~1.4 km:
virtual potential temperature falls from 282.95 K at 998 hPa to 279.81 K at
865 hPa, which is upside down. A real marine boundary layer is well mixed and
capped by an inversion.

This records the virtual-potential-temperature gradient at every interface
immediately before and immediately after each physics scheme runs, so the
question "which scheme destabilises the column and which restores it" is
answered by measurement rather than by perturbing parameters and guessing.
It changes no physics and runs a short integration only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import scm.column_model as model
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params
from scm.thermo import Rd, g, kappa, make_grid, p0

CONFIG = root / 'scm/configs/atm407.toml'
REFERENCE = root / 'notebooks/data/atm407_equilibrium_20level.npz'
# Two days at the production step. Long enough for the stability budget to
# average out step-to-step noise, short enough to stay a diagnostic.
DAYS = 2

params = default_params()
params.update(extract_param_overrides(load_run_config(CONFIG)))
params.update(dt=900.0, ocean_depth=5.0, use_slab_ocean=True,
              profile_diagnostics=True)
grid = make_grid(20)

state = model.initial_state(1, grid, params)
with np.load(REFERENCE) as reference:
    for name in ('t', 'q', 'qc', 'cloud_fraction'):
        state[name][0] = torch.as_tensor(reference[name], dtype=state[name].dtype)
    state['ts'][0] = float(reference['ts'])
    state['ps'][0] = float(reference['ps'])
state['slab_ts_ref'] = state['ts'].clone()
state['slab_energy'].zero_()
model.update_derived(state, grid)


def stability(current):
    """Virtual-potential-temperature lapse in K/km at each interface.

    Positive means theta_v increases upward, which is stable. Negative is
    upside down. Normalising by layer thickness keeps the number comparable
    across a stretched grid.
    """
    t, q, qc, p = (current[k] for k in ('t', 'q', 'qc', 'p'))
    theta_v = t / (p / p0) ** kappa * (1.0 + 0.608 * q - qc)
    mean_t = (t[:, :-1] + t[:, 1:]) / 2.0
    thickness = Rd * mean_t / g * torch.log(p[:, 1:] / p[:, :-1])
    return ((theta_v[:, :-1] - theta_v[:, 1:]) / thickness * 1000.0)[0].numpy()


# Wrap every scheme so each one is bracketed by a stability reading. The
# difference across a scheme is that scheme's contribution to static
# stability, which is exactly what no previous trace measured.
# Some schemes are module-level functions; the rest are dispatched by category
# through run_physics_scheme, so both routes have to be intercepted or the
# boundary layer -- the scheme most likely to be at fault -- is invisible.
SCHEMES = ('radiation', 'surface_fluxes', 'dry_adjustment',
           'dispatch_convection', 'cloud_microphysics_step')
DISPATCHED = ('boundary_layer', 'shallow_convection', 'condensation')
contributions = {}
originals = {}


def wrap(function, label):
    def call(current, *args, **kwargs):
        before = stability(current)
        result = function(current, *args, **kwargs)
        # Schemes return tendencies rather than a new state, so re-read the
        # state after column_model has applied them: bracket at the call site
        # instead by stashing `before` and differencing on the next bracket.
        contributions.setdefault(label, {'before': []})['before'].append(before)
        return result
    return call


for name in SCHEMES:
    if hasattr(model, name):
        originals[name] = getattr(model, name)
        setattr(model, name, wrap(originals[name], name))

dispatch = model.run_physics_scheme


def dispatch_wrapper(category, scheme, current, *args, **kwargs):
    if category in DISPATCHED:
        contributions.setdefault(category, {'before': []})['before'].append(
            stability(current))
    return dispatch(category, scheme, current, *args, **kwargs)


model.run_physics_scheme = dispatch_wrapper

# Physics order in column_model: radiation, surface, boundary layer, dry
# adjustment, shallow, deep convection, condensation, cloud microphysics.
ordered = ['radiation', 'surface_fluxes', 'boundary_layer', 'dry_adjustment',
           'shallow_convection', 'dispatch_convection', 'condensation',
           'cloud_microphysics_step']
ordered = [n for n in ordered if n in originals or n in DISPATCHED]
end_of_step = []
steps = int(DAYS * 86400 / params['dt'])
for _ in range(steps):
    state, diagnostics, _ = model.physics_step(state, grid, params)
    end_of_step.append(stability(state))

for name, function in originals.items():
    setattr(model, name, function)
model.run_physics_scheme = dispatch

# Each scheme's effect is the change from its own "before" reading to the
# "before" reading of the next scheme in the sequence (and for the last
# scheme, to the end-of-step reading).
pressure = ((state['p'][0, :-1] + state['p'][0, 1:]) / 200.0).numpy()
effects = {}
for index, name in enumerate(ordered):
    start = np.array(contributions[name]['before'])
    if index + 1 < len(ordered):
        finish = np.array(contributions[ordered[index + 1]]['before'])
    else:
        finish = np.array(end_of_step)
    n = min(len(start), len(finish))
    effects[name] = (finish[:n] - start[:n]).mean(axis=0)

print(f'static stability budget, {DAYS} days, K/km per interface')
print('positive = stabilising, negative = destabilising\n')
header = f"{'p hPa':>8} " + ' '.join(f'{n[:9]:>10}' for n in ordered) + f" {'net':>9} {'mean':>9}"
print(header)
mean_state = np.array(end_of_step).mean(axis=0)
for i in range(len(pressure)):
    if pressure[i] < 700:
        continue
    row = ' '.join(f'{effects[n][i]:10.3f}' for n in ordered)
    net = sum(effects[n][i] for n in ordered)
    print(f'{pressure[i]:8.1f} {row} {net:9.3f} {mean_state[i]:9.3f}')

output = root / 'outputs/column/diagnostics/static_stability_budget.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    'reference': str(REFERENCE), 'config': str(CONFIG), 'days': DAYS,
    'interface_pressure_hpa': pressure.tolist(),
    'scheme_order': ordered,
    'stability_change_kkm': {n: effects[n].tolist() for n in ordered},
    'mean_stability_kkm': mean_state.tolist(),
}, indent=2) + '\n')
print(f'\nwritten to {output.relative_to(root)}')
