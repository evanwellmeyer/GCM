"""Trace actual UW diffusivity and total-water flux in the ATM407 column."""
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import scm.boundary_layer_uw as uw
from scm.thermo import make_grid, g

samples = []
original = uw.uw_diffusivity
solver = uw.solve_scalar
mass = make_grid(20)['dsigma'].double() * 100000 / g
calls = 0


def diffusivity(t, q, u, v, p, height, theta, depth, buoyancy, params):
    global calls
    output = original(t, q, u, v, p, height, theta, depth, buoyancy, params)
    interface = (height[:, :-1] + height[:, 1:]) / 2
    samples.append({'height_m': interface[0].tolist(),
                    'pressure_hpa': ((p[0, :-1] + p[0, 1:]) / 200).tolist(),
                    'diffusivity_m2s': output[0][0].tolist(),
                    'above_bl': (interface[0] > depth[0]).tolist(),
                    'theta_gradient_kkm': ((theta[0, :-1] - theta[0, 1:]) /
                        (height[0, :-1] - height[0, 1:]) * 1000).tolist(),
                    'depth_m': depth.item(), 'dt_s': params['dt']})
    calls = 0
    return output


def solve(original, rhs, coefficients):
    global calls
    result = solver(original, rhs, coefficients)
    if calls == 0:
        # Positive upward flux follows from column storage above each interface;
        # surface injection only enters the bottom cell, below these integrals.
        tendency = (result[0] - original[0]).double() / samples[-1]['dt_s']
        samples[-1]['water_flux_kgm2day'] = (torch.cumsum(tendency * mass, 0)[:-1] * 86400).tolist()
    calls += 1
    return result


uw.uw_diffusivity = diffusivity
uw.solve_scalar = solve
sys.argv = ['diagnose_atm407_budget.py', '--reference',
            str(root / 'notebooks/data/atm407_equilibrium_20level.npz'),
            '--config', str(root / 'scm/configs/atm407.toml'), '--ocean-depth', '5',
            '--days', '2', '--surface-coupling', 'boundary_layer',
            '--bl-scheme', 'uw_moist', '--output',
            str(root / 'outputs/column/diagnostics/uw_transport_trace_budget.json')]
runpy.run_path(str(root / 'scripts/diagnose_atm407_budget.py'), run_name='__main__')
result = {'samples': samples, 'means': {
    key: np.mean([sample[key] for sample in samples], axis=0).tolist()
    for key in samples[0]}}
path = root / 'outputs/column/diagnostics/uw_transport_trace.json'
path.write_text(json.dumps(result, indent=2) + '\n')
print(path)
