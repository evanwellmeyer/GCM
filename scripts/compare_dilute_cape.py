"""Dilute CAPE on the BOMEX sounding and the lab column, across parcel mixing rates.

As run on 12 Sep 2026. Settings are in the block near the top; edit them to
re-run a variant. See docs/column_open_problems.md.
"""
import sys
import numpy as np
import torch
sys.path.insert(0, '/Users/evanwellmeyer/Documents/GCM')
sys.path.insert(0, '/Users/evanwellmeyer/Documents/GCM/scripts')
import bomex_observed_steady_state as case
from scm.case_benchmarks import initialize_bomex
from scm.column_model import initial_state, update_derived
from scm.configuration import extract_param_overrides, load_run_config
from scm.convection_mf import _column_param, dilute_cape
from scm.ensemble import default_params
from scm.thermo import make_grid

params = default_params(); params.update(extract_param_overrides(load_run_config('scm/configs/atm407.toml')))
step = params.get('mf_cape_max_pressure_step', 1000.0)
print('production entrainment_rate', params.get('entrainment_rate'), 'per Pa; max pressure step', step)

grid = make_grid(20)
bomex, _ = initialize_bomex(grid); bomex = case.extend_above_case_top(bomex, grid)

ref = np.load('notebooks/data/atm407_equilibrium_20level.npz')
print('reference keys:', {k: ref[k].shape for k in ref.files})
rce = update_derived(initial_state(1, grid, params), grid)
for name in ['t', 'q', 'ps', 'qc']:
    if name in ref.files:
        rce[name] = torch.as_tensor(ref[name], dtype=rce[name].dtype).reshape(rce[name].shape)
rce = update_derived(rce, grid)

def cape(state, rate):
    local = dict(params, entrainment_rate=rate)
    t, q, p = state['t'], state['q'], state['p']
    entrainment = _column_param(local, 'entrainment_rate', 5.0e-6, t, 1)
    retain = torch.zeros_like(_column_param(local, 'mf_condensate_retention', 0.25, t, 1))
    fallout = torch.ones_like(retain)
    return float(dilute_cape(t, q, p, entrainment, condensate_retention=retain,
                             condensate_fallout=fallout, max_pressure_step=step)[0])

print(' rate per Pa  ~per km near sfc   BOMEX CAPE   RCE reference CAPE')
for rate in [5e-6, 2e-5, 5e-5, 1e-4, 1.5e-4]:
    print(f'   {rate:8.1e}     {rate * 11.3e3:6.2f}          {cape(bomex, rate):8.0f}     {cape(rce, rate):8.0f}')
