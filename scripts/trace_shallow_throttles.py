"""Simple shallow scheme tendencies at the BOMEX start, removing one throttle at a time.

As run on 12 Sep 2026. Settings are in the block near the top; edit them to
re-run a variant. See docs/column_open_problems.md.
"""
import sys
sys.path.insert(0, '/Users/evanwellmeyer/Documents/GCM')
sys.path.insert(0, '/Users/evanwellmeyer/Documents/GCM/scripts')
import torch
import bomex_observed_steady_state as case
from scm.case_benchmarks import initialize_bomex, model_height
from scm.configuration import extract_param_overrides, load_run_config
from scm.convection_shallow import shallow_convection
from scm.ensemble import default_params
from scm.thermo import cape, make_grid, relative_humidity

params = default_params(); params.update(extract_param_overrides(load_run_config('scm/configs/atm407.toml')))
params['dt'] = 900.0
grid = make_grid(20)
state, _ = initialize_bomex(grid); state = case.extend_above_case_top(state, grid)
height = model_height(state, grid)[0]
rh = relative_humidity(state['q'], state['t'], state['p'])[0]
undilute = float(cape(state['t'], state['q'], state['p'], grid)[0])
print(f"undilute CAPE {undilute:.0f} J/kg, CAPE factor {1 / (1 + undilute / params['shallow_cape_suppress']):.3f}, "
      f"detrain_rh {params['shallow_detrain_rh']}, caps {params['shallow_max_dt_day']} K/day {params['shallow_max_dq_day']} g/kg/day")
variants = {
    'production': {},
    'detrain_rh=1': {'shallow_detrain_rh': 1.0},
    'no CAPE cut': {'shallow_cape_suppress': 1.0e12},
    'no caps': {'shallow_max_dt_day': 1.0e9, 'shallow_max_dq_day': 1.0e9},
    'all three': {'shallow_detrain_rh': 1.0, 'shallow_cape_suppress': 1.0e12,
                  'shallow_max_dt_day': 1.0e9, 'shallow_max_dq_day': 1.0e9},
}
outputs = {name: shallow_convection(state, grid, dict(params, **update)) for name, update in variants.items()}
for field, scale, label in [('dq', 86400e3, 'water g/kg/day'), ('dt', 86400.0, 'temperature K/day')]:
    print(f'\nshallow {label} at the BOMEX start, 20 levels')
    print('  height    RH ' + ''.join(f'{name:>14}' for name in variants))
    for i in range(19, -1, -1):
        if float(height[i]) > 2500: continue
        print(f'  {float(height[i]):6.0f}  {float(rh[i]):4.2f} ' + ''.join(f'{float(outputs[n][field][0, i]) * scale:14.2f}' for n in variants))
