"""UW diffusivity, boundary-layer depth and shallow mass flux through a BOMEX run.

As run on 12 Sep 2026. Settings are in the block near the top; edit them to
re-run a variant. See docs/column_open_problems.md.
"""
import sys
sys.path.insert(0, '/Users/evanwellmeyer/Documents/GCM')
sys.path.insert(0, '/Users/evanwellmeyer/Documents/GCM/scripts')
import torch
import bomex_observed_steady_state as case
import scm.column_model as column_model
from scm.case_benchmarks import initialize_bomex, model_height
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params
from scm.thermo import make_grid

base = default_params(); base.update(extract_param_overrides(load_run_config('scm/configs/uw_candidate_v1.toml')))
base.update({'dt': 900.0, 'use_slab_ocean': False, 'ps0': 101500.0, 'surface_temperature': case.SEA_SURFACE_TEMPERATURE,
             'cloud_ls_precip_fraction': 0.0, 'entrainment_rate': 5.0e-5})
column_model.radiation = case.prescribed_radiation
column_model.surface_fluxes = case.observed_surface_fluxes
original = column_model.run_physics_scheme
step = [0]
captured = {}
def scheme(category, name, state, grid, local):
    out = original(category, name, state, grid, local)
    if step[0] in (0, 12, 23):
        if category == 'boundary_layer':
            captured[step[0], 'K'] = out['heat_diffusivity'][0].clone()
            captured[step[0], 'depth'] = float(out['boundary_layer_depth_m'][0])
            captured[step[0], 'height'] = model_height(state, grid)[0].clone()
            captured[step[0], 'water'] = (state['q'] + state['qc'])[0].clone()
        if category == 'shallow_convection':
            mb = out.get('cloud_base_mass_flux')
            captured[step[0], 'mb'] = float(mb[0]) if mb is not None else float('nan')
    return out
column_model.run_physics_scheme = scheme

for levels in [20, 40]:
    grid = make_grid(levels)
    state, _ = initialize_bomex(grid); state = case.extend_above_case_top(state, grid)
    captured.clear()
    for n in range(24):
        step[0] = n
        heating, moistening = case.forcing_tendencies(state, grid)
        state, diag, _ = column_model.physics_step(state, grid, base, ls_forcing={'dt': heating, 'dq': moistening})
    print(f'\n{levels} levels')
    for n in (0, 12, 23):
        height = captured[n, 'height']; k = captured[n, 'K']; water = captured[n, 'water']
        print(f'  hour {n * 0.25:4.2f}: BL depth {captured[n, "depth"]:.0f} m, shallow cloud-base mass flux {captured[n, "mb"]:.4f} kg/m2/s')
        cells = []
        for i in range(k.shape[0] - 1, -1, -1):
            z = 0.5 * (float(height[i]) + float(height[i + 1]))
            if z > 1500: break
            cells.append(f'{z:5.0f}m:{float(k[i]):6.1f}')
        print('    K m2/s at interfaces: ' + '  '.join(cells))
        print('    water g/kg by level:  ' + '  '.join(f'{float(height[i]):5.0f}m:{float(water[i]) * 1e3:5.2f}' for i in range(levels - 1, -1, -1) if float(height[i]) <= 1500))
