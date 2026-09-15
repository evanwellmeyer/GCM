"""Count how many BOMEX steps keep the UW shallow plume, with the crossing switch off and on.

As run on 12 Sep 2026. Settings are in the block near the top; edit them to
re-run a variant. See docs/column_open_problems.md.
"""
import sys
sys.path.insert(0, '/Users/evanwellmeyer/Documents/GCM')
sys.path.insert(0, '/Users/evanwellmeyer/Documents/GCM/scripts')
import torch
import bomex_observed_steady_state as case
import scm.column_model as column_model
from scm.case_benchmarks import initialize_bomex
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params
from scm.thermo import make_grid

original = column_model.run_physics_scheme
captured = []
def scheme(category, name, state, grid, local):
    out = original(category, name, state, grid, local)
    if category == 'shallow_convection':
        captured.append((float(out['plume_top_height_m'][0]), float(out['cloud_base_mass_flux'][0]),
                         float(out['plume_cloud_base_height_m'][0])))
    return out
column_model.run_physics_scheme = scheme
column_model.radiation = case.prescribed_radiation
column_model.surface_fluxes = case.observed_surface_fluxes

for keep in (False, True):
    base = default_params(); base.update(extract_param_overrides(load_run_config('scm/configs/uw_candidate_v1.toml')))
    base.update({'dt': 900.0, 'use_slab_ocean': False, 'ps0': 101500.0, 'surface_temperature': case.SEA_SURFACE_TEMPERATURE,
                 'cloud_ls_precip_fraction': 0.0, 'entrainment_rate': 5.0e-5, 'uw_shallow_keep_crossed_interface': keep})
    grid = make_grid(20)
    state, _ = initialize_bomex(grid); state = case.extend_above_case_top(state, grid)
    captured.clear()
    for n in range(24):
        heating, moistening = case.forcing_tendencies(state, grid)
        state, diag, _ = column_model.physics_step(state, grid, base, ls_forcing={'dt': heating, 'dq': moistening})
    tops = [c[0] for c in captured]
    kept = sorted(top for top in tops if top > 0)
    median = kept[len(kept) // 2] if kept else 0.0
    print(f'keep crossed {keep}: kept in {len(kept)} of {len(tops)} steps, median top {median:.0f} m, '
          f'tops {[round(x) for x in tops]}, mean cloud-base mass flux {sum(c[1] for c in captured) / len(captured):.4f}')
