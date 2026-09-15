"""One 20-level BOMEX screen with the conserved-environment correction."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

import bomex_observed_steady_state as case
from scm import column_model as column
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=6.)
    parser.add_argument('--internal-step', type=float, default=225.)
    parser.add_argument('--partition', choices=['host', 'saturation'], default='host')
    parser.add_argument('--diagnose-sorting', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    params = default_params()
    params.update(extract_param_overrides(load_run_config(root / 'scm/configs/uw_candidate_v1.toml')))
    params.update(cloud_ls_precip_fraction=0., entrainment_rate=5e-5, uw_layer_closure=True,
                  uw_shallow_keep_crossed_interface=True, uw_shallow_conserved_environment=True,
                  uw_shallow_environment_partition=args.partition,
                  uw_shallow_diagnose_sorting_distance=args.diagnose_sorting,
                  uw_shallow_maximum_timestep_s=args.internal_step)
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
              ['scm/convection_uw.py', 'scm/column_model.py', 'scm/boundary_layer_uw.py',
               'scm/case_benchmarks.py', 'scripts/bomex_observed_steady_state.py']}
    original = column.physics_step
    count = 0

    def progress(*values, **kwargs):
        nonlocal count
        result = original(*values, **kwargs)
        count += 1
        print(f'completed {count * .25:g} / {args.hours:g} model hours', flush=True)
        return result

    with patch.object(column, 'surface_fluxes', case.observed_surface_fluxes), \
         patch.object(column, 'radiation', case.prescribed_radiation), \
         patch.object(column, 'physics_step', progress):
        result = case.run_case(20, args.hours, 900., params)
    result['internal_step_s'] = args.internal_step
    result['environment_partition'] = args.partition
    result['diagnosed_sorting'] = args.diagnose_sorting
    result['hours'] = args.hours
    if args.hours != 6.:
        result['rain_late_window_mmday'] = result.pop('rain_hours_3_to_6_mmday')
        result['short_window_threshold_checks'] = result.pop('passes')
        result['six_hour_validation'] = False
    result['sha256'] = hashes
    suffix = '_diagnosed' if args.diagnose_sorting else ''
    output = root / f'outputs/column/diagnostics/bomex_conserved_environment_{args.hours:g}h_{args.partition}{suffix}.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
