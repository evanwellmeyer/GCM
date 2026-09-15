"""Compare conserved environmental reconstruction on saved BOMEX states."""

import argparse
import json
from pathlib import Path

import torch

from test_bomex_launch_closure import run_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hour', type=float, default=3.)
    parser.add_argument('--timestep', type=float, default=225.)
    parser.add_argument('--only-conserved', action='store_true')
    parser.add_argument('--double', action='store_true')
    parser.add_argument('--internal-step', type=float, default=0.)
    parser.add_argument('--partition', choices=['host', 'saturation'], default='host')
    parser.add_argument('--sorting-distance', type=float)
    parser.add_argument('--diagnose-sorting', action='store_true')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / 'outputs/column/diagnostics'
    saved = json.loads((directory / 'bomex_launch_budget_20_6h.json').read_text())
    sample = next(item for item in saved['samples'] if item['hour'] == args.hour)
    sample['plume_params']['uw_shallow_maximum_timestep_s'] = args.internal_step
    sample['plume_params']['uw_shallow_environment_partition'] = args.partition
    if args.sorting_distance is not None:
        sample['plume_params']['uw_shallow_sorting_distance_m'] = args.sorting_distance
    sample['plume_params']['uw_shallow_diagnose_sorting_distance'] = args.diagnose_sorting
    torch.set_num_threads(1)
    results = []
    for enabled in ((True,) if args.only_conserved else (False, True)):
        sample['plume_params']['uw_shallow_conserved_environment'] = enabled
        evolved = None
        if args.double:
            fields = sample['plume_input']
            evolved = {name: torch.tensor(fields[name], dtype=torch.float64)
                       for name in ('t', 'q', 'qc', 'u', 'v', 'p', 'dp', 'tke')}
            evolved['boundary_layer_depth_m'] = torch.tensor(fields['boundary_depth'], dtype=torch.float64)
            evolved['tke_interfaces'] = torch.tensor(fields['interfaces'], dtype=torch.float64)
        result = run_trial(sample, 'baseline', evolving=evolved, timestep=args.timestep)
        result['conserved_environment'] = enabled
        result['double_precision'] = args.double
        result['timestep_s'] = args.timestep
        result['internal_step_s'] = args.internal_step
        result['environment_partition'] = args.partition
        result['sorting_distance_m'] = sample['plume_params'].get('uw_shallow_sorting_distance_m', 100.)
        result['diagnosed_sorting'] = args.diagnose_sorting
        results.append(result)
        print(f"conserved={enabled}: water {result['water_tendency_825_gkgday']:.6f} g/kg/day, energy residual {result['energy_residual_wm2']:.6f} W/m2", flush=True)
        suffix = f'{args.hour:g}_{args.timestep:g}' + ('_double' if args.double else '')
        if args.internal_step:
            suffix += f'_internal{args.internal_step:g}'
        suffix += '_' + args.partition
        if args.sorting_distance is not None:
            suffix += f'_distance{args.sorting_distance:g}'
        if args.diagnose_sorting:
            suffix += '_diagnosed'
        (directory / f'bomex_conserved_environment_{suffix}.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
