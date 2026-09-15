"""Evolve the archived late BOMEX shallow column for one 900-second interval.

Only shallow physics evolves T, vapor, condensate and winds. Pressure, native
TKE and boundary-layer depth stay fixed: this is not a full-column integration.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'scripts'))
from test_bomex_launch_closure import run_trial
from scm.thermo import cp, Lv as latent, g, p0, kappa


def compare(sample, mode, timestep):
    initial = {key: torch.tensor(sample['plume_input'][key], dtype=torch.float32)
               for key in ('t', 'q', 'qc', 'u', 'v', 'p', 'dp', 'tke')}
    initial['boundary_layer_depth_m'] = torch.tensor(sample['plume_input']['boundary_depth'])
    initial['tke_interfaces'] = torch.tensor(sample['plume_input']['interfaces'])
    state = {key: value.clone() for key, value in initial.items()}
    steps = []
    count = round(900 / timestep)
    if abs(count * timestep - 900) > 1e-8:
        raise ValueError('substeps must exactly cover 900 seconds')
    for index in range(count):
        trial, state = run_trial(sample, mode, state, timestep, return_state=True)
        if trial['source_liquid_kgkg'] <= 0.:
            raise RuntimeError('unsaturated source needs an LCL-aware closure')
        steps.append(trial)
        print(f'{mode} dt={timestep:g}: step {index + 1}/{count}', flush=True)
    mass = initial['dp'].double() / g
    water = state['q'].double() + state['qc'].double() - initial['q'].double() - initial['qc'].double()
    energy = cp * (state['t'].double() - initial['t'].double()) + latent * (state['q'].double() - initial['q'].double())
    exner = (initial['p'].double() / p0) ** kappa
    theta = ((state['t'].double() - initial['t'].double()) - latent / cp * (state['qc'].double() - initial['qc'].double())) / exner
    precipitation = sum(trial['precipitation_kgm2s'] * timestep for trial in steps)
    return {'mode': mode, 'timestep_s': timestep, 'water_change_gkg': (water[0] * 1000).tolist(),
            'theta_change_k': theta[0].tolist(), 'water_change_825_gkg': float(water[0, 15] * 1000),
            'precipitation_kgm2': precipitation,
            'integrated_water_residual_kgm2s': (float((water * mass).sum()) + precipitation) / 900,
            'integrated_energy_residual_wm2': float((energy * mass).sum()) / 900, 'steps': steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--substeps', type=float, nargs='+', default=[900., 450., 225.])
    parser.add_argument('--internal-step', type=float, default=0., help='maximum internal shallow step; zero keeps the original path')
    parser.add_argument('--joint-only', action='store_true')
    parser.add_argument('--output', type=Path, default=root / 'outputs/column/diagnostics/bomex_inversion_substeps.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = json.loads((root / 'outputs/column/diagnostics/bomex_launch_budget_20_6h.json').read_text())
    sample = saved['samples'][-1]
    sample['plume_params']['uw_shallow_maximum_timestep_s'] = args.internal_step
    results = []
    trials = [] if args.joint_only else [('baseline', 900.)]
    for mode, timestep in trials + [('joint', value) for value in args.substeps]:
        result = compare(sample, mode, timestep)
        results.append(result)
        args.output.write_text(json.dumps({'saved_hour': sample['hour'], 'duration_s': 900., 'internal_step_s': args.internal_step, 'results': results}, indent=2) + '\n')
        print(f"water change at 825 m: {result['water_change_825_gkg']:+.6f} g/kg", flush=True)


if __name__ == '__main__':
    main()
