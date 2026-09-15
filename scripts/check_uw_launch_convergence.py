"""Replay fixed failing plume states at different numerical vertical steps."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from scm.convection_uw import _integrate_columns


def check():
    fixtures = json.loads((root / 'scm/testdata/uw_launch_frozen_20260914.json').read_text())
    results = []
    for fixture in fixtures:
        arrays = [torch.tensor(value, dtype=torch.float32) for value in fixture['input'].values()]
        trials = []
        for spacing in (50., 25., 10.):
            start = time.monotonic()
            output = _integrate_columns(*arrays, dict(fixture['params'], uw_shallow_vertical_step_m=spacing))
            trials.append({'step_m': spacing, 'top_m': float(output['plume_top_height'][0]),
                           'water_flux_kgm2day': (output['water_flux'][0] * 86400).tolist(),
                           'mse_flux_wm2': output['mse_flux'][0].tolist(),
                           'rain_source_kgm2day': (output['precipitation_source'][0] * 86400).tolist(),
                           'runtime_s': time.monotonic() - start})
            print(f"{fixture['name']} {spacing:g} m: top {trials[-1]['top_m']:.3f} m, max flux "
                  f"{max(map(abs, trials[-1]['water_flux_kgm2day'])):.4f} kg/m2/day", flush=True)
        flux = torch.tensor([trial['water_flux_kgm2day'] for trial in trials])
        energy = torch.tensor([trial['mse_flux_wm2'] for trial in trials])
        tops = torch.tensor([trial['top_m'] for trial in trials])
        watererror = float((flux - flux[-1]).abs().max())
        energyerror = float((energy - energy[-1]).abs().max())
        toperror = float((tops - tops[-1]).abs().max())
        passed = watererror < .05 and energyerror < 2. and toperror < 2.
        results.append({'name': fixture['name'], 'trials': trials, 'max_water_error_kgm2day': watererror,
                        'max_energy_error_wm2': energyerror, 'max_top_error_m': toperror, 'passed': passed})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root / 'outputs/column/diagnostics/uw_launch_convergence_corrected.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    if not all(result['passed'] for result in results):
        raise SystemExit('frozen-state convergence gate failed')
    print('All frozen-state convergence gates passed.')


if __name__ == '__main__':
    main()
