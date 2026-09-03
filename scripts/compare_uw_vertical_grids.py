from pathlib import Path
import argparse
import json
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scm.case_benchmarks import run_bomex, run_dry_mixed_layer
from scm.thermo import make_grid, make_smooth_test_grid


parser = argparse.ArgumentParser()
parser.add_argument('--hours', type=float, default=6.0)
parser.add_argument('--timestep', type=float, default=900.0)
parser.add_argument('--output', type=Path)
args = parser.parse_args()

grids = {
    'historical_20level': make_grid(20),
    'smooth_20level_v1': make_smooth_test_grid(),
}
results = {}
for name, grid in grids.items():
    dry = run_dry_mixed_layer(
        grid,
        hours=args.hours,
        timestep=args.timestep,
        scheme='uw',
    )
    bomex = run_bomex(
        grid,
        hours=args.hours,
        timestep=args.timestep,
        scheme='uw',
        shallow_scheme='uw',
        parameter_updates={'uw_shallow_layer_mean_saturation': True},
    )
    results[name] = {
        'layer_thickness_hpa': (grid['dsigma'] * 1000.0).tolist(),
        'dry_mixed_layer': dry,
        'bomex': bomex,
    }

output = args.output
if output is None:
    output = root / 'outputs' / 'column' / 'diagnostics' / 'uw_vertical_grid_comparison.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + '\n')
print(output)
print(json.dumps(results, indent=2))
