import json
from pathlib import Path
import sys

import torch

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scm.case_benchmarks import run_bomex, run_dry_mixed_layer
from scm.thermo import make_grid


results = {}
for levels in [20, 40, 80]:
    grid = make_grid(levels)
    results[str(levels)] = {
        'dry_mixed_layer': run_dry_mixed_layer(grid),
        'bomex_boundary_layer_only': run_bomex(grid, use_shallow=False),
        'bomex_with_shallow': run_bomex(grid, use_shallow=True),
    }

output = root / 'outputs' / 'column' / 'benchmarks' / 'scm_case_benchmarks.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + '\n')
print(output)
print(json.dumps(results, indent=2))
