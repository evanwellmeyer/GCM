from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import time

import torch


root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scm.column_model import initial_state, run
from scm.configuration import extract_param_overrides, load_run_config
from scm.diagnostics import equilibrium_metrics, equilibrium_stats
from scm.ensemble import default_params
from scm.thermo import g, make_grid, relative_humidity


parser = ArgumentParser()
parser.add_argument('--levels', nargs='+', type=int, default=[10, 20, 40])
parser.add_argument('--days', type=int, default=100)
parser.add_argument('--surface-temperature', type=float, default=290.45)
parser.add_argument('--mb-max', type=float)
parser.add_argument('--detrain-rh', type=float)
parser.add_argument('--max-dq-day', type=float)
parser.add_argument('--output', type=Path)
args = parser.parse_args()

results = []
for nlevels in args.levels:
    grid = make_grid(nlevels, device='cpu')
    params = default_params(device='cpu')
    params.update(extract_param_overrides(load_run_config()))
    params.update({
        'dt': 900.0,
        'ts_init': args.surface_temperature,
        'use_slab_ocean': False,
        'convection_scheme': 'mass_flux',
        'radiation_scheme': 'multiband',
    })
    if args.mb_max is not None:
        params['mf_mb_max'] = args.mb_max
    if args.detrain_rh is not None:
        params['mf_detrain_rh'] = args.detrain_rh
    if args.max_dq_day is not None:
        params['mf_max_dq_day'] = args.max_dq_day

    state = initial_state(1, grid, params, device='cpu')
    stepsperday = round(86400 / params['dt'])
    start = time.perf_counter()
    state, history = run(
        state,
        grid,
        params,
        args.days * stepsperday,
        rad_interval=8,
        diag_interval=stepsperday,
    )
    elapsed = time.perf_counter() - start

    window = min(50, len(history))
    stats = equilibrium_stats(history, last_n=window)
    metrics = equilibrium_metrics(history, window=window)
    rh = relative_humidity(state['q'], state['t'], state['p'])[0]
    saturated_mass = torch.sum((rh >= 0.95) * state['dp'][0] / g)
    column_mass = torch.sum(state['dp'][0] / g)

    result = {
        'levels': nlevels,
        'days': args.days,
        'surface_temperature_k': args.surface_temperature,
        'cape_jkg': stats['cape_mean'][0].item(),
        'deep_precipitation_mmday': stats['precip_conv_mean'][0].item() * 86400,
        'large_scale_precipitation_mmday': stats['precip_ls_mean'][0].item() * 86400,
        'cloud_precipitation_mmday': stats['precip_cloud_mean'][0].item() * 86400,
        'total_precipitation_mmday': stats['precip_total_mean'][0].item() * 86400,
        'toa_net_wm2': stats['toa_net_mean'][0].item(),
        'surface_total_flux_wm2': stats['surface_total_flux_mean'][0].item(),
        'atmospheric_energy_residual_wm2': stats['atmos_energy_residual_mean'][0].item(),
        'fixed_surface_column_residual_wm2': stats['column_energy_residual_mean'][0].item(),
        'rh95_mass_fraction': (saturated_mass / column_mass).item(),
        'cloud_base_mass_flux_kgm2s': stats['cloud_base_mass_flux_mean'][0].item(),
        'mass_flux_cap_fraction': stats['mass_flux_cap_active_mean'][0].item(),
        'temperature_cap_fraction': stats['temperature_cap_fraction_mean'][0].item(),
        'moisture_cap_fraction': stats['moisture_cap_fraction_mean'][0].item(),
        'temperature_drift_k': metrics['max_ts_window_drift'],
        'runtime_s': elapsed,
    }
    results.append(result)
    print(json.dumps(result, indent=2), flush=True)

if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
