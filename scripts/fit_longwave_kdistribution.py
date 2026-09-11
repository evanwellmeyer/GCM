"""Fit a k-distribution longwave (bands x g-points) to matched-column clear-sky RRTMG.

Each spectral band is split into g-points that share its Planck emission in fitted
fractions and carry their own water-vapour and CO2 absorption strengths, as RRTMG
does. Trained on the 15 perturbed profiles of fit_multiband_rrtmg.py plus a CO2
doubling of the base profile, so the CO2 forcing is fitted, not tuned afterwards.
Acceptance criteria are recorded in docs/column_open_problems.md (Tier 2 item 6).
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scripts.compare_radiation_rrtmg import rrtmg
from scripts.fit_multiband_rrtmg import combine, profiles
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params
from scm.radiation_schemes.multiband import compute_longwave_multiband
from scm.thermo import make_grid

EDGES = {
    'four': [10, 500, 820, 1180, 3250],
    'rrtmg16': [10, 350, 500, 630, 700, 820, 980, 1080, 1180,
                1390, 1480, 1800, 2080, 2250, 2380, 2600, 3250],
}


def bounded(raw, low, high):
    """Map an unbounded parameter to [low, high] on a logarithmic scale."""
    return torch.exp(math.log(low) + (math.log(high) - math.log(low)) * torch.sigmoid(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edges', choices=sorted(EDGES), default='four')
    parser.add_argument('--gpoints', type=int, default=3)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--shares', choices=('per-band', 'shared'), default='per-band')
    parser.add_argument('--pressure-exponent', action='store_true',
                        help='also fit a per-g-point pressure exponent on water-vapour absorption')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(7)
    params = default_params()
    params.update(extract_param_overrides(load_run_config(root / 'scm/configs/atm407.toml')))
    co2 = float(params.get('co2', 400.0))
    with np.load(root / 'notebooks/data/atm407_equilibrium_20level.npz') as reference:
        grid = make_grid(len(reference['sigma_full']), dtype=torch.float64)
        cases = profiles(reference, grid, params)
    names = [name for name, _ in cases] + ['base_2xco2']
    targets = [rrtmg(state, grid, params) for _, state in cases]
    targets.append(rrtmg(cases[0][1], grid, dict(params, co2=2 * co2)))
    target_olr = torch.tensor([t['olr_wm2'] for t in targets], dtype=torch.float64)
    target_down = torch.tensor([t['downward_flux_wm2'][-1] for t in targets], dtype=torch.float64)
    target_heat = torch.tensor([t['heating_kday'] for t in targets], dtype=torch.float64)
    rrtmg_forcing = targets[0]['olr_wm2'] - targets[-1]['olr_wm2']
    state = combine(cases + [('base_2xco2', cases[0][1])])
    co2_by_case = torch.full((len(names),), co2, dtype=torch.float64)
    co2_by_case[-1] = 2 * co2

    edges = EDGES[args.edges]
    bands, points = len(edges) - 1, args.gpoints
    spread = torch.logspace(-4, 2, bands * points, dtype=torch.float64).view(points, bands).T.flatten()
    raw_kappa = torch.nn.Parameter(torch.logit((torch.log(spread) - math.log(1e-6)) / (math.log(2000.) - math.log(1e-6))))
    raw_carbon = torch.nn.Parameter(torch.zeros(bands * points, dtype=torch.float64))
    raw_continuum = torch.nn.Parameter(torch.full((bands,), -4.0, dtype=torch.float64))
    raw_trace = torch.nn.Parameter(torch.full((bands,), -2.0, dtype=torch.float64))
    raw_log = torch.nn.Parameter(torch.full((bands,), -2.0, dtype=torch.float64))
    raw_fraction = torch.nn.Parameter(torch.zeros(bands if args.shares == 'per-band' else 1, points, dtype=torch.float64))
    raw_exponent = torch.nn.Parameter(torch.zeros(bands * points, dtype=torch.float64))
    variables = [raw_kappa, raw_carbon, raw_continuum, raw_trace, raw_log, raw_fraction]
    if args.pressure_exponent:
        variables.append(raw_exponent)

    def coefficients():
        return dict(
            lw_band_edges_cm1=edges,
            lw_gpoint_fractions=torch.softmax(raw_fraction, 1).flatten(),
            lw_band_wv_kappa=bounded(raw_kappa, 1e-6, 2000.),
            lw_band_co2_base_tau=bounded(raw_carbon, 1e-5, 2000.),
            lw_band_wv_continuum=500 * torch.sigmoid(raw_continuum),
            lw_band_trace_scale=20 * torch.sigmoid(raw_trace),
            lw_band_co2_log_factor=5 * torch.sigmoid(raw_log),
            lw_band_o3_scale=[0.] * bands,
            **({'lw_band_wv_pressure_exponent': 1.5 * torch.tanh(raw_exponent)} if args.pressure_exponent else {}),
        )

    def model(state, co2_values):
        return compute_longwave_multiband(
            state, grid, dict(params, **coefficients(), o3_lw_tau=0., co2=co2_values),
            force_clear_sky=True)

    mass = state['dp'].double()
    weight = mass / mass.sum(dim=1, keepdim=True)
    training = torch.tensor([i % 3 != 2 for i in range(len(cases))] + [True])
    optimizer = torch.optim.Adam(variables, lr=.02)
    for step in range(args.steps):
        optimizer.zero_grad()
        heating, down, olr = model(state, co2_by_case)
        difference = heating * 86400 - target_heat
        loss = (torch.sum(difference.square() * weight, dim=1) + .25 * difference.square().mean(dim=1)
                + ((olr - target_olr) / 2).square() + ((down - target_down) / 5).square())
        forcing_error = (olr[0] - olr[-1]) - rrtmg_forcing
        total = loss[training].mean() + (forcing_error / .2).square()
        total.backward()
        optimizer.step()

    with torch.no_grad():
        heating, down, olr = model(state, co2_by_case)
        heating = heating * 86400
        fitted = {key: (value.tolist() if torch.is_tensor(value) else value) for key, value in coefficients().items()}
    pressure = np.array(targets[0]['pressure_hpa'])
    records = []
    for index, name in enumerate(names):
        difference = (heating[index] - target_heat[index]).numpy()
        records.append({
            'case': name, 'set': 'training' if training[index] else 'validation',
            'olr_error_wm2': float(olr[index] - target_olr[index]),
            'surface_down_error_wm2': float(down[index] - target_down[index]),
            'heating_rms_kday': float(np.sqrt(np.sum(difference ** 2 * weight[index].numpy()))),
        })
    base = (heating[0] - target_heat[0]).numpy()
    upper, strat = (pressure >= 230) & (pressure <= 470), pressure <= 80
    single = combine([cases[0]])
    start = time.perf_counter()
    for _ in range(50):
        compute_longwave_multiband(single, grid, dict(params, **{k: torch.as_tensor(v) if isinstance(v, list) and k != 'lw_band_edges_cm1' else v for k, v in fitted.items()}, o3_lw_tau=0.), force_clear_sky=True)
    cost_ms = (time.perf_counter() - start) / 50 * 1000
    profiles_only = records[:-1]
    checks = {
        'heating_rms_all_profiles_max_kday': max(r['heating_rms_kday'] for r in profiles_only),
        'olr_error_all_profiles_max_wm2': max(abs(r['olr_error_wm2']) for r in profiles_only),
        'surface_down_error_all_profiles_max_wm2': max(abs(r['surface_down_error_wm2']) for r in profiles_only),
        'reference_230_470hpa_max_error_kday': float(np.abs(base[upper]).max()),
        'reference_10_80hpa_max_error_kday': float(np.abs(base[strat]).max()),
        'reference_lowest_layer_error_kday': float(base[-1]),
        'co2_doubling_forcing_wm2': float(olr[0] - olr[-1]),
        'rrtmg_co2_doubling_forcing_wm2': rrtmg_forcing,
        'cost_ms_per_call': cost_ms,
    }
    report = {'description': __doc__.splitlines()[0], 'edges': args.edges, 'gpoints': points, 'shares': args.shares, 'pressure_exponent': args.pressure_exponent,
              'steps': args.steps, 'coefficients': fitted, 'checks': checks, 'cases': records,
              'reference_heating_error_kday': dict(zip([f'{p:.0f}' for p in pressure], base.tolist()))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
