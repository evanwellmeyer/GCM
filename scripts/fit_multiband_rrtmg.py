"""Fit an eight-stream grey longwave approximation to matched-column RRTMG."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scripts.compare_radiation_rrtmg import rrtmg
from scm.column_model import initial_state, update_derived
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params
from scm.radiation_schemes.multiband import compute_longwave_multiband
from scm.thermo import make_grid, saturation_specific_humidity


def inverse(value, maximum):
    fraction = torch.as_tensor(value, dtype=torch.float64) / maximum
    fraction = fraction.clamp(1e-6, 1 - 1e-6)
    return torch.log(fraction / (1 - fraction))


def profiles(reference, grid, params):
    base = initial_state(1, grid, params)
    for name in ('t', 'q', 'qc', 'cloud_fraction'):
        base[name][0] = torch.as_tensor(reference[name], dtype=base[name].dtype)
    base['ts'][0] = float(reference['ts'])
    base['ps'][0] = float(reference['ps'])
    base = update_derived(base, grid)
    sigma = grid['sigma_full'].unsqueeze(0)
    free = ((sigma >= .20) & (sigma <= .75)).to(base['t'].dtype)
    cases = []
    definitions = [
        ('base', 0., 0., 1.),
        ('warm', 4., 0., 1.), ('cold', -4., 0., 1.),
        ('surface_warm', 0., 4., 1.), ('surface_cold', 0., -4., 1.),
        ('free_warm', 3., 0., 1.), ('free_cold', -3., 0., 1.),
        ('dry_half', 0., 0., .5), ('dry_three_quarters', 0., 0., .75),
        ('moist_quarter', 0., 0., 1.25), ('moist_half', 0., 0., 1.5),
        ('warm_moist', 4., 0., 1.25), ('cold_dry', -4., 0., .75),
        ('free_warm_dry', 3., 0., .75), ('free_cold_moist', -3., 0., 1.25),
    ]
    for name, atmospheric, surface, humidity in definitions:
        state = {key: value.clone() if torch.is_tensor(value) else value
                 for key, value in base.items()}
        if name.startswith('free_'):
            state['t'] = state['t'] + atmospheric * free
        else:
            state['t'] = state['t'] + atmospheric
            state['ts'] = state['ts'] + atmospheric
        state['ts'] = state['ts'] + surface
        saturation = saturation_specific_humidity(state['t'], state['p'])
        state['q'] = torch.minimum(state['q'] * humidity, .98 * saturation)
        cases.append((name, update_derived(state, grid)))
    return cases


def combine(cases):
    combined = {}
    for key, value in cases[0][1].items():
        if torch.is_tensor(value):
            combined[key] = torch.cat([state[key] for _, state in cases], dim=0)
        else:
            combined[key] = value
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(7)
    configpath = root / 'scm/configs/atm407.toml'
    referencepath = root / 'notebooks/data/atm407_equilibrium_20level.npz'
    params = default_params()
    params.update(extract_param_overrides(load_run_config(configpath)))
    with np.load(referencepath) as reference:
        grid = make_grid(len(reference['sigma_full']), dtype=torch.float64)
        cases = profiles(reference, grid, params)
    targets = []
    for name, state in cases:
        target = rrtmg(state, grid, params)
        targets.append((target['olr_wm2'], target['downward_flux_wm2'][-1],
                        target['heating_kday']))
    targetolr = torch.tensor([item[0] for item in targets], dtype=torch.float64)
    targetdown = torch.tensor([item[1] for item in targets], dtype=torch.float64)
    targetheat = torch.tensor([item[2] for item in targets], dtype=torch.float64)
    state = combine(cases)
    edges = [10, 350, 500, 630, 700, 820, 980, 1080, 1180,
             1390, 1480, 1800, 2080, 2250, 2380, 2600, 3250]
    bands = len(edges) - 1
    kappas = torch.logspace(-6, 1, bands, dtype=torch.float64)
    carbons = torch.logspace(-5, 1, bands, dtype=torch.float64)
    traces = torch.logspace(-5, .5, bands, dtype=torch.float64)
    continua = torch.logspace(-4, 2, bands, dtype=torch.float64)
    rawkappas = torch.nn.Parameter(inverse(kappas, 20.))
    rawcarbons = torch.nn.Parameter(inverse(carbons, 20.))
    rawtraces = torch.nn.Parameter(inverse(traces, 20.))
    rawcontinua = torch.nn.Parameter(inverse(continua, 500.))
    variables = [rawkappas, rawcarbons, rawtraces, rawcontinua]
    optimizer = torch.optim.Adam(variables, lr=.015)
    mass = state['dp'].double()
    massweight = mass / mass.sum(dim=1, keepdim=True)
    training = torch.tensor([i % 3 != 2 for i in range(len(cases))])
    for step in range(args.steps):
        optimizer.zero_grad()
        fitted = dict(params,
                      lw_band_edges_cm1=edges,
                      lw_band_wv_kappa=20 * torch.sigmoid(rawkappas),
                      lw_band_co2_base_tau=20 * torch.sigmoid(rawcarbons),
                      lw_band_trace_scale=20 * torch.sigmoid(rawtraces),
                      lw_band_wv_continuum=500 * torch.sigmoid(rawcontinua),
                      lw_band_co2_log_factor=[0.] * bands,
                      lw_band_o3_scale=[0.] * bands,
                      o3_lw_tau=0.)
        heating, down, olr = compute_longwave_multiband(state, grid, fitted, force_clear_sky=True)
        heating = heating * 86400
        difference = heating - targetheat
        heatloss = torch.sum(difference.square() * massweight, dim=1)
        heatloss = heatloss + .25 * difference.square().mean(dim=1)
        lossbycase = heatloss + ((olr - targetolr) / 5).square() + ((down - targetdown) / 5).square()
        loss = lossbycase[training].mean()
        loss.backward()
        optimizer.step()
    coefficients = {
        'lw_band_edges_cm1': edges,
        'lw_band_wv_kappa': (20 * torch.sigmoid(rawkappas)).detach().tolist(),
        'lw_band_co2_base_tau': (20 * torch.sigmoid(rawcarbons)).detach().tolist(),
        'lw_band_trace_scale': (20 * torch.sigmoid(rawtraces)).detach().tolist(),
        'lw_band_wv_continuum': (500 * torch.sigmoid(rawcontinua)).detach().tolist(),
        'lw_band_co2_log_factor': [0.] * bands,
        'lw_band_o3_scale': [0.] * bands,
    }
    fitted = dict(params, **coefficients, o3_lw_tau=0.)
    heating, down, olr = compute_longwave_multiband(state, grid, fitted, force_clear_sky=True)
    heating = heating.detach() * 86400
    records = []
    for index, (name, _) in enumerate(cases):
        difference = heating[index] - targetheat[index]
        records.append({
            'case': name,
            'set': 'training' if training[index] else 'validation',
            'multiband_olr_wm2': olr[index].item(), 'rrtmg_olr_wm2': targetolr[index].item(),
            'olr_error_wm2': (olr[index] - targetolr[index]).item(),
            'surface_down_error_wm2': (down[index] - targetdown[index]).item(),
            'heating_rms_kday': torch.sqrt(torch.sum(difference.square() * massweight[index])).item(),
            'heating_bias_kday': torch.sum(difference * massweight[index]).item(),
        })
    summary = {}
    for subset in ('training', 'validation'):
        selected = [record for record in records if record['set'] == subset]
        summary[subset] = {
            'maximum_absolute_olr_error_wm2': max(abs(item['olr_error_wm2']) for item in selected),
            'maximum_heating_rms_kday': max(item['heating_rms_kday'] for item in selected),
            'maximum_absolute_surface_down_error_wm2': max(abs(item['surface_down_error_wm2']) for item in selected),
        }
    report = {'description': 'Multi-stream grey fit to matched-column clear-sky RRTMG',
              'steps': args.steps, 'bands': bands, 'coefficients': coefficients,
              'cases': records, 'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(json.dumps(coefficients, indent=2))


if __name__ == '__main__':
    main()
