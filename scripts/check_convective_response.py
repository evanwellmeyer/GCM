"""Test the conservative convection response to controlled profile changes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scm.column_model import initial_state, update_derived
from scm.configuration import extract_param_overrides, load_run_config
from scm.convection_mf import mass_flux_convection
from scm.ensemble import default_params
from scm.thermo import Lv, cp, g, make_grid, relative_humidity, saturation_specific_humidity


def loadstate(reference, grid, params):
    state = initial_state(1, grid, params)
    for name in ('t', 'q', 'qc', 'cloud_fraction'):
        state[name][0] = torch.as_tensor(reference[name], dtype=state[name].dtype)
    state['ts'][0] = float(reference['ts'])
    state['ps'][0] = float(reference['ps'])
    return update_derived(state, grid)


def perturb(state, grid, cooling, moistening):
    changed = {name: value.clone() if torch.is_tensor(value) else value
               for name, value in state.items()}
    sigma = grid['sigma_full'].unsqueeze(0)
    free = (sigma >= .30) & (sigma <= .80)
    changed['t'] = changed['t'] - cooling * free
    saturation = saturation_specific_humidity(changed['t'], changed['p'])
    target = torch.minimum(changed['q'] * (1 + moistening), .90 * saturation)
    changed['q'] = torch.where(free, torch.maximum(changed['q'], target), changed['q'])
    return update_derived(changed, grid)


def runexperiment(state, grid, params, hours):
    timestep = float(params['dt'])
    steps = round(hours * 3600 / timestep)
    mass = state['dp'] / g
    initial = mass_flux_convection(state, grid, params)
    rain = []
    evaporation = []
    massflux = []
    limiter = []
    energyerror = []
    watererror = []
    for step in range(steps):
        output = mass_flux_convection(state, grid, params)
        rain.append(output['precip'])
        evaporation.append(output['rain_evaporation'])
        massflux.append(output['cloud_base_mass_flux'])
        limiter.append(output['transport_limiter'])
        energyerror.append(torch.sum((cp * output['dt'] + Lv * output['dq']) * mass, dim=1))
        watererror.append(torch.sum(output['dq'] * mass, dim=1) + output['precip'])
        state['t'] = state['t'] + timestep * output['dt']
        state['q'] = state['q'] + timestep * output['dq']
        state = update_derived(state, grid)
    final = mass_flux_convection(state, grid, params)
    return {
        'initial_cape_jkg': initial['cape'].item(),
        'final_cape_jkg': final['cape'].item(),
        'cape_change_jkg': final['cape'].item() - initial['cape'].item(),
        'mean_rain_mmday': torch.stack(rain).mean().item() * 86400,
        'mean_rain_evaporation_mmday': torch.stack(evaporation).mean().item() * 86400,
        'mean_cloud_base_mass_flux_kgm2s': torch.stack(massflux).mean().item(),
        'minimum_transport_limiter': torch.stack(limiter).min().item(),
        'maximum_energy_error_wm2': torch.stack(energyerror).abs().max().item(),
        'maximum_water_error_kgm2s': torch.stack(watererror).abs().max().item(),
        'maximum_relative_humidity': relative_humidity(state['q'], state['t'], state['p']).max().item(),
        'final_temperature_k': state['t'][0].tolist(),
        'final_vapor_kgkg': state['q'][0].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=6.)
    parser.add_argument('--config', type=Path,
                        default=root / 'scm/configs/atm407_flux_v1.toml')
    parser.add_argument('--reference', type=Path,
                        default=root / 'notebooks/data/atm407_equilibrium_20level.npz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configpath = args.config
    referencepath = args.reference
    params = default_params()
    params.update(extract_param_overrides(load_run_config(configpath)))
    params.update(convection_scheme='mass_flux')
    cases = [('control', 0., 0.), ('cool', 2., 0.),
             ('moist', 0., .20), ('cool_moist', 2., .20)]
    results = []
    with np.load(referencepath) as reference:
        grid = make_grid(len(reference['sigma_full']))
        for timestep in (900., 300.):
            local = dict(params, dt=timestep)
            base = loadstate(reference, grid, local)
            for name, cooling, moistening in cases:
                state = perturb(base, grid, cooling, moistening)
                result = runexperiment(state, grid, local, args.hours)
                result.update(case=name, timestep_s=timestep,
                              cooling_k=cooling, moisture_fraction=moistening)
                results.append(result)
                print(json.dumps({key: value for key, value in result.items()
                                  if not isinstance(value, list)}, indent=2), flush=True)
    sourcepaths = [configpath, root / 'scm/convection_mf.py',
                   root / 'scm/convective_transport.py']
    report = {
        'description': 'Convection-only response from one common ATM407 checkpoint',
        'hours': args.hours,
        'reference': str(referencepath),
        'reference_sha256': hashlib.sha256(referencepath.read_bytes()).hexdigest(),
        'source_sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sourcepaths},
        'results': results,
    }
    indexed = {(item['case'], item['timestep_s']): item for item in results}
    gates = {
        'cooling_increases_initial_cape': all(
            indexed['cool', step]['initial_cape_jkg']
            > indexed['control', step]['initial_cape_jkg'] + 200 for step in (900., 300.)),
        'convection_reduces_cape': all(item['cape_change_jkg'] < -100 for item in results),
        'cooling_strengthens_cape_removal': all(
            indexed['cool', step]['cape_change_jkg']
            < indexed['control', step]['cape_change_jkg'] - 40 for step in (900., 300.)),
        'cooling_increases_mass_flux': all(
            indexed['cool', step]['mean_cloud_base_mass_flux_kgm2s']
            > 1.1 * indexed['control', step]['mean_cloud_base_mass_flux_kgm2s']
            for step in (900., 300.)),
        'cooling_increases_explicit_rain': all(
            indexed['cool', step]['mean_rain_mmday']
            > 1.1 * indexed['control', step]['mean_rain_mmday'] for step in (900., 300.)),
        'timestep_response_is_close': all(
            abs(indexed[name, 900.]['cape_change_jkg'] - indexed[name, 300.]['cape_change_jkg']) < 5
            and abs(indexed[name, 900.]['mean_rain_mmday']
                    - indexed[name, 300.]['mean_rain_mmday']) < .02
            for name, _, _ in cases),
        'budgets_close': all(item['maximum_energy_error_wm2'] < 1e-4
                             and item['maximum_water_error_kgm2s'] < 1e-10
                             for item in results),
        'limiter_is_inactive': all(item['minimum_transport_limiter'] > .99 for item in results),
    }
    report['gates'] = gates
    report['passed'] = all(gates.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'gates': gates, 'passed': report['passed']}, indent=2))
    if not report['passed']:
        raise SystemExit('Convection response gates failed')


if __name__ == '__main__':
    main()
