"""Per-process heat and water budget for the BOMEX observed-balance check.

Runs the same setup as scripts/bomex_observed_steady_state.py and records each
scheme's tendency at every level over the second half of the run. The last two
columns are the sum of the schemes and the actual change; they should agree.
"""

from collections import defaultdict
from pathlib import Path
import argparse
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import bomex_observed_steady_state as case
import scm.column_model as column_model
from scm.case_benchmarks import initialize_bomex, model_height
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params
from scm.thermo import make_grid

NAMES = ['forcing', 'surface', 'BL+dry', 'shallow', 'deep', 'cond+cloud']
RAIN = ['precip_conv', 'precip_shallow', 'precip_ls', 'precip_cloud']


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default=str(ROOT / 'scm/configs/atm407.toml'))
    parser.add_argument('--levels', type=int, default=20)
    parser.add_argument('--hours', type=float, default=6.0)
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='override a production parameter, e.g. cloud_ls_precip_fraction=0')
    args = parser.parse_args()

    config = load_run_config(args.config)
    params = default_params()
    params.update(extract_param_overrides(config))
    for item in args.set:
        key, value = item.split('=', 1)
        params[key] = json.loads(value)
    timestep = float(config.get('numerics', {}).get('dt', 900.0))
    params.update({
        'dt': timestep,
        'use_slab_ocean': False,
        'ps0': 101500.0,
        'surface_temperature': case.SEA_SURFACE_TEMPERATURE,
    })

    sums = defaultdict(float)
    recording = [False]

    def add(name, heating, water):
        if recording[0]:
            sums[name, 'T'] = sums[name, 'T'] + heating[0].detach().double()
            sums[name, 'W'] = sums[name, 'W'] + water[0].detach().double()

    def surface(state, grid, local):
        output = case.observed_surface_fluxes(state, grid, local)
        add('surface', output['dt'], output['dq'])
        return output

    original_scheme = column_model.run_physics_scheme

    def scheme(category, name, state, grid, local):
        output = original_scheme(category, name, state, grid, local)
        if category == 'condensation':
            add('cond+cloud', output['dt'] / timestep, output['dq'] / timestep)
        elif category == 'boundary_layer':
            add('BL+dry', output['dt'], output['dq'] + output.get('dqc', torch.zeros_like(output['dq'])))
        elif category == 'shallow_convection':
            add('shallow', output['dt'], output['dq'] + output.get('dqc', torch.zeros_like(output['dq'])))
        return output

    original_dry = column_model.dry_adjustment

    def dry(state, grid, local):
        output = original_dry(state, grid, local)
        add('BL+dry', output['dt'], output['dq'] + output['dqc'])
        return output

    original_convection = column_model.dispatch_convection

    def convection(state, grid, local):
        output = original_convection(state, grid, local)
        add('deep', output['dt'], output['dq'])
        return output

    original_cloud = column_model.cloud_microphysics_step

    def cloud(state, grid, local, cond_out, conv_out, shallow_out=None):
        before = state['qc'].clone()
        output = original_cloud(state, grid, local, cond_out, conv_out, shallow_out=shallow_out)
        add('cond+cloud',
            output.get('dt', torch.zeros_like(state['t'])) / timestep,
            (output.get('dq', torch.zeros_like(state['q'])) + output['qc'] - before) / timestep)
        return output

    column_model.radiation = case.prescribed_radiation
    column_model.surface_fluxes = surface
    column_model.run_physics_scheme = scheme
    column_model.dry_adjustment = dry
    column_model.dispatch_convection = convection
    column_model.cloud_microphysics_step = cloud

    grid = make_grid(args.levels)
    state, _ = initialize_bomex(grid)
    state = case.extend_above_case_top(state, grid)
    height = model_height(state, grid)[0]
    steps = round(args.hours * 3600.0 / timestep)
    first = steps // 2
    rain = defaultdict(float)
    for step in range(steps):
        recording[0] = step >= first
        if step == first:
            start_temperature = state['t'][0].double().clone()
            start_water = (state['q'] + state['qc'])[0].double().clone()
        heating, moistening = case.forcing_tendencies(state, grid)
        add('forcing', heating, moistening)
        state, diagnostics, _ = column_model.physics_step(
            state, grid, params, ls_forcing={'dt': heating, 'dq': moistening}
        )
        if recording[0]:
            for name in RAIN:
                rain[name] += float(diagnostics[name][0]) * 86400.0

    count = steps - first
    elapsed = count * timestep
    actual = {
        'T': (state['t'][0].double() - start_temperature) / elapsed,
        'W': ((state['q'] + state['qc'])[0].double() - start_water) / elapsed,
    }
    if args.set:
        print(f'overrides: {args.set}')
    print('rain, mm/day: ' + ', '.join(f'{name[7:]} {rain[name] / count:.2f}' for name in RAIN))
    for kind, scale, label in [('T', 86400.0, 'temperature, K/day'), ('W', 86400.0e3, 'total water, g/kg/day')]:
        print(f'\n{label}, hours {args.hours / 2:g}-{args.hours:g}, {args.levels} levels')
        print('  height ' + ''.join(f'{name:>11}' for name in NAMES) + '      sum   actual')
        for index in range(len(height) - 1, -1, -1):
            if float(height[index]) > case.CHECK_TOP:
                continue
            values = []
            for name in NAMES:
                total = sums[name, kind]
                values.append(float(total[index]) / count * scale if torch.is_tensor(total) else 0.0)
            print(f'  {float(height[index]):6.0f} ' + ''.join(f'{value:11.2f}' for value in values)
                  + f'{sum(values):9.2f}{float(actual[kind][index]) * scale:9.2f}')


if __name__ == '__main__':
    main()
