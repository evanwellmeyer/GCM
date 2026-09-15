"""Diagnostic-only joint launch closure trials on archived BOMEX inputs.

The Gaussian source equations follow CAM6_3_000. This is not a CAM port:
host density, CIN, and the existing ascent remain fixed. The additional LCL
area constraint is unnecessary only when the source is already saturated.
"""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import scm.convection_uw as plume
from scm.shallow_plume_v2 import partition_plume
from scm.thermo import g, make_grid


def joint_launch(density, tke, cin, thickness, timestep, maximum):
    """Mass, conditional mean velocity and area from one Gaussian tail."""
    variance = float(tke) + 5e-4
    sigma = math.sqrt(variance)
    mu = math.sqrt(max(0., float(cin)) / variance)
    if mu >= 3.:
        return 0., 0., 0.
    lower, upper = 0., 8.
    for _ in range(60):
        middle = .5 * (lower + upper)
        if .5 * math.erfc(middle) > maximum:
            lower = middle
        else:
            upper = middle
    mu = max(mu, upper)
    unconstrained = float(density) * sigma / math.sqrt(2 * math.pi)
    limit = .9 * float(thickness) / (g * timestep)
    if unconstrained > limit:
        mu = max(mu, math.sqrt(math.log(unconstrained / limit)))
    area = .5 * math.erfc(mu)
    mass = unconstrained * math.exp(-mu * mu)
    velocity = mass / (float(density) * area)
    return mass, velocity, area


def run_trial(sample, mode, evolving=None, timestep=None, return_state=False):
    state = {key: torch.tensor(value, dtype=torch.float32) for key, value in sample['plume_input'].items() if value is not None}
    host = {key: state[key] for key in ('t', 'q', 'qc', 'u', 'v', 'p', 'dp', 'tke')}
    host['boundary_layer_depth_m'] = state['boundary_depth']
    host['tke_interfaces'] = state['interfaces']
    if evolving is not None:
        host = {key: value.clone() for key, value in evolving.items()}
    grid = make_grid(20)
    grid['height_surface_pressure_pa'] = 101500.
    params = dict(sample['plume_params'])
    if timestep is not None:
        params['dt'] = timestep
    launch = {}

    def select(density, tke, cin, thickness, timestep, maximum, mass, speed):
        gaussian, velocity, area = joint_launch(density, tke, cin, thickness, timestep, maximum)
        oldmass, oldspeed = float(mass), math.sqrt(float(speed))
        selectedmass = oldmass if mode in ('baseline', 'velocity_only') else gaussian
        selectedspeed = oldspeed if mode == 'baseline' else velocity
        launch.update(cin_jkg=float(cin), tke_m2s2=float(tke), density_kgm3=float(density),
                      mass_flux_kgm2s=selectedmass, velocity_ms=selectedspeed,
                      area=selectedmass / (float(density) * selectedspeed) if selectedspeed else 0.,
                      gaussian_mass_flux_kgm2s=gaussian, gaussian_velocity_ms=velocity, gaussian_area=area)
        return mass.new_tensor(selectedmass), speed.new_tensor(selectedspeed ** 2)

    original = inspect.getsource(plume._integrate_one_column)
    anchor = '    if mass_flux <= 1.0e-10:\n'
    if original.count(anchor) != 1:
        raise RuntimeError('source closure changed; review diagnostic injection')
    injection = ('    mass_flux, velocity_squared = select(source_density, source_tke, cin, dp[column, source],\n'
                 '        float(params.get("dt", 900.)), area_max, mass_flux, velocity_squared)\n')
    namespace = dict(plume.__dict__, select=select)
    exec(compile(original.replace(anchor, injection + anchor), '<diagnostic launch trial>', 'exec'), namespace)
    integrate = plume._integrate_columns
    raw = {}

    def capture(*args):
        result = integrate(*args)
        raw['flux'] = result['water_flux'][0].clone()
        raw['inputs'] = args
        raw['result'] = result
        return result

    with patch.object(plume, '_integrate_one_column', namespace['_integrate_one_column']), patch.object(plume, '_integrate_columns', capture):
        result = plume.uw_shallow_convection(host, grid, params)
    inputs = raw['inputs']
    reconstructed = plume.source_properties(*(inputs[index][0] for index in (5, 6, 7, 8, 9, 3, 4, 11, 12)), inputs[14][0])
    source = reconstructed['index']
    parcel = partition_plume(reconstructed['theta'], reconstructed['water'], reconstructed['faces'][source], iterations=32)
    launch['source_liquid_kgkg'] = float(parcel[2])
    water = (result['dq'] + result['dqc'])[0] * 86400e3
    flux = raw['result']['water_flux'][0] * 86400
    launch.update(hour=sample['hour'], mode=mode, water_tendency_825_gkgday=float(water[15]),
                  flux_below_kgm2day=float(flux[16]), flux_above_kgm2day=float(flux[15]),
                  plume_top_m=float(result['plume_top_height_m'][0]),
                  diagnosed_sorting_distance_m=float(result.get('plume_sorting_distance_m', torch.zeros(1))[0]),
                  sorting_scale_relative_error=float(result.get('plume_sorting_scale_relative_error', torch.zeros(1))[0]),
                  implicit_factor=float(result['implicit_cin_factor'][0]),
                  water_residual_kgm2s=float(result['water_residual'][0]),
                  energy_residual_wm2=float(result['energy_residual'][0]),
                  raw_replay_difference_kgm2day=float((raw['flux'] - torch.tensor(sample['raw_flux'])).abs().max() * 86400),
                  water_profile_gkgday=water.tolist())
    if return_state:
        updated = dict(host)
        for name in ('t', 'q', 'qc', 'u', 'v'):
            updated[name] = host[name] + params['dt'] * result['d' + name]
        updated['qc'] = updated['qc'].clamp(min=0.)
        launch['precipitation_kgm2s'] = float(result['precip'][0])
        return launch, updated
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root / 'outputs/column/diagnostics/bomex_launch_budget_20_6h.json')
    parser.add_argument('--output', type=Path, default=root / 'outputs/column/diagnostics/bomex_joint_launch_trials.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = json.loads(args.input.read_text())
    trials = []
    for hour in (0., 3., 5.75):
        sample = next(item for item in saved['samples'] if item['hour'] == hour)
        for mode in (('baseline', 'joint', 'velocity_only') if hour == 3. else ('baseline', 'joint')):
            trial = run_trial(sample, mode)
            if trial['source_liquid_kgkg'] <= 0.:
                raise RuntimeError('unsaturated source requires the additional LCL constraint; trial is unsupported')
            if abs(trial['water_residual_kgm2s']) >= 2e-8 or abs(trial['energy_residual_wm2']) >= .1:
                raise RuntimeError('trial failed the unchanged conservation tolerances')
            if mode == 'baseline' and trial['raw_replay_difference_kgm2day'] >= .01:
                raise RuntimeError('baseline does not reproduce the saved plume')
            trials.append(trial)
            args.output.write_text(json.dumps({'trials': trials}, indent=2) + '\n')
            print(f"hour {hour:g} {mode}: water {trial['water_tendency_825_gkgday']:.4f} g/kg/day; mass {trial['mass_flux_kgm2s']:.5f}; speed {trial['velocity_ms']:.4f}", flush=True)
    result = {'reference': 'CAM cam6_3_000 Gaussian source; source density/CIN and ascent held at host definitions',
              'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'physics_sha256': hashlib.sha256((root / 'scm/convection_uw.py').read_bytes()).hexdigest(), 'trials': trials}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
