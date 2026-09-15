"""Trace the corrected six-hour BOMEX experiment without changing its closures.

This is a screening audit, not a reproduction of the published LES experiment.
Liquid forcing uses theta_l for subsidence and leaves condensate unchanged
during the forcing step, so dT = Exner * dtheta_l is exact for that step.
--dry-forcing isolates the old temperature-variable error, not the entire old
benchmark setup. Production defaults and checkpoints are never modified.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'scripts'))

import bomex_observed_steady_state as case
import scm.column_model as column
import scm.convection_uw as plume
from scm.case_benchmarks import initialize_bomex, linear_profile, model_height, vertical_gradient, bomex_momentum_forcing
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params
from scm.thermo import Lv as latent, Rd as gas, cp, g, kappa, make_grid, p0


def liquidtheta(state):
    exner = (state['p'] / p0) ** kappa
    return (state['t'] - latent * state['qc'] / cp) / exner


def forcing(state, grid, liquid):
    heating, water = case.forcing_tendencies(state, grid)
    if not liquid:
        height = model_height(state, grid)
        velocity = linear_profile(height, [0, 1500, 2100, 3500], [0, -.0065, 0, 0])
        cooling = linear_profile(height, [0, 1500, 3000, 3500], [-2 / 86400, -2 / 86400, 0, 0])
        heating = (cooling - velocity * vertical_gradient(state['t'] * (p0 / state['p']) ** kappa, height))
        heating = heating * (state['p'] / p0) ** kappa
    return heating, water


def run(timestep, liquid):
    params = default_params()
    params.update(extract_param_overrides(load_run_config(root / 'scm/configs/uw_candidate_v1.toml')))
    # Match the latest documented UW test, rather than silently mixing in ATM407 defaults.
    params.update(dt=timestep, use_slab_ocean=False, ps0=101500.,
                  surface_temperature=case.SEA_SURFACE_TEMPERATURE,
                  cloud_ls_precip_fraction=0., entrainment_rate=5e-5,
                  uw_layer_closure=True, uw_shallow_keep_crossed_interface=True)
    grid = make_grid(20)
    state, _ = initialize_bomex(grid)
    state = case.extend_above_case_top(state, grid)
    initial = {name: value.clone() if torch.is_tensor(value) else value for name, value in state.items()}
    samples = []
    current = {}
    captured = {}
    integrate = plume._integrate_columns
    scheme = column.run_physics_scheme
    deep = column.dispatch_convection
    clouds = column.cloud_microphysics_step
    dry = column.dry_adjustment
    lines, firstline = inspect.getsourcelines(plume._integrate_one_column)
    stopline = next((firstline + index for index, line in enumerate(lines)
                     if line.strip() == 'if velocity_squared <= 0.0:'), None)

    def trace(frame, event, arg):
        if frame.f_code is not plume._integrate_one_column.__code__:
            return None
        if event == 'line' and frame.f_lineno == stopline:
            local = frame.f_locals
            if local['lower'] == local['source']:
                names = ['subheight', 'step_height', 'buoyancy', 'previous_velocity_squared',
                         'velocity_squared', 'accepted', 'mass_flux', 'plume_water', 'environment_water']
                current['launch_steps'].append({name: float(local[name]) for name in names})
        return trace

    def transport(*args):
        current['launch_steps'] = []
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            result = integrate(*args)
        finally:
            sys.settrace(previous)
        captured['plume'] = result
        t, q, qc, u, v, pressure, dp, height, theta, water, mse, tke, depth, local = args
        inside = torch.nonzero(height[0] <= depth[0]).flatten()
        source = int(inside[0]) if len(inside) else -1
        current['source_index'] = source
        current['height_m'] = height[0].clone()
        current['raw_water_flux_kgm2s'] = result['water_flux'][0].clone()
        current['raw_mass_flux_kgm2s'] = result['cloud_base_mass_flux'][0].clone()
        current['depth_m'] = depth[0].clone()
        current['cin_jkg'] = result['cin'][0].clone()
        if source >= 0:
            current['source_tke'] = tke[0, source].clone()
            current['layer_mean_tke_proxy'] = (tke[0, inside] * dp[0, inside]).sum() / dp[0, inside].sum()
            current['source_theta_k'] = theta[0, inside].min()
            # CAM uses reconstructed interface theta_vl and interface TKE.
            # This full-level proxy shows the omitted virtual correction only.
            virtual = theta[0, inside] * (1 + .61 * water[0, inside])
            current['cam_source_theta_proxy_k'] = virtual.min() / (1 + .61 * water[0, -1])
        if abs(current['hour'] - 3.5) < 1e-8:
            # Frozen-state replays separate inner plume numerics from column feedback.
            names = ['t', 'q', 'qc', 'u', 'v', 'p', 'dp', 'height', 'theta_liquid',
                     'total_water', 'mse', 'tke', 'boundary_depth']
            current['frozen_plume_input'] = {name: value.detach().cpu().tolist() for name, value in zip(names, args[:-1])}
            current['frozen_plume_dtype'] = str(t.dtype)
            current['frozen_vertical_step_replay'] = []
            for spacing in [50., 25., 10.]:
                replay = integrate(*args[:-1], dict(local, uw_shallow_vertical_step_m=spacing))
                current['frozen_vertical_step_replay'].append({
                    'step_m': spacing,
                    'raw_mass_flux_kgm2s': float(replay['cloud_base_mass_flux'][0]),
                    'top_m': float(replay['plume_top_height'][0]),
                    'max_water_flux_kgm2day': float(replay['water_flux'].abs().max()) * 86400,
                })
        return result

    def surface(state, grid, local):
        result = case.observed_surface_fluxes(state, grid, local)
        current['surface_flux_kgm2s'] = result['lhf'][0] / latent
        current['surface'] = result['dq'][0].clone()
        return result

    def physics(category, name, state, grid, local):
        result = scheme(category, name, state, grid, local)
        water = result['dq'] + result.get('dqc', torch.zeros_like(state['q']))
        current[category] = water[0].clone() / (timestep if category == 'condensation' else 1)
        if category == 'shallow_convection':
            current['final_water_flux_kgm2s'] = captured['plume']['water_flux'][0].clone()
            for name in ['cloud_base_mass_flux', 'implicit_cin_factor', 'plume_top_height_m',
                         'water_residual', 'precip', 'plume_mass_flux_profile', 'condensate_detrainment']:
                current[name] = result[name][0].clone()
        return result

    def convection(state, grid, local):
        result = deep(state, grid, local)
        current['deep'] = result['dq'][0].clone()
        return result

    def adjustment(state, grid, local):
        result = dry(state, grid, local)
        current['dry'] = (result['dq'] + result['dqc'])[0].clone()
        return result

    def cloud(state, grid, local, *args, **kwargs):
        before = state['qc'].clone()
        result = clouds(state, grid, local, *args, **kwargs)
        current['cloud'] = ((result.get('dq', torch.zeros_like(before)) + result['qc'] - before) / timestep)[0].clone()
        return result

    with patch.object(plume, '_integrate_columns', transport), \
         patch.object(column, 'surface_fluxes', surface), \
         patch.object(column, 'radiation', case.prescribed_radiation), \
         patch.object(column, 'run_physics_scheme', physics), \
         patch.object(column, 'dispatch_convection', convection), \
         patch.object(column, 'dry_adjustment', adjustment), \
         patch.object(column, 'cloud_microphysics_step', cloud):
        for step in range(round(21600 / timestep)):
            current = {'hour': step * timestep / 3600}
            before = (state['q'] + state['qc'])[0].double().clone()
            mass = state['dp'][0].double() / g
            heating, water = forcing(state, grid, liquid)
            zonal, meridional = bomex_momentum_forcing(state, grid)
            current['forcing'] = water[0].clone()
            # Compare forcing in conserved temperature units at this same state.
            other, _ = forcing(state, grid, not liquid)
            current['forcing_temperature_difference_kday'] = (heating - other)[0] * 86400
            state, diagnostics, _ = column.physics_step(state, grid, params, ls_forcing={'dt': heating, 'dq': water, 'du': zonal, 'dv': meridional})
            current['actual'] = ((state['q'] + state['qc'])[0].double() - before) / timestep
            names = ['forcing', 'surface', 'boundary_layer', 'dry', 'shallow_convection', 'deep', 'condensation', 'cloud']
            summed = sum(current.get(name, torch.zeros_like(mass)).double() for name in names)
            current['budget_error_gkgday'] = (summed - current['actual']) * 86400e3
            # Positive upwards; zero top boundary. Includes the surface source in BL.
            current['bl_water_flux_kgm2s'] = torch.cat((torch.zeros(1), torch.cumsum(current['boundary_layer'].double() * mass, 0)))
            current['bl_surface_error_kgm2s'] = current['bl_water_flux_kgm2s'][-1] - current['surface_flux_kgm2s']
            current['rain_mmday'] = diagnostics['precip_total'][0] * 86400
            # Flux needed to balance only prescribed forcing below each interface.
            below = torch.flip(torch.cumsum(torch.flip(current['forcing'].double() * mass, [0]), 0), [0])
            current['steady_required_flux_kgm2s'] = torch.cat((below, torch.zeros(1))) + current['surface_flux_kgm2s']
            samples.append({name: value.detach().cpu().tolist() if torch.is_tensor(value) else value for name, value in current.items()})

    checked = model_height(initial, grid)[0] <= 2500
    waterchange = ((state['q'] + state['qc']) - (initial['q'] + initial['qc']))[0] * 1000
    thetachange = (liquidtheta(state) - liquidtheta(initial))[0]
    late = [sample for sample in samples if sample['hour'] >= 3]
    keys = ['surface_flux_kgm2s', 'bl_surface_error_kgm2s', 'rain_mmday', 'raw_mass_flux_kgm2s',
            'cloud_base_mass_flux', 'implicit_cin_factor', 'depth_m', 'plume_top_height_m',
            'source_tke', 'layer_mean_tke_proxy', 'source_theta_k', 'cam_source_theta_proxy_k',
            'height_m', 'bl_water_flux_kgm2s', 'raw_water_flux_kgm2s', 'final_water_flux_kgm2s', 'steady_required_flux_kgm2s',
            'forcing', 'boundary_layer', 'shallow_convection', 'deep', 'condensation', 'cloud', 'actual',
            'condensate_detrainment', 'forcing_temperature_difference_kday']
    means = {name: torch.tensor([sample[name] for sample in late], dtype=torch.float64).mean(0).tolist() for name in keys}
    summary = {'dt_s': timestep, 'liquid_forcing': liquid,
               'max_water_change_gkg': float(waterchange[checked].abs().max()),
               'max_liquid_theta_change_k': float(thetachange[checked].abs().max()),
               'max_budget_error_gkgday': max(torch.tensor(sample['budget_error_gkgday']).abs().max().item() for sample in samples),
               'late': means}
    return {'parameters': params, 'summary': summary, 'samples': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt', type=float, default=900.)
    parser.add_argument('--liquid-forcing', action='store_true', default=True)
    parser.add_argument('--dry-forcing', action='store_false', dest='liquid_forcing', help='diagnostic dry-theta forcing counterfactual')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(args.dt, args.liquid_forcing)
    paths = [Path(__file__), root / 'scripts/bomex_observed_steady_state.py',
             root / 'scm/case_benchmarks.py', root / 'scm/convection_uw.py',
             root / 'scm/boundary_layer_uw.py', root / 'scm/uw_layers.py',
             root / 'scm/column_model.py', root / 'scm/configs/uw_candidate_v1.toml',
             root / 'outputs/cam_reference/uwshcu.F90']
    result['provenance'] = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                            'sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({key: value for key, value in result['summary'].items() if key != 'late'}, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
