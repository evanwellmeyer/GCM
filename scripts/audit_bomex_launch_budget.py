"""Attribute the corrected BOMEX launch-layer drift without changing physics."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'scripts'))

import bomex_observed_steady_state as case
import scm.column_model as column
import scm.convection_uw as plume
from scm.case_benchmarks import initialize_bomex, model_height, bomex_momentum_forcing
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params
from scm.thermo import Lv as latent, cp, g, kappa, make_grid, p0


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def theta(state):
    return (state['t'] - latent * state['qc'] / cp) / (state['p'] / p0) ** kappa


def divergence(flux, mass):
    return (flux[1:] - flux[:-1]) / mass


def run(hours, output):
    params = default_params()
    params.update(extract_param_overrides(load_run_config(root / 'scm/configs/uw_candidate_v1.toml')))
    params.update(dt=900., use_slab_ocean=False, ps0=101500., surface_temperature=case.SEA_SURFACE_TEMPERATURE,
                  cloud_ls_precip_fraction=0., entrainment_rate=5e-5, uw_layer_closure=True,
                  uw_shallow_keep_crossed_interface=True)
    grid = make_grid(20)
    state, _ = initialize_bomex(grid)
    initial = plain(state)
    heights = model_height(state, grid)[0].clone()
    samples, current, captured = [], {}, {}
    integrate, subcloud = plume._integrate_columns, plume.subcloud_transport
    scheme, deep, dry, clouds = column.run_physics_scheme, column.dispatch_convection, column.dry_adjustment, column.cloud_microphysics_step

    def record(name, state, heating, vapor, liquid):
        exner = (state['p'][0].double() / p0) ** kappa
        current['water'][name] = (vapor[0].double() + liquid[0].double())
        current['theta'][name] = (heating[0].double() - latent * liquid[0].double() / cp) / exner

    def subtransport(result, index, launch, massflux, liquidtheta, water, u, v, pressure, timestep):
        subcloud(result, index, launch, massflux, liquidtheta, water, u, v, pressure, timestep)
        current['subcloud_raw_flux'] = result['water_flux'][index].clone()
        current['subcloud_raw_mse_flux'] = result['mse_flux'][index].clone()
        current['source_index'] = launch['index']
        current['source'] = plain(launch)
        # Split the moving-inversion term from the pressure-linear subcloud flux.
        source, faces = launch['index'], launch['faces']
        slope = plume.source_slope(water, pressure)
        upper = max(0, source - 2)
        bottom = water[source] + slope[source] * (faces[source] - pressure[source])
        top = water[upper] + slope[upper] * (faces[source - 1] - pressure[upper])
        contrast = bottom - top
        denominator = torch.where(contrast >= 0., contrast.clamp(min=1e-20), contrast.clamp(max=-1e-20))
        position = ((water[source - 1] - top) / denominator).clamp(0., 1.)
        fraction = massflux * g * timestep / (faces[source] - faces[source - 1])
        correction = torch.zeros_like(faces)
        correction[source] = (1. - position / fraction).clamp(min=0.) * massflux * contrast
        current['inversion_raw_flux'] = correction
        current['inversion_position_fraction'] = float(position)
        current['inversion_swept_fraction'] = float(fraction)

    def transport(*args):
        names = ['t', 'q', 'qc', 'u', 'v', 'p', 'dp', 'height', 'theta_liquid', 'total_water', 'mse', 'tke', 'boundary_depth']
        current['plume_input'] = plain(dict(zip(names, args[:13])))
        current['plume_input']['interfaces'] = plain(args[14]) if len(args) > 14 else None
        current['plume_params'] = plain(args[13])
        result = integrate(*args)
        captured['plume'] = result
        current['raw_flux'] = result['water_flux'][0].clone()
        current['raw_mass_flux'] = float(result['cloud_base_mass_flux'][0])
        return result

    def physics(category, name, state, grid, local):
        result = scheme(category, name, state, grid, local)
        scale = params['dt'] if category == 'condensation' else 1.
        zero = torch.zeros_like(state['q'])
        if category in ('boundary_layer', 'shallow_convection', 'condensation'):
            record(category, state, result['dt'] / scale, result['dq'] / scale, result.get('dqc', zero) / scale)
        if category == 'shallow_convection':
            raw = current.get('raw_mass_flux', 0.)
            factor = float(result['cloud_base_mass_flux'][0]) / raw if raw > 0. else 0.
            current['transport_scale'] = factor
            current['implicit_cin_factor'] = float(result['implicit_cin_factor'][0])
            current['applied_flux'] = captured['plume']['water_flux'][0].clone()
            current['applied_mse_flux'] = captured['plume']['mse_flux'][0].clone()
            current['rain_evaporation'] = result['precipitation_evaporation'][0].clone()
            current['rain_source'] = captured['plume']['precipitation_source'][0].clone()
            current['shallow_water_residual'] = float(result['water_residual'][0])
            current['shallow_energy_residual'] = float(result['energy_residual'][0])
        return result

    def convection(state, grid, local):
        result = deep(state, grid, local)
        record('deep', state, result['dt'], result['dq'], torch.zeros_like(state['q']))
        return result

    def adjustment(state, grid, local):
        result = dry(state, grid, local)
        record('dry', state, result['dt'], result['dq'], result['dqc'])
        return result

    def cloud(state, grid, local, *args, **kwargs):
        before = state['qc'].clone()
        result = clouds(state, grid, local, *args, **kwargs)
        zero = torch.zeros_like(before)
        record('cloud', state, result.get('dt', zero) / params['dt'], result.get('dq', zero) / params['dt'],
               (result['qc'] - before) / params['dt'])
        return result

    def surface(state, grid, local):
        result = case.observed_surface_fluxes(state, grid, local)
        record('surface', state, result['dt'], result['dq'], torch.zeros_like(state['q']))
        return result

    with patch.object(plume, '_integrate_columns', transport), patch.object(plume, 'subcloud_transport', subtransport), \
         patch.object(column, 'run_physics_scheme', physics), patch.object(column, 'dispatch_convection', convection), \
         patch.object(column, 'dry_adjustment', adjustment), patch.object(column, 'cloud_microphysics_step', cloud), \
         patch.object(column, 'surface_fluxes', surface), patch.object(column, 'radiation', case.prescribed_radiation):
        for step in range(round(hours * 3600 / params['dt'])):
            current = {'hour': step * params['dt'] / 3600, 'water': {}, 'theta': {}}
            beforewater = state['q'][0].double() + state['qc'][0].double()
            beforetheta = theta({key: value.double() if torch.is_tensor(value) else value for key, value in state.items()})[0]
            heating, moistening = case.forcing_tendencies(state, grid)
            zonal, meridional = bomex_momentum_forcing(state, grid)
            record('forcing', state, heating, moistening, torch.zeros_like(state['q']))
            state, diagnostics, _ = column.physics_step(state, grid, params,
                ls_forcing={'dt': heating, 'dq': moistening, 'du': zonal, 'dv': meridional})
            mass = state['dp'][0].double() / g
            actualwater = (state['q'][0].double() + state['qc'][0].double() - beforewater) / params['dt']
            actualtheta = (theta({key: value.double() if torch.is_tensor(value) else value for key, value in state.items()})[0] - beforetheta) / params['dt']
            for name, actual in [('water', actualwater), ('theta', actualtheta)]:
                current[name]['unattributed'] = actual - sum(current[name].values())
                current[name]['actual'] = actual
            zero = torch.zeros(21, dtype=torch.float64)
            factor = current.get('transport_scale', 0.)
            subflux = current.get('subcloud_raw_flux', zero).double() * factor
            inversion = current.get('inversion_raw_flux', zero).double() * factor
            applied = current.get('applied_flux', zero).double()
            current['flux_components'] = {'subcloud_linear': subflux - inversion,
                                           'inversion_displacement': inversion,
                                           'above_source': applied - subflux}
            current['transport_budget'] = {name: divergence(value, mass) for name, value in current['flux_components'].items()}
            current['transport_budget']['precipitation'] = current.get('rain_evaporation', torch.zeros(20)).double() - current.get('rain_source', torch.zeros(20)).double() / mass
            current['transport_budget']['partition_roundoff'] = current['water']['shallow_convection'] - sum(current['transport_budget'].values())
            current['bl_flux'] = torch.cat((torch.zeros(1), (current['water']['boundary_layer'] * mass).cumsum(0)))
            current['rain_mmday'] = float(diagnostics['precip_total'][0]) * 86400
            samples.append(plain(current))
            output.with_suffix('.partial.json').write_text(json.dumps({'parameters': plain(params), 'height_m': plain(heights), 'samples': samples}))
            print(f"hour {(step + 1) * params['dt'] / 3600:.2f}: recorded", flush=True)
    return plain({'parameters': params, 'height_m': heights, 'initial_state': initial, 'final_state': state, 'samples': samples})


def summarize(result):
    """Integrate process rates and retain a check against endpoint storage."""
    samples = result['samples']
    # An initial recorder version logged deep convection both at dispatch and
    # registry entry. They are aliases, not two physical tendencies.
    for sample in samples:
        for kind in ('water', 'theta'):
            values = sample[kind]
            values.pop('convection', None)
            values['unattributed'] = (torch.tensor(values['actual'], dtype=torch.float64) -
                sum(torch.tensor(value, dtype=torch.float64) for key, value in values.items()
                    if key not in ('actual', 'unattributed'))).tolist()
    windows = {}
    for name, selected in [('all', samples), ('late', [sample for sample in samples if sample['hour'] >= 3.])]:
        if not selected:
            continue
        means = {}
        for kind in ('water', 'theta', 'transport_budget', 'flux_components'):
            keys = sorted(set().union(*(sample[kind].keys() for sample in selected)))
            scale = 86400. if kind in ('theta', 'flux_components') else 86400e3
            means[kind] = {key: (torch.tensor([sample[kind].get(key, [0.] * 20) for sample in selected],
                                   dtype=torch.float64).mean(0) * scale).tolist() for key in keys}
        means['bl_flux_kgm2day'] = (torch.tensor([sample['bl_flux'] for sample in selected], dtype=torch.float64).mean(0) * 86400).tolist()
        means['rain_mmday'] = sum(sample['rain_mmday'] for sample in selected) / len(selected)
        means['max_water_closure_gkgday'] = max(max(map(abs, sample['water']['unattributed'])) for sample in selected) * 86400e3
        means['max_theta_closure_kday'] = max(max(map(abs, sample['theta']['unattributed'])) for sample in selected) * 86400
        means['max_shallow_decomposition_gkgday'] = max(max(map(abs, sample['transport_budget']['partition_roundoff'])) for sample in selected) * 86400e3
        means['minimum_transport_scale'] = min(sample.get('transport_scale', 0.) for sample in selected)
        windows[name] = means
    output = {'height_m': result['height_m'], 'windows': windows,
              'source_indices': [sample.get('source_index') for sample in samples]}
    if 'final_state' in result:
        initial, final = result['initial_state'], result['final_state']
        waterchange = (torch.tensor(final['q'], dtype=torch.float64) + torch.tensor(final['qc'], dtype=torch.float64)
                       - torch.tensor(initial['q'], dtype=torch.float64) - torch.tensor(initial['qc'], dtype=torch.float64))[0] * 1000
        integrated = torch.tensor(windows['all']['water']['actual']) * len(samples) * result['parameters']['dt'] / 86400
        output['water_change_gkg'] = waterchange.tolist()
        output['max_integrated_storage_error_gkg'] = float((integrated - waterchange).abs().max())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=6.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        result = summarize(json.loads(args.summarize.read_text()))
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(args.output)
        return
    paths = ['scm/convection_uw.py', 'scm/column_model.py', 'scm/boundary_layer_uw.py', 'scm/case_benchmarks.py', 'scripts/bomex_observed_steady_state.py', 'scripts/audit_bomex_launch_budget.py']
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
    result = run(args.hours, args.output)
    result['sha256'] = hashes
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
