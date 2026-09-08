"""Screen the conservative convection closure in one batched column run."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scm.column_model import initial_state, run, update_derived
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params
from scm.thermo import g, make_grid, relative_humidity


def mean(history, name, start):
    return torch.stack([item[name] for item in history[start:]]).mean(dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path,
                        default=root / 'notebooks/data/atm407_equilibrium_20level.npz')
    parser.add_argument('--config', type=Path,
                        default=root / 'scm/configs/atm407_flux_v1.toml')
    parser.add_argument('--days', type=int, default=5)
    parser.add_argument('--timescales-hours', type=float, nargs='+',
                        default=[24, 12, 6, 3])
    parser.add_argument('--entrainment-rates', type=float, nargs='+',
                        default=[5e-6])
    parser.add_argument('--detrainment-rates', type=float, nargs='+',
                        default=[3e-5])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    params = default_params()
    config = load_run_config(args.config)
    params.update(extract_param_overrides(config))
    cases = [(hours, entrainment, detrainment)
             for entrainment in args.entrainment_rates
             for detrainment in args.detrainment_rates
             for hours in args.timescales_hours]
    count = len(cases)
    params.update(
        dt=900., ocean_depth=5., use_slab_ocean=True,
        tau_cape=torch.tensor([case[0] for case in cases]) * 3600,
        entrainment_rate=torch.tensor([case[1] for case in cases]),
        mf_detrainment_rate=torch.tensor([case[2] for case in cases]),
    )
    with np.load(args.reference) as reference:
        grid = make_grid(len(reference['sigma_full']))
        state = initial_state(count, grid, params)
        for name in ('t', 'q', 'qc', 'cloud_fraction'):
            values = torch.as_tensor(reference[name], dtype=state[name].dtype)
            state[name][:] = values.unsqueeze(0)
        state['ts'][:] = float(reference['ts'])
        state['ps'][:] = float(reference['ps'])
    state['slab_ts_ref'] = state['ts'].clone()
    state['slab_energy'].zero_()
    state = update_derived(state, grid)
    initialtemperature = state['ts'].clone()
    stepsperday = round(86400 / params['dt'])
    state, history = run(
        state, grid, params, args.days * stepsperday,
        rad_interval=8, diag_interval=1,
    )
    start = max(0, len(history) - stepsperday)
    rh = relative_humidity(state['q'], state['t'], state['p'])
    mass = state['dp'] / g
    saturated = torch.sum((rh >= .95) * mass, dim=1) / mass.sum(dim=1)
    records = []
    for index, (hours, entrainment, detrainment) in enumerate(cases):
        deep = mean(history, 'precip_conv', start)[index].item() * 86400
        large = mean(history, 'precip_ls', start)[index].item() * 86400
        cloud = mean(history, 'precip_cloud', start)[index].item() * 86400
        records.append({
            'timescale_hours': hours,
            'entrainment_rate_pa1': entrainment,
            'detrainment_rate_pa1': detrainment,
            'surface_temperature_k': state['ts'][index].item(),
            'surface_temperature_drift_k': (state['ts'][index] - initialtemperature[index]).item(),
            'toa_net_wm2': mean(history, 'toa_net', start)[index].item(),
            'surface_total_flux_wm2': mean(history, 'surface_total_flux', start)[index].item(),
            'cape_jkg': mean(history, 'cape', start)[index].item(),
            'rh95_mass_fraction': saturated[index].item(),
            'deep_precipitation_mmday': deep,
            'large_scale_precipitation_mmday': large,
            'cloud_precipitation_mmday': cloud,
            'deep_precipitation_fraction': deep / max(deep + large + cloud, 1e-12),
            'cloud_base_mass_flux_kgm2s': mean(history, 'cloud_base_mass_flux', start)[index].item(),
            'mass_flux_cap_fraction': mean(history, 'mass_flux_cap_active', start)[index].item(),
            'transport_limiter': mean(history, 'deep_transport_limiter', start)[index].item(),
            'column_energy_residual_wm2': mean(history, 'column_energy_residual', start)[index].item(),
            'column_water_residual_kgm2s': mean(history, 'column_water_residual', start)[index].item(),
            'relative_humidity_percent': (rh[index] * 100).tolist(),
        })
    report = {
        'description': 'Batched closure-timescale screen from one common checkpoint',
        'configuration_label': config['run']['label'],
        'days': args.days,
        'final_day_means': records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for record in records:
        print(json.dumps({key: value for key, value in record.items()
                          if key != 'relative_humidity_percent'}, indent=2))


if __name__ == '__main__':
    main()
