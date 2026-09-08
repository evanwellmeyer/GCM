"""Compare multiband longwave radiation with RRTMG on the same column."""

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
from scm.ensemble import default_params
from scm.radiation_schemes.multiband import compute_longwave_multiband
from scm.thermo import make_grid


def rrtmg(state, grid, params):
    try:
        import climlab
        from climlab.domain.axis import Axis
    except ImportError as error:
        raise SystemExit('climlab and climlab-rrtmg are required') from error

    pressure = state['p'][0].double().cpu().numpy() / 100
    bounds = grid['sigma_half'].double().cpu().numpy() * state['ps'].item() / 100
    # RRTMG requires a positive top pressure. One millipascal changes the top
    # layer mass by 1e-7 and is recorded in the output rather than hidden.
    bounds[0] = max(bounds[0], 1e-5)
    axis = Axis(axis_type='lev', points=pressure, bounds=bounds)
    column = climlab.column_state(lev=axis)
    column['Tatm'][:] = state['t'][0].double().cpu().numpy()
    column['Ts'][:] = state['ts'].item()
    gases = {
        'CO2': float(params.get('co2', 400)) * 1e-6,
        'CH4': float(params.get('ch4', 1.8)) * 1e-6,
        'N2O': float(params.get('n2o', .332)) * 1e-6,
        'O2': .2095,
        'CFC11': 0., 'CFC12': 0., 'CFC22': 0., 'CCL4': 0., 'O3': 0.,
    }
    radiation = climlab.radiation.RRTMG_LW(
        state=column,
        specific_humidity=state['q'][0].double().cpu().numpy(),
        absorber_vmr=gases,
        icld=0,
        emissivity=1.,
    )
    radiation.compute_diagnostics()
    return {
        'pressure_hpa': pressure.tolist(),
        'pressure_bounds_hpa': bounds.tolist(),
        'temperature_k': np.asarray(column['Tatm']).tolist(),
        'specific_humidity_kgkg': state['q'][0].double().cpu().numpy().tolist(),
        'surface_temperature_k': state['ts'].item(),
        'absorber_vmr': gases,
        'olr_wm2': float(np.asarray(radiation.OLR).reshape(-1)[0]),
        'heating_kday': np.asarray(radiation.TdotLW).reshape(-1).tolist(),
        'upward_flux_wm2': np.asarray(radiation.LW_flux_up).reshape(-1).tolist(),
        'downward_flux_wm2': np.asarray(radiation.LW_flux_down).reshape(-1).tolist(),
        'climlab_version': climlab.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path,
                        default=root / 'notebooks/data/atm407_equilibrium_20level.npz')
    parser.add_argument('--config', type=Path, default=root / 'scm/configs/atm407.toml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    params = default_params()
    params.update(extract_param_overrides(load_run_config(args.config)))
    with np.load(args.reference) as reference:
        grid = make_grid(len(reference['sigma_full']), dtype=torch.float64)
        state = initial_state(1, grid, params)
        for name in ('t', 'q', 'qc', 'cloud_fraction'):
            state[name][0] = torch.as_tensor(reference[name], dtype=state[name].dtype)
        state['ts'][0] = float(reference['ts'])
        state['ps'][0] = float(reference['ps'])
        state = update_derived(state, grid)
    multibandparams = dict(params, o3_lw_tau=0.)
    multibandheating, multibanddown, multibandolr = compute_longwave_multiband(
        state, grid, multibandparams, force_clear_sky=True)
    reference = rrtmg(state, grid, params)
    multibandheating = multibandheating[0].double().cpu().numpy() * 86400
    rrtmgheating = np.asarray(reference['heating_kday'])
    difference = multibandheating - rrtmgheating
    mass = state['dp'][0].double().cpu().numpy()
    report = {
        'description': 'Clear-sky longwave comparison on matched pressure centers and interfaces',
        'assumptions': [
            'Both schemes use the same T, q, surface temperature and pressure grid.',
            'Cloud and ozone effects are zero in both schemes.',
            'RRTMG includes physical CO2, CH4, N2O and O2; multiband uses its configured abstract gas optical depths.',
            'This diagnoses structural error; it is not a coefficient calibration.',
        ],
        'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'multiband': {
            'olr_wm2': multibandolr.item(),
            'surface_downward_longwave_wm2': multibanddown.item(),
            'heating_kday': multibandheating.tolist(),
        },
        'rrtmg': reference,
        'difference': {
            'olr_wm2': multibandolr.item() - reference['olr_wm2'],
            'heating_kday': difference.tolist(),
            'mass_weighted_heating_bias_kday': float(np.sum(difference * mass) / np.sum(mass)),
            'heating_rms_kday': float(np.sqrt(np.sum(difference ** 2 * mass) / np.sum(mass))),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({
        'multiband_olr_wm2': report['multiband']['olr_wm2'],
        'rrtmg_olr_wm2': report['rrtmg']['olr_wm2'],
        **report['difference'],
        'output': str(args.output),
    }, indent=2))


if __name__ == '__main__':
    main()
