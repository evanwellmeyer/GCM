"""Short matched-checkpoint screening of the conservative updraft candidate."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scm.column_model import initial_state, update_derived, run
from scm.configuration import load_run_config, extract_param_overrides
from scm.ensemble import default_params
from scm.thermo import make_grid, relative_humidity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=6.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configpath = root / 'scm/configs/atm407.toml'
    referencepath = root / 'notebooks/data/atm407_equilibrium_20level.npz'
    config = load_run_config(configpath)
    params = default_params(device='cpu')
    params.update(extract_param_overrides(config))
    params.update(convection_scheme='mass_flux', ocean_depth=5., use_slab_ocean=True)
    results = []
    with np.load(referencepath) as reference:
        grid = make_grid(len(reference['sigma_full']))
        for form, timestep in [('flux', 900.), ('flux', 300.)]:
            local = dict(params, mf_transport_form=form, dt=timestep)
            state = initial_state(1, grid, local)
            for name in ('t', 'q', 'qc', 'cloud_fraction'):
                state[name][0] = torch.as_tensor(reference[name], dtype=state[name].dtype)
            state['ts'][0] = float(reference['ts'])
            state['ps'][0] = float(reference['ps'])
            state['slab_ts_ref'] = state['ts'].clone()
            state['slab_energy'].zero_()
            state = update_derived(state, grid)
            initialtemperature = state['ts'].item()
            state, history = run(state, grid, local, round(args.hours * 3600 / timestep),
                                 rad_interval=1, diag_interval=1)
            names = ('toa_net', 'surface_total_flux', 'cape', 'precip_conv', 'precip_ls',
                     'column_energy_residual', 'column_water_residual',
                     'deep_transport_energy_residual', 'deep_raw_energy_residual',
                     'deep_downdraft_energy_residual', 'deep_export_energy_residual',
                     'deep_transport_limiter',
                     'temperature_cap_fraction', 'moisture_cap_fraction')
            summary = {name: torch.stack([item[name] for item in history]).mean().item()
                       for name in names}
            summary.update(transport=form, timestep=timestep,
                           temperaturedrift=state['ts'].item() - initialtemperature,
                           humidity=relative_humidity(state['q'], state['t'], state['p'])[0].tolist(),
                           pressure=state['p'][0].tolist(), temperature=state['t'][0].tolist(),
                           finite=bool(torch.isfinite(state['t']).all() and torch.isfinite(state['q']).all()))
            results.append(summary)
            print(json.dumps({key: value for key, value in summary.items()
                              if key not in ('humidity', 'pressure', 'temperature')}, indent=2), flush=True)
    sources = ['scm/convective_transport.py', 'scm/convection_mf.py', 'scm/column_model.py',
               'scm/configs/default.toml', 'scm/configs/atm407.toml']
    report = dict(hours=args.hours, reference=str(referencepath),
                  referencehash=hashlib.sha256(referencepath.read_bytes()).hexdigest(),
                  sourcehashes={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in sources},
                  notes='Flux transport at two time steps. It uses a conservative rain-fed '
                        'downdraft, no artificial export and no global energy repair. Radiation every step. '
                        'This is a transient screen, not an equilibrium or realism test.',
                  results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
