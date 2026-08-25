from pathlib import Path
import json
import sys
import time

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scm.column_model import initial_state, run
from scm.configuration import extract_param_overrides, load_run_config
from scm.diagnostics import equilibrium_metrics, equilibrium_stats
from scm.ensemble import default_params
from scm.thermo import make_grid


outputdir = root / 'notebooks' / 'data'
outputdir.mkdir(parents=True, exist_ok=True)
referencepath = outputdir / 'atm407_equilibrium_20level.npz'

device = torch.device('cpu')
grid = make_grid(20, device=device)
params = default_params(device=device)
params.update(extract_param_overrides(load_run_config()))
params.update({
    'dt': 1800.0,
    'ps0': 100000.0,
    'ts_init': 290.0,
    'solar_constant': 1360.0,
    'zenith_factor': 0.25,
    'ocean_depth': 5.0,
    'wind_speed': 5.0,
    'convection_scheme': 'mass_flux',
    'use_slab_ocean': True,
})

start = time.perf_counter()
if referencepath.exists():
    reference = np.load(referencepath)
    state = initial_state(1, grid, params, device=device)
    state['t'][0] = torch.as_tensor(reference['t'], dtype=state['t'].dtype)
    state['q'][0] = torch.as_tensor(reference['q'], dtype=state['q'].dtype)
    state['qc'][0] = torch.as_tensor(reference['qc'], dtype=state['qc'].dtype)
    state['cloud_fraction'][0] = torch.as_tensor(
        reference['cloud_fraction'], dtype=state['cloud_fraction'].dtype
    )
    state['ts'][0] = float(reference['ts'])
    state['ps'][0] = float(reference['ps'])
    state['slab_ts_ref'] = state['ts'].clone()
    state['slab_energy'].zero_()
    history = []
else:
    state = initial_state(1, grid, params, device=device)
    stepsperday = round(86400 / params['dt'])
    state, history = run(
        state,
        grid,
        params,
        500 * stepsperday,
        rad_interval=4,
        diag_interval=stepsperday,
    )

params['dt'] = 900.0
params['ocean_depth'] = 50.0
stepsperday = round(86400 / params['dt'])
state, finalhistory = run(
    state,
    grid,
    params,
    100 * stepsperday,
    rad_interval=8,
    diag_interval=stepsperday,
)
history.extend(finalhistory)

metrics = equilibrium_metrics(history, window=50)
stats = equilibrium_stats(history, last_n=50)
np.savez_compressed(
    referencepath,
    sigma_full=grid['sigma_full'].cpu().numpy(),
    t=state['t'][0].cpu().numpy(),
    q=state['q'][0].cpu().numpy(),
    qc=state['qc'][0].cpu().numpy(),
    cloud_fraction=state['cloud_fraction'][0].cpu().numpy(),
    ts=np.array(state['ts'][0].item()),
    ps=np.array(state['ps'][0].item()),
)

metadata = {
    'description': 'Pre-equilibrated ATM407 mass-flux SCM reference state',
    'reference_levels': 20,
    'accelerated_spinup_days': 500,
    'final_adjustment_days': 100,
    'spinup_ocean_depth_m': 5.0,
    'final_ocean_depth_m': 50.0,
    'final_dt_s': 900.0,
    'radiation_scheme': params['radiation_scheme'],
    'convection_scheme': 'mass_flux',
    'surface_albedo': params['albedo'],
    'surface_temperature_k': stats['ts_mean'][0].item(),
    'toa_net_wm2': stats['toa_net_mean'][0].item(),
    'surface_total_flux_wm2': stats['surface_total_flux_mean'][0].item(),
    'precipitation_mmday': stats['precip_total_mean'][0].item() * 86400,
    'equilibrium_metrics': metrics,
    'generation_runtime_s': time.perf_counter() - start,
}

metadatapath = outputdir / 'atm407_equilibrium_20level.json'
metadatapath.write_text(json.dumps(metadata, indent=2) + '\n')
print(referencepath)
print(metadatapath)
print(json.dumps(metadata, indent=2))
