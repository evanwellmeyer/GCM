from pathlib import Path
import argparse
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
from scm.thermo import g, make_grid, relative_humidity


parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true')
parser.add_argument('--adjustment-days', type=int, default=100)
parser.add_argument('--config', type=Path)
parser.add_argument('--output-label', default='')
parser.add_argument('--initial-reference', type=Path)
parser.add_argument('--levels', type=int, default=20)
args = parser.parse_args()

if args.output_label and not args.output_label.replace('_', '').isalnum():
    parser.error('--output-label may contain only letters, numbers, and underscores')
if args.levels < 4:
    parser.error('--levels must be at least 4')

outputdir = root / 'notebooks' / 'data'
outputdir.mkdir(parents=True, exist_ok=True)
suffix = f'_{args.output_label}' if args.output_label else ''
referencepath = outputdir / f'atm407_equilibrium_{args.levels}level{suffix}.npz'
metadatapath = outputdir / f'atm407_equilibrium_{args.levels}level{suffix}.json'

device = torch.device('cpu')
grid = make_grid(args.levels, device=device)
params = default_params(device=device)
config = load_run_config(args.config)
params.update(extract_param_overrides(config))
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
sourcepath = None
if not args.force:
    if args.initial_reference is not None:
        sourcepath = args.initial_reference
    elif referencepath.exists():
        sourcepath = referencepath

if sourcepath is not None:
    if not sourcepath.exists():
        raise FileNotFoundError(sourcepath)
    resumingoutput = sourcepath.resolve() == referencepath.resolve()
    previousmetadata = (
        json.loads(metadatapath.read_text())
        if resumingoutput and metadatapath.exists()
        else {}
    )
    previousadjustment = int(previousmetadata.get('final_adjustment_days', 0))
    initialreference = previousmetadata.get('initial_reference', str(sourcepath))
    reference = np.load(sourcepath)
    state = initial_state(1, grid, params, device=device)
    sourcesigma = np.asarray(reference['sigma_full'])
    targetsigma = grid['sigma_full'].cpu().numpy()

    def seedprofile(name):
        values = np.asarray(reference[name])
        if values.shape[0] == args.levels:
            return values
        return np.interp(targetsigma, sourcesigma, values)

    temperature = seedprofile('t')
    vapor = seedprofile('q')
    condensate = seedprofile('qc')
    cloudfraction = seedprofile('cloud_fraction')
    if len(sourcesigma) != args.levels:
        sourcegrid = make_grid(len(sourcesigma), device=device)
        sourceweights = torch.diff(sourcegrid['sigma_half']).cpu().numpy()
        targetweights = torch.diff(grid['sigma_half']).cpu().numpy()
        sourcewater = np.sum((reference['q'] + reference['qc']) * sourceweights)
        targetwater = np.sum((vapor + condensate) * targetweights)
        waterscale = sourcewater / max(targetwater, 1.0e-12)
        vapor = vapor * waterscale
        condensate = condensate * waterscale

    state['t'][0] = torch.as_tensor(temperature, dtype=state['t'].dtype)
    state['q'][0] = torch.as_tensor(vapor, dtype=state['q'].dtype)
    state['qc'][0] = torch.as_tensor(condensate, dtype=state['qc'].dtype)
    state['cloud_fraction'][0] = torch.as_tensor(
        cloudfraction, dtype=state['cloud_fraction'].dtype
    )
    if 'tke' in reference.files:
        state['tke'] = torch.as_tensor(
            seedprofile('tke'), dtype=state['t'].dtype
        ).unsqueeze(0)
    state['ts'][0] = float(reference['ts'])
    state['ps'][0] = float(reference['ps'])
    state['slab_ts_ref'] = state['ts'].clone()
    state['slab_energy'].zero_()
    history = []
else:
    previousadjustment = 0
    initialreference = None
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
state['slab_ts_ref'] = state['ts'].clone()
state['slab_energy'].zero_()
stepsperday = round(86400 / params['dt'])
state, finalhistory = run(
    state,
    grid,
    params,
    args.adjustment_days * stepsperday,
    rad_interval=8,
    diag_interval=stepsperday,
)
history.extend(finalhistory)

metrics = equilibrium_metrics(history, window=50)
stats = equilibrium_stats(history, last_n=50)
rh = relative_humidity(state['q'], state['t'], state['p'])[0]
rh95mass = torch.sum((rh >= 0.95) * state['dp'][0] / g) / torch.sum(state['dp'][0] / g)
referencearrays = dict(
    sigma_full=grid['sigma_full'].cpu().numpy(),
    t=state['t'][0].cpu().numpy(),
    q=state['q'][0].cpu().numpy(),
    qc=state['qc'][0].cpu().numpy(),
    cloud_fraction=state['cloud_fraction'][0].cpu().numpy(),
    ts=np.array(state['ts'][0].item()),
    ps=np.array(state['ps'][0].item()),
)
if 'tke' in state:
    referencearrays['tke'] = state['tke'][0].cpu().numpy()
np.savez_compressed(referencepath, **referencearrays)

metadata = {
    'description': 'Near-equilibrium ATM407 mass-flux SCM reference state',
    'configuration_label': config['run']['label'],
    'initial_reference': initialreference,
    'reference_levels': args.levels,
    'accelerated_spinup_days': 500,
    'final_adjustment_days': previousadjustment + args.adjustment_days,
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
    'deep_precipitation_mmday': stats['precip_conv_mean'][0].item() * 86400,
    'large_scale_precipitation_mmday': stats['precip_ls_mean'][0].item() * 86400,
    'cloud_precipitation_mmday': stats['precip_cloud_mean'][0].item() * 86400,
    'cape_jkg': stats['cape_mean'][0].item(),
    'rh95_mass_fraction': rh95mass.item(),
    'cloud_base_mass_flux_kgm2s': stats['cloud_base_mass_flux_mean'][0].item(),
    'mass_flux_cap_fraction': stats['mass_flux_cap_active_mean'][0].item(),
    'temperature_cap_fraction': stats['temperature_cap_fraction_mean'][0].item(),
    'moisture_cap_fraction': stats['moisture_cap_fraction_mean'][0].item(),
    'column_water_residual_kgm2s': stats['column_water_residual_mean'][0].item(),
    'equilibrium_metrics': metrics,
    'generation_runtime_s': time.perf_counter() - start,
}

metadatapath.write_text(json.dumps(metadata, indent=2) + '\n')
print(referencepath)
print(metadatapath)
print(json.dumps(metadata, indent=2))
