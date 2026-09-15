"""Six-hour BOMEX screening with the production column physics.

Inputs follow Siebesma et al. (2003), Table B1 and Appendix B: specific water,
liquid-water potential temperature, prescribed scalar and momentum fluxes,
subsidence and geostrophic forcing. Initial profiles and the full-column
extension live in scm.case_benchmarks. Radiation is replaced by prescribed
cooling; the selected configuration retains its other physics.

The pointwise 0.5 K / 0.5 g/kg and 0.1 mm/day limits are local screening gates,
not published LES acceptance bands. Passing them alone is not validation.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scm.column_model as column_model
from scm.case_benchmarks import bomex_forcing, bomex_momentum_forcing, initialize_bomex, model_height
from scm.column_model import update_derived
from scm.configuration import extract_param_overrides, load_run_config
from scm.ensemble import default_params
from scm.surface import surface_fluxes
from scm.thermo import Lv, Rd, cp, kappa, make_grid, p0, g, saturation_specific_humidity

SEA_SURFACE_TEMPERATURE = 300.4
SURFACE_THETA_FLUX = 8.0e-3
SURFACE_WATER_FLUX = 5.2e-5
THETA_LIMIT = 0.5
WATER_LIMIT = 0.5e-3
RAIN_LIMIT = 0.1
CHECK_TOP = 2500.0
CASE_TOP = 3000.0
THETA_AT_CASE_TOP = 311.85
WATER_AT_CASE_TOP = 3.0e-3
THETA_GRADIENT = (311.85 - 308.2) / 1000.0
TROPOPAUSE_TEMPERATURE = 195.0


def extend_above_case_top(state, grid):
    """Compatibility helper: initialize_bomex now constructs the extension."""
    return state


def observed_surface_fluxes(state, grid, params):
    # Call the bulk formula with the surface 10 K warmer than the lowest air, so
    # both of its fluxes are positive, then rescale its tendencies to the
    # observed fluxes. The tendencies are linear in the fluxes, so this keeps
    # exactly the layers production puts the fluxes in.
    local = dict(params)
    local['surface_temperature'] = float(state['t'][0, -1]) + 10.0
    output = dict(surface_fluxes(state, grid, local))
    density = state['p'][:, -1] / (Rd * state['t'][:, -1])
    sensible = (density * cp * (state['ps'] / p0) ** kappa * SURFACE_THETA_FLUX).to(output['shf'].dtype)
    latent = (density * Lv * SURFACE_WATER_FLUX).to(output['lhf'].dtype)
    output['dt'] = output['dt'] * (sensible / output['shf']).unsqueeze(1).to(output['dt'].dtype)
    output['dq'] = output['dq'] * (latent / output['lhf']).unsqueeze(1).to(output['dq'].dtype)
    output['shf'] = sensible
    for name in ['lhf', 'lhf_potential', 'ocean_lhf', 'land_lhf']:
        output[name] = latent
    return output


def prescribed_radiation(state, grid, params):
    column = torch.zeros_like(state['t'])
    surface = torch.zeros_like(state['ts'])
    output = {'dt': column, 'dq': torch.zeros_like(state['q'])}
    for name in ['olr', 'asr', 'toa_net', 'sw_absorbed_sfc', 'lw_down_sfc',
                 'lw_up_sfc', 'sw_reflected_toa']:
        output[name] = surface
    return output


def potential_temperature(state):
    """Liquid-water potential temperature, conserved under phase changes."""
    return (state['t'] - Lv * state['qc'] / cp) * (p0 / state['p']) ** kappa


def forcing_tendencies(state, grid):
    theta_tendency, water_tendency = bomex_forcing(state, grid)
    return theta_tendency * (state['p'] / p0) ** kappa, water_tendency


def run_case(levels, hours, timestep, production):
    grid = make_grid(levels)
    state, _ = initialize_bomex(grid)
    state = extend_above_case_top(state, grid)
    params = dict(production)
    params.update({
        'dt': timestep,
        'use_slab_ocean': False,
        'ps0': 101500.0,
        'surface_temperature': SEA_SURFACE_TEMPERATURE,
    })
    height = model_height(state, grid)[0]
    theta_observed = potential_temperature(state)[0].clone()
    water_observed = (state['q'] + state['qc'])[0].clone()
    wind_observed = state['u'][0].clone()

    # The same forcing with no physics and no surface fluxes: the drift the
    # physics has to cancel.
    idle = {name: value.clone() if torch.is_tensor(value) else value for name, value in state.items()}

    steps = round(hours * 3600.0 / timestep)
    rain = []
    for step in range(steps):
        heating, moistening = forcing_tendencies(state, grid)
        zonal, meridional = bomex_momentum_forcing(state, grid)
        state, diagnostics, _ = column_model.physics_step(
            state, grid, params, ls_forcing={'dt': heating, 'dq': moistening, 'du': zonal, 'dv': meridional}
        )
        if (step + 1) * timestep > 0.5 * hours * 3600.0:
            rain.append(float(diagnostics['precip_total'][0]) * 86400.0)

        heating, moistening = forcing_tendencies(idle, grid)
        zonal, meridional = bomex_momentum_forcing(idle, grid)
        idle['u'] = idle['u'] + zonal * timestep
        idle['v'] = idle['v'] + meridional * timestep
        idle['t'] = idle['t'] + heating * timestep
        idle['q'] = torch.clamp(idle['q'] + moistening * timestep, min=1.0e-7)
        idle = update_derived(idle, grid)

    theta_change = potential_temperature(state)[0] - theta_observed
    water_change = (state['q'] + state['qc'])[0] - water_observed
    idle_theta_change = potential_temperature(idle)[0] - theta_observed
    idle_water_change = (idle['q'] + idle['qc'])[0] - water_observed
    checked = height <= CHECK_TOP

    def worst(change):
        values = change.masked_fill(~checked, 0.0).abs()
        index = int(torch.argmax(values))
        return float(change[index]), float(height[index])

    theta_worst, theta_height = worst(theta_change)
    water_worst, water_height = worst(water_change)
    rain_mean = sum(rain) / max(len(rain), 1)
    profile = []
    for index in torch.nonzero(checked).flatten().tolist()[::-1]:
        profile.append({
            'height_m': round(float(height[index]), 1),
            'theta_observed_k': round(float(theta_observed[index]), 3),
            'theta_change_k': round(float(theta_change[index]), 3),
            'theta_change_no_physics_k': round(float(idle_theta_change[index]), 3),
            'water_observed_gkg': round(float(water_observed[index]) * 1.0e3, 3),
            'water_change_gkg': round(float(water_change[index]) * 1.0e3, 3),
            'water_change_no_physics_gkg': round(float(idle_water_change[index]) * 1.0e3, 3),
            'cloud_fraction': round(float(state['cloud_fraction'][0, index]), 4),
        })
    return {
        'levels': levels,
        'levels_below_check_top': int(checked.sum()),
        'theta_worst_change_k': theta_worst,
        'theta_worst_height_m': theta_height,
        'water_worst_change_gkg': water_worst * 1.0e3,
        'water_worst_height_m': water_height,
        'rain_hours_3_to_6_mmday': rain_mean,
        'cloud_water_path_kgm2': float(torch.sum(state['qc'][0] * state['dp'][0] / g)),
        'maximum_cloud_fraction': float(state['cloud_fraction'][0].max()),
        'boundary_layer_depth_m': float(state.get('boundary_layer_depth_m', torch.zeros(1))[0]),
        'lowest_level_wind_change_ms': float(state['u'][0, -1] - wind_observed[-1]),
        'passes': {
            'theta': abs(theta_worst) <= THETA_LIMIT,
            'water': abs(water_worst) <= WATER_LIMIT,
            'rain': rain_mean < RAIN_LIMIT,
        },
        'profile': profile,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default=str(ROOT / 'scm/configs/atm407.toml'))
    parser.add_argument('--levels', type=int, nargs='+', default=[20, 40, 80])
    parser.add_argument('--hours', type=float, default=6.0)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='override a production parameter, e.g. cloud_ls_precip_fraction=0')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    config = load_run_config(args.config)
    production = default_params()
    production.update(extract_param_overrides(config))
    overrides = {}
    for item in args.set:
        key, value = item.split('=', 1)
        overrides[key] = json.loads(value)
    production.update(overrides)
    timestep = args.dt or float(config.get('numerics', {}).get('dt', 900.0))
    if args.output is None:
        suffix = ''.join(f'_{key}-{value}' for key, value in overrides.items())
        args.output = str(ROOT / f'outputs/column/diagnostics/bomex_observed_steady_state{suffix}.json')
    if overrides:
        print(f'overrides: {overrides}')

    column_model.surface_fluxes = observed_surface_fluxes
    column_model.radiation = prescribed_radiation

    results = []
    for levels in args.levels:
        result = run_case(levels, args.hours, timestep, production)
        results.append(result)
        passes = ' '.join(f"{name}={'pass' if ok else 'FAIL'}" for name, ok in result['passes'].items())
        print(
            f"{levels:3d} levels: theta {result['theta_worst_change_k']:+.2f} K at "
            f"{result['theta_worst_height_m']:.0f} m, water {result['water_worst_change_gkg']:+.2f} g/kg at "
            f"{result['water_worst_height_m']:.0f} m, rain {result['rain_hours_3_to_6_mmday']:.2f} mm/day, "
            f"max cloud {result['maximum_cloud_fraction']:.2f}, water path "
            f"{result['cloud_water_path_kgm2']:.3f} kg/m2 | {passes}"
        )

    first = results[0]
    print(f"\n{first['levels']} levels, change after {args.hours:g} h (no-physics change in brackets)")
    print('  height   theta obs  d theta         water obs  d water           cloud')
    for row in first['profile']:
        print(
            f"  {row['height_m']:6.0f}   {row['theta_observed_k']:7.2f}  "
            f"{row['theta_change_k']:+6.2f} ({row['theta_change_no_physics_k']:+5.2f})   "
            f"{row['water_observed_gkg']:6.2f}   {row['water_change_gkg']:+6.2f} "
            f"({row['water_change_no_physics_gkg']:+5.2f})   {row['cloud_fraction']:.3f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'benchmark': 'Siebesma et al. 2003 Table B1; theta_l and specific water; prescribed momentum forcing',
        'config': args.config,
        'overrides': overrides,
        'hours': args.hours,
        'timestep_s': timestep,
        'criteria': {
            'theta_limit_k': THETA_LIMIT,
            'water_limit_gkg': WATER_LIMIT * 1.0e3,
            'rain_limit_mmday': RAIN_LIMIT,
            'check_top_m': CHECK_TOP,
        },
        'results': results,
    }, indent=2))
    print(f"\nwrote {output}")


if __name__ == '__main__':
    main()
