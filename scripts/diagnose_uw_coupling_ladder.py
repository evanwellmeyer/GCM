from pathlib import Path
import argparse
import copy
import json
import sys

import torch

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scm.case_benchmarks import initialize_bomex
from scm.column_model import physics_step
from scm.thermo import g, make_grid, make_smooth_test_grid, relative_humidity


parser = argparse.ArgumentParser()
parser.add_argument('--hours', type=float, default=2.0)
parser.add_argument('--timestep', type=float, default=900.0)
parser.add_argument('--output', type=Path)
args = parser.parse_args()


def clone_state(state):
    return {
        name: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for name, value in state.items()
    }


def zero_radiation(state):
    column = torch.zeros_like(state['ts'])
    profile = torch.zeros_like(state['t'])
    return {
        'dt': profile,
        'dq': profile,
        'olr': column,
        'asr': column,
        'toa_net': column,
        'sw_absorbed_sfc': column,
        'sw_reflected_toa': column,
        'lw_down_sfc': column,
        'lw_up_sfc': column,
    }


stages = {
    'turbulence': {
        'shallow_convection_scheme': 'none',
        'uw_shallow_condensate_maximum_kgkg': 1.0,
        'condensation_scheme': 'none',
        'cloud_microphysics_enabled': False,
    },
    'shallow_transport': {
        'shallow_convection_scheme': 'uw_shallow',
        'uw_shallow_condensate_maximum_kgkg': 1.0,
        'condensation_scheme': 'none',
        'cloud_microphysics_enabled': False,
    },
    'shallow_precipitation': {
        'shallow_convection_scheme': 'uw_shallow',
        'uw_shallow_condensate_maximum_kgkg': 1.0e-3,
        'condensation_scheme': 'none',
        'cloud_microphysics_enabled': False,
    },
    'saturation_adjustment': {
        'shallow_convection_scheme': 'uw_shallow',
        'uw_shallow_condensate_maximum_kgkg': 1.0e-3,
        'condensation_scheme': 'large_scale',
        'cloud_microphysics_enabled': False,
    },
    'cloud_microphysics': {
        'shallow_convection_scheme': 'uw_shallow',
        'uw_shallow_condensate_maximum_kgkg': 1.0e-3,
        'condensation_scheme': 'large_scale',
        'cloud_microphysics_enabled': True,
    },
}


def run_stage(grid, updates):
    initial, params = initialize_bomex(grid)
    state = clone_state(initial)
    state = {
        name: value.to(dtype=torch.float64)
        if torch.is_tensor(value) and value.is_floating_point()
        else value
        for name, value in state.items()
    }
    params.update({
        'dt': args.timestep,
        'use_slab_ocean': False,
        'surface_flux_coupling': 'boundary_layer',
        'boundary_layer_scheme': 'uw_moist',
        'convection_scheme': 'none',
        'bl_min_depth_m': 50.0,
        'bl_max_depth_m': 3000.0,
        'uw_maximum_turbulent_height_m': 5000.0,
        'include_precip_enthalpy_flux': False,
    })
    params.update(updates)
    steps = round(args.hours * 3600.0 / args.timestep)
    diagnostics = []
    for _ in range(steps):
        state, diagnostic, _ = physics_step(
            state,
            grid,
            params,
            rad_cache=zero_radiation(state),
        )
        diagnostics.append(diagnostic)

    mass = state['dp'] / g
    rh = relative_humidity(state['q'], state['t'], state['p'])
    total_precipitation = sum(
        item['precip_total'][0] * 86400.0 for item in diagnostics
    ) / len(diagnostics)
    return {
        'maximum_water_residual_kgm2s': max(
            abs(float(item['column_water_residual'][0])) for item in diagnostics
        ),
        'maximum_mse_residual_wm2': max(
            abs(float(item['column_mse_residual'][0])) for item in diagnostics
        ),
        'maximum_energy_residual_wm2': max(
            abs(float(item['column_energy_residual'][0])) for item in diagnostics
        ),
        'boundary_layer_depth_m': float(diagnostics[-1]['boundary_layer_depth_m'][0]),
        'maximum_tke_m2s2': float(state.get('tke', torch.zeros_like(state['t'])).max()),
        'rh95_mass_fraction': float(
            torch.sum((rh >= 0.95) * mass) / torch.sum(mass)
        ),
        'cloud_water_path_kgm2': float(torch.sum(state['qc'] * mass)),
        'precipitation_mmday': float(total_precipitation),
    }


grids = {
    'historical_20level': make_grid(20, dtype=torch.float64),
    'smooth_20level_v1': make_smooth_test_grid(dtype=torch.float64),
}
results = {
    grid_name: {
        stage_name: run_stage(grid, updates)
        for stage_name, updates in stages.items()
    }
    for grid_name, grid in grids.items()
}

output = args.output
if output is None:
    output = root / 'outputs' / 'column' / 'diagnostics' / 'uw_coupling_ladder.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + '\n')
print(output)
print(json.dumps(results, indent=2))
