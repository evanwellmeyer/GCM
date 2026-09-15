"""Replay one saved plume and inspect the first flux above its launch layer."""

import argparse
import json
import inspect
from pathlib import Path
import sys

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from scm.convection_uw import _integrate_columns, _integrate_one_column, source_properties, source_slope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--hour', type=float, default=3.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = json.loads(args.input.read_text())
    sample = min(saved['samples'], key=lambda item: abs(item['hour'] - args.hour))
    state = {key: torch.tensor(value, dtype=torch.float32) for key, value in sample['plume_input'].items() if value is not None}
    names = ['t', 'q', 'qc', 'u', 'v', 'p', 'dp', 'height', 'theta_liquid', 'total_water', 'mse', 'tke', 'boundary_depth']
    lines, first = inspect.getsourcelines(_integrate_one_column)
    capline = first + next(index for index, line in enumerate(lines) if 'flux = torch.minimum(torch.exp(values[4])' in line)
    sourcecapline = first + next(index for index, line in enumerate(lines) if 'mass_flux = torch.minimum(mass_flux, area_max' in line)
    caps = []

    def trace(frame, event, arg):
        if frame.f_code is not _integrate_one_column.__code__:
            return None
        if event == 'line' and frame.f_lineno in (capline, sourcecapline):
            local = frame.f_locals
            if frame.f_lineno == capline:
                requested = float(torch.exp(local['values'][4]))
                capacity = float(local['area_max'] * local['density'] * local['velocity'])
                face = local['lower']
            else:
                requested = float(local['mass_flux'])
                capacity = float(local['area_max'] * local['source_density'] * local['velocity'])
                face = local['source']
            caps.append({'face': face, 'requested_mass_flux': requested, 'area_limited_capacity': capacity,
                         'active': requested > capacity})
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        result = _integrate_columns(*(state[name] for name in names), sample['plume_params'], state.get('interfaces'))
    finally:
        sys.settrace(previous)
    launch = source_properties(*(state[name][0] for name in ['p', 'dp', 'height', 'theta_liquid', 'total_water', 'u', 'v', 'tke', 'boundary_depth']),
                               None if 'interfaces' not in state else state['interfaces'][0])
    source = launch['index']
    water, pressure, height = state['total_water'][0], state['p'][0], state['height'][0]
    slope = source_slope(water, pressure)
    rows = []
    for face in range(source - 1, max(0, source - 4), -1):
        upper, lower = face - 1, face
        share = (launch['heights'][face] - height[lower]) / (height[upper] - height[lower])
        environment = water[lower] + share * (water[upper] - water[lower])
        reconstructed = water[upper] + slope[upper] * (launch['faces'][face] - pressure[upper])
        massflux = result['mass_flux_profile'][0, upper]
        flux = result['water_flux'][0, face]
        parcel = environment + flux / massflux.clamp(min=1e-20)
        # This changes only the compensating-environment sample, not the
        # plume or its mass flux. It is not execution of CAM's full closure.
        comparison = massflux * (parcel - reconstructed)
        rows.append({'face': face, 'height_m': float(launch['heights'][face]),
                     'mass_flux_kgm2s': float(massflux),
                     'mass_flux_relative_to_launch': float(massflux / result['cloud_base_mass_flux'][0]),
                     'environment_total_water_gkg': float(environment * 1000),
                     'upper_cell_reconstruction_gkg': float(reconstructed * 1000),
                     'inferred_plume_total_water_gkg': float(parcel * 1000),
                     'water_flux_kgm2day': float(flux * 86400),
                     'held_plume_upwind_flux_kgm2day': float(comparison * 86400)})
    error = (result['water_flux'][0] - torch.tensor(sample['raw_flux'])).abs().max() * 86400
    output = {'hour': sample['hour'], 'max_replay_flux_error_kgm2day': float(error), 'area_caps': caps,
              'launch_mass_flux_kgm2s': float(result['cloud_base_mass_flux'][0]),
              'launch_height_m': float(launch['heights'][source]), 'rows': rows}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
