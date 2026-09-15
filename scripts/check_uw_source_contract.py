"""Audit frozen UW source states against CAM6_3_000 scalar contracts.

This is a diagnostic reference translation, not a replacement closure.
Interface TKE is absent from the fixtures, so its layer-weighted comparison
is explicitly a proxy, not an exact CAM mass-flux prediction.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from scm.convection_uw import _integrate_columns, source_properties


def slope(field, pressure):
    """CAM slope routine with bottom-to-top arrays."""
    result = np.zeros_like(field)
    below = (field[1] - field[0]) / (pressure[1] - pressure[0])
    for k in range(1, len(field)):
        above = (field[k] - field[k - 1]) / (pressure[k] - pressure[k - 1])
        result[k - 1] = max(0., min(above, below)) if above > 0 else min(0., max(above, below))
        below = above
    result[-1] = result[-2]
    return result


def belowflux(massflux, faces, inversion, dt, source, mean, top, bottom):
    """CAM fluxbelowinv, retaining its one-based inversion index."""
    flux = np.zeros_like(faces)
    thickness = faces[inversion - 1] - faces[inversion]
    fraction = massflux * 9.81 * dt / thickness
    contrast = bottom - top
    denominator = max(1.e-20, contrast) if contrast >= 0 else min(-1.e-20, contrast)
    position = np.clip((mean - top) / denominator, 0., 1.)
    original = contrast
    if position == 0. or position == 1.:
        bottom = mean
    pressure = faces[inversion - 1] - position * thickness
    flux[:inversion] = massflux * (source - bottom) * (faces[0] - faces[:inversion]) / (faces[0] - pressure)
    if position <= fraction:
        flux[inversion - 1] += (1. - position / fraction) * massflux * original
    return flux


def audit(fixture):
    data = {key: np.asarray(value, dtype=float)[0] for key, value in fixture['input'].items()}
    height = data['height'][::-1]
    pressure = data['p'][::-1]
    thickness = data['dp'][::-1]
    faces = np.r_[pressure[0] + thickness[0] / 2., pressure[0] + thickness[0] / 2. - np.cumsum(thickness)]
    # Matched source support isolates reconstruction; this does not emulate
    # CAM's separate interface-height inversion diagnosis.
    tensor = {key: torch.tensor(value, dtype=torch.float64) for key, value in data.items()}
    launch = source_properties(tensor['p'], tensor['dp'], tensor['height'], tensor['theta_liquid'],
                               tensor['total_water'], tensor['u'], tensor['v'], tensor['tke'],
                               tensor['boundary_depth'])
    if launch is None:
        raise ValueError('fixture has no valid subcloud source')
    count = len(height) - launch['index']
    inversion = count + 1
    water = data['total_water'][::-1]
    theta = data['theta_liquid'][::-1]
    watergradient = slope(water, pressure)
    thetagradient = slope(theta, pressure)
    virtual = []
    for k in range(count):
        for face in (faces[k], faces[k + 1]):
            liquid = theta[k] + thetagradient[k] * (face - pressure[k])
            total = water[k] + watergradient[k] * (face - pressure[k])
            virtual.append(liquid * (1. + .608 * total))
    sourcetheta = min(virtual) / (1. + .608 * water[0])
    arrays = [torch.tensor(value, dtype=torch.float32) for value in fixture['input'].values()]
    output = _integrate_columns(*arrays, fixture['params'])
    massflux = float(output['cloud_base_mass_flux'][0])
    bottom = water[count - 1] + watergradient[count - 1] * (faces[count] - pressure[count - 1])
    top = water[count + 1] + watergradient[count + 1] * (faces[count + 1] - pressure[count + 1])
    dt = float(fixture['params']['dt'])
    reference = belowflux(massflux, faces, inversion, dt, water[0], water[count], top, bottom)
    actual = output['water_flux'][0].numpy()[::-1]
    tke = data['tke'][::-1]
    return {'name': fixture['name'], 'matched_source_layers': count,
            'legacy_minimum_theta_l_k': float(min(theta[:count])),
            'current_source_theta_l_k': float(launch['theta']),
            'cam_reconstruction_theta_l_k': float(sourcetheta),
            'legacy_source_level_tke_m2s2': float(tke[count - 1]),
            'current_source_tke_m2s2': float(launch['tke']),
            'layer_weighted_tke_proxy_m2s2': float(np.average(tke[:count], weights=thickness[:count])),
            'held_mass_flux_kgm2s': massflux,
            'cam_subcloud_water_flux_kgm2day': (reference[:inversion] * 86400).tolist(),
            'current_subcloud_water_flux_kgm2day': (actual[:inversion] * 86400).tolist(),
            'surface_internal_flux': float(reference[0]),
            'reference_closed_column_water_residual': float(np.sum(reference[:-1] - reference[1:]))}


def main():
    torch.set_num_threads(1)
    fixtures = json.loads((root / 'scm/testdata/uw_launch_frozen_20260914.json').read_text())
    results = [audit(fixture) for fixture in fixtures]
    output = root / 'outputs/column/diagnostics/uw_source_contract_corrected.json'
    output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
