"""Separate the saved launch-layer flux jump without changing any physics."""

import json
from pathlib import Path


def decompose(massbelow, massabove, sourcewater, plumewater, environment, fluxbelow):
    # Effective environment exactly represents the reconstructed lower flux.
    # It is not a measured environmental humidity at a physical interface.
    effective = sourcewater - fluxbelow / massbelow
    terms = {
        'mass_flux_change': (massabove - massbelow) * (sourcewater - effective),
        'plume_water_change': massabove * (plumewater - sourcewater),
        'environment_contrast_change': -massabove * (environment - effective),
    }
    return effective, terms


def main():
    directory = Path(__file__).resolve().parents[1] / 'outputs/column/diagnostics'
    saved = json.loads((directory / 'bomex_launch_budget_20_6h.json').read_text())
    replay = json.loads((directory / 'bomex_launch_flux_hour3_caps.json').read_text())
    sample = next(row for row in saved['samples'] if row['hour'] == replay['hour'])
    face = replay['rows'][0]
    below = sample['raw_flux'][sample['source_index']]
    source = sample['source']['water']
    effective, terms = decompose(sample['raw_mass_flux'], face['mass_flux_kgm2s'],
                                source, face['inferred_plume_total_water_gkg'] / 1000,
                                face['environment_total_water_gkg'] / 1000, below)
    observed = face['water_flux_kgm2day'] - below * 86400
    terms = {name: value * 86400 for name, value in terms.items()}
    residual = sum(terms.values()) - observed
    if abs(residual) > 1e-5:
        raise RuntimeError('flux decomposition does not reproduce the archived jump')
    output = {'hour': replay['hour'], 'effective_lower_environment_gkg': effective * 1000,
              'source_water_gkg': source * 1000, 'upper_environment_gkg': face['environment_total_water_gkg'],
              'upper_plume_water_gkg': face['inferred_plume_total_water_gkg'],
              'terms_kgm2day': terms, 'upward_flux_increase_kgm2day': observed,
              'residual_kgm2day': residual,
              'interpretation': 'Algebraic flux decomposition, not independent causal interventions or validation of a coefficient.'}
    (directory / 'bomex_launch_contrast.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
