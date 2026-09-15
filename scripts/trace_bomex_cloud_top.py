"""Attribute archived cloud-top water storage without changing the physics."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / 'outputs/column/diagnostics'
    source = directory / 'bomex_launch_budget_20_6h.json'
    saved = json.loads(source.read_text())
    rows = []
    for index, height in enumerate(saved['height_m']):
        if not 700 < height < 2500:
            continue
        changes = {name: sum(sample['water'][name][index] * 900 * 1000
                             for sample in saved['samples'])
                   for name in saved['samples'][0]['water']}
        rows.append({'height_m': height, 'water_change_gkg': changes})
    sample = next(value for value in saved['samples'] if value['hour'] == 3.)
    index = min(range(len(saved['height_m'])),
                key=lambda value: abs(saved['height_m'][value] - 1823))
    result = {'source': str(source.relative_to(root)), 'configuration': 'archived baseline, not combined repair',
              'profiles': rows, 'hour3_cloud_top_flux_kgm2day': {
                  'entering': sample['applied_flux'][index + 1] * 86400,
                  'leaving': sample['applied_flux'][index] * 86400},
              'limitation': 'Identifies the depositing operator, not which closure causes excessive transport.'}
    (directory / 'bomex_cloud_top_trace.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
