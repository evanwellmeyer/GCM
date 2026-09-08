"""Check whether UW and partial condensation agree without external forcing."""
import json
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from scm.thermo import cp, Lv, saturation_specific_humidity
from scm.shallow_plume_v2 import partition_mse
from scm.condensation import partial_condensation

temperature = torch.tensor([[280.0]], dtype=torch.float64)
pressure = torch.full_like(temperature, 85000.0)
vapor = 0.98 * saturation_specific_humidity(temperature, pressure)
liquid = torch.zeros_like(vapor)
height = torch.zeros_like(vapor)
water = vapor.clone()
enthalpy = cp * temperature + Lv * vapor
history = []
for cycle in range(5):
    temperature, vapor, liquid = partition_mse(
        vapor + liquid, cp * temperature + Lv * vapor, height, pressure)
    full = liquid.item() * 1000
    output = partial_condensation(
        {'t': temperature, 'q': vapor, 'qc': liquid, 'p': pressure,
         'dp': torch.full_like(pressure, 5000.0)},
        {'cloud_ls_precip_fraction': 0.0}, 0.95)
    temperature += output['dt']
    vapor += output['dq']
    liquid += output['cloud_source']
    history.append({'cycle': cycle + 1, 'full_saturation_qc_gkg': full,
                    'partial_cloud_qc_gkg': liquid.item() * 1000,
                    'enthalpy_error_jkg': (cp * temperature + Lv * vapor - enthalpy).item(),
                    'water_error_kgkg': (vapor + liquid - water).item()})
path = root / 'outputs/column/diagnostics/uw_partition_handoff.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(history, indent=2) + '\n')
print(json.dumps(history, indent=2))
