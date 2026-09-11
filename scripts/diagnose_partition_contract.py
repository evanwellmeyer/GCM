"""Check that the phase-partition routines agree with each other.

When two schemes split the same total water into vapour and condensate, they must
agree about where the split lies. If they do not, condensate cycles between them
on every call while water and energy still conserve exactly, so conservation
checks cannot see it.

Two pairs are tested:

1. The pair production actually uses: ``phase_partition.partition_water`` at the
   configured critical relative humidity, alternating with
   ``condensation.partial_condensation`` at the same value. This must be stable.
2. The older pair: ``partition_mse`` (grid-mean full saturation) alternating with
   ``partial_condensation`` at the configured value. This cycles, and is kept as a known gap. The
   ATM407 path never calls partition_mse; only the inactive UW convection and
   shallow-plume-v2 schemes do, and they would recycle condensate if run with a
   critical relative humidity below one.

The older ``scripts/diagnose_partition_handoff.py`` (removed 10 Sep) tested only
the older pair. Its failure did not describe the production model.
"""
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scm.condensation import partial_condensation
from scm.configuration import extract_param_overrides, load_run_config
from scm.convection_uw import partition_mse
from scm.phase_partition import partition_water
from scm.thermo import Lv, cp, saturation_specific_humidity

# Use the value production runs with, not a copy of it.
CRITICAL = float(extract_param_overrides(load_run_config(
    str(root / 'scm' / 'configs' / 'atm407.toml')))['condensation_rh_crit'])
CYCLES = 5


def column():
    temperature = torch.tensor([[280.0]], dtype=torch.float64)
    pressure = torch.full_like(temperature, 85000.0)
    vapor = 0.98 * saturation_specific_humidity(temperature, pressure)
    return temperature, pressure, vapor, torch.zeros_like(vapor)


def alternate(first):
    temperature, pressure, vapor, liquid = column()
    water, enthalpy = vapor.clone(), cp * temperature + Lv * vapor
    history = []
    for _ in range(CYCLES):
        temperature, vapor, liquid = first(temperature, pressure, vapor, liquid)
        after_first = float(liquid) * 1000.0
        output = partial_condensation(
            {'t': temperature, 'q': vapor, 'qc': liquid, 'p': pressure,
             'dp': torch.full_like(pressure, 5000.0)},
            {'cloud_ls_precip_fraction': 0.0}, CRITICAL)
        temperature = temperature + output['dt']
        vapor = vapor + output['dq']
        liquid = liquid + output['cloud_source']
        history.append((after_first, float(liquid) * 1000.0))
    water_error = float(vapor + liquid - water)
    enthalpy_error = float(cp * temperature + Lv * vapor - enthalpy)
    spread = max(abs(a - b) for pair in history for a in pair for b in pair)
    return history, spread, water_error, enthalpy_error


def production(temperature, pressure, vapor, liquid):
    t, q, l, _ = partition_water(vapor + liquid, cp * temperature + Lv * vapor,
                                 pressure, CRITICAL)
    return t, q, l


def older(temperature, pressure, vapor, liquid):
    height = torch.zeros_like(vapor)
    return partition_mse(vapor + liquid, cp * temperature + Lv * vapor, height, pressure)


for label, first, must_pass in (('production pair', production, True),
                                ('older pair (inactive schemes only)', older, False)):
    history, spread, werr, herr = alternate(first)
    stable = spread < 1.0e-9
    print(f'{label}: condensate g/kg per cycle', [round(b, 7) for _, b in history])
    print(f'  spread {spread:.2e} g/kg | water error {werr:.1e} | enthalpy error {herr:.1e} | '
          f'{"STABLE" if stable else "RECYCLES"}')
    if must_pass and not stable:
        raise SystemExit('production partition pair recycles condensate')
