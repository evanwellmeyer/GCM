import pytest
import torch

from scm.phase_partition import partition_water
from scm.condensation import partial_condensation
from scm.thermo import cp, Lv, saturation_specific_humidity


@pytest.mark.parametrize('humidity', [0.5, 0.98, 1.05])
def test_shared_partition_is_stationary_without_forcing(humidity):
    temperature = torch.tensor([[280.0]], dtype=torch.float64)
    pressure = torch.full_like(temperature, 85000)
    water = humidity * saturation_specific_humidity(temperature, pressure)
    enthalpy = cp * temperature + Lv * water
    temperature, vapor, liquid, fraction = partition_water(water, enthalpy, pressure, 0.95)
    expected = liquid.clone()
    for cycle in range(10):
        temperature, vapor, liquid, fraction = partition_water(
            vapor + liquid, cp * temperature + Lv * vapor, pressure, 0.95)
        output = partial_condensation(
            {'t': temperature, 'q': vapor, 'qc': liquid, 'p': pressure,
             'dp': torch.full_like(pressure, 5000)},
            {'cloud_ls_precip_fraction': 0.0}, 0.95)
        temperature += output['dt']
        vapor += output['dq']
        liquid += output['cloud_source']
        torch.testing.assert_close(liquid, expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(vapor + liquid, water, atol=1e-14, rtol=0)
        torch.testing.assert_close(cp * temperature + Lv * vapor, enthalpy, atol=1e-8, rtol=0)
        assert output['precip'].item() == 0
