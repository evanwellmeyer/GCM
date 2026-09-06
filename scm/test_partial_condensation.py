import pytest
import torch

from scm.condensation import condensation
from scm.cloud_microphysics import cloud_microphysics_step
from scm.thermo import cp, Lv, g, make_grid, saturation_specific_humidity


@pytest.mark.parametrize('critical', [0.9, 0.95])
@pytest.mark.parametrize('humidity,liquid', [(0.99, 0.0), (0.5, 0.001), (1.1, 0.001)])
@pytest.mark.parametrize('rain', [0.0, 0.95])
def test_partial_partition_contract(critical, humidity, liquid, rain):
    grid = make_grid(4, dtype=torch.float64)
    temperature = torch.full((1, 4), 280.0, dtype=torch.float64)
    pressure = torch.full_like(temperature, 80000.0)
    state = {
        't': temperature,
        'q': humidity * saturation_specific_humidity(temperature, pressure),
        'qc': torch.full_like(temperature, liquid),
        'p': pressure,
        'dp': torch.full_like(temperature, 5000.0),
    }
    params = {
        'condensation_rh_crit': critical,
        'cloud_microphysics_enabled': True,
        'cloud_ls_precip_fraction': rain,
        'cloud_autoconv_qc_thresh': 1.0,
        'cloud_qc_max': 1.0,
        'dt': 900.0,
    }
    for iteration in range(3):
        water = state['q'] + state['qc']
        energy = cp * state['t'] + Lv * state['q']
        output = condensation(state, grid, params)
        if iteration and rain == 0.0:
            assert output['dq'].abs().max() < 1.0e-12
        state['t'] = state['t'] + output['dt']
        state['q'] = state['q'] + output['dq']
        clouds = cloud_microphysics_step(state, grid, params, output, {})
        state['t'] = state['t'] + clouds['dt']
        state['q'] = state['q'] + clouds['dq']
        state['qc'] = clouds['qc']
        loss = ((water - state['q'] - state['qc']) * state['dp'] / g).sum(1)
        assert torch.allclose(loss, output['precip'] + clouds['precip'], atol=1.0e-12)
        assert torch.allclose(cp * state['t'] + Lv * state['q'], energy, atol=1.0e-8, rtol=0)
        assert torch.all(state['q'] >= 0) and torch.all(state['qc'] >= 0)
