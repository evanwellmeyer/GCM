"""Check conservative internal shallow updates independently of plume cost."""

import pytest
import torch

from scm import convection_uw as shallow
from scm.thermo import g


def state():
    return {'t': torch.full((2, 3), 280., dtype=torch.float64),
            'q': torch.full((2, 3), .01, dtype=torch.float64),
            'dp': torch.tensor([[100., 200., 300.]] * 2, dtype=torch.float64)}


def test_substeps_evolve_conserve_and_do_not_mutate(monkeypatch):
    original = state()
    saved = {key: value.clone() for key, value in original.items()}
    calls = []

    def exchange(current, grid, params):
        calls.append((params['dt'], current['q'].clone()))
        tendency = torch.zeros_like(current['q'])
        tendency[:, 0] = -current['q'][:, 0] * 1e-4
        tendency[:, 1] = -tendency[:, 0] * .5
        result = {'d' + name: torch.zeros_like(tendency) for name in ('t', 'q', 'qc', 'u', 'v')}
        result['dq'] = tendency
        result['precip'] = torch.zeros(2, dtype=torch.float64)
        return result

    monkeypatch.setattr(shallow, 'shallow_step', exchange)
    params = {'dt': 900., 'uw_shallow_maximum_timestep_s': 225.}
    result = shallow.uw_shallow_convection(original, {}, params)
    assert len(calls) == 4
    assert all(step == 225. for step, _ in calls)
    assert all(torch.all(a[1][:, 0] > b[1][:, 0]) for a, b in zip(calls, calls[1:]))
    expected = .01 * (1 - 225.e-4) ** 4
    assert torch.allclose(original['q'][:, 0] + 900. * result['dq'][:, 0], torch.full((2,), expected, dtype=torch.float64))
    assert result['water_residual'].abs().max() < 1e-15
    assert result['energy_residual'].abs().max() < 1e-9
    assert params['dt'] == 900.
    assert original.keys() == saved.keys()
    assert all(torch.equal(original[key], saved[key]) for key in saved)


def test_substeps_average_rain_and_close_endpoint_budget(monkeypatch):
    original = state()
    rates = []

    def rain(current, grid, params):
        rate = current['q'] * 1e-5
        rates.append((rate * current['dp'] / g).sum(dim=1))
        result = {'d' + name: torch.zeros_like(rate) for name in ('t', 'q', 'qc', 'u', 'v')}
        result['dq'] = -rate
        result['dt'] = shallow.Lv / shallow.cp * rate
        result['precip'] = rates[-1]
        return result

    monkeypatch.setattr(shallow, 'shallow_step', rain)
    result = shallow.uw_shallow_convection(original, {}, {'dt': 500., 'uw_shallow_maximum_timestep_s': 225.})
    assert len(rates) == 3
    assert torch.allclose(result['precip'], torch.stack(rates).mean(dim=0))
    assert result['water_residual'].abs().max() < 1e-15
    assert result['energy_residual'].abs().max() < 1e-9


@pytest.mark.parametrize('maximum', [0., 900., 1000.])
def test_single_step_compatibility(monkeypatch, maximum):
    marker = object()
    monkeypatch.setattr(shallow, 'shallow_step', lambda *args: marker)
    assert shallow.uw_shallow_convection(state(), {}, {'dt': 900., 'uw_shallow_maximum_timestep_s': maximum}) is marker


@pytest.mark.parametrize('params', [{'dt': 0.}, {'dt': float('nan')},
                                   {'uw_shallow_maximum_timestep_s': -1.},
                                   {'uw_shallow_maximum_timestep_s': float('inf')}])
def test_reject_invalid_time_settings(params):
    with pytest.raises(ValueError):
        shallow.uw_shallow_convection(state(), {}, params)
