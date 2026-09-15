"""Regression cases for the numerical step-dependent launch failure."""

import json
from pathlib import Path

import pytest
import torch

from scm.convection_uw import _integrate_columns, critical_mixing_fraction, uw_shallow_convection
from scm.thermo import g, kappa, p0, make_grid


def test_dry_buoyancy_sorting_has_the_analytic_stopping_root():
    pressure = torch.tensor(90000., dtype=torch.float64)
    theta = torch.tensor(299., dtype=torch.float64)
    environment = theta + 1.
    water = torch.tensor(.001, dtype=torch.float64)
    virtual = environment * (pressure / p0) ** kappa * (1. + .61 * water)
    velocity = torch.tensor(10., dtype=torch.float64)
    fraction = critical_mixing_fraction(theta, water, environment, water, pressure, virtual, velocity, 100.)
    expected = 1. + 200. * g * (theta - environment) / (environment * velocity)
    assert torch.allclose(fraction, expected, atol=1e-8)


@pytest.mark.parametrize('index', [0, 1, 2])
def test_frozen_launch_converges_when_the_vertical_step_changes(index):
    fixture = json.loads((Path(__file__).parent / 'testdata/uw_launch_frozen_20260914.json').read_text())[index]
    arrays = [torch.tensor(value, dtype=torch.float32) for value in fixture['input'].values()]
    coarse = _integrate_columns(*arrays, dict(fixture['params'], uw_shallow_vertical_step_m=50.))
    fine = _integrate_columns(*arrays, dict(fixture['params'], uw_shallow_vertical_step_m=10.))
    assert torch.max(torch.abs(coarse['water_flux'] - fine['water_flux'])) * 86400 < .05
    assert torch.max(torch.abs(coarse['mse_flux'] - fine['mse_flux'])) < 2.
    assert torch.max(torch.abs(coarse['plume_top_height'] - fine['plume_top_height'])) < 2.
    if index == 0:
        assert torch.max(torch.abs(fine['water_flux'])) * 86400 > 1.
    assert torch.all(coarse['precipitation_source'] >= 0.)
    assert torch.all(fine['water_flux'][:, [0, -1]] == 0.)


@pytest.mark.parametrize('ceiling', [4000., 1400.])
def test_frozen_plume_conserves_water_and_energy_with_partial_top_cells(ceiling):
    fixture = json.loads((Path(__file__).parent / 'testdata/uw_launch_frozen_20260914.json').read_text())[0]
    arrays = {name: torch.tensor(value) for name, value in fixture['input'].items()}
    state = {name: arrays[name] for name in ['t', 'q', 'qc', 'u', 'v', 'p', 'dp', 'tke']}
    state['boundary_layer_depth_m'] = arrays['boundary_depth']
    result = uw_shallow_convection(state, make_grid(20), dict(fixture['params'], uw_shallow_maximum_height_m=ceiling))
    assert result['water_residual'].abs().max() < 2e-8
    assert result['energy_residual'].abs().max() < .1
    assert torch.all(result['precip'] >= 0.)
