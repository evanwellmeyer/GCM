"""A layer reconstruction must describe one thermodynamic state."""

import pytest
import torch
import json
from pathlib import Path

from scm.convection_uw import conserved_environment
from scm.thermo import cp, Lv as latent, g, kappa, p0


@pytest.mark.parametrize('water', [.001, .012, .025])
def test_conserved_reconstruction_recovers_heat_water_and_winds(water):
    pressure = torch.tensor(85000., dtype=torch.float64)
    center = pressure + 1000.
    values = [torch.tensor(value, dtype=torch.float64) for value in [300., water, 5., -2.]]
    slopes = [torch.tensor(value, dtype=torch.float64) for value in [-1e-4, 1e-7, .001, -.001]]
    _, temperature, vapor, liquid, theta, total, wind, crosswind, energy = conserved_environment(
        pressure, center, values, slopes, 1500.)
    exner = (pressure / p0) ** kappa
    assert float(theta) == pytest.approx(300.1)
    assert float(total) == pytest.approx(water - .0001)
    assert torch.allclose(vapor + liquid, total, atol=1e-14, rtol=0.)
    assert torch.allclose((temperature - latent * liquid / cp) / exner, theta, atol=1e-12, rtol=0.)
    assert float(wind) == pytest.approx(4.)
    assert float(crosswind) == pytest.approx(-1.)
    assert float(energy) == pytest.approx(float(cp * exner * theta + latent * total + g * 1500.))
    assert vapor >= 0. and liquid >= 0.


def test_constant_conserved_profiles_survive_pressure_change():
    pressure = torch.tensor([70000., 80000., 90000.], dtype=torch.float64)
    values = [torch.full_like(pressure, value) for value in [300., .016, 0., 0.]]
    result = conserved_environment(pressure, pressure.new_tensor(85000.), values,
                                   [torch.zeros_like(pressure)] * 4, 1000.)
    assert torch.all(result[4] == 300.)
    assert torch.all(result[5] == .016)
    assert torch.allclose(result[2] + result[3], result[5], atol=1e-14, rtol=0.)


def test_partial_cloud_center_reconstructs_the_existing_state():
    from scm.phase_partition import partition_water
    pressure = torch.tensor(90000., dtype=torch.float64)
    water = torch.tensor(.015, dtype=torch.float64)
    energy = torch.tensor(330000., dtype=torch.float64)
    temperature, vapor, liquid, fraction = partition_water(water, energy, pressure, .9)
    theta = (temperature - latent * liquid / cp) / (pressure / p0) ** kappa
    zero = torch.zeros_like(water)
    result = conserved_environment(pressure, pressure, [theta, water, zero, zero], [zero] * 4, 0., .9)
    assert 0. < fraction < 1.
    assert torch.allclose(result[1], temperature, atol=1e-8, rtol=0.)
    assert torch.allclose(result[2], vapor, atol=1e-10, rtol=0.)
    assert torch.allclose(result[3], liquid, atol=1e-10, rtol=0.)


def test_conserved_ascent_converges_across_internal_vertical_steps():
    from scm.convection_uw import _integrate_columns
    fixture = json.loads((Path(__file__).parent / 'testdata/uw_launch_frozen_20260914.json').read_text())[0]
    arrays = [torch.tensor(value, dtype=torch.float64) for value in fixture['input'].values()]
    settings = dict(fixture['params'], uw_shallow_conserved_environment=True)
    coarse = _integrate_columns(*arrays, dict(settings, uw_shallow_vertical_step_m=50.))
    fine = _integrate_columns(*arrays, dict(settings, uw_shallow_vertical_step_m=25.))
    assert (coarse['water_flux'] - fine['water_flux']).abs().max() * 86400 < .05
    assert (coarse['mse_flux'] - fine['mse_flux']).abs().max() < 2.
    assert (coarse['plume_top_height'] - fine['plume_top_height']).abs().max() < 2.
    assert torch.all(fine['water_flux'][:, [0, -1]] == 0.)
