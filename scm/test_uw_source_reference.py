"""Manufactured checks for the isolated CAM comparison routines."""

import numpy as np
import pytest
import torch

from scripts.check_uw_source_contract import belowflux, slope
from scm.convection_uw import source_slope, subcloud_flux, source_properties


def test_linear_pressure_reconstruction():
    pressure = np.array([95000., 85000., 75000., 65000.])
    np.testing.assert_allclose(slope(3. + pressure * .002, pressure), .002)


@pytest.mark.parametrize('dt', [100., 100000.])
def test_subcloud_flux_and_inversion_crossing(dt):
    faces = np.array([100000., 90000., 80000., 70000.])
    flux = belowflux(.01, faces, 2, dt, .018, .012, .010, .014)
    expected = .01 * .004 * 10000. / 15000.
    fraction = .01 * 9.81 * dt / 10000.
    if fraction >= .5:
        expected += (1. - .5 / fraction) * .01 * .004
    assert flux[1] == pytest.approx(expected)
    assert flux[0] == 0.
    assert sum(flux[:-1] - flux[1:]) == pytest.approx(0., abs=1.e-18)


def test_uniform_scalar_has_no_flux():
    faces = np.array([100000., 90000., 80000., 70000.])
    np.testing.assert_array_equal(belowflux(.01, faces, 2, 900., .012, .012, .012, .012), 0.)


@pytest.mark.parametrize('dt', [100., 100000.])
def test_production_subcloud_flux_matches_reference(dt):
    faces = np.array([100000., 90000., 80000., 70000.])
    expected = belowflux(.01, faces, 2, dt, .018, .012, .010, .014)
    values = [torch.tensor(value, dtype=torch.float64) for value in [.01, .018, .012, .010, .014]]
    actual = subcloud_flux(values[0], torch.tensor(faces[::-1].copy()), 2, dt, *values[1:])
    np.testing.assert_allclose(actual.numpy(), expected[:2][::-1], rtol=2e-4, atol=1e-12)


def test_source_reconstruction_and_native_tke_weights():
    pressure = torch.tensor([65000., 75000., 85000., 95000.], dtype=torch.float64)
    height = torch.tensor([3500., 2500., 1500., 500.], dtype=torch.float64)
    theta = torch.tensor([310., 305., 300., 299.], dtype=torch.float64)
    water = torch.tensor([.004, .008, .012, .016], dtype=torch.float64)
    wind = torch.zeros_like(theta)
    native = torch.tensor([.1, .2, .8], dtype=torch.float64)
    source = source_properties(pressure, torch.full_like(pressure, 10000.), height,
                               theta, water, wind, wind, wind, 2100., native)
    assert source['index'] == 2
    # Surface (nearest-interior), interior, and top-interface pressure weights.
    assert source['tke'] == pytest.approx((.8 * 5000 + .8 * 10000 + .2 * 5000) / 20000)
    for field in [theta, water]:
        np.testing.assert_allclose(source_slope(field, pressure).numpy()[::-1],
                                   slope(field.numpy()[::-1], pressure.numpy()[::-1]))
    reconstructed = []
    for k in [2, 3]:
        for face in source['faces'][k:k + 2]:
            offset = face - pressure[k]
            reconstructed.append((theta[k] + source_slope(theta, pressure)[k] * offset) *
                                 (1 + .608 * (water[k] + source_slope(water, pressure)[k] * offset)))
    assert source['theta'] == pytest.approx(float(min(reconstructed) / (1 + .608 * water[-1])))
