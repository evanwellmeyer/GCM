"""Analytic checks for the diagnostic Gaussian source closure."""

import math

import pytest

from scripts.test_bomex_launch_closure import joint_launch
from scm.thermo import g


def test_unrestricted_zero_cin_matches_half_normal_moments():
    mass, velocity, area = joint_launch(1.2, .25, 0., 10000., 900., .5)
    sigma = math.sqrt(.2505)
    assert area == pytest.approx(.5)
    assert mass == pytest.approx(1.2 * sigma / math.sqrt(2 * math.pi))
    assert velocity == pytest.approx(sigma * math.sqrt(2 / math.pi))


@pytest.mark.parametrize('cin', [0., .1, .5])
def test_joint_area_and_mass_velocity_identity(cin):
    mass, velocity, area = joint_launch(1.1, .3, cin, 4000., 900., .1)
    assert 0 < area <= .1 + 1e-12
    assert mass == pytest.approx(1.1 * area * velocity)


def test_thin_layer_mass_constraint_preserves_joint_identity():
    mass, velocity, area = joint_launch(1.1, .3, 0., 20., 900., .1)
    assert mass <= .9 * 20 / (g * 900) * (1 + 1e-12)
    assert mass == pytest.approx(1.1 * area * velocity)


def test_strong_cin_prevents_launch():
    assert joint_launch(1.1, .1, 10., 4000., 900., .1) == (0., 0., 0.)
