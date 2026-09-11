import pytest
import torch

from scm.column_model import initial_state, update_derived
from scm.ensemble import default_params
from scm.radiation_schemes.multiband import compute_longwave_multiband
from scm.thermo import make_grid


def _column():
    grid = make_grid(20, dtype=torch.float64)
    params = default_params()
    state = update_derived(initial_state(1, grid, params), grid)
    return state, grid, params


def test_identical_gpoints_reproduce_the_grey_band():
    state, grid, params = _column()
    kappa = [0.0, 0.1, 0.22, 0.4]
    carbon = [0.0, 0.15, 0.65, 0.35]
    grey = dict(params, lw_band_weights=[0.1, 0.25, 0.35, 0.3],
                lw_band_wv_kappa=kappa, lw_band_co2_base_tau=carbon)
    split = dict(grey, lw_gpoint_fractions=[0.5, 0.3, 0.2],
                 lw_band_wv_kappa=[k for k in kappa for _ in range(3)],
                 lw_band_co2_base_tau=[c for c in carbon for _ in range(3)])
    for expected, actual in zip(compute_longwave_multiband(state, grid, grey, force_clear_sky=True),
                                compute_longwave_multiband(state, grid, split, force_clear_sky=True)):
        torch.testing.assert_close(actual, expected)


def test_gpoint_strengths_must_cover_every_band():
    state, grid, params = _column()
    bad = dict(params, lw_band_weights=[0.1, 0.25, 0.35, 0.3], lw_band_wv_kappa=[0.1] * 4,
               lw_band_co2_base_tau=[0.1] * 12, lw_gpoint_fractions=[0.5, 0.3, 0.2])
    with pytest.raises(ValueError):
        compute_longwave_multiband(state, grid, bad, force_clear_sky=True)


def test_per_band_shares_reproduce_the_grey_band():
    state, grid, params = _column()
    kappa = [0.0, 0.1, 0.22, 0.4]
    carbon = [0.0, 0.15, 0.65, 0.35]
    grey = dict(params, lw_band_weights=[0.1, 0.25, 0.35, 0.3],
                lw_band_wv_kappa=kappa, lw_band_co2_base_tau=carbon)
    split = dict(grey, lw_gpoint_fractions=[0.9, 0.1, 0.5, 0.5, 0.2, 0.8, 0.3, 0.7],
                 lw_band_wv_kappa=[k for k in kappa for _ in range(2)],
                 lw_band_co2_base_tau=[c for c in carbon for _ in range(2)])
    for expected, actual in zip(compute_longwave_multiband(state, grid, grey, force_clear_sky=True),
                                compute_longwave_multiband(state, grid, split, force_clear_sky=True)):
        torch.testing.assert_close(actual, expected)


def test_pressure_exponent_zero_changes_nothing_and_negative_adds_opacity_aloft():
    state, grid, params = _column()
    base = dict(params, lw_band_weights=[0.1, 0.25, 0.35, 0.3],
                lw_band_wv_kappa=[0.0, 0.1, 0.22, 0.4], lw_band_co2_base_tau=[0.0, 0.15, 0.65, 0.35])
    flat = compute_longwave_multiband(state, grid, base, force_clear_sky=True)
    zero = compute_longwave_multiband(state, grid, dict(base, lw_band_wv_pressure_exponent=[0.0] * 4),
                                      force_clear_sky=True)
    for expected, actual in zip(flat, zero):
        torch.testing.assert_close(actual, expected)
    aloft = compute_longwave_multiband(state, grid, dict(base, lw_band_wv_pressure_exponent=[-1.0] * 4),
                                       force_clear_sky=True)
    # Stronger absorption at low pressure moves emission to space higher and colder.
    assert aloft[2].item() < flat[2].item()
