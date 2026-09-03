import torch

from scm.thermo import make_smooth_test_grid


def test_smooth_grid_preserves_column_and_limits_layer_thickness():
    grid = make_smooth_test_grid(dtype=torch.float64)
    thickness = grid['dsigma'] * 1000.0

    assert grid['nlevels'] == 20
    assert torch.isclose(thickness.sum(), torch.tensor(1000.0, dtype=thickness.dtype))
    assert thickness.min() >= 20.0 - 1.0e-10
    assert thickness.max() <= 70.0 + 1.0e-10
    assert torch.isclose(thickness[-1], torch.tensor(20.0, dtype=thickness.dtype))
