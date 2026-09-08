import torch

from scm.radiation_schemes.common import forward_flux_sweep


def test_flux_sweep_matches_layer_recurrence():
    transmissivity = torch.tensor([[.8, .6, .3], [.9, .7, .5]], dtype=torch.float64)
    source = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
    boundary = torch.tensor([10., 20.], dtype=torch.float64)
    result = forward_flux_sweep(transmissivity, source, boundary)
    for level in range(3):
        torch.testing.assert_close(
            result[:, level + 1],
            result[:, level] * transmissivity[:, level] + source[:, level],
        )


def test_flux_sweep_retains_sources_below_an_opaque_layer():
    transmissivity = torch.tensor([[1e-20, .5, .5]], dtype=torch.float64)
    source = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    result = forward_flux_sweep(transmissivity, source, torch.tensor([100.], dtype=torch.float64))
    expected = torch.tensor([[100., 1., 2.5, 4.25]], dtype=torch.float64)
    torch.testing.assert_close(result, expected)


def test_flux_sweep_has_finite_gradients_when_optically_thick():
    transmissivity = torch.full((1, 20), 1e-4, dtype=torch.float64, requires_grad=True)
    source = torch.linspace(1., 2., 20, dtype=torch.float64).unsqueeze(0)
    result = forward_flux_sweep(transmissivity, source, torch.tensor([300.], dtype=torch.float64))
    result.sum().backward()
    assert torch.isfinite(result).all()
    assert torch.isfinite(transmissivity.grad).all()
