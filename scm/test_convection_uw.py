import torch

from scm.case_benchmarks import initialize_bomex, run_bomex
from scm.convection_uw import (
    cloud_base_mass_flux,
    conservative_positivity_factor,
    implicit_cin_factor,
    lateral_mixing_rate,
    partition_layer_mean,
    uw_shallow_convection,
)
from scm.thermo import Lv, cp, g, geopotential, make_grid


def test_uw_lateral_mixing_weakens_with_height():
    height = torch.tensor([500.0, 1000.0, 2000.0])
    density = torch.ones_like(height)
    mixing = lateral_mixing_rate(height, density)

    assert torch.all(mixing[:-1] > mixing[1:])
    assert torch.allclose(mixing[0], 2.0 * mixing[1])


def test_uw_implicit_cin_factor_satisfies_closure_equation():
    change = torch.tensor([0.0, 10.0, 100.0], dtype=torch.float64)
    tke = torch.tensor([1.0, 50.0, 50.0], dtype=torch.float64)
    factor = implicit_cin_factor(change, tke, iterations=30)
    expected = torch.exp(-factor * change / tke)

    assert factor[0] == 1.0
    assert torch.allclose(factor, expected, atol=2.0e-9)
    assert factor[1] > factor[2]


def test_uw_mass_flux_responds_to_cin_and_implicit_stabilization():
    density = torch.tensor([1.1, 1.1, 1.1])
    tke = torch.tensor([1.0, 1.0, 1.0])
    cin = torch.tensor([0.0, 1.0, 5.0])
    explicit = cloud_base_mass_flux(density, tke, cin)
    implicit = cloud_base_mass_flux(
        density,
        tke,
        cin,
        cin_change=torch.full_like(cin, 2.0),
    )

    assert torch.all(explicit[:-1] > explicit[1:])
    assert torch.all(implicit < explicit)
    assert torch.all(explicit >= 0.0)


def test_uw_positivity_factor_scales_the_whole_column():
    water = torch.tensor([[2.0e-3, 1.0e-4]])
    tendency = torch.tensor([[1.0e-6, -2.0e-6]])
    factor = conservative_positivity_factor(water, tendency, 100.0)
    updated = water + factor.unsqueeze(1) * tendency * 100.0

    assert 0.0 < factor[0] < 1.0
    assert torch.all(updated >= 0.99e-8)


def test_uw_closures_preserve_torch_gradients():
    tke = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    mass_flux = cloud_base_mass_flux(
        torch.tensor([1.1], dtype=torch.float64),
        tke,
        torch.tensor([0.5], dtype=torch.float64),
        cin_change=torch.tensor([1.0], dtype=torch.float64),
    )
    mass_flux.sum().backward()

    assert tke.grad is not None
    assert torch.isfinite(tke.grad).all()


def test_uw_shallow_step_conserves_water_and_energy():
    for levels in (20, 40):
        grid = make_grid(levels)
        state, params = initialize_bomex(grid)
        params["dt"] = 900.0
        output = uw_shallow_convection(state, grid, params)

        assert torch.max(torch.abs(output["water_residual"])) < 2.0e-8
        assert torch.max(torch.abs(output["energy_residual"])) < 0.1
        assert torch.all(output["precip"] >= 0.0)
        assert torch.all((output["cloud_fraction"] >= 0.0) & (output["cloud_fraction"] <= 1.0))


def test_layer_mean_partition_preserves_water_and_mse():
    grid = make_grid(20)
    state, _ = initialize_bomex(grid)
    water = state['q'] + state['qc']
    height = geopotential(state['t'], state['q'], state['p'], grid)
    mse = cp * state['t'] + Lv * state['q'] + g * height

    temperature, vapor, liquid = partition_layer_mean(
        water,
        mse,
        height,
        state['p'],
        state['dp'],
    )

    assert torch.max(torch.abs(vapor + liquid - water)) < 2.0e-8
    reconstructed = cp * temperature + Lv * vapor + g * height
    assert torch.max(torch.abs(reconstructed - mse)) < 0.2


def test_uw_bomex_long_timestep_is_bounded_at_development_resolutions():
    results = []
    for levels in (20, 40):
        result = run_bomex(
            make_grid(levels),
            hours=6.0,
            timestep=900.0,
            use_shallow=True,
            scheme="uw",
            shallow_scheme="uw",
        )
        results.append(result)
        assert result["cloud_water_path_kgm2"] < 0.10
        assert result["maximum_cloud_fraction"] < 0.15
        assert 500.0 < result["boundary_layer_depth_m"] < 1200.0

    assert abs(
        results[0]["cloud_water_path_kgm2"]
        - results[1]["cloud_water_path_kgm2"]
    ) < 0.02
    assert abs(
        results[0]["shallow_mass_flux_kgm2s"]
        - results[1]["shallow_mass_flux_kgm2s"]
    ) < 0.02
