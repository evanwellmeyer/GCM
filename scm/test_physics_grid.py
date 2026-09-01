import pytest
import torch

from scm.configuration import extract_param_overrides
from scm.physics_grid import conservative_remap, layer_overlap, make_physics_grid
from scm.thermo import dp_from_ps, make_grid, pressure_at_full, pressure_at_half


def test_overlap_finds_pressure_shared_by_layers():
    source = torch.tensor([0.0, 400.0, 1000.0])
    target = torch.tensor([0.0, 250.0, 700.0, 1000.0])
    overlap = layer_overlap(source, target)

    expected = torch.tensor([[
        [250.0, 0.0],
        [150.0, 300.0],
        [0.0, 300.0],
    ]])
    assert torch.equal(overlap, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_remap_preserves_pressure_integral_for_batches(dtype):
    source = torch.tensor([
        [0.0, 20000.0, 60000.0, 100000.0],
        [0.0, 18000.0, 54000.0, 90000.0],
    ], dtype=dtype)
    target = torch.tensor([
        [0.0, 10000.0, 30000.0, 70000.0, 100000.0],
        [0.0, 9000.0, 27000.0, 63000.0, 90000.0],
    ], dtype=dtype)
    values = torch.tensor([
        [1.0, 2.0, 4.0],
        [3.0, 2.0, 1.0],
    ], dtype=dtype)

    mapped = conservative_remap(values, source, target)
    source_integral = torch.sum(values * (source[:, 1:] - source[:, :-1]), dim=1)
    target_integral = torch.sum(mapped * (target[:, 1:] - target[:, :-1]), dim=1)

    assert mapped.dtype == dtype
    assert torch.allclose(target_integral, source_integral, rtol=2.0e-6, atol=2.0e-3)


def test_nested_grid_round_trip_recovers_host_layer_means():
    host = make_grid(20, dtype=torch.float64)
    ps = torch.tensor([100000.0, 93000.0], dtype=torch.float64)
    mapping = make_physics_grid(host, ps, sublevels=5, top=0.70)
    values = torch.stack([
        torch.linspace(200.0, 290.0, 20, dtype=torch.float64),
        torch.linspace(210.0, 300.0, 20, dtype=torch.float64),
    ])

    restored = mapping.to_host(mapping.to_physics(values))

    assert mapping.grid["nlevels"] > host["nlevels"]
    assert torch.equal(mapping.parent_layer.unique(), torch.arange(20))
    assert torch.allclose(restored, values, atol=1.0e-12)


def test_nested_grid_preserves_total_water_and_energy_integrals():
    host = make_grid(20, dtype=torch.float64)
    ps = torch.tensor([100000.0, 97000.0], dtype=torch.float64)
    mapping = make_physics_grid(host, ps, sublevels=4, top=0.65)
    generator = torch.Generator().manual_seed(7)
    temperature = 210.0 + 80.0 * torch.rand((2, 20), generator=generator, dtype=torch.float64)
    total_water = 0.02 * torch.rand((2, 20), generator=generator, dtype=torch.float64)
    host_depth = mapping.host_interfaces[:, 1:] - mapping.host_interfaces[:, :-1]
    physics_depth = mapping.physics_interfaces[:, 1:] - mapping.physics_interfaces[:, :-1]

    for values in (temperature, total_water):
        fine = mapping.to_physics(values)
        host_integral = torch.sum(values * host_depth, dim=1)
        physics_integral = torch.sum(fine * physics_depth, dim=1)
        assert torch.allclose(physics_integral, host_integral, atol=1.0e-8)


def test_remap_keeps_torch_gradient():
    source = torch.tensor([0.0, 300.0, 1000.0], dtype=torch.float64)
    target = torch.tensor([0.0, 100.0, 500.0, 1000.0], dtype=torch.float64)
    values = torch.tensor([[2.0, 5.0]], dtype=torch.float64, requires_grad=True)

    conservative_remap(values, source, target).square().sum().backward()

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_remap_rejects_different_column_boundaries():
    with pytest.raises(ValueError, match="same pressure column"):
        conservative_remap(
            torch.ones(1, 2),
            torch.tensor([0.0, 500.0, 1000.0]),
            torch.tensor([10.0, 500.0, 1000.0]),
        )


def test_physics_grid_configuration_is_opt_in():
    assert extract_param_overrides({}) == {}
    params = extract_param_overrides({
        "physics_grid": {
            "enabled": True,
            "sublevels": 5,
            "top": 0.65,
            "categories": ["boundary_layer", "shallow_convection"],
        }
    })
    assert params == {
        "physics_grid_enabled": True,
        "physics_grid_sublevels": 5,
        "physics_grid_top": 0.65,
        "physics_grid_categories": ["boundary_layer", "shallow_convection"],
    }


def test_host_grid_is_not_modified():
    host = make_grid(20)
    original = pressure_at_half(host, torch.tensor([100000.0])).clone()
    make_physics_grid(host, torch.tensor([100000.0]), sublevels=4, top=0.70)
    after = pressure_at_half(host, torch.tensor([100000.0]))

    assert torch.equal(after, original)


def test_state_mapping_updates_derived_pressure_fields():
    host = make_grid(20, dtype=torch.float64)
    ps = torch.tensor([100000.0, 95000.0], dtype=torch.float64)
    mapping = make_physics_grid(host, ps, sublevels=3, top=0.70)
    temperature = torch.full((2, 20), 280.0, dtype=torch.float64)
    state = {
        "t": temperature,
        "q": torch.full_like(temperature, 0.01),
        "ts": torch.tensor([285.0, 284.0], dtype=torch.float64),
        "ps": ps,
        "p": pressure_at_full(host, ps),
        "dp": dp_from_ps(host, ps),
    }

    refined = mapping.state_to_physics(state)

    assert refined["t"].shape == (2, mapping.grid["nlevels"])
    assert refined["p"].shape == refined["t"].shape
    assert refined["dp"].shape == refined["t"].shape
    assert refined["ts"] is state["ts"]


def test_scheme_dispatch_uses_physics_grid_only_when_enabled():
    from scm.column_model import initial_state, update_derived
    from scm.ensemble import default_params
    from scm.physics_suites import run_physics_scheme

    host = make_grid(20)
    params = default_params()
    state = update_derived(initial_state(1, host, params), host)
    direct = run_physics_scheme("boundary_layer", "constant", state, host, params)

    refined_params = dict(params)
    refined_params.update({
        "physics_grid_enabled": True,
        "physics_grid_sublevels": 4,
        "physics_grid_top": 0.70,
        "physics_grid_categories": ["boundary_layer"],
    })
    refined = run_physics_scheme(
        "boundary_layer",
        "constant",
        state,
        host,
        refined_params,
    )

    assert direct["dt"].shape == state["t"].shape
    assert refined["dt"].shape == state["t"].shape
    layer_mass = state["dp"] / 9.81
    assert torch.allclose(
        torch.sum(refined["dq"] * layer_mass, dim=1),
        torch.zeros(1),
        atol=2.0e-8,
    )
