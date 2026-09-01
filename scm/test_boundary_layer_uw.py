import torch

from scm.boundary_layer_uw import galperin_functions, uw_moist_turbulence
from scm.case_benchmarks import initialize_bomex, run_dry_mixed_layer
from scm.thermo import Lv, cp, g, make_grid


def test_galperin_functions_are_neutral_and_stop_at_critical_ri():
    ri = torch.tensor([[-0.1, 0.0, 0.10, 0.19, 0.30]])
    heat, momentum = galperin_functions(ri, {})

    assert torch.all(heat[:, :3] > 0.0)
    assert torch.all(momentum[:, :3] > 0.0)
    assert torch.all(heat[:, 3:] == 0.0)
    assert torch.all(momentum[:, 3:] == 0.0)


def test_uw_moist_turbulence_conserves_surface_water_and_energy():
    grid = make_grid(40, dtype=torch.float64)
    state, params = initialize_bomex(grid)
    state = {name: value.to(torch.float64) if torch.is_tensor(value) else value for name, value in state.items()}
    sensible = torch.tensor([12.0], dtype=torch.float64)
    moisture = torch.tensor([4.0e-5], dtype=torch.float64)
    params.update({
        "dt": 300.0,
        "_surface_sensible_heat_flux": sensible,
        "_surface_moisture_flux": moisture,
    })

    output = uw_moist_turbulence(state, grid, params)
    mass = state["dp"] / g
    water = torch.sum((output["dq"] + output["dqc"]) * mass, dim=1)
    energy = torch.sum(
        (cp * output["dt"] + Lv * output["dq"]) * mass,
        dim=1,
    )

    assert torch.allclose(water, moisture, atol=2.0e-10)
    assert torch.allclose(energy, sensible + Lv * moisture, atol=0.2)
    assert torch.max(torch.abs(output["water_residual"])) < 2.0e-10
    assert torch.max(torch.abs(output["energy_residual"])) < 0.2


def test_uw_dry_convective_layer_is_bounded_across_host_grids():
    depths = []
    spreads = []
    for levels in (20, 40, 80):
        result = run_dry_mixed_layer(
            make_grid(levels),
            hours=1.0,
            timestep=60.0,
            scheme="uw",
        )
        depths.append(result["boundary_layer_depth_m"])
        spreads.append(result["mixed_layer_theta_spread_k"])
        assert abs(result["energy_error_wm2"]) < 0.5
        assert 300.0 < result["boundary_layer_depth_m"] < 2000.0

    assert max(depths) - min(depths) < 250.0
    assert max(spreads) < 1.5


def test_uw_bomex_boundary_layer_is_bounded_without_shallow_convection():
    depths = []
    paths = []
    for levels in (20, 40, 80):
        from scm.case_benchmarks import run_bomex

        result = run_bomex(
            make_grid(levels),
            hours=1.0,
            timestep=60.0,
            use_shallow=False,
            scheme="uw",
        )
        depths.append(result["boundary_layer_depth_m"])
        paths.append(result["cloud_water_path_kgm2"])
        assert 400.0 < result["boundary_layer_depth_m"] < 1500.0
        assert result["cloud_water_path_kgm2"] < 0.15

    assert max(depths) - min(depths) < 150.0
    assert max(paths) - min(paths) < 0.08


def test_registered_uw_scheme_returns_host_grid_tendencies_from_physics_grid():
    from scm.physics_suites import run_physics_scheme

    grid = make_grid(20)
    state, params = initialize_bomex(grid)
    params.update({
        "dt": 300.0,
        "physics_grid_enabled": True,
        "physics_grid_sublevels": 4,
        "physics_grid_top": 0.70,
        "physics_grid_categories": ["boundary_layer"],
        "_surface_sensible_heat_flux": torch.tensor([12.0]),
        "_surface_moisture_flux": torch.tensor([4.0e-5]),
    })

    output = run_physics_scheme(
        "boundary_layer",
        "uw_moist",
        state,
        grid,
        params,
    )

    assert output["dt"].shape == state["t"].shape
    assert output["dq"].shape == state["q"].shape
    assert output["dqc"].shape == state["qc"].shape
    assert torch.isfinite(output["dt"]).all()


def test_column_step_applies_uw_momentum_tendencies():
    from scm.column_model import initial_state, physics_step, update_derived
    from scm.ensemble import default_params

    grid = make_grid(20)
    params = default_params()
    params.update({
        "dt": 60.0,
        "use_slab_ocean": False,
        "boundary_layer_scheme": "uw_moist",
        "surface_flux_coupling": "boundary_layer",
        "physics_grid_enabled": True,
        "physics_grid_sublevels": 3,
        "physics_grid_top": 0.70,
        "physics_grid_categories": ["boundary_layer"],
        "profile_diagnostics": True,
    })
    state = update_derived(initial_state(1, grid, params), grid)
    state["u"][0] = torch.linspace(0.0, 10.0, 20)
    before = state["u"].clone()

    state, diagnostics, _ = physics_step(state, grid, params)

    assert not torch.equal(state["u"], before)
    assert "boundary_layer_zonal_momentum_tendency" in diagnostics
