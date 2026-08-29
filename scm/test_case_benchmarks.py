import torch

from scm.case_benchmarks import run_bomex, run_dry_mixed_layer
from scm.case_benchmarks import initialize_bomex
from scm.shallow_plume_v2 import partition_plume, shallow_plume
from scm.boundary_layer_edmf_v3 import edmf_boundary_layer
from scm.boundary_layer_tke_v2 import (
    pressure_diffusion_coefficients,
    solve_scalar,
    tke_boundary_layer,
    tke_boundary_layer_depth,
)
from scm.thermo import make_grid


def test_dry_mixed_layer_conserves_surface_energy():
    result = run_dry_mixed_layer(make_grid(40), hours=1.0)
    assert abs(result['energy_error_wm2']) < 0.2
    assert result['surface_theta_change_k'] > 0.0


def test_bomex_case_has_resolved_lower_atmosphere():
    coarse = run_bomex(make_grid(20), hours=0.25)
    fine = run_bomex(make_grid(80), hours=0.25)
    assert fine['levels_below_2000m'] > coarse['levels_below_2000m']
    assert fine['levels_below_2000m'] >= 10


def test_tke_closure_improves_dry_mixed_layer_resolution_response():
    richardson_spreads = []
    tke_spreads = []
    for levels in [20, 40, 80]:
        grid = make_grid(levels)
        richardson = run_dry_mixed_layer(grid, hours=1.0)
        tke = run_dry_mixed_layer(grid, hours=1.0, scheme='tke')
        richardson_spreads.append(richardson['mixed_layer_theta_spread_k'])
        tke_spreads.append(tke['mixed_layer_theta_spread_k'])
        assert abs(tke['energy_error_wm2']) < 0.5

    assert max(tke_spreads) < 0.5 * max(richardson_spreads)
    assert max(tke_spreads) - min(tke_spreads) < 0.25


def test_shallow_plume_conserves_water_and_energy():
    grid = make_grid(40)
    state, params = initialize_bomex(grid)
    params['dt'] = 60.0
    output = shallow_plume(state, grid, params)
    assert torch.max(torch.abs(output['water_residual'])) < 2.0e-8
    assert torch.max(torch.abs(output['energy_residual'])) < 0.1


def test_tke_plume_bomex_is_bounded_and_mass_flux_convergent():
    depths = []
    mass_fluxes = []
    for levels in [20, 40, 80]:
        result = run_bomex(
            make_grid(levels), hours=1.0, scheme='tke', shallow_scheme='plume'
        )
        depths.append(result['boundary_layer_depth_m'])
        mass_fluxes.append(result['shallow_mass_flux_kgm2s'])
        assert result['cloud_layer_max_rh'] <= 1.001
        assert 0.0 <= result['maximum_cloud_fraction'] <= 0.301
        assert 300.0 <= result['boundary_layer_depth_m'] <= 1600.0
    assert max(mass_fluxes) - min(mass_fluxes) < 0.015


def test_unified_edmf_conserves_surface_water_and_energy():
    grid = make_grid(40)
    state, params = initialize_bomex(grid)
    params.update({
        'dt': 60.0,
        '_surface_sensible_heat_flux': torch.tensor([10.0]),
        '_surface_moisture_flux': torch.tensor([5.0e-5]),
    })
    output = edmf_boundary_layer(state, grid, params)
    assert torch.max(torch.abs(output['water_residual'])) < 1.0e-7
    assert torch.max(torch.abs(output['energy_residual'])) < 0.1


def test_unified_edmf_bomex_depth_is_resolution_convergent():
    depths = []
    for levels in [20, 40, 80]:
        result = run_bomex(
            make_grid(levels), hours=1.0, use_shallow=False, scheme='edmf'
        )
        depths.append(result['boundary_layer_depth_m'])
        assert result['cloud_layer_max_rh'] <= 1.001
        assert result['cloud_water_path_kgm2'] < 0.1
    assert max(depths) - min(depths) < 75.0


def test_plume_saturation_solver_produces_liquid_water():
    temperature, vapor, liquid = partition_plume(
        torch.tensor(300.0), torch.tensor(0.03), torch.tensor(90000.0)
    )
    assert 295.0 < float(temperature) < 305.0
    assert float(vapor) < 0.03
    assert float(liquid) > 0.0


def test_edmf_detrainment_hands_cloud_water_to_the_grid():
    paths = []
    for levels in [20, 40, 80]:
        result = run_bomex(
            make_grid(levels),
            hours=1.0,
            use_shallow=False,
            scheme='edmf',
            parameter_updates={'shallow_plume_grid_saturation_adjustment': False},
        )
        paths.append(result['cloud_water_path_kgm2'])
        assert result['cloud_layer_max_rh'] < 1.0
        assert 0.0 < result['cloud_water_path_kgm2'] < 0.6
    assert max(paths) - min(paths) < 0.1


def test_distributed_detrainment_reduces_mass_flux_near_plume_top():
    grid = make_grid(40)
    state, params = initialize_bomex(grid)
    params.update({
        'dt': 60.0,
        '_surface_sensible_heat_flux': torch.tensor([10.0]),
        '_surface_moisture_flux': torch.tensor([5.0e-5]),
        'shallow_plume_grid_saturation_adjustment': False,
        'shallow_plume_detrainment_depth_m': 500.0,
        'shallow_plume_detrainment_strength': 8.0,
        'shallow_plume_buoyancy_detrainment_constant': 1.0,
    })
    distributed = edmf_boundary_layer(state, grid, params)
    baseline_params = dict(params)
    baseline_params['shallow_plume_detrainment_strength'] = 0.0
    baseline_params['shallow_plume_buoyancy_detrainment_constant'] = 0.0
    baseline = edmf_boundary_layer(state, grid, baseline_params)
    distributed_active = distributed['plume_mass_flux_profile'][0]
    distributed_active = distributed_active[distributed_active > 0.0]
    baseline_active = baseline['plume_mass_flux_profile'][0]
    baseline_active = baseline_active[baseline_active > 0.0]
    assert distributed_active.numel() >= 2
    assert distributed_active[0] < baseline_active[0]


def test_semi_implicit_tke_is_bounded_at_long_physics_timestep():
    grid = make_grid(40)
    state, params = initialize_bomex(grid)
    params.update({
        'dt': 900.0,
        '_surface_sensible_heat_flux': torch.tensor([10.0]),
        '_surface_moisture_flux': torch.tensor([5.0e-5]),
        'tke_time_integration': 'semi_implicit',
        'tke_stability_factor_max': 1.5,
    })
    output = tke_boundary_layer(state, grid, params)
    assert torch.max(output['tke']) < 1.0
    assert torch.max(output['diffusivity']) < 100.0


def test_tke_vertical_transport_is_column_conservative():
    grid = make_grid(20)
    state, _ = initialize_bomex(grid)
    diffusivity = torch.full((1, 19), 20.0)
    coefficients = pressure_diffusion_coefficients(
        diffusivity, state['t'], state['p'], state['dp'], 900.0
    )
    original = torch.linspace(0.02, 0.20, 20).unsqueeze(0)
    transported = solve_scalar(original, original, coefficients)
    mass = state['dp'] / 9.80665
    before = torch.sum(original * mass, dim=1)
    after = torch.sum(transported * mass, dim=1)
    assert torch.allclose(after, before, rtol=1.0e-6, atol=1.0e-5)


def test_tke_vertical_transport_reaches_surface_layer():
    grid = make_grid(20)
    state, params = initialize_bomex(grid)
    state['tke'] = torch.full_like(state['t'], 1.0e-4)
    state['tke'][:, -2] = 0.20
    params.update({
        'dt': 900.0,
        'tke_time_integration': 'semi_implicit',
        'tke_dissipation_constant': 0.0,
        'tke_vertical_transport': True,
    })
    transported = tke_boundary_layer(state, grid, params)
    params['tke_vertical_transport'] = False
    local = tke_boundary_layer(state, grid, params)
    assert transported['tke'][0, -1] > local['tke'][0, -1]


def test_tke_depth_diagnostic_uses_turbulent_layer_top():
    height = torch.tensor([[2500.0, 1800.0, 1200.0, 600.0, 100.0]])
    tke = torch.tensor([[1.0e-4, 5.0e-3, 2.0e-2, 0.1, 0.2]])
    depth = tke_boundary_layer_depth(
        height,
        tke,
        {
            'tke_boundary_layer_threshold_m2s2': 0.01,
            'bl_min_depth_m': 100.0,
            'tke_boundary_layer_max_m': 4000.0,
        },
    )
    assert torch.allclose(depth, torch.tensor([1600.0]))


def test_tke_depth_diagnostic_converges_across_vertical_grids():
    depths = []
    for levels in (10, 20, 40):
        height = torch.linspace(4000.0, 0.0, levels).unsqueeze(0)
        tke = 0.0275 - 5.0e-6 * height
        depth = tke_boundary_layer_depth(
            height,
            tke,
            {
                'tke_boundary_layer_threshold_m2s2': 0.02,
                'bl_min_depth_m': 100.0,
                'tke_boundary_layer_max_m': 4000.0,
            },
        )
        depths.append(depth[0])

    depths = torch.stack(depths)
    assert torch.max(torch.abs(depths - 1500.0)) < 1.0


def test_tke_depth_ignores_disconnected_turbulence_aloft():
    height = torch.tensor([[3000.0, 2000.0, 1000.0, 200.0]])
    tke = torch.tensor([[0.20, 0.001, 0.001, 0.10]])
    depth = tke_boundary_layer_depth(
        height,
        tke,
        {
            'tke_boundary_layer_threshold_m2s2': 0.01,
            'bl_min_depth_m': 100.0,
            'tke_boundary_layer_max_m': 4000.0,
        },
    )
    assert 200.0 < depth.item() < 1000.0
