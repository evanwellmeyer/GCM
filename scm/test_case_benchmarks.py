import torch

from scm.case_benchmarks import run_bomex, run_dry_mixed_layer
from scm.case_benchmarks import initialize_bomex
from scm.shallow_plume_v2 import partition_plume, shallow_plume
from scm.boundary_layer_edmf_v3 import edmf_boundary_layer
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
