import torch

from scm.boundary_layer import boundary_layer_mixing, diagnose_boundary_layer_depth

from scm.column_model import initial_state, update_derived
from scm.convection_mf import dilute_cape, mass_flux_convection
from scm.ensemble import default_params
from scm.surface import surface_fluxes
from scm.thermo import (
    Lv,
    cp,
    g,
    make_grid,
    pressure_at_full,
    saturation_specific_humidity,
)


def test_dilute_cape_converges_across_teaching_grids():
    values = []
    for nlevels in [10, 20, 40]:
        grid = make_grid(nlevels)
        pressure = pressure_at_full(grid, torch.tensor([100000.0]))
        sigma = grid['sigma_full'].unsqueeze(0)
        temperature = torch.clamp(295.0 * sigma ** 0.17, min=205.0)
        humidity = (
            0.85
            * saturation_specific_humidity(temperature, pressure)
            * sigma ** 0.7
        )
        cape = dilute_cape(
            temperature,
            humidity,
            pressure,
            entrainment=torch.tensor([5.0e-6]),
            condensate_retention=0.25,
            condensate_fallout=0.45,
        )
        values.append(cape[0])

    values = torch.stack(values)
    spread = (values.max() - values.min()) / values.mean()
    assert spread < 0.07


def test_surface_flux_distribution_conserves_flux_across_grids():
    for nlevels in [10, 20, 40]:
        grid = make_grid(nlevels)
        params = default_params()
        params.update({
            'surface_heat_sigma_depth': 0.02,
            'surface_moisture_sigma_depth': 0.005,
        })
        state = initial_state(1, grid, params)
        state = update_derived(state, grid)
        output = surface_fluxes(state, grid, params)
        layer_mass = state['dp'] / g
        heat_flux = torch.sum(cp * output['dt'] * layer_mass, dim=1)
        moisture_flux = torch.sum(Lv * output['dq'] * layer_mass, dim=1)

        assert torch.allclose(heat_flux, output['shf'], rtol=1.0e-5, atol=1.0e-5)
        assert torch.allclose(moisture_flux, output['lhf'], rtol=1.0e-5, atol=1.0e-5)


def test_diagnosed_boundary_depth_is_grid_independent():
    depths = []
    for nlevels in [10, 20, 40]:
        height = torch.linspace(2000.0, 0.0, nlevels).unsqueeze(0)
        theta = 300.0 + 0.001 * height
        depth = diagnose_boundary_layer_depth(
            height,
            theta,
            torch.tensor([25.0]),
            torch.tensor([0.25]),
            {'bl_min_depth_m': 100.0, 'bl_max_depth_m': 1500.0},
        )
        depths.append(depth[0])

    depths = torch.stack(depths)
    assert depths.max() - depths.min() < 20.0
    assert torch.all((depths > 400.0) & (depths < 480.0))


def test_boundary_layer_mse_mixing_conserves_column_energy():
    grid = make_grid(20)
    params = default_params()
    params.update({
        'dt': 900.0,
        'bl_diagnose_depth': True,
        'bl_min_depth_m': 100.0,
        'bl_max_depth_m': 900.0,
        'bl_mix_moist_static_energy': True,
    })
    state = initial_state(1, grid, params)
    state = update_derived(state, grid)
    output = boundary_layer_mixing(state, grid, params)
    layer_mass = state['dp'] / g
    energy_tendency = torch.sum(
        (cp * output['dt'] + Lv * output['dq']) * layer_mass,
        dim=1,
    )

    assert torch.allclose(energy_tendency, torch.zeros_like(energy_tendency), atol=5.0e-2)


def test_mass_flux_cape_response_converges_across_teaching_grids():
    responses = []
    limits = []
    for nlevels in [10, 20, 40]:
        grid = make_grid(nlevels)
        params = default_params()
        params.update({
            'dt': 900.0,
            'mf_closure_mode': 'cape_response',
            'mf_trial_mass_flux': 0.01,
            'mf_minimum_cape_response': 1.0,
            'mf_available_mass_fraction': 0.25,
            'mf_source_top_sigma': 0.90,
            'mf_mb_max': 100.0,
        })
        state = initial_state(1, grid, params)
        state = update_derived(state, grid)
        output = mass_flux_convection(state, grid, params)
        responses.append(output['cape_response_per_mass_flux'][0])
        limits.append(output['cloud_base_mass_flux_limit'][0])

    responses = torch.stack(responses)
    limits = torch.stack(limits)
    responsespread = (responses.max() - responses.min()) / responses.mean()
    limitspread = (limits.max() - limits.min()) / limits.mean()

    assert torch.all(responses > 0.0)
    assert responsespread < 0.15
    assert limitspread < 0.02
