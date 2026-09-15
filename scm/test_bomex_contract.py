"""Published BOMEX inputs, distinct from model-performance acceptance tests."""

import torch

from scm.case_benchmarks import (
    bomex_forcing, bomex_momentum_forcing, initialize_bomex, linear_profile,
    model_height, vertical_gradient,
)
from scm.thermo import Lv as latent, Rd as gas, cp, g, kappa, make_grid, p0
from scripts.bomex_observed_steady_state import observed_surface_fluxes


def test_bomex_uses_published_specific_water_and_liquid_temperature():
    grid = make_grid(20)
    state, _ = initialize_bomex(grid)
    height = model_height(state, grid)
    checked = height <= 3000.
    water = linear_profile(height, [0., 520., 1480., 2000., 3000.], [.017, .0163, .0107, .0042, .003])
    theta = linear_profile(height, [0., 520., 1480., 2000., 3000.], [298.7, 298.7, 302.4, 308.2, 311.85])
    actual = (state['t'] - latent * state['qc'] / cp) * (p0 / state['p']) ** kappa
    assert torch.max(torch.abs((state['q'] + state['qc'] - water)[checked])) < 2e-7
    assert torch.max(torch.abs((actual - theta)[checked])) < 2e-3
    assert height[0, -1] > 0.
    assert state['q'][0, -1] < .017


def test_bomex_forcing_advects_liquid_temperature_and_total_water():
    grid = make_grid(20)
    state, _ = initialize_bomex(grid)
    state = {name: value.double() if torch.is_tensor(value) else value for name, value in state.items()}
    state['qc'][0, -5:-2] += 1e-4
    state['q'][0, -5:-2] -= 1e-4
    state['t'][0, -5:-2] += latent * 1e-4 / cp
    height = model_height(state, grid)
    theta = (state['t'] - latent * state['qc'] / cp) * (p0 / state['p']) ** kappa
    water = state['q'] + state['qc']
    sinking = linear_profile(height, [0., 1500., 2100., 3500.], [0., -.0065, 0., 0.])
    cooling = linear_profile(height, [0., 1500., 3000., 3500.], [-2/86400, -2/86400, 0., 0.])
    drying = linear_profile(height, [0., 300., 500., 3500.], [-1.2e-8, -1.2e-8, 0., 0.])
    heating, moistening = bomex_forcing(state, grid)
    assert torch.allclose(heating, cooling - sinking * vertical_gradient(theta, height), atol=1e-12)
    assert torch.allclose(moistening, drying - sinking * vertical_gradient(water, height), atol=1e-12)


def test_bomex_prescribed_momentum_stress_has_correct_column_integral():
    grid = make_grid(20)
    state, _ = initialize_bomex(grid)
    height = model_height(state, grid)
    sinking = linear_profile(height, [0., 1500., 2100., 3500.], [0., -.0065, 0., 0.])
    zonal, meridional = bomex_momentum_forcing(state, grid)
    stress = zonal + sinking * vertical_gradient(state['u'], height) - .376e-4 * state['v']
    density = state['p'][0, -1] / (gas * state['t'][0, -1])
    assert torch.allclose((stress * state['dp'] / g).sum(), density * .28**2, atol=1e-6)
    assert torch.isfinite(meridional).all()


def test_bomex_surface_heat_flux_includes_surface_exner():
    grid = make_grid(20)
    state, params = initialize_bomex(grid)
    output = observed_surface_fluxes(state, grid, params)
    density = state['p'][:, -1] / (gas * state['t'][:, -1])
    exner = (state['ps'] / p0) ** kappa
    assert torch.allclose(output['shf'] / (density * cp * exner), torch.full_like(output['shf'], .008), atol=1e-8)
    assert torch.allclose(output['lhf'] / (density * latent), torch.full_like(output['lhf'], 5.2e-5), atol=1e-10)
