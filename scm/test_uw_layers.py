import torch

from scm.uw_layers import moist_stability, layer_diffusivity
from scm.phase_partition import partition_water
from scm.thermo import cp, Lv, g, kappa, p0, make_grid
from scm.case_benchmarks import initialize_bomex
from scm.boundary_layer_uw import uw_moist_turbulence


def test_uniform_moist_conserved_variables_are_neutral():
    height = torch.tensor([[1200., 800., 400., 0.]], dtype=torch.float64)
    pressure = 100000 * torch.exp(-height / 8500)
    water = torch.full_like(height, 0.020)
    liquidenergy = torch.full_like(height, cp * 285)
    t, q, liquid, fraction = partition_water(
        water, liquidenergy + Lv * water - g * height, pressure)
    stability = moist_stability(t, q, liquid, torch.ones_like(t), pressure, height)
    assert stability.abs().max() < 1e-12


def test_surface_energy_does_not_power_detached_layer():
    height = torch.tensor([[2000., 1600., 1200., 800., 400., 0.]], dtype=torch.float64)
    pressure = 100000 * torch.exp(-height / 8500)
    theta = torch.tensor([[310., 311., 312., 302., 303., 304.]], dtype=torch.float64)
    temperature = theta * (pressure / p0) ** kappa
    zero = torch.zeros_like(height)
    thickness = torch.full_like(height, 4000)
    outputs = [layer_diffusivity(temperature, zero, zero, zero, zero, zero,
                                pressure, thickness, height, torch.tensor([flux]), {})
               for flux in (0.001, 0.01)]
    for output in outputs:
        assert output['labels'][0, 2] == -1
        assert output['labels'][0, 0] != output['labels'][0, -1]
    torch.testing.assert_close(outputs[0]['heat'][0, :2], outputs[1]['heat'][0, :2])
    torch.testing.assert_close(outputs[0]['tke'][0, :2], outputs[1]['tke'][0, :2])


def test_subcycled_layer_closure_preserves_surface_budgets():
    grid = make_grid(20, dtype=torch.float64)
    state, params = initialize_bomex(grid)
    state = {key: value.double() if torch.is_tensor(value) else value
             for key, value in state.items()}
    params.update(dt=900., uw_layer_closure=True,
                  _surface_sensible_heat_flux=12., _surface_moisture_flux=4e-5)
    output = uw_moist_turbulence(state, grid, params)
    mass = state['dp'] / g
    assert abs(((output['dq'] + output['dqc']) * mass).sum().item() - 4e-5) < 1e-10
    assert abs(((cp * output['dt'] + Lv * output['dq']) * mass).sum().item() - (12 + Lv * 4e-5)) < 0.01
