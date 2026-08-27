import torch

from scm.column_model import initial_state, update_derived
from scm.convection_mf import dilute_cape
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
