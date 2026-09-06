import pytest
import torch

from scm.column_model import initial_state, update_derived, physics_step
from scm.convection_bm import conservative_adjustment
from scm.ensemble import default_params
from scm.thermo import cp, Lv, g, make_grid, saturation_specific_humidity


@pytest.mark.parametrize('levels', [10, 20, 40])
@pytest.mark.parametrize('limit', [0.1, 10.0])
def test_adjustment_preserves_enthalpy_and_precipitation_water(levels, limit):
    grid = make_grid(levels, dtype=torch.float64)
    params = default_params()
    params.update(dt=900.0, bm_conserve_enthalpy=True, bm_max_dt_day=limit)
    state = initial_state(1, grid, params)
    state = {key: value.double() if torch.is_tensor(value) else value for key, value in state.items()}
    update_derived(state, grid)
    state['t'] = (300 * (state['p'] / state['ps'].reshape(-1, 1)) ** 0.22).clamp(min=200)
    state['q'] = 1.5 * saturation_specific_humidity(state['t'], state['p'])
    output = conservative_adjustment(state, grid, params)
    mass = state['dp'] / g
    assert output['precip'].item() > 0
    energy = ((cp * output['dt'] + Lv * output['dq']) * mass).sum()
    assert abs(energy.item()) < 1e-7
    water = (output['dq'] * mass).sum() + output['precip'].sum()
    assert abs(water.item()) < 1e-12
    assert output['cloud_condensate'].item() == 0
    assert torch.all(state['q'] + 900 * output['dq'] >= 0)
    assert output['dt'].abs().max() <= limit / 86400 * 1.00001


def test_adjustment_cloud_handoff_closes_column_water():
    grid = make_grid(20)
    params = default_params()
    params.update(convection_scheme='betts_miller', bm_conserve_enthalpy=True,
                  cloud_microphysics_enabled=True, condensation_rh_crit=0.95,
                  dt=900.0)
    state = initial_state(1, grid, params)
    update_derived(state, grid)
    state['t'] = (300 * (state['p'] / state['ps'].reshape(-1, 1)) ** 0.22).clamp(min=200)
    state['q'] = 1.5 * saturation_specific_humidity(state['t'], state['p'])
    _, diagnostic, _ = physics_step(state, grid, params)
    assert diagnostic['precip_conv'].item() > 0
    assert diagnostic['conv_cloud_source_kgm2s'].item() == 0
    assert diagnostic['column_water_residual'].abs().max() < 1e-8


def test_shallow_branch_redistributes_water_without_rain():
    grid = make_grid(20, dtype=torch.float64)
    params = default_params()
    params.update(dt=900.0, bm_conserve_enthalpy=True)
    state = initial_state(1, grid, params)
    state = {key: value.double() if torch.is_tensor(value) else value for key, value in state.items()}
    update_derived(state, grid)
    state['t'] = (300 * (state['p'] / state['ps'].reshape(-1, 1)) ** 0.22).clamp(min=200)
    state['q'] = 0.5 * saturation_specific_humidity(state['t'], state['p'])
    output = conservative_adjustment(state, grid, params)
    mass = state['dp'] / g
    assert output['precip'].item() == 0
    assert output['dq'].abs().max().item() > 0
    assert abs((output['dq'] * mass).sum().item()) < 1e-12
    assert abs(((cp * output['dt'] + Lv * output['dq']) * mass).sum().item()) < 1e-7
