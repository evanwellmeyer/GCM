import pytest
import torch

from scm.convective_transport import transport, updraft, downdraft
from scm.convection_mf import mass_flux_convection
from scm.column_model import initial_state, update_derived
from scm.ensemble import default_params
from scm.thermo import cp, Lv, g, make_grid


def test_uniform_scalars_do_not_change_with_divergent_mass_flux():
    energy = torch.full((2, 4), 300000., dtype=torch.float64)
    water = torch.full_like(energy, 0.001)
    mass = torch.tensor([[50., 200., 400., 100.]], dtype=torch.float64).expand(2, -1)
    flow = torch.tensor([[0., 0.1, 0.3, 0.2, 0.]], dtype=torch.float64).expand(2, -1)
    heating, moistening, _, _ = transport(
        energy, water, mass, flow, torch.full_like(flow, 300000.),
        torch.full_like(flow, 0.001), torch.zeros_like(water))
    assert heating.abs().max() == 0
    assert moistening.abs().max() == 0


def test_each_interface_has_equal_opposite_heat_and_water_exchanges():
    energy = torch.tensor([[300000., 290000.]], dtype=torch.float64)
    water = torch.tensor([[0.001, 0.01]], dtype=torch.float64)
    mass = torch.tensor([[70., 350.]], dtype=torch.float64)
    flow = torch.tensor([[0., 0.02, 0.]], dtype=torch.float64)
    plumeenergy = torch.tensor([[0., 290000., 0.]], dtype=torch.float64)
    plumewater = torch.tensor([[0., 0.01, 0.]], dtype=torch.float64)
    heating, moistening, heatflux, waterflux = transport(
        energy, water, mass, flow, plumeenergy, plumewater, torch.zeros_like(water))
    assert heating[0, 1] > 0 and moistening[0, 1] < 0
    torch.testing.assert_close((cp * heating * mass).sum(), torch.tensor(0., dtype=torch.float64))
    torch.testing.assert_close((moistening * mass).sum(), torch.tensor(0., dtype=torch.float64))
    torch.testing.assert_close(cp * heating * mass, heatflux[:, 1:] - heatflux[:, :-1])
    torch.testing.assert_close(moistening * mass, waterflux[:, 1:] - waterflux[:, :-1])


@pytest.mark.parametrize('levels', [10, 20, 40])
def test_flux_divergence_integrates_over_unequal_layers(levels):
    edges = torch.linspace(0, 1, levels + 1, dtype=torch.float64).square()[None]
    mass = 10000 * (edges[:, 1:] - edges[:, :-1])
    energy = torch.full_like(mass, 300000.)
    water = torch.full_like(mass, 0.001)
    flow = torch.sin(torch.pi * edges)
    plumeenergy = torch.full_like(flow, 301000.)
    plumewater = torch.full_like(flow, 0.002)
    rain = mass * 1e-8
    heating, moistening, _, _ = transport(energy, water, mass, flow, plumeenergy, plumewater, rain)
    assert abs(((cp * heating + Lv * moistening) * mass).sum()) < 1e-9
    assert abs((moistening * mass + rain).sum()) < 1e-15


def test_updraft_closes_mass_water_and_energy_before_any_correction():
    height = torch.tensor([[5000., 3000., 1000., 0.]], dtype=torch.float64)
    pressure = 100000 * torch.exp(-height / 8000)
    temperature = 300 - 0.007 * height
    water = torch.tensor([[0.001, 0.004, 0.012, 0.018]], dtype=torch.float64)
    thickness = torch.tensor([[20000., 20000., 15000., 5000.]], dtype=torch.float64)
    result = updraft(temperature, water, height, pressure, thickness,
                     5e-6, 3e-5, 1.5e-4, 1.)
    mass = thickness / g
    assert result['rain'].sum() > 0
    assert abs(((cp * result['dt'] + Lv * result['dq']) * mass).sum()) < 1e-9
    assert abs((result['dq'] * mass + result['rain']).sum()) < 1e-15
    torch.testing.assert_close(result['exchange'], result['massflux'][:, 1:] - result['massflux'][:, :-1])
    assert result['exchange'].sum().abs() < 1e-15


def test_dry_neutral_plume_does_not_need_energy_repair():
    height = torch.tensor([[3000., 1500., 500., 0.]], dtype=torch.float64)
    temperature = 300 - g * height / cp
    pressure = 100000 * (temperature / 300) ** (cp / 287.05)
    water = torch.zeros_like(height)
    result = updraft(temperature, water, height, pressure, torch.full_like(height, 10000),
                     5e-6, 3e-5, 1.5e-4, 1.)
    assert result['dt'].abs().max() < 1e-14
    assert result['dq'].abs().max() == 0


def test_candidate_is_used_by_convection_and_exposes_uncorrected_budget():
    grid = make_grid(20, dtype=torch.float64)
    params = default_params()
    params.update(mf_transport_form='flux', mf_bl_export_fraction=0.,
                  mf_downdraft_fraction=0., mf_enforce_mse_conservation=False,
                  mf_max_dt_day=1e8, mf_max_dq_day=1e8)
    state = update_derived(initial_state(1, grid, params), grid)
    state = {key: value.double() if torch.is_tensor(value) else value for key, value in state.items()}
    result = mass_flux_convection(state, grid, params)
    assert result['transport_mse_residual_per_mass_flux'].abs().max() < 1e-8
    assert result['raw_mse_residual_per_mass_flux'].abs().max() < 1e-8
    assert torch.isfinite(result['dt']).all()
    assert result['mse_residual'].abs().max() < 1e-8


@pytest.mark.parametrize('supply', [0., 1e-5, .01])
def test_downdraft_accounts_for_every_evaporated_drop(supply):
    height = torch.tensor([[5000., 3000., 1000., 0.]], dtype=torch.float64)
    pressure = 100000 * torch.exp(-height / 8000)
    temperature = 300 - .006 * height
    water = torch.tensor([[.001, .002, .004, .006]], dtype=torch.float64)
    thickness = torch.tensor([[20000., 20000., 15000., 5000.]], dtype=torch.float64)
    rain = torch.zeros_like(water)
    rain[:, 0] = supply
    result = downdraft(temperature, water, height, pressure, thickness, pressure / 100000,
                       rain, torch.tensor([.4], dtype=torch.float64), {})
    mass = thickness / g
    evaporation = result['evaporation']
    assert evaporation.sum() <= .5 * rain.sum() + 1e-15
    assert torch.all(evaporation.cumsum(1) <= .5 * rain.cumsum(1) + 1e-15)
    torch.testing.assert_close(result['dq'] * mass,
                              result['waterflux'][:, 1:] - result['waterflux'][:, :-1] + evaporation)
    torch.testing.assert_close(cp * result['dt'] * mass,
                              result['heatflux'][:, 1:] - result['heatflux'][:, :-1] - Lv * evaporation)
    assert abs((result['dq'] * mass).sum() - evaporation.sum()) < 1e-14
    assert abs(((cp * result['dt'] + Lv * result['dq']) * mass).sum()) < 1e-9
    if supply == 0:
        assert result['dt'].abs().max() == 0
        assert result['dq'].abs().max() == 0


def test_complete_candidate_conserves_without_repair_even_when_limited():
    import scm.convection_mf as convection

    # The global energy correction left with the legacy transport (10 Sep 2026).
    assert not hasattr(convection, '_conserve_mse')
    grid = make_grid(20, dtype=torch.float64)
    params = default_params()
    params.update(mf_transport_form='flux', mf_downdraft_fraction=.4,
                  mf_max_dt_day=.01, mf_max_dq_day=.01, mf_closure_mode='heating_proxy')
    state = update_derived(initial_state(1, grid, params), grid)
    state = {key: value.double() if torch.is_tensor(value) else value for key, value in state.items()}
    result = mass_flux_convection(state, grid, params)
    assert result['cloud_base_mass_flux'].item() > 0
    assert result['transport_limiter'].item() < 1
    assert result['raw_mse_residual_per_mass_flux'].abs().max() < 1e-8
    assert result['export_mse_residual_per_mass_flux'].abs().max() == 0
    mass = state['dp'] / g
    assert abs(((cp * result['dt'] + Lv * result['dq']) * mass).sum()) < 1e-9
    assert abs((result['dq'] * mass).sum() + result['precip'].sum()) < 1e-14
    torch.testing.assert_close(result['precip'], result['rain_production'] - result['rain_evaporation'])


def test_vapor_floor_does_not_turn_off_an_active_plume():
    grid = make_grid(20)
    params = default_params()
    params.update(mf_transport_form='flux', mf_downdraft_fraction=.4,
                  mf_max_dt_day=1e8, mf_max_dq_day=1e8)
    state = update_derived(initial_state(1, grid, params), grid)
    state['q'][:, :5] = 1e-7
    result = mass_flux_convection(state, grid, params)
    assert result['cloud_base_mass_flux'].item() > 0
    assert result['transport_limiter'].item() > .99
    assert result['precip'].item() > 0


def test_plume_stops_at_neutral_buoyancy_when_asked():
    height = torch.tensor([[16000., 13000., 10000., 7000., 4000., 1500., 0.]], dtype=torch.float64)
    temperature = torch.where(height > 12000, 216.5 + 0.002 * (height - 12000), 300 - 0.007 * height)
    pressure = 100000 * torch.exp(-height / 7500)
    water = 0.018 * torch.exp(-height / 2500)
    thickness = torch.tensor([[8000., 10000., 12000., 15000., 20000., 20000., 15000.]], dtype=torch.float64)
    args = (temperature, water, height, pressure, thickness, 5e-6, 3e-5, 1.5e-4, 1.)
    decaying = updraft(*args)
    stopped = updraft(*args, stop_at_neutral_buoyancy=True)
    # By default the plume only decays past neutral buoyancy, so it reaches the top level.
    assert decaying['massflux'][0, 1] > 0 and decaying['dt'][0, 0] < 0
    # Stopped, nothing crosses into the top level, and the plume below is unchanged.
    assert stopped['massflux'][0, :2].abs().max() == 0
    assert stopped['dt'][0, 0] == 0 and stopped['dq'][0, 0] == 0
    torch.testing.assert_close(stopped['massflux'][:, 2:], decaying['massflux'][:, 2:])
    mass = thickness / g
    assert abs(((cp * stopped['dt'] + Lv * stopped['dq']) * mass).sum()) < 1e-9
    assert abs((stopped['dq'] * mass + stopped['rain']).sum()) < 1e-15


def test_mass_flux_scheme_passes_plume_top_switch():
    grid = make_grid(20)
    top = {}
    for stop in (False, True):
        params = default_params()
        params.update(mf_transport_form='flux', mf_plume_stop_at_neutral_buoyancy=stop)
        state = update_derived(initial_state(1, grid, params), grid)
        top[stop] = mass_flux_convection(state, grid, params)['dt'][0, 0].item()
    assert top[False] < 0 and top[True] == 0


def test_legacy_transport_is_rejected():
    grid = make_grid(20)
    params = default_params()
    params.update(mf_transport_form='legacy')
    state = update_derived(initial_state(1, grid, params), grid)
    with pytest.raises(ValueError):
        mass_flux_convection(state, grid, params)
