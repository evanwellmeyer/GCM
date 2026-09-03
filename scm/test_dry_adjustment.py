import torch

from scm.column_model import initial_state, physics_step, update_derived
from scm.dry_adjustment import dry_adjustment
from scm.ensemble import default_params
from scm.thermo import cp, g, kappa, make_grid, p0


def _column(nlevels=20):
    grid = make_grid(nlevels)
    params = default_params()
    params['dt'] = 900.0
    state = initial_state(1, grid, params)
    state = update_derived(state, grid)
    return grid, params, state


def _theta_v(state):
    exner = (state['p'] / p0) ** kappa
    theta = state['t'] / exner
    return theta * (1.0 + 0.608 * state['q'] - state['qc'])


def _worst_deficit(state):
    """largest amount by which theta_v decreases upward. <= 0 means stable."""

    theta_v = _theta_v(state)
    return (theta_v[:, 1:] - theta_v[:, :-1]).max().item()


def test_dry_adjustment_removes_a_superadiabatic_layer():
    grid, params, state = _column()
    # drive one interface far past the dry adiabat, the way the boundary layer
    # scheme's fixed mixing depth allows just above bl_top_sigma.
    state['t'][:, -4:] = state['t'][:, -4:] + 10.0
    before = _worst_deficit(state)
    assert before > 5.0

    for _ in range(20):
        output = dry_adjustment(state, grid, params)
        state['t'] = state['t'] + output['dt'] * params['dt']
        state['q'] = state['q'] + output['dq'] * params['dt']
        state['qc'] = state['qc'] + output['dqc'] * params['dt']

    after = _worst_deficit(state)
    assert after < before
    assert after <= params.get('dry_adjustment_tolerance', 1.0) + 0.5


def test_dry_adjustment_conserves_enthalpy_and_water():
    grid, params, state = _column()
    state['t'][:, -4:] = state['t'][:, -4:] + 10.0
    mass = state['dp'] / g

    energy_before = torch.sum(cp * state['t'] * mass).item()
    water_before = torch.sum((state['q'] + state['qc']) * mass).item()

    output = dry_adjustment(state, grid, params)
    t_after = state['t'] + output['dt'] * params['dt']
    q_after = state['q'] + output['dq'] * params['dt']
    qc_after = state['qc'] + output['dqc'] * params['dt']

    energy_after = torch.sum(cp * t_after * mass).item()
    water_after = torch.sum((q_after + qc_after) * mass).item()

    assert abs(energy_after - energy_before) <= 1.0e-6 * abs(energy_before)
    assert abs(water_after - water_before) <= 1.0e-6 * abs(water_before)
    assert q_after.min().item() >= 0.0


def test_dry_adjustment_is_inert_in_a_stable_column():
    grid, params, state = _column()
    assert _worst_deficit(state) <= params.get('dry_adjustment_tolerance', 1.0)

    output = dry_adjustment(state, grid, params)

    assert torch.allclose(output['dt'], torch.zeros_like(output['dt']))
    assert torch.allclose(output['dq'], torch.zeros_like(output['dq']))


def test_physics_step_leaves_column_unchanged_when_adjustment_disabled():
    grid, params, state = _column()
    params['dry_adjustment_enabled'] = False
    params['profile_diagnostics'] = True
    disabled, diagnostics, _ = physics_step(
        {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()},
        grid,
        params,
    )
    assert torch.allclose(
        diagnostics['dry_adjustment_temperature_tendency'],
        torch.zeros_like(disabled['t']),
    )
