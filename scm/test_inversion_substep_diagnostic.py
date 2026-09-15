"""Check the evolving-state driver independently of the expensive plume."""

import pytest
import torch

from scripts import check_bomex_inversion_substeps as diagnostic


def sample():
    fields = {key: [[value] * 20] for key, value in
              [('t', 280.), ('q', .01), ('qc', 0.), ('u', 0.), ('v', 0.),
               ('p', 90000.), ('dp', 1000.), ('tke', .1)]}
    fields.update(boundary_depth=[600.], interfaces=[[.1] * 19])
    return {'plume_input': fields}


def test_driver_evolves_state_and_closes_exchange_budget(monkeypatch):
    observed = []

    def exchange(saved, mode, state, timestep, return_state):
        observed.append(float(state['q'][0, 15]))
        updated = {key: value.clone() for key, value in state.items()}
        transfer = state['q'][0, 15] * 1e-5 * timestep
        updated['q'][0, 15] -= transfer
        updated['q'][0, 14] += transfer
        return {'source_liquid_kgkg': .001, 'precipitation_kgm2s': 0.}, updated

    monkeypatch.setattr(diagnostic, 'run_trial', exchange)
    result = diagnostic.compare(sample(), 'joint', 225.)
    assert len(observed) == 4
    assert all(left > right for left, right in zip(observed, observed[1:]))
    expected = .01 * ((1 - 225.e-5) ** 4 - 1) * 1000
    assert result['water_change_825_gkg'] == pytest.approx(expected, abs=2e-6)
    assert abs(result['integrated_water_residual_kgm2s']) < 1e-8
    assert abs(result['integrated_energy_residual_wm2']) < .1


def test_driver_rejects_incomplete_interval():
    with pytest.raises(ValueError, match='exactly cover'):
        diagnostic.compare(sample(), 'joint', 400.)
