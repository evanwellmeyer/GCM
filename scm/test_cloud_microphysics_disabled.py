import torch

from scm.cloud_microphysics import cloud_microphysics_step
from scm.column_model import initial_state, update_derived
from scm.ensemble import default_params
from scm.thermo import make_grid


def test_disabled_cloud_microphysics_preserves_condensate_state():
    grid = make_grid(20)
    params = default_params()
    params['cloud_microphysics_enabled'] = False
    state = initial_state(1, grid, params)
    state = update_derived(state, grid)
    state['qc'][:, -4:] = 2.0e-4
    state['cloud_fraction'][:, -4:] = 0.15
    zeros = torch.zeros_like(state['q'])
    process = {'cloud_source': zeros, 'dt': zeros, 'dq': zeros}

    output = cloud_microphysics_step(state, grid, params, process, process)

    assert torch.equal(output['qc'], state['qc'])
    assert torch.equal(output['cloud_fraction'], state['cloud_fraction'])
