import pytest
import torch

from scm.cloud_optics import cloud_optical_properties, clouds_enabled
from scm.configuration import extract_param_overrides, load_run_config
from scm.thermo import make_grid


@pytest.mark.parametrize('scheme', ['auto', 'microphysics', 'microphysics_linear', 'prescribed'])
def test_explicit_cloud_radiation_switch_overrides_optics(scheme):
    grid = make_grid(20)
    state = {
        't': torch.full((1, 20), 280.0),
        'cloud_fraction': torch.full((1, 20), 0.5),
        'cloud_sw_tau_layer': torch.ones(1, 20),
        'cloud_lw_tau_layer': torch.ones(1, 20),
    }
    params = extract_param_overrides(load_run_config('scm/configs/atm407.toml'))
    params['cloud_optics_scheme'] = scheme
    assert params['cloud_microphysics_enabled']
    assert not clouds_enabled(params)
    original = state['cloud_fraction'].clone()
    optics = cloud_optical_properties(state, grid, params, 1, torch.float32)
    assert all(torch.count_nonzero(value) == 0 for value in optics)
    assert torch.equal(state['cloud_fraction'], original)

    params['cloud_radiative_effects_enabled'] = True
    params['cloud_optics_scheme'] = 'auto'
    assert clouds_enabled(params)
    optics = cloud_optical_properties(state, grid, params, 1, torch.float32)
    assert all(torch.count_nonzero(value) > 0 for value in optics)
    clear = cloud_optical_properties(state, grid, params, 1, torch.float32, force_clear_sky=True)
    assert all(torch.count_nonzero(value) == 0 for value in clear)
