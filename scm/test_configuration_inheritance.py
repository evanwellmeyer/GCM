from pathlib import Path

import pytest

from scm.configuration import extract_param_overrides, load_run_config


def test_default_and_teaching_configuration_are_identical():
    default = load_run_config()
    teaching = load_run_config('scm/configs/atm407.toml')
    teaching.pop('_config_path')
    assert default == teaching


def test_explicit_legacy_experiment_does_not_inherit_new_teaching_params(tmp_path):
    experiment = tmp_path / 'experiment.toml'
    experiment.write_text('[run]\nlabel = "isolated"\n')
    params = extract_param_overrides(load_run_config(experiment))
    assert params['cloud_ls_precip_fraction'] == .95
    assert 'mf_cape_entrainment_rate' not in params


def test_flux_candidate_inherits_complete_atm407_configuration():
    config = load_run_config('scm/configs/atm407_flux_v1.toml')
    params = extract_param_overrides(config)

    assert config['run']['label'] == 'atm407_flux_v1'
    assert config['numerics']['nlevels'] == 20
    assert params['condensation_rh_crit'] == .90
    assert params['bl_diagnose_depth'] is True
    assert params['radiation_scheme'] == 'multiband_ozone_profile'
    assert params['bl_mix_moist_static_energy'] is True
    assert params['mf_transport_form'] == 'flux'
    assert params['mf_downdraft_entrainment_pa'] == 1e-5


def test_configuration_inheritance_cycle_is_rejected(tmp_path):
    first = tmp_path / 'first.toml'
    second = tmp_path / 'second.toml'
    first.write_text('extends = "second.toml"\n')
    second.write_text('extends = "first.toml"\n')

    with pytest.raises(ValueError, match='inheritance cycle'):
        load_run_config(first)
