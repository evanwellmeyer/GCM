"""The closure's CAPE parcel mixes separately from the plume.

CESM2 dilutes ZM's CAPE parcel at about 1 per km while the plume carries its own
entrainment. Ours was one knob, so the rate that stopped deep convection firing in
BOMEX also moved the plume's detrainment and cooled the upper troposphere 6-7 K.
`mf_cape_entrainment_rate` splits them; unset, it is the plume's rate.
"""

from pathlib import Path

import torch

from scm.case_benchmarks import initialize_bomex
from scm.configuration import extract_param_overrides, load_run_config
from scm.convection_mf import mass_flux_convection
from scm.ensemble import default_params
from scm.thermo import make_grid

CONFIG = Path(__file__).resolve().parents[1] / 'scm/configs/atm407.toml'


def _production_params():
    params = default_params()
    params.update(extract_param_overrides(load_run_config(CONFIG)))
    return params


def _column():
    grid = make_grid(20)
    state, _ = initialize_bomex(grid)
    return state, grid


def test_matching_the_plume_rate_reproduces_the_single_knob():
    state, grid = _column()
    params = _production_params()
    params.pop('mf_cape_entrainment_rate', None)
    baseline = mass_flux_convection(state, grid, params)
    matched = mass_flux_convection(
        state, grid, dict(params, mf_cape_entrainment_rate=params['entrainment_rate'])
    )
    for name in ['cape', 'cloud_base_mass_flux', 'dt', 'dq']:
        torch.testing.assert_close(matched[name], baseline[name])


def test_diluting_the_cape_parcel_leaves_the_plume_alone():
    state, grid = _column()
    params = _production_params()
    params.pop('mf_cape_entrainment_rate', None)
    baseline = mass_flux_convection(state, grid, params)
    diluted = mass_flux_convection(
        state, grid, dict(params, mf_cape_entrainment_rate=10.0 * params['entrainment_rate'])
    )
    # The closure sees a much less buoyant parcel. Its mass flux is not a one-way
    # function of that: the CAPE response per unit mass flux falls too, so on a column
    # with large CAPE the diagnosed mass flux can rise even as CAPE drops.
    assert float(diluted['cape'][0]) < 0.5 * float(baseline['cape'][0])
    # ... while the plume's own transport, per unit mass flux, is untouched.
    torch.testing.assert_close(
        diluted['transport_mse_residual_per_mass_flux'],
        baseline['transport_mse_residual_per_mass_flux'],
    )


def test_production_config_carries_the_split_and_no_instant_rainout():
    params = _production_params()
    assert params['mf_cape_entrainment_rate'] > params['entrainment_rate']
    assert params['cloud_ls_precip_fraction'] == 0.0
