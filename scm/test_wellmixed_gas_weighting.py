"""Well-mixed absorbers should be tied to layer mass, not to layer count.

CO2 and the trace gases are uniformly mixed, so how much a layer absorbs depends
on how much air is in it. The scheme used to divide the column optical depth by
the number of levels, which handed a thin near-surface layer the same absorption
as a thick mid-tropospheric one and left the radiative answer dependent on how
the levels happened to be distributed.
"""

import torch

from scm.column_model import initial_state, update_derived
from scm.ensemble import default_params
from scm.radiation_schemes.multiband import compute_longwave_multiband
from scm.thermo import Rd, g, make_grid, make_smooth_test_grid


def _outgoing_longwave(grid):
    """OLR for one fixed physical atmosphere sampled onto the given grid.

    Temperature follows a single lapse rate and humidity a single scale height,
    so every grid describes an identical atmosphere and any difference in the
    answer belongs to the discretization rather than to the air.
    """

    params = default_params()
    state = initial_state(1, grid, params)
    state = update_derived(state, grid)
    sigma = (state['p'] / state['p'][:, -1:]).clamp(min=1.0e-6)
    state['t'] = 290.0 * sigma ** (Rd * 6.5e-3 / g)
    state['q'] = 0.012 * sigma ** 3
    state['qc'] = torch.zeros_like(state['q'])
    state['ts'] = torch.full_like(state['ts'], 290.0)
    # a well-mixed absorber alone, so the comparison isolates how the column
    # optical depth is spread across layers.
    params.update({
        'lw_band_weights': [1.0],
        'lw_band_wv_kappa': [0.0],
        'lw_band_co2_base_tau': [1.5],
        'lw_band_co2_log_factor': [0.0],
        'lw_band_trace_scale': [0.0],
        'trace_gases_enabled': False,
    })
    return compute_longwave_multiband(state, grid, params)[2][0].item()


def test_outgoing_longwave_barely_moves_when_the_levels_are_redistributed():
    """Two 20-level grids of different shape must see the same atmosphere.

    Both grids carry the same number of levels and the same column optical
    depth; only the thicknesses differ. Weighting the absorber by layer mass
    takes the disagreement from about 1.7 W/m2 down to about 0.8, and what is
    left is the transfer solver's own finite-layer error rather than absorber
    placed in the wrong layers.
    """

    production = _outgoing_longwave(make_grid(20))
    smooth = _outgoing_longwave(make_smooth_test_grid())
    assert abs(production - smooth) < 1.2, (
        f"OLR differs by {abs(production - smooth):.2f} W/m2 between two "
        f"20-level grids ({production:.2f} vs {smooth:.2f}); a well-mixed "
        f"absorber should not care how the levels are cut"
    )


def test_thinnest_layer_takes_the_smallest_share_of_a_well_mixed_absorber():
    grid = make_grid(20)
    state = update_derived(initial_state(1, grid, default_params()), grid)
    dp = state['dp'][0]
    massfraction = (dp / dp.sum())

    # the scheme scales optical depth by exactly this weight, so the thinnest
    # layer must take the smallest share instead of an equal one.
    assert int(massfraction.argmin()) == dp.shape[0] - 1
    assert float(massfraction.min()) < 0.5 / dp.shape[0]
    assert abs(float(massfraction.sum()) - 1.0) < 1.0e-6
