import torch

from scm.case_benchmarks import run_bomex, run_dry_mixed_layer
from scm.thermo import make_grid


def test_dry_mixed_layer_conserves_surface_energy():
    result = run_dry_mixed_layer(make_grid(40), hours=1.0)
    assert abs(result['energy_error_wm2']) < 0.2
    assert result['surface_theta_change_k'] > 0.0


def test_bomex_case_has_resolved_lower_atmosphere():
    coarse = run_bomex(make_grid(20), hours=0.25)
    fine = run_bomex(make_grid(80), hours=0.25)
    assert fine['levels_below_2000m'] > coarse['levels_below_2000m']
    assert fine['levels_below_2000m'] >= 10


def test_tke_closure_improves_dry_mixed_layer_resolution_response():
    richardson_spreads = []
    tke_spreads = []
    for levels in [20, 40, 80]:
        grid = make_grid(levels)
        richardson = run_dry_mixed_layer(grid, hours=1.0)
        tke = run_dry_mixed_layer(grid, hours=1.0, scheme='tke')
        richardson_spreads.append(richardson['mixed_layer_theta_spread_k'])
        tke_spreads.append(tke['mixed_layer_theta_spread_k'])
        assert abs(tke['energy_error_wm2']) < 0.5

    assert max(tke_spreads) < 0.5 * max(richardson_spreads)
    assert max(tke_spreads) - min(tke_spreads) < 0.25
