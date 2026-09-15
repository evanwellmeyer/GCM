"""Cloud-depth iteration must not accumulate trial-plume tendencies."""

import pytest
import torch

from scm import convection_uw as plume


def test_sorting_scale_converges_per_column_without_accumulation(monkeypatch):
    calls = []

    def trial(column, result, *args):
        params = args[-2]
        distance = params.get('uw_shallow_sorting_distance_m', 100.)
        calls.append((column, distance))
        result['plume_top_height'][column] = 2000. + 1000. * column
        result['sorting_distance'][column] = distance
        result['water_flux'][column] += 1.

    monkeypatch.setattr(plume, '_integrate_one_column', trial)
    array = torch.ones((2, 3), dtype=torch.float64)
    params = {'uw_shallow_diagnose_sorting_distance': True}
    result = plume._integrate_columns(*([array] * 12), torch.ones(2), params)
    assert calls == [(0, 100.), (0, 200.), (1, 100.), (1, 300.)]
    assert torch.all(result['water_flux'] == 1.)
    assert torch.all(result['sorting_scale_error'] == 0.)
    assert params == {'uw_shallow_diagnose_sorting_distance': True}


def test_zero_iteration_budget_is_rejected():
    array = torch.ones((1, 3))
    with pytest.raises(ValueError):
        plume._integrate_columns(*([array] * 12), torch.ones(1),
                                 {'uw_shallow_diagnose_sorting_distance': True,
                                  'uw_shallow_sorting_iterations': 0})
