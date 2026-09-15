"""Diagnostic arithmetic must not manufacture a launch-layer imbalance."""

import torch

from scripts.audit_bomex_launch_budget import divergence, plain, summarize, theta
from scm.thermo import Lv as latent, cp


def test_flux_divergence_telescopes_on_unequal_layers():
    flux = torch.tensor([0., .002, -.001, 0.], dtype=torch.float64)
    mass = torch.tensor([50., 100., 300.], dtype=torch.float64)
    assert abs(float((divergence(flux, mass) * mass).sum())) < 1e-15


def test_diagnostic_theta_is_invariant_under_liquid_condensation():
    state = {'t': torch.tensor([[290.]], dtype=torch.float64),
             'p': torch.tensor([[90000.]], dtype=torch.float64),
             'qc': torch.tensor([[0.]], dtype=torch.float64)}
    before = theta(state)
    state['t'] = state['t'] + latent * .001 / cp
    state['qc'] = state['qc'] + .001
    assert torch.allclose(theta(state), before, atol=1e-12)


def test_summary_does_not_count_the_deep_dispatch_alias_twice():
    zero = [0.] * 20
    process = {'deep': [1e-8] * 20, 'convection': [1e-8] * 20,
               'actual': [1e-8] * 20, 'unattributed': [-1e-8] * 20}
    sample = {'hour': 3., 'water': dict(process), 'theta': dict(process),
              'transport_budget': {'partition_roundoff': zero},
              'flux_components': {'above_source': [0.] * 21}, 'bl_flux': [0.] * 21,
              'rain_mmday': 0., 'source_index': 16, 'transport_scale': 1.}
    result = summarize({'samples': [sample], 'height_m': list(range(20))})
    assert result['windows']['late']['max_water_closure_gkgday'] == 0.
    assert 'convection' not in result['windows']['late']['water']


def test_serialization_does_not_change_tensors():
    value = torch.tensor([[1., 2.]])
    saved = plain({'value': value})
    saved['value'][0][0] = 99.
    assert value[0, 0] == 1.
