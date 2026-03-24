# ensemble driver.
# handles parameter sampling (latin hypercube), structural configuration
# (which convection scheme per member), and batched execution.

import torch
import math


# parameter ranges for each tunable quantity. each entry is
# (default, low, high, description). these ranges are deliberately
# wide to explore structural-like behavior from parameter perturbation.
parameter_ranges = {
    # radiation
    'f_window':       (0.15, 0.10, 0.25, 'longwave window fraction'),
    'kappa_wv':       (0.15, 0.08, 0.30, 'lw water vapor absorption coeff'),
    'co2_log_factor': (0.14, 0.08, 0.25, 'co2 optical depth change per doubling'),
    'co2_base_tau':   (1.50, 0.80, 2.50, 'baseline co2 optical depth'),
    'sw_kappa_wv':    (0.01, 0.005, 0.02, 'sw water vapor absorption'),
    'albedo':         (0.28, 0.20, 0.35, 'surface albedo (includes missing cloud reflection)'),

    # surface
    'cd':             (1.2e-3, 0.5e-3, 2.5e-3, 'surface drag coefficient'),
    'wind_speed':     (5.0, 2.0, 10.0, 'prescribed surface wind speed'),
    'ocean_depth':    (50.0, 20.0, 100.0, 'slab ocean mixed layer depth'),

    # boundary layer
    'k_diff':         (10.0, 2.0, 30.0, 'boundary layer diffusivity'),

    # betts-miller convection
    'tau_bm':         (7200.0, 1800.0, 14400.0, 'bm relaxation timescale'),
    'rhbm':           (0.70, 0.50, 0.90, 'bm reference relative humidity'),

    # mass-flux convection
    'entrainment_rate': (1.5e-4, 0.5e-4, 5.0e-4, 'mf entrainment rate per Pa'),
    'tau_cape':         (3600.0, 1800.0, 7200.0, 'mf cape closure timescale'),
    'precip_efficiency': (0.80, 0.50, 0.95, 'mf precip efficiency'),

    # shared convection
    'cape_threshold': (100.0, 10.0, 500.0, 'cape threshold for triggering'),

    # large-scale condensation
    'ls_precip_fraction': (0.1, 0.05, 0.3, 'fraction of condensate removed per step'),
}


def latin_hypercube(n, d, device='cpu'):
    """latin hypercube sample: n points in d dimensions, each in [0,1].
    better space-filling than pure random for parameter exploration."""

    result = torch.zeros(n, d, device=device)
    for j in range(d):
        perm = torch.randperm(n, device=device).float()
        u = torch.rand(n, device=device)
        result[:, j] = (perm + u) / n

    return result


def sample_parameters(n_members, scheme='betts_miller', param_names=None,
                      device='cpu'):
    """sample parameters using latin hypercube. returns a dict of (n_members,)
    tensors. only samples parameters relevant to the specified scheme plus
    shared parameters.

    param_names: if given, only sample these parameters (others get defaults).
    """

    # figure out which parameters to sample
    if param_names is None:
        # pick scheme-relevant params
        shared = ['f_window', 'kappa_wv', 'co2_log_factor', 'co2_base_tau',
                  'sw_kappa_wv', 'albedo', 'cd', 'wind_speed', 'ocean_depth',
                  'k_diff', 'cape_threshold', 'ls_precip_fraction']
        if scheme == 'betts_miller':
            scheme_params = ['tau_bm', 'rhbm']
        elif scheme == 'mass_flux':
            scheme_params = ['entrainment_rate', 'tau_cape', 'precip_efficiency']
        else:
            scheme_params = []
        param_names = shared + scheme_params

    d = len(param_names)
    lhs = latin_hypercube(n_members, d, device=device)

    params = {}
    for i, name in enumerate(param_names):
        default, lo, hi, desc = parameter_ranges[name]
        # map from [0,1] to [lo, hi]
        params[name] = lo + (hi - lo) * lhs[:, i]

    # fill in defaults for anything not sampled
    for name, (default, lo, hi, desc) in parameter_ranges.items():
        if name not in params:
            params[name] = torch.full((n_members,), default, device=device)

    return params


def make_ensemble_params(n_bm, n_mf, base_params=None, device='cpu'):
    """create parameters for a mixed structural-parametric ensemble.
    n_bm members use betts-miller, n_mf use mass-flux. returns a single
    params dict with (n_bm + n_mf,) tensors and a scheme_mask."""

    n_total = n_bm + n_mf

    bm_params = sample_parameters(n_bm, scheme='betts_miller', device=device)
    mf_params = sample_parameters(n_mf, scheme='mass_flux', device=device)

    # concatenate along the batch dimension
    params = {}
    for name in parameter_ranges:
        bm_val = bm_params.get(name, torch.full((n_bm,), parameter_ranges[name][0], device=device))
        mf_val = mf_params.get(name, torch.full((n_mf,), parameter_ranges[name][0], device=device))
        params[name] = torch.cat([bm_val, mf_val], dim=0)

    # scheme mask: 0 for betts-miller, 1 for mass-flux
    params['scheme_mask'] = torch.cat([
        torch.zeros(n_bm, device=device),
        torch.ones(n_mf, device=device),
    ])

    # merge in any base params that aren't per-member
    if base_params is not None:
        for k, v in base_params.items():
            if k not in params:
                params[k] = v

    return params


def make_fixed_ensemble_params(n_bm, n_mf, base_params=None, device='cpu'):
    """create a deterministic mixed ensemble using the default parameter
    values for every member. useful for debugging structural differences
    without introducing random parameter spread."""

    n_total = n_bm + n_mf

    params = {}
    for name, (default, lo, hi, desc) in parameter_ranges.items():
        params[name] = torch.full((n_total,), default, device=device)

    params['scheme_mask'] = torch.cat([
        torch.zeros(n_bm, device=device),
        torch.ones(n_mf, device=device),
    ])

    if base_params is not None:
        for k, v in base_params.items():
            if k not in params:
                params[k] = v

    return params


def default_params(device='cpu'):
    """single-member default parameters for testing."""
    params = {}
    for name, (default, lo, hi, desc) in parameter_ranges.items():
        params[name] = default
    params['dt'] = 900.0
    params['ps0'] = 1e5
    params['ts_init'] = 290.0
    params['solar_constant'] = 1360.0
    params['zenith_factor'] = 0.25
    params['co2'] = 400.0
    params['co2_ref'] = 400.0
    params['use_slab_ocean'] = True
    params['convection_scheme'] = 'betts_miller'
    return params
