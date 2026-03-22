# diagnostics for the single column model.
# equilibrium detection, climate sensitivity computation, and
# basic analysis of ensemble output.

import torch
from scm.thermo import Lv, g


def check_equilibrium(diag_history, window=50, threshold=0.1):
    """check if the model has reached radiative-convective equilibrium by
    looking at the trend in surface temperature over the last `window`
    diagnostic snapshots. returns True if the trend is below threshold K
    per window length for all ensemble members."""

    if len(diag_history) < window:
        return False

    recent_ts = torch.stack([d['ts'] for d in diag_history[-window:]])  # (window, batch)
    # linear trend per member
    x = torch.arange(window, dtype=recent_ts.dtype, device=recent_ts.device)
    x = x - x.mean()
    slopes = (recent_ts * x.unsqueeze(1)).sum(dim=0) / (x * x).sum()
    max_trend = slopes.abs().max().item()

    return max_trend < threshold


def equilibrium_stats(diag_history, last_n=50):
    """compute time-averaged diagnostics over the last n snapshots.
    returns a dict of (batch,) tensors."""

    if len(diag_history) < last_n:
        last_n = len(diag_history)

    recent = diag_history[-last_n:]

    stats = {}
    for key in ['ts', 'olr', 'precip_total', 'precip_conv', 'precip_ls',
                'shf', 'lhf']:
        if key in recent[0]:
            vals = torch.stack([d[key] for d in recent])
            stats[key + '_mean'] = vals.mean(dim=0)
            stats[key + '_std'] = vals.std(dim=0)

    # mean temperature profile
    t_profiles = torch.stack([d['t'] for d in recent])
    stats['t_mean'] = t_profiles.mean(dim=0)

    # mean moisture profile
    q_profiles = torch.stack([d['q'] for d in recent])
    stats['q_mean'] = q_profiles.mean(dim=0)

    return stats


def climate_sensitivity(stats_1x, stats_2x):
    """compute effective climate sensitivity and hydrological sensitivity
    from control and 2xCO2 equilibrium statistics.

    returns a dict with:
      ecs: equilibrium climate sensitivity (K)
      delta_precip: precipitation change (mm/day)
      hydro_sensitivity: percent precip change per K warming
    """

    dts = stats_2x['ts_mean'] - stats_1x['ts_mean']

    # precipitation in mm/day: kg/m2/s * 86400 s/day
    precip_1x = stats_1x['precip_total_mean'] * 86400.0
    precip_2x = stats_2x['precip_total_mean'] * 86400.0
    dprecip = precip_2x - precip_1x

    # hydrological sensitivity: % change in precip per K warming
    hydro = 100.0 * dprecip / precip_1x.clamp(min=0.01) / dts.clamp(min=0.01)

    return {
        'ecs': dts,
        'delta_precip': dprecip,
        'precip_1x': precip_1x,
        'precip_2x': precip_2x,
        'hydro_sensitivity': hydro,
    }


def energy_balance(state, diag):
    """check the top-of-atmosphere and surface energy balance.
    useful for debugging: at equilibrium both should be near zero."""

    # toa: absorbed solar minus olr
    # we don't have absorbed solar directly in the diag, but we can
    # compute it from olr + surface balance + column tendency
    # for now just return what we have
    return {
        'olr': diag['olr'],
        'shf': diag['shf'],
        'lhf': diag['lhf'],
    }


def summarize_ensemble(sensitivity_results, scheme_mask=None):
    """print a summary of the ensemble results. if scheme_mask is provided,
    break down by convection scheme."""

    ecs = sensitivity_results['ecs']
    dp = sensitivity_results['delta_precip']
    hs = sensitivity_results['hydro_sensitivity']

    print(f"ensemble size: {ecs.shape[0]}")
    print(f"ecs:  mean={ecs.mean():.2f} K, std={ecs.std():.2f}, "
          f"range=[{ecs.min():.2f}, {ecs.max():.2f}]")
    print(f"dP:   mean={dp.mean():.3f} mm/day, std={dp.std():.3f}, "
          f"range=[{dp.min():.3f}, {dp.max():.3f}]")
    print(f"HS:   mean={hs.mean():.2f} %/K, std={hs.std():.2f}")

    if scheme_mask is not None:
        bm_mask = scheme_mask < 0.5
        mf_mask = scheme_mask >= 0.5

        if bm_mask.any():
            print(f"\nbetts-miller ({bm_mask.sum()} members):")
            print(f"  ecs: mean={ecs[bm_mask].mean():.2f}, std={ecs[bm_mask].std():.2f}")
            print(f"  dP:  mean={dp[bm_mask].mean():.3f}, std={dp[bm_mask].std():.3f}")

        if mf_mask.any():
            print(f"\nmass-flux ({mf_mask.sum()} members):")
            print(f"  ecs: mean={ecs[mf_mask].mean():.2f}, std={ecs[mf_mask].std():.2f}")
            print(f"  dP:  mean={dp[mf_mask].mean():.3f}, std={dp[mf_mask].std():.3f}")
