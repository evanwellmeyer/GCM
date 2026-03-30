# diagnostics for the single column model.
# equilibrium detection, climate sensitivity computation, and
# basic analysis of ensemble output.

import torch
from scm.thermo import Lv, g


def equilibrium_metrics(diag_history, window=50):
    """summarize late-window equilibrium metrics for all members.

    Returns a dict of scalar worst-member metrics over the last `window`
    diagnostic snapshots. If there is not enough history yet, returns None.
    """

    if len(diag_history) < window:
        return None

    recent = diag_history[-window:]
    recent_ts = torch.stack([d['ts'] for d in recent])  # (window, batch)
    x = torch.arange(window, dtype=recent_ts.dtype, device=recent_ts.device)
    x = x - x.mean()
    slopes = (recent_ts * x.unsqueeze(1)).sum(dim=0) / (x * x).sum()
    max_ts_slope = slopes.abs().max().item()

    metrics = {
        'max_ts_slope': max_ts_slope,
        'max_ts_window_drift': max_ts_slope * (window - 1),
    }

    def add_mean_abs_metric(source_key, metric_key):
        if source_key in recent[0]:
            vals = torch.stack([d[source_key] for d in recent])
            metrics[metric_key] = vals.mean(dim=0).abs().max().item()

    add_mean_abs_metric('toa_net', 'max_toa_imbalance')
    add_mean_abs_metric('surface_total_flux', 'max_surface_total_imbalance')
    add_mean_abs_metric('column_energy_residual', 'max_column_residual')
    add_mean_abs_metric('column_mse_residual', 'max_column_mse_residual')

    return metrics


def check_equilibrium(
    diag_history,
    window=50,
    ts_threshold=0.05,
    toa_threshold=1.0,
    surface_threshold=1.0,
    column_residual_threshold=1.0,
):
    """check whether the column is close to radiative-convective equilibrium.

    The late-window surface-temperature trend must be small, and any available
    column-closure diagnostics must also be small. This is intentionally stricter
    than the earlier check that only looked at temperature trend and slab/column
    tendency.
    """

    metrics = equilibrium_metrics(diag_history, window=window)
    if metrics is None:
        return False

    if metrics['max_ts_window_drift'] >= ts_threshold:
        return False
    if metrics.get('max_toa_imbalance', 0.0) >= toa_threshold:
        return False
    if metrics.get('max_surface_total_imbalance', 0.0) >= surface_threshold:
        return False
    if metrics.get('max_column_residual', 0.0) >= column_residual_threshold:
        return False

    return True


def equilibrium_stats(diag_history, last_n=50):
    """compute time-averaged diagnostics over the last n snapshots.
    returns a dict of (batch,) tensors."""

    if len(diag_history) < last_n:
        last_n = len(diag_history)

    recent = diag_history[-last_n:]

    stats = {}
    for key in ['ts', 'olr', 'asr', 'toa_net', 'precip_total', 'precip_conv',
                'precip_ls', 'precip_cloud', 'precip_heat_flux',
                'shf', 'lhf', 'sw_absorbed_sfc', 'sw_reflected_toa',
                'lw_down_sfc', 'lw_up_sfc',
                'clear_sky_olr', 'clear_sky_asr', 'clear_sky_toa_net',
                'cloud_lw_cre', 'cloud_sw_cre', 'cloud_toa_cre',
                'surface_net_flux', 'surface_total_flux',
                'forcing_energy_tendency', 'forcing_mse_tendency',
                'rad_energy_tendency', 'surface_energy_tendency',
                'bl_energy_tendency', 'shallow_energy_tendency',
                'conv_energy_tendency', 'condensation_energy_tendency',
                'atmos_flux_convergence', 'atmos_energy_tendency',
                'atmos_energy_residual', 'atmos_mse_tendency',
                'atmos_mse_residual', 'slab_energy_tendency',
                'column_energy_tendency', 'column_energy_residual',
                'column_mse_tendency', 'column_mse_residual',
                'conv_mse_residual', 'shallow_mse_residual', 'tau_cape_eff',
                'cloud_cover', 'lwp', 'iwp']:
        if key in recent[0]:
            vals = torch.stack([d[key] for d in recent])
            stats[key + '_mean'] = vals.mean(dim=0)
            stats[key + '_std'] = vals.std(dim=0, unbiased=False)

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


def forcing_breakdown(stats_1x, stats_2x):
    """Summarize fixed-SST forcing and cloud-adjustment diagnostics."""

    summary = {
        'delta_toa_net': float((stats_2x['toa_net_mean'] - stats_1x['toa_net_mean']).mean().item()),
        'delta_asr': float((stats_2x['asr_mean'] - stats_1x['asr_mean']).mean().item()),
        'delta_olr': float((stats_2x['olr_mean'] - stats_1x['olr_mean']).mean().item()),
    }

    if 'clear_sky_toa_net_mean' in stats_1x and 'clear_sky_toa_net_mean' in stats_2x:
        summary.update({
            'delta_clear_sky_toa_net': float(
                (stats_2x['clear_sky_toa_net_mean'] - stats_1x['clear_sky_toa_net_mean']).mean().item()
            ),
            'delta_clear_sky_asr': float(
                (stats_2x['clear_sky_asr_mean'] - stats_1x['clear_sky_asr_mean']).mean().item()
            ),
            'delta_clear_sky_olr': float(
                (stats_2x['clear_sky_olr_mean'] - stats_1x['clear_sky_olr_mean']).mean().item()
            ),
            'delta_cloud_toa_cre': float(
                (stats_2x['cloud_toa_cre_mean'] - stats_1x['cloud_toa_cre_mean']).mean().item()
            ),
            'delta_cloud_sw_cre': float(
                (stats_2x['cloud_sw_cre_mean'] - stats_1x['cloud_sw_cre_mean']).mean().item()
            ),
            'delta_cloud_lw_cre': float(
                (stats_2x['cloud_lw_cre_mean'] - stats_1x['cloud_lw_cre_mean']).mean().item()
            ),
        })

    return summary


def energy_balance(state, diag):
    """check the top-of-atmosphere and surface energy balance.
    useful for debugging: at equilibrium both should be near zero."""

    return {
        'asr': diag['asr'],
        'olr': diag['olr'],
        'toa_net': diag['toa_net'],
        'clear_sky_asr': diag.get('clear_sky_asr'),
        'clear_sky_olr': diag.get('clear_sky_olr'),
        'clear_sky_toa_net': diag.get('clear_sky_toa_net'),
        'cloud_lw_cre': diag.get('cloud_lw_cre'),
        'cloud_sw_cre': diag.get('cloud_sw_cre'),
        'cloud_toa_cre': diag.get('cloud_toa_cre'),
        'shf': diag['shf'],
        'lhf': diag['lhf'],
        'surface_net_flux': diag['surface_net_flux'],
        'precip_heat_flux': diag.get('precip_heat_flux'),
        'surface_total_flux': diag.get('surface_total_flux'),
        'forcing_energy_tendency': diag.get('forcing_energy_tendency'),
        'forcing_mse_tendency': diag.get('forcing_mse_tendency'),
        'rad_energy_tendency': diag.get('rad_energy_tendency'),
        'surface_energy_tendency': diag.get('surface_energy_tendency'),
        'bl_energy_tendency': diag.get('bl_energy_tendency'),
        'shallow_energy_tendency': diag.get('shallow_energy_tendency'),
        'conv_energy_tendency': diag.get('conv_energy_tendency'),
        'condensation_energy_tendency': diag.get('condensation_energy_tendency'),
        'atmos_flux_convergence': diag.get('atmos_flux_convergence'),
        'atmos_energy_tendency': diag.get('atmos_energy_tendency'),
        'atmos_energy_residual': diag.get('atmos_energy_residual'),
        'atmos_mse_tendency': diag.get('atmos_mse_tendency'),
        'atmos_mse_residual': diag.get('atmos_mse_residual'),
        'slab_energy_tendency': diag.get('slab_energy_tendency'),
        'column_energy_tendency': diag.get('column_energy_tendency'),
        'column_energy_residual': diag.get('column_energy_residual'),
        'column_mse_tendency': diag.get('column_mse_tendency'),
        'column_mse_residual': diag.get('column_mse_residual'),
        'conv_mse_residual': diag.get('conv_mse_residual'),
        'shallow_mse_residual': diag.get('shallow_mse_residual'),
    }


def summarize_ensemble(sensitivity_results, scheme_mask=None):
    """print a summary of the ensemble results. if scheme_mask is provided,
    break down by convection scheme."""

    ecs = sensitivity_results['ecs']
    dp = sensitivity_results['delta_precip']
    hs = sensitivity_results['hydro_sensitivity']

    print(f"ensemble size: {ecs.shape[0]}")
    print(f"ecs:  mean={ecs.mean():.2f} K, std={ecs.std(unbiased=False):.2f}, "
          f"range=[{ecs.min():.2f}, {ecs.max():.2f}]")
    print(f"dP:   mean={dp.mean():.3f} mm/day, std={dp.std(unbiased=False):.3f}, "
          f"range=[{dp.min():.3f}, {dp.max():.3f}]")
    print(f"HS:   mean={hs.mean():.2f} %/K, std={hs.std(unbiased=False):.2f}")

    if scheme_mask is not None:
        bm_mask = scheme_mask < 0.5
        mf_mask = scheme_mask >= 0.5

        if bm_mask.any():
            print(f"\nbetts-miller ({bm_mask.sum()} members):")
            print(f"  ecs: mean={ecs[bm_mask].mean():.2f}, std={ecs[bm_mask].std(unbiased=False):.2f}")
            print(f"  dP:  mean={dp[bm_mask].mean():.3f}, std={dp[bm_mask].std(unbiased=False):.3f}")

        if mf_mask.any():
            print(f"\nmass-flux ({mf_mask.sum()} members):")
            print(f"  ecs: mean={ecs[mf_mask].mean():.2f}, std={ecs[mf_mask].std(unbiased=False):.2f}")
            print(f"  dP:  mean={dp[mf_mask].mean():.3f}, std={dp[mf_mask].std(unbiased=False):.3f}")
