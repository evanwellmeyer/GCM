# plotting utilities for the single column model.
# visualize profiles, time series, and ensemble spread.
# all functions take torch tensors and convert to numpy internally.

import torch
import numpy as np


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def plot_profile(ax, sigma, values, label=None, **kwargs):
    """plot a vertical profile with pressure-like coordinates (high sigma
    at bottom, low sigma at top)."""
    ax.plot(to_np(values), to_np(sigma), label=label, **kwargs)
    ax.invert_yaxis()
    ax.set_ylabel('sigma')


def plot_temperature_profiles(ax, grid, state, n_show=10):
    """plot temperature profiles for a subset of ensemble members."""
    sigma = to_np(grid['sigma_full'])
    t = to_np(state['t'])
    batch = t.shape[0]
    step = max(1, batch // n_show)

    for i in range(0, batch, step):
        ax.plot(t[i], sigma, alpha=0.5, linewidth=0.8)

    ax.invert_yaxis()
    ax.set_xlabel('temperature (K)')
    ax.set_ylabel('sigma')
    ax.set_title('temperature profiles')


def plot_moisture_profiles(ax, grid, state, n_show=10):
    """plot specific humidity profiles in g/kg."""
    sigma = to_np(grid['sigma_full'])
    q = to_np(state['q']) * 1000  # g/kg
    batch = q.shape[0]
    step = max(1, batch // n_show)

    for i in range(0, batch, step):
        ax.plot(q[i], sigma, alpha=0.5, linewidth=0.8)

    ax.invert_yaxis()
    ax.set_xlabel('specific humidity (g/kg)')
    ax.set_ylabel('sigma')
    ax.set_title('moisture profiles')


def plot_ts_timeseries(ax, diag_history, scheme_mask=None, dt_days=10):
    """plot surface temperature evolution over time."""
    n_diags = len(diag_history)
    time_days = np.arange(n_diags) * dt_days

    ts_all = np.stack([to_np(d['ts']) for d in diag_history])  # (time, batch)
    batch = ts_all.shape[1]

    if scheme_mask is not None:
        mask = to_np(scheme_mask)
        bm = mask < 0.5
        mf = mask >= 0.5

        if bm.any():
            ts_bm = ts_all[:, bm]
            ax.fill_between(time_days, ts_bm.min(axis=1), ts_bm.max(axis=1),
                            alpha=0.2, color='C0', label='BM range')
            ax.plot(time_days, ts_bm.mean(axis=1), color='C0',
                    label='BM mean')

        if mf.any():
            ts_mf = ts_all[:, mf]
            ax.fill_between(time_days, ts_mf.min(axis=1), ts_mf.max(axis=1),
                            alpha=0.2, color='C1', label='MF range')
            ax.plot(time_days, ts_mf.mean(axis=1), color='C1',
                    label='MF mean')
    else:
        ax.fill_between(time_days, ts_all.min(axis=1), ts_all.max(axis=1),
                        alpha=0.2, color='C0')
        ax.plot(time_days, ts_all.mean(axis=1), color='C0')

    ax.set_xlabel('time (days)')
    ax.set_ylabel('surface temperature (K)')
    ax.set_title('surface temperature evolution')
    ax.legend(fontsize=8)


def plot_precip_timeseries(ax, diag_history, scheme_mask=None, dt_days=10):
    """plot precipitation evolution in mm/day."""
    n_diags = len(diag_history)
    time_days = np.arange(n_diags) * dt_days

    precip = np.stack([to_np(d['precip_total']) * 86400
                       for d in diag_history])

    if scheme_mask is not None:
        mask = to_np(scheme_mask)
        bm = mask < 0.5
        mf = mask >= 0.5

        if bm.any():
            p_bm = precip[:, bm]
            ax.plot(time_days, p_bm.mean(axis=1), color='C0', label='BM mean')
            ax.fill_between(time_days, p_bm.min(axis=1), p_bm.max(axis=1),
                            alpha=0.15, color='C0')
        if mf.any():
            p_mf = precip[:, mf]
            ax.plot(time_days, p_mf.mean(axis=1), color='C1', label='MF mean')
            ax.fill_between(time_days, p_mf.min(axis=1), p_mf.max(axis=1),
                            alpha=0.15, color='C1')
    else:
        ax.plot(time_days, precip.mean(axis=1), color='C0')
        ax.fill_between(time_days, precip.min(axis=1), precip.max(axis=1),
                        alpha=0.15, color='C0')

    ax.set_xlabel('time (days)')
    ax.set_ylabel('precipitation (mm/day)')
    ax.set_title('precipitation evolution')
    ax.legend(fontsize=8)


def plot_ecs_vs_precip(ax, sensitivity, scheme_mask=None):
    """the money plot: ECS vs precipitation change, colored by scheme."""
    ecs = to_np(sensitivity['ecs'])
    dp = to_np(sensitivity['delta_precip'])

    if scheme_mask is not None:
        mask = to_np(scheme_mask)
        bm = mask < 0.5
        mf = mask >= 0.5

        if bm.any():
            ax.scatter(ecs[bm], dp[bm], c='C0', alpha=0.6, s=20,
                       label='Betts-Miller', edgecolors='none')
        if mf.any():
            ax.scatter(ecs[mf], dp[mf], c='C1', alpha=0.6, s=20,
                       label='Mass-Flux', edgecolors='none')
        ax.legend(fontsize=9)
    else:
        ax.scatter(ecs, dp, c='C0', alpha=0.6, s=20, edgecolors='none')

    ax.set_xlabel('equilibrium climate sensitivity (K)')
    ax.set_ylabel('precipitation change (mm/day)')
    ax.set_title('ECS vs precipitation response to 2xCO2')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')


def plot_hydro_sensitivity(ax, sensitivity, scheme_mask=None):
    """histogram of hydrological sensitivity (%/K)."""
    hs = to_np(sensitivity['hydro_sensitivity'])

    if scheme_mask is not None:
        mask = to_np(scheme_mask)
        bm = mask < 0.5
        mf = mask >= 0.5

        bins = np.linspace(hs.min() - 0.5, hs.max() + 0.5, 25)
        if bm.any():
            ax.hist(hs[bm], bins=bins, alpha=0.5, color='C0',
                    label='Betts-Miller')
        if mf.any():
            ax.hist(hs[mf], bins=bins, alpha=0.5, color='C1',
                    label='Mass-Flux')
        ax.legend(fontsize=9)
    else:
        ax.hist(hs, bins=25, alpha=0.6, color='C0')

    ax.set_xlabel('hydrological sensitivity (%/K)')
    ax.set_ylabel('count')
    ax.set_title('hydrological sensitivity distribution')

    # CMIP range for reference: roughly 1-3 %/K
    ax.axvline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(3.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(2.0, ax.get_ylim()[1] * 0.9, 'CMIP range', ha='center',
            fontsize=7, color='gray')


def plot_energy_balance(ax, diag_history, dt_days=10):
    """plot OLR, SHF, LHF over time to check energy balance convergence."""
    n = len(diag_history)
    time_days = np.arange(n) * dt_days

    olr = np.array([to_np(d['olr']).mean() for d in diag_history])
    shf = np.array([to_np(d['shf']).mean() for d in diag_history])
    lhf = np.array([to_np(d['lhf']).mean() for d in diag_history])

    ax.plot(time_days, olr, label='OLR', color='C3')
    ax.plot(time_days, shf, label='SHF', color='C0')
    ax.plot(time_days, lhf, label='LHF', color='C2')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('flux (W/m2)')
    ax.set_title('energy balance components')
    ax.legend(fontsize=8)


def full_diagnostic_figure(diag_history, sensitivity, grid, state,
                           scheme_mask=None, dt_days=10, savepath=None):
    """create the full 6-panel diagnostic figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    plot_ts_timeseries(axes[0, 0], diag_history, scheme_mask, dt_days)
    plot_precip_timeseries(axes[0, 1], diag_history, scheme_mask, dt_days)
    plot_energy_balance(axes[0, 2], diag_history, dt_days)
    plot_temperature_profiles(axes[1, 0], grid, state)
    plot_ecs_vs_precip(axes[1, 1], sensitivity, scheme_mask)
    plot_hydro_sensitivity(axes[1, 2], sensitivity, scheme_mask)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"saved figure to {savepath}")
    return fig
