# run a mixed structural-parametric ensemble experiment.
#
# two modes:
#   python -m scm.run_scm --demo     fast 10-member, 200-day test
#   python -m scm.run_scm            full 100-member, 2000-day experiment
#
# spins up to radiative-convective equilibrium under 1xCO2,
# then branches to 2xCO2 and runs to new equilibrium.

import torch
import time
import argparse
import sys
sys.path.insert(0, '/home/claude')

from scm.thermo import make_grid
from scm.column_model import initial_state, run, update_derived
from scm.ensemble import make_ensemble_params, default_params
from scm.diagnostics import (
    equilibrium_stats, climate_sensitivity, summarize_ensemble
)


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def progress_callback(step, state, diag):
    ts_mean = state['ts'].mean().item()
    ts_std = state['ts'].std().item()
    olr_mean = diag['olr'].mean().item()
    precip_mean = (diag['precip_total'] * 86400).mean().item()
    print(f"  step {step:6d}  Ts={ts_mean:.2f}+/-{ts_std:.2f} K  "
          f"OLR={olr_mean:.1f} W/m2  P={precip_mean:.2f} mm/day")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true',
                        help='quick 10-member, 200-day test run')
    parser.add_argument('--no-plot', action='store_true',
                        help='skip plotting (if matplotlib not available)')
    parser.add_argument('--fixed-sst', action='store_true',
                        help='fixed SST mode (fast equilibration for debugging)')
    parser.add_argument('--device', type=str, default=None,
                        help='force device (cpu/cuda/mps)')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"using device: {device}")

    # configuration
    if args.demo:
        n_bm, n_mf = 5, 5
        spinup_days, perturb_days = 200, 200
        print("demo mode: 10 members, 200-day spinup")
    else:
        n_bm, n_mf = 50, 50
        spinup_days, perturb_days = 2000, 2000
        print(f"full mode: {n_bm + n_mf} members, {spinup_days}-day spinup")

    n_total = n_bm + n_mf
    dt = 900.0
    steps_per_day = int(86400 / dt)
    spinup_steps = spinup_days * steps_per_day
    perturb_steps = perturb_days * steps_per_day
    diag_every = steps_per_day * 10
    rad_every = 8

    # grid
    grid = make_grid(nlevels=20, device=device)

    # ensemble parameters
    base = {
        'dt': dt,
        'ps0': 1e5,
        'ts_init': 290.0,
        'solar_constant': 1360.0,
        'zenith_factor': 0.25,
        'co2': 400.0,
        'co2_ref': 400.0,
        'use_slab_ocean': not args.fixed_sst,
    }
    params = make_ensemble_params(n_bm, n_mf, base_params=base, device=device)
    state = initial_state(n_total, grid, params, device=device)

    # --- 1xCO2 spinup ---
    print(f"\nspinup: {n_total} members, {spinup_days} days...")
    t0 = time.time()
    state, history_1x = run(
        state, grid, params, spinup_steps,
        rad_interval=rad_every, diag_interval=diag_every,
        callback=progress_callback,
    )
    elapsed = time.time() - t0
    sim_speed = spinup_days * n_total / elapsed
    print(f"done in {elapsed:.1f} s ({sim_speed:.0f} member-days/s)")

    stats_1x = equilibrium_stats(history_1x, last_n=50)
    print(f"\n1xCO2 equilibrium:")
    print(f"  Ts = {stats_1x['ts_mean'].mean():.2f} +/- "
          f"{stats_1x['ts_mean'].std():.2f} K")
    print(f"  OLR = {stats_1x['olr_mean'].mean():.1f} W/m2")
    print(f"  precip = {(stats_1x['precip_total_mean'] * 86400).mean():.2f} mm/day")

    # --- branch to 2xCO2 ---
    print(f"\nbranching to 2xCO2 for {perturb_days} days...")
    params['co2'] = 800.0

    t0 = time.time()
    state, history_2x = run(
        state, grid, params, perturb_steps,
        rad_interval=rad_every, diag_interval=diag_every,
        callback=progress_callback,
    )
    elapsed = time.time() - t0
    sim_speed = perturb_days * n_total / elapsed
    print(f"done in {elapsed:.1f} s ({sim_speed:.0f} member-days/s)")

    stats_2x = equilibrium_stats(history_2x, last_n=50)
    print(f"\n2xCO2 equilibrium:")
    print(f"  Ts = {stats_2x['ts_mean'].mean():.2f} +/- "
          f"{stats_2x['ts_mean'].std():.2f} K")
    print(f"  OLR = {stats_2x['olr_mean'].mean():.1f} W/m2")
    print(f"  precip = {(stats_2x['precip_total_mean'] * 86400).mean():.2f} mm/day")

    # --- climate sensitivity ---
    print("\n--- climate sensitivity ---")
    results = climate_sensitivity(stats_1x, stats_2x)
    summarize_ensemble(results, scheme_mask=params.get('scheme_mask'))

    # save results
    output = {
        'stats_1x': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                     for k, v in stats_1x.items()},
        'stats_2x': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                     for k, v in stats_2x.items()},
        'sensitivity': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in results.items()},
        'params': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                   for k, v in params.items()},
        'scheme_mask': params.get('scheme_mask').cpu()
                       if params.get('scheme_mask') is not None else None,
        'history_1x': [{k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in d.items()} for d in history_1x],
        'history_2x': [{k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in d.items()} for d in history_2x],
    }
    torch.save(output, 'scm_ensemble_results.pt')
    print("\nresults saved to scm_ensemble_results.pt")

    # --- plotting ---
    if not args.no_plot:
        try:
            from scm.plotting import full_diagnostic_figure

            combined_history = history_1x + history_2x

            fig = full_diagnostic_figure(
                combined_history, results, grid, state,
                scheme_mask=params.get('scheme_mask'),
                dt_days=10,
                savepath='scm_ensemble_diagnostics.png',
            )
            print("diagnostic figure saved to scm_ensemble_diagnostics.png")
        except ImportError:
            print("matplotlib not available, skipping plots")
        except Exception as e:
            print(f"plotting failed: {e}")


if __name__ == '__main__':
    main()
