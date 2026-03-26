# run a mixed structural-parametric ensemble experiment.
#
# two modes:
#   python -m scm.run_scm --demo     fast 10-member, 500-day test
#   python -m scm.run_scm            full 100-member, 2000-day experiment
#
# spins up to radiative-convective equilibrium under 1xCO2,
# then branches to 2xCO2 and runs to a new equilibrium.
# if calibration is enabled in the config, the driver runs a short
# fixed-parameter radiation sweep instead of the 1x/2x experiment.

import torch
import time
import argparse
import sys
sys.path.insert(0, '/home/claude')

from scm.calibration import run_radiation_calibration
from scm.configuration import (
    load_run_config, DEFAULT_CONFIG_PATH, extract_param_overrides
)
from scm.experiment import build_output_stem, apply_param_overrides, member_counts
from scm.thermo import make_grid
from scm.column_model import initial_state, run
from scm.ensemble import make_ensemble_params, make_fixed_ensemble_params
from scm.diagnostics import (
    check_equilibrium, equilibrium_stats, climate_sensitivity, summarize_ensemble
)


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def progress_callback(step, state, diag):
    ts_mean = state['ts'].mean().item()
    ts_std = state['ts'].std(unbiased=False).item()
    olr_mean = diag['olr'].mean().item()
    toa_mean = diag['toa_net'].mean().item()
    precip_mean = (diag['precip_total'] * 86400).mean().item()
    print(f"  step {step:6d}  Ts={ts_mean:.2f}+/-{ts_std:.2f} K  "
          f"OLR={olr_mean:.1f} W/m2  TOA={toa_mean:+.1f} W/m2  "
          f"P={precip_mean:.2f} mm/day")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help=f'path to TOML run config (default: {DEFAULT_CONFIG_PATH})')
    parser.add_argument('--demo', action='store_true', default=None,
                        help='10-member diagnostic run')
    parser.add_argument('--scheme', choices=['mixed', 'bm', 'mf'],
                        default=None,
                        help='convection configuration: mixed, bm only, or mf only')
    parser.add_argument('--fixed-params', action='store_true', default=None,
                        help='use default parameter values instead of sampling')
    parser.add_argument('--spinup-days', type=int, default=None,
                        help='override spinup length in days')
    parser.add_argument('--perturb-days', type=int, default=None,
                        help='override 2xCO2 branch length in days')
    parser.add_argument('--no-plot', action='store_true', default=None,
                        help='skip plotting (if matplotlib not available)')
    parser.add_argument('--fixed-sst', action='store_true', default=None,
                        help='fixed SST mode (fast equilibration for debugging)')
    parser.add_argument('--device', type=str, default=None,
                        help='force device (cpu/cuda/mps)')
    args = parser.parse_args()

    config = load_run_config(args.config)
    run_cfg = config.get('run', {})
    numerics_cfg = config.get('numerics', {})
    initial_cfg = config.get('initial', {})
    forcing_cfg = config.get('forcing', {})
    param_overrides = extract_param_overrides(config)

    if args.demo is not None:
        run_cfg['mode'] = 'demo'
    if args.scheme is not None:
        run_cfg['scheme'] = args.scheme
    if args.fixed_params is not None:
        run_cfg['sampling'] = 'fixed'
    if args.spinup_days is not None:
        run_cfg['spinup_days'] = args.spinup_days
    if args.perturb_days is not None:
        run_cfg['perturb_days'] = args.perturb_days
    if args.no_plot is not None:
        run_cfg['plot'] = False
    if args.fixed_sst is not None:
        run_cfg['fixed_sst'] = True
    if args.device is not None:
        run_cfg['device'] = args.device

    calibration_cfg = config.get('calibration', {})
    if args.scheme is not None:
        calibration_cfg['scheme'] = args.scheme
    if args.spinup_days is not None:
        calibration_cfg['spinup_days'] = args.spinup_days
    if args.fixed_sst is not None:
        calibration_cfg['fixed_sst'] = True
    calibration_enabled = bool(calibration_cfg.get('enabled', False))
    mode = run_cfg.get('mode', 'full')
    scheme = run_cfg.get('scheme', 'mixed')
    sampling_mode = run_cfg.get('sampling', 'random')
    fixed_params = (sampling_mode == 'fixed') or (mode == 'demo')
    fixed_sst = bool(run_cfg.get('fixed_sst', False))
    plot_enabled = bool(run_cfg.get('plot', True))
    spinup_days = int(run_cfg.get('spinup_days', 500 if mode == 'demo' else 2000))
    perturb_days = int(run_cfg.get('perturb_days', 500 if mode == 'demo' else 2000))
    label = run_cfg.get('label', '')
    preserve_ensemble_shape = bool(run_cfg.get('preserve_ensemble_shape', False))

    device_name = run_cfg.get('device', 'auto')
    if args.device:
        device = torch.device(args.device)
    elif device_name == 'auto':
        device = pick_device()
    else:
        device = torch.device(device_name)

    print(f"using device: {device}")

    if calibration_enabled:
        run_radiation_calibration(config, device)
        return

    n_bm, n_mf = member_counts(
        mode, scheme, fixed_params=fixed_params,
        preserve_ensemble_shape=preserve_ensemble_shape,
    )

    if mode == 'demo':
        sampling = 'fixed parameters' if fixed_params else 'sampled parameters'
        print(f"demo mode: {n_bm + n_mf} members, scheme={scheme}, "
              f"{sampling}, {spinup_days}-day spinup")
    else:
        sampling = 'fixed parameters' if fixed_params else 'sampled parameters'
        print(f"full mode: {n_bm + n_mf} members, scheme={scheme}, "
              f"{sampling}, {spinup_days}-day spinup")

    n_total = n_bm + n_mf
    dt = float(numerics_cfg.get('dt', 900.0))
    steps_per_day = int(86400 / dt)
    spinup_steps = spinup_days * steps_per_day
    perturb_steps = perturb_days * steps_per_day
    diag_every = steps_per_day * int(numerics_cfg.get('diag_interval_days', 10))
    rad_every = int(numerics_cfg.get('rad_interval_steps', 8))

    # grid
    grid = make_grid(nlevels=int(numerics_cfg.get('nlevels', 20)), device=device)

    # ensemble parameters
    base = {
        'dt': dt,
        'ps0': float(initial_cfg.get('ps0', 1e5)),
        'ts_init': float(initial_cfg.get('ts_init', 290.0)),
        'solar_constant': float(forcing_cfg.get('solar_constant', 1360.0)),
        'zenith_factor': float(forcing_cfg.get('zenith_factor', 0.25)),
        'co2': float(forcing_cfg.get('co2', 400.0)),
        'co2_ref': float(forcing_cfg.get('co2_ref', 400.0)),
        'use_slab_ocean': not fixed_sst,
        'rad_interval_microphysics_steps': int(
            numerics_cfg.get('rad_interval_microphysics_steps', 1)
        ),
    }
    if mode == 'demo' or fixed_params:
        params = make_fixed_ensemble_params(n_bm, n_mf, base_params=base, device=device)
    else:
        params = make_ensemble_params(n_bm, n_mf, base_params=base, device=device)
    apply_param_overrides(params, param_overrides, n_total, device)
    params['use_slab_ocean'] = not fixed_sst
    state = initial_state(n_total, grid, params, device=device)
    output_stem = build_output_stem(
        mode, scheme, 'fixed' if (mode == 'demo' or fixed_params) else 'random',
        fixed_sst, spinup_days, perturb_days, label=label,
    )

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
    eq_1x = check_equilibrium(history_1x)
    print(f"\n1xCO2 equilibrium:")
    print(f"  Ts = {stats_1x['ts_mean'].mean():.2f} +/- "
          f"{stats_1x['ts_mean'].std(unbiased=False):.2f} K")
    print(f"  ASR = {stats_1x['asr_mean'].mean():.1f} W/m2")
    print(f"  OLR = {stats_1x['olr_mean'].mean():.1f} W/m2")
    print(f"  TOA net = {stats_1x['toa_net_mean'].mean():+.2f} W/m2")
    print(f"  surface net = {stats_1x['surface_net_flux_mean'].mean():+.2f} W/m2")
    if 'precip_heat_flux_mean' in stats_1x:
        print(f"  precip heat = {stats_1x['precip_heat_flux_mean'].mean():+.2f} W/m2")
    if 'surface_total_flux_mean' in stats_1x:
        print(f"  surface total = {stats_1x['surface_total_flux_mean'].mean():+.2f} W/m2")
    if 'column_energy_tendency_mean' in stats_1x:
        print(f"  column tendency = {stats_1x['column_energy_tendency_mean'].mean():+.2f} W/m2")
    if 'column_energy_residual_mean' in stats_1x:
        print(f"  column residual = {stats_1x['column_energy_residual_mean'].mean():+.2f} W/m2")
    print(f"  precip = {(stats_1x['precip_total_mean'] * 86400).mean():.2f} mm/day")
    print(f"  equilibrium check = {'PASS' if eq_1x else 'NOT CONVERGED'}")

    # --- branch to 2xCO2 ---
    print(f"\nbranching to 2xCO2 for {perturb_days} days...")
    params['co2'] = float(forcing_cfg.get('co2_2x', 800.0))

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
    eq_2x = check_equilibrium(history_2x)
    print(f"\n2xCO2 equilibrium:")
    print(f"  Ts = {stats_2x['ts_mean'].mean():.2f} +/- "
          f"{stats_2x['ts_mean'].std(unbiased=False):.2f} K")
    print(f"  ASR = {stats_2x['asr_mean'].mean():.1f} W/m2")
    print(f"  OLR = {stats_2x['olr_mean'].mean():.1f} W/m2")
    print(f"  TOA net = {stats_2x['toa_net_mean'].mean():+.2f} W/m2")
    print(f"  surface net = {stats_2x['surface_net_flux_mean'].mean():+.2f} W/m2")
    if 'precip_heat_flux_mean' in stats_2x:
        print(f"  precip heat = {stats_2x['precip_heat_flux_mean'].mean():+.2f} W/m2")
    if 'surface_total_flux_mean' in stats_2x:
        print(f"  surface total = {stats_2x['surface_total_flux_mean'].mean():+.2f} W/m2")
    if 'column_energy_tendency_mean' in stats_2x:
        print(f"  column tendency = {stats_2x['column_energy_tendency_mean'].mean():+.2f} W/m2")
    if 'column_energy_residual_mean' in stats_2x:
        print(f"  column residual = {stats_2x['column_energy_residual_mean'].mean():+.2f} W/m2")
    print(f"  precip = {(stats_2x['precip_total_mean'] * 86400).mean():.2f} mm/day")
    print(f"  equilibrium check = {'PASS' if eq_2x else 'NOT CONVERGED'}")

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
        'config': config,
        'scheme_mask': params.get('scheme_mask').cpu()
                       if params.get('scheme_mask') is not None else None,
        'history_1x': [{k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in d.items()} for d in history_1x],
        'history_2x': [{k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in d.items()} for d in history_2x],
    }
    output_path = f'{output_stem}_results.pt'
    torch.save(output, output_path)
    print(f"\nresults saved to {output_path}")

    # --- plotting ---
    if plot_enabled:
        try:
            from scm.plotting import full_diagnostic_figure

            combined_history = history_1x + history_2x
            figure_path = f'{output_stem}_diagnostics.png'

            fig = full_diagnostic_figure(
                combined_history, results, grid, state,
                scheme_mask=params.get('scheme_mask'),
                dt_days=10,
                savepath=figure_path,
            )
            print(f"diagnostic figure saved to {figure_path}")
        except ImportError:
            print("matplotlib not available, skipping plots")
        except Exception as e:
            print(f"plotting failed: {e}")


if __name__ == '__main__':
    main()
