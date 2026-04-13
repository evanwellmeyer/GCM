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
from scm.experiment import (
    apply_param_overrides, build_output_stem, build_restart_path,
    expand_member_params_to_columns, load_restart_bundle, member_counts,
    save_restart_bundle,
)
from scm.thermo import make_grid
from scm.column_model import initial_state, run
from scm.ensemble import make_ensemble_params, make_fixed_ensemble_params
from scm.diagnostics import (
    check_equilibrium, equilibrium_metrics, equilibrium_stats,
    climate_sensitivity, forcing_breakdown, summarize_ensemble
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


def print_phase_summary(title, stats, eq_metrics, eq):
    print(f"\n{title}:")
    print(f"  Ts = {stats['ts_mean'].mean():.2f} +/- "
          f"{stats['ts_mean'].std(unbiased=False):.2f} K")
    print(f"  ASR = {stats['asr_mean'].mean():.1f} W/m2")
    print(f"  OLR = {stats['olr_mean'].mean():.1f} W/m2")
    print(f"  TOA net = {stats['toa_net_mean'].mean():+.2f} W/m2")
    if 'clear_sky_toa_net_mean' in stats:
        print(f"  clear-sky TOA = {stats['clear_sky_toa_net_mean'].mean():+.2f} W/m2")
    if 'cloud_toa_cre_mean' in stats:
        print(f"  cloud TOA CRE = {stats['cloud_toa_cre_mean'].mean():+.2f} W/m2")
    print(f"  surface net = {stats['surface_net_flux_mean'].mean():+.2f} W/m2")
    if 'precip_heat_flux_mean' in stats:
        print(f"  precip heat = {stats['precip_heat_flux_mean'].mean():+.2f} W/m2")
    if 'surface_total_flux_mean' in stats:
        print(f"  surface total = {stats['surface_total_flux_mean'].mean():+.2f} W/m2")
    if 'forcing_energy_tendency_mean' in stats:
        print(f"  forcing energy = {stats['forcing_energy_tendency_mean'].mean():+.2f} W/m2")
    if 'column_energy_tendency_mean' in stats:
        print(f"  column tendency = {stats['column_energy_tendency_mean'].mean():+.2f} W/m2")
    if 'column_energy_residual_mean' in stats:
        print(f"  column residual = {stats['column_energy_residual_mean'].mean():+.2f} W/m2")
    if 'column_mse_residual_mean' in stats:
        print(f"  column mse residual = {stats['column_mse_residual_mean'].mean():+.2f} W/m2")
    if 'conv_energy_tendency_mean' in stats:
        print(f"  conv energy = {stats['conv_energy_tendency_mean'].mean():+.2f} W/m2")
    if 'shallow_energy_tendency_mean' in stats:
        print(f"  shallow energy = {stats['shallow_energy_tendency_mean'].mean():+.2f} W/m2")
    if 'tau_cape_eff_mean' in stats:
        print(f"  tau_cape eff = {stats['tau_cape_eff_mean'].mean():.0f} s")
    if 'conv_mse_residual_mean' in stats:
        print(f"  conv mse residual = {stats['conv_mse_residual_mean'].mean():+.2e} W/m2")
    if 'shallow_mse_residual_mean' in stats:
        print(f"  shallow mse residual = {stats['shallow_mse_residual_mean'].mean():+.2e} W/m2")
    print(f"  precip = {(stats['precip_total_mean'] * 86400).mean():.2f} mm/day")
    if eq_metrics is not None:
        print(f"  late Ts drift = {eq_metrics['max_ts_window_drift']:.3f} K/window")
        if 'max_toa_imbalance' in eq_metrics:
            print(f"  late |TOA net| = {eq_metrics['max_toa_imbalance']:.2f} W/m2")
        if 'max_surface_total_imbalance' in eq_metrics:
            print(f"  late |surface total| = {eq_metrics['max_surface_total_imbalance']:.2f} W/m2")
        if 'max_column_residual' in eq_metrics:
            print(f"  late |column residual| = {eq_metrics['max_column_residual']:.2f} W/m2")
    print(f"  equilibrium check = {'PASS' if eq else 'NOT CONVERGED'}")


def print_forcing_summary(title, forcing):
    print(f"\n{title}:")
    print(f"  dTOA net = {forcing['delta_toa_net']:+.2f} W/m2")
    print(f"  dASR = {forcing['delta_asr']:+.2f} W/m2")
    print(f"  dOLR = {forcing['delta_olr']:+.2f} W/m2")
    if 'delta_clear_sky_toa_net' in forcing:
        print(f"  dclear-sky TOA = {forcing['delta_clear_sky_toa_net']:+.2f} W/m2")
        print(f"  dclear-sky ASR = {forcing['delta_clear_sky_asr']:+.2f} W/m2")
        print(f"  dclear-sky OLR = {forcing['delta_clear_sky_olr']:+.2f} W/m2")
        print(f"  dcloud TOA CRE = {forcing['delta_cloud_toa_cre']:+.2f} W/m2")
        print(f"  dcloud SW CRE = {forcing['delta_cloud_sw_cre']:+.2f} W/m2")
        print(f"  dcloud LW CRE = {forcing['delta_cloud_lw_cre']:+.2f} W/m2")


def build_large_scale_forcing(ls_cfg, state, grid):
    if not ls_cfg or not bool(ls_cfg.get('enabled', False)):
        return None

    batch = state['t'].shape[0]
    nlevels = grid['nlevels']

    def expand_field(name, reference, surface=False):
        if name not in ls_cfg:
            return None

        value = ls_cfg[name]
        if value is None:
            return None

        tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if tensor.ndim == 0:
            if surface:
                return tensor.expand(batch).clone()
            return tensor.expand(batch, nlevels).clone()
        if surface:
            if tensor.shape == (batch,):
                return tensor.clone()
        else:
            if tensor.shape == (nlevels,):
                return tensor.unsqueeze(0).expand(batch, -1).clone()
            if tensor.shape == (batch, nlevels):
                return tensor.clone()

        target = "(batch,)" if surface else f"({nlevels},) or (batch, {nlevels})"
        raise ValueError(f"large_scale_forcing.{name} must be scalar or {target}, got {tuple(tensor.shape)}")

    forcing = {}
    dt_force = expand_field('dt', state['t'], surface=False)
    dq_force = expand_field('dq', state['q'], surface=False)
    du_force = expand_field('du', state['u'], surface=False) if 'u' in state else None
    dv_force = expand_field('dv', state['v'], surface=False) if 'v' in state else None
    dps_force = expand_field('dps', state['ps'], surface=True)
    if dt_force is not None:
        forcing['dt'] = dt_force
    if dq_force is not None:
        forcing['dq'] = dq_force
    if du_force is not None:
        forcing['du'] = du_force
    if dv_force is not None:
        forcing['dv'] = dv_force
    if dps_force is not None:
        forcing['dps'] = dps_force
    return forcing or None


def make_restart_bundle(
    phase,
    output_stem,
    mode,
    scheme,
    sampling,
    fixed_sst,
    config,
    grid,
    params,
    state,
    diag_every,
    rad_every,
    completed_spinup_days,
    completed_perturb_days,
    history_1x,
    stats_1x,
    eq_metrics_1x,
    eq_1x,
    history_2x=None,
):
    return {
        'kind': 'scm_restart',
        'phase': phase,
        'output_stem': output_stem,
        'mode': mode,
        'scheme': scheme,
        'sampling': sampling,
        'fixed_sst': fixed_sst,
        'config': config,
        'nlevels': int(grid['sigma_full'].numel()),
        'diag_every': int(diag_every),
        'rad_every': int(rad_every),
        'completed_spinup_days': int(completed_spinup_days),
        'completed_perturb_days': int(completed_perturb_days),
        'params': params,
        'state': state,
        'history_1x': history_1x,
        'stats_1x': stats_1x,
        'eq_metrics_1x': eq_metrics_1x,
        'eq_1x': bool(eq_1x),
        'history_2x': history_2x or [],
    }


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
    parser.add_argument('--restart-from', type=str, default=None,
                        help='resume from a saved 1x or 2x restart bundle')
    args = parser.parse_args()

    config = load_run_config(args.config)
    run_cfg = config.get('run', {})
    numerics_cfg = config.get('numerics', {})
    initial_cfg = config.get('initial', {})
    forcing_cfg = config.get('forcing', {})
    ls_forcing_cfg = config.get('large_scale_forcing', {})
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
    if args.restart_from is not None:
        run_cfg['restart_from'] = args.restart_from

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
    ncol = int(run_cfg.get('ncol', 1))
    label = run_cfg.get('label', '')
    preserve_ensemble_shape = bool(run_cfg.get('preserve_ensemble_shape', False))
    restart_from = run_cfg.get('restart_from') or None
    save_restarts = bool(run_cfg.get('save_restarts', True))

    if ncol < 1:
        raise ValueError(f"run.ncol must be >= 1, got {ncol}")

    device_name = run_cfg.get('device', 'auto')
    if args.device:
        device = torch.device(args.device)
    elif device_name == 'auto':
        device = pick_device()
    else:
        device = torch.device(device_name)

    print(f"using device: {device}")

    if calibration_enabled and not restart_from:
        run_radiation_calibration(config, device)
        return

    dt = float(numerics_cfg.get('dt', 900.0))
    steps_per_day = int(86400 / dt)
    spinup_steps = spinup_days * steps_per_day
    perturb_steps = perturb_days * steps_per_day
    diag_every = steps_per_day * int(numerics_cfg.get('diag_interval_days', 10))
    rad_every = int(numerics_cfg.get('rad_interval_steps', 8))
    stats_1x = None
    eq_metrics_1x = None
    eq_1x = False
    history_1x = []
    history_2x = []
    completed_spinup_days = spinup_days
    completed_perturb_days = 0
    sampling_key = 'fixed' if (mode == 'demo' or fixed_params) else 'random'

    if restart_from:
        restart = load_restart_bundle(restart_from, device=device)
        if restart.get('kind') != 'scm_restart':
            raise ValueError(f"{restart_from} is not a valid SCM restart bundle")

        mode = restart.get('mode', mode)
        scheme = restart.get('scheme', scheme)
        sampling_key = restart.get('sampling', sampling_key)
        fixed_sst = bool(restart.get('fixed_sst', fixed_sst))
        completed_spinup_days = int(restart.get('completed_spinup_days', spinup_days))
        completed_perturb_days = int(restart.get('completed_perturb_days', 0))
        diag_every = int(restart.get('diag_every', diag_every))
        rad_every = int(restart.get('rad_every', rad_every))
        label = restart.get('config', {}).get('run', {}).get('label', label) or label

        grid = make_grid(nlevels=int(restart.get('nlevels', numerics_cfg.get('nlevels', 20))),
                         device=device)
        params = restart['params']
        state = restart['state']
        history_1x = restart.get('history_1x', [])
        stats_1x = restart.get('stats_1x')
        eq_metrics_1x = restart.get('eq_metrics_1x')
        eq_1x = bool(restart.get('eq_1x', False))
        history_2x = restart.get('history_2x', [])
        ncol = int(state.get('ncol', ncol))
        nmember = int(state.get('nmember', state['ts'].shape[0]))
        n_total = int(state['ts'].shape[0])
        ls_forcing = build_large_scale_forcing(ls_forcing_cfg, state, grid)

        if mode == 'demo':
            if ncol == 1:
                print(f"demo mode restart: {nmember} members, scheme={scheme}, "
                      f"{sampling_key} parameters")
            else:
                print(f"demo mode restart: {ncol} columns x {nmember} members "
                      f"({n_total} total columns), scheme={scheme}, {sampling_key} parameters")
        else:
            if ncol == 1:
                print(f"full mode restart: {nmember} members, scheme={scheme}, "
                      f"{sampling_key} parameters")
            else:
                print(f"full mode restart: {ncol} columns x {nmember} members "
                      f"({n_total} total columns), scheme={scheme}, {sampling_key} parameters")
        print(f"loaded restart from {restart_from} (phase={restart['phase']})")

        if stats_1x is None and history_1x:
            stats_1x = equilibrium_stats(history_1x, last_n=50)
            eq_metrics_1x = equilibrium_metrics(history_1x)
            eq_1x = check_equilibrium(history_1x)

        if restart['phase'] == '1x':
            print_phase_summary("1xCO2 equilibrium (from restart)", stats_1x, eq_metrics_1x, eq_1x)
            params['co2'] = float(
                restart.get('config', {}).get('forcing', {}).get(
                    'co2_2x', forcing_cfg.get('co2_2x', 800.0)
                )
            )
            total_perturb_days = perturb_days
            output_stem = build_output_stem(
                mode, scheme, sampling_key, fixed_sst,
                completed_spinup_days, total_perturb_days, label=label,
            )
            print(f"\nbranching to 2xCO2 from restart for {perturb_days} days...")
        elif restart['phase'] == '2x':
            total_perturb_days = completed_perturb_days + perturb_days
            output_stem = build_output_stem(
                mode, scheme, sampling_key, fixed_sst,
                completed_spinup_days, total_perturb_days, label=label,
            )
            print(f"\ncontinuing 2xCO2 branch from restart for {perturb_days} days "
                  f"(completed {completed_perturb_days} days already)...")
        else:
            raise ValueError(f"unsupported restart phase: {restart['phase']}")

        t0 = time.time()
        state, new_history_2x = run(
            state, grid, params, perturb_steps,
            rad_interval=rad_every, diag_interval=diag_every,
            callback=progress_callback,
            ls_forcing=ls_forcing,
        )
        elapsed = time.time() - t0
        sim_speed = perturb_days * n_total / max(elapsed, 1.0e-6)
        print(f"done in {elapsed:.1f} s ({sim_speed:.0f} member-days/s)")
        history_2x = history_2x + new_history_2x
        completed_perturb_days = total_perturb_days
    else:
        n_bm, n_mf = member_counts(
            mode, scheme, fixed_params=fixed_params,
            preserve_ensemble_shape=preserve_ensemble_shape,
        )
        nmember = n_bm + n_mf

        if mode == 'demo':
            sampling = 'fixed parameters' if fixed_params else 'sampled parameters'
            if ncol == 1:
                print(f"demo mode: {nmember} members, scheme={scheme}, "
                      f"{sampling}, {spinup_days}-day spinup")
            else:
                print(f"demo mode: {ncol} columns x {nmember} members "
                      f"({ncol * nmember} total columns), scheme={scheme}, "
                      f"{sampling}, {spinup_days}-day spinup")
        else:
            sampling = 'fixed parameters' if fixed_params else 'sampled parameters'
            if ncol == 1:
                print(f"full mode: {nmember} members, scheme={scheme}, "
                      f"{sampling}, {spinup_days}-day spinup")
            else:
                print(f"full mode: {ncol} columns x {nmember} members "
                      f"({ncol * nmember} total columns), scheme={scheme}, "
                      f"{sampling}, {spinup_days}-day spinup")

        n_total = ncol * nmember
        grid = make_grid(nlevels=int(numerics_cfg.get('nlevels', 20)), device=device)

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
        apply_param_overrides(params, param_overrides, nmember, device)
        params = expand_member_params_to_columns(params, ncol, nmember)
        params['use_slab_ocean'] = not fixed_sst
        state = initial_state(n_total, grid, params, device=device, ncol=ncol, nmember=nmember)
        ls_forcing = build_large_scale_forcing(ls_forcing_cfg, state, grid)
        output_stem = build_output_stem(
            mode, scheme, sampling_key, fixed_sst, spinup_days, perturb_days, label=label,
        )

        print(f"\nspinup: {n_total} column states, {spinup_days} days...")
        t0 = time.time()
        state, history_1x = run(
            state, grid, params, spinup_steps,
            rad_interval=rad_every, diag_interval=diag_every,
            callback=progress_callback,
            ls_forcing=ls_forcing,
        )
        elapsed = time.time() - t0
        sim_speed = spinup_days * n_total / elapsed
        print(f"done in {elapsed:.1f} s ({sim_speed:.0f} member-days/s)")

        stats_1x = equilibrium_stats(history_1x, last_n=50)
        eq_metrics_1x = equilibrium_metrics(history_1x)
        eq_1x = check_equilibrium(history_1x)
        print_phase_summary("1xCO2 equilibrium", stats_1x, eq_metrics_1x, eq_1x)

        if save_restarts:
            restart_path_1x = build_restart_path(output_stem, '1x')
            save_restart_bundle(
                restart_path_1x,
                make_restart_bundle(
                    '1x',
                    output_stem,
                    mode,
                    scheme,
                    sampling_key,
                    fixed_sst,
                    config,
                    grid,
                    params,
                    state,
                    diag_every,
                    rad_every,
                    spinup_days,
                    0,
                    history_1x,
                    stats_1x,
                    eq_metrics_1x,
                    eq_1x,
                ),
            )
            print(f"saved 1x restart to {restart_path_1x}")

        print(f"\nbranching to 2xCO2 for {perturb_days} days...")
        params['co2'] = float(forcing_cfg.get('co2_2x', 800.0))

        t0 = time.time()
        state, history_2x = run(
            state, grid, params, perturb_steps,
            rad_interval=rad_every, diag_interval=diag_every,
            callback=progress_callback,
            ls_forcing=ls_forcing,
        )
        elapsed = time.time() - t0
        sim_speed = perturb_days * n_total / elapsed
        print(f"done in {elapsed:.1f} s ({sim_speed:.0f} member-days/s)")
        completed_perturb_days = perturb_days

    stats_2x = equilibrium_stats(history_2x, last_n=50)
    eq_metrics_2x = equilibrium_metrics(history_2x)
    eq_2x = check_equilibrium(history_2x)
    print_phase_summary("2xCO2 equilibrium", stats_2x, eq_metrics_2x, eq_2x)

    # --- climate sensitivity ---
    print("\n--- climate sensitivity ---")
    results = climate_sensitivity(stats_1x, stats_2x)
    summarize_ensemble(results, scheme_mask=params.get('scheme_mask'))
    forcing = forcing_breakdown(stats_1x, stats_2x)
    if fixed_sst or 'delta_clear_sky_toa_net' in forcing:
        print_forcing_summary("forcing / adjustment", forcing)

    if save_restarts:
        restart_path_2x = build_restart_path(output_stem, '2x')
        save_restart_bundle(
            restart_path_2x,
            make_restart_bundle(
                '2x',
                output_stem,
                mode,
                scheme,
                sampling_key,
                fixed_sst,
                config,
                grid,
                params,
                state,
                diag_every,
                rad_every,
                completed_spinup_days,
                completed_perturb_days,
                history_1x,
                stats_1x,
                eq_metrics_1x,
                eq_1x,
                history_2x=history_2x,
            ),
        )
        print(f"saved 2x restart to {restart_path_2x}")

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
