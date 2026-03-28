import argparse
import time
import tomllib
from pathlib import Path

import torch

from scm.column_model import initial_state, run
from scm.configuration import deep_merge, extract_param_overrides, load_run_config
from scm.diagnostics import check_equilibrium, climate_sensitivity, equilibrium_metrics, equilibrium_stats
from scm.ensemble import make_fixed_ensemble_params
from scm.experiment import apply_param_overrides
from scm.thermo import make_grid


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _resolve_path(path_str, suite_path):
    path = Path(path_str)
    if path.is_absolute():
        return path
    suite_parent = Path(suite_path).resolve().parent
    candidate = suite_parent / path
    if candidate.exists():
        return candidate
    return Path(path_str)


def _stats_summary(stats, eq_metrics, eq):
    summary = {
        'ts': float(stats['ts_mean'].mean().item()),
        'asr': float(stats['asr_mean'].mean().item()),
        'olr': float(stats['olr_mean'].mean().item()),
        'toa_net': float(stats['toa_net_mean'].mean().item()),
        'surface_total': float(stats['surface_total_flux_mean'].mean().item()),
        'column_residual': float(stats['column_energy_residual_mean'].mean().item()),
        'precip_mm_day': float((stats['precip_total_mean'] * 86400.0).mean().item()),
        'equilibrium': bool(eq),
    }
    if eq_metrics is not None:
        summary.update({
            'late_ts_drift': float(eq_metrics.get('max_ts_window_drift', 0.0)),
            'late_toa_abs': float(eq_metrics.get('max_toa_imbalance', 0.0)),
            'late_surface_total_abs': float(eq_metrics.get('max_surface_total_imbalance', 0.0)),
            'late_column_residual_abs': float(eq_metrics.get('max_column_residual', 0.0)),
        })
    return summary


def _print_case_summary(case_name, phase_name, summary):
    print(
        f"  {case_name} {phase_name}: "
        f"Ts={summary['ts']:.2f} K  "
        f"TOA={summary['toa_net']:+.2f} W/m2  "
        f"P={summary['precip_mm_day']:.2f} mm/day  "
        f"eq={'PASS' if summary['equilibrium'] else 'NO'}"
    )


def run_benchmark_case(case_name, case_cfg, base_config, device):
    config = deep_merge(base_config, {})
    run_cfg = config.get('run', {})
    numerics_cfg = config.get('numerics', {})
    initial_cfg = config.get('initial', {})
    forcing_cfg = config.get('forcing', {})

    spinup_days = int(case_cfg.get('spinup_days', run_cfg.get('spinup_days', 2000)))
    perturb_days = int(case_cfg.get('perturb_days', run_cfg.get('perturb_days', 0)))
    fixed_sst = bool(case_cfg.get('fixed_sst', run_cfg.get('fixed_sst', False)))
    scheme = str(case_cfg.get('scheme', run_cfg.get('scheme', 'mf')))
    mode = str(run_cfg.get('mode', 'full'))
    label = str(case_cfg.get('label', case_name))

    forcing_override = {}
    if 'co2' in case_cfg:
        forcing_override['co2'] = float(case_cfg['co2'])
    if 'co2_2x' in case_cfg:
        forcing_override['co2_2x'] = float(case_cfg['co2_2x'])
    if forcing_override:
        config['forcing'] = deep_merge(forcing_cfg, forcing_override)
        forcing_cfg = config['forcing']

    param_overrides = extract_param_overrides(config)

    dt = float(numerics_cfg.get('dt', 900.0))
    steps_per_day = int(86400 / dt)
    spinup_steps = spinup_days * steps_per_day
    perturb_steps = perturb_days * steps_per_day
    diag_every = steps_per_day * int(numerics_cfg.get('diag_interval_days', 10))
    rad_every = int(numerics_cfg.get('rad_interval_steps', 8))

    grid = make_grid(nlevels=int(numerics_cfg.get('nlevels', 20)), device=device)

    if scheme == 'bm':
        n_bm, n_mf = 1, 0
    elif scheme == 'mf':
        n_bm, n_mf = 0, 1
    else:
        n_bm, n_mf = 1, 1
    n_total = n_bm + n_mf

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

    params = make_fixed_ensemble_params(n_bm, n_mf, base_params=base, device=device)
    apply_param_overrides(params, param_overrides, n_total, device)
    params['use_slab_ocean'] = not fixed_sst
    state = initial_state(n_total, grid, params, device=device)

    t0 = time.time()
    state, history_1x = run(
        state, grid, params, spinup_steps,
        rad_interval=rad_every, diag_interval=diag_every,
        callback=None,
    )
    elapsed_1x = time.time() - t0

    stats_1x = equilibrium_stats(history_1x, last_n=50)
    eq_metrics_1x = equilibrium_metrics(history_1x)
    eq_1x = check_equilibrium(history_1x)
    summary_1x = _stats_summary(stats_1x, eq_metrics_1x, eq_1x)
    _print_case_summary(case_name, '1x', summary_1x)

    case_result = {
        'name': case_name,
        'description': case_cfg.get('description', ''),
        'label': label,
        'mode': mode,
        'scheme': scheme,
        'fixed_sst': fixed_sst,
        'spinup_days': spinup_days,
        'perturb_days': perturb_days,
        'elapsed_1x_s': elapsed_1x,
        'one_x': summary_1x,
    }

    if perturb_days <= 0:
        return case_result

    params['co2'] = float(forcing_cfg.get('co2_2x', 800.0))
    t0 = time.time()
    state, history_2x = run(
        state, grid, params, perturb_steps,
        rad_interval=rad_every, diag_interval=diag_every,
        callback=None,
    )
    elapsed_2x = time.time() - t0

    stats_2x = equilibrium_stats(history_2x, last_n=50)
    eq_metrics_2x = equilibrium_metrics(history_2x)
    eq_2x = check_equilibrium(history_2x)
    summary_2x = _stats_summary(stats_2x, eq_metrics_2x, eq_2x)
    _print_case_summary(case_name, '2x', summary_2x)

    sensitivity = climate_sensitivity(stats_1x, stats_2x)
    case_result['elapsed_2x_s'] = elapsed_2x
    case_result['two_x'] = summary_2x
    case_result['sensitivity'] = {
        'ecs': float(sensitivity['ecs'].mean().item()),
        'delta_precip': float(sensitivity['delta_precip'].mean().item()),
        'hydro_sensitivity': float(sensitivity['hydro_sensitivity'].mean().item()),
    }
    return case_result


def render_markdown_report(label, base_config_path, case_results):
    lines = [
        f"# SCM Benchmark Report: {label}",
        "",
        f"- Baseline config: `{base_config_path}`",
        "",
    ]
    for case in case_results:
        lines.extend([
            f"## {case['name']}",
            "",
            case.get('description', '') or "No description provided.",
            "",
            f"- Scheme: `{case['scheme']}`",
            f"- Fixed SST: `{case['fixed_sst']}`",
            f"- 1x spinup days: `{case['spinup_days']}`",
            f"- 2x branch days: `{case['perturb_days']}`",
            "",
            "### 1x Summary",
            "",
            f"- Ts: `{case['one_x']['ts']:.2f} K`",
            f"- TOA net: `{case['one_x']['toa_net']:+.2f} W/m2`",
            f"- Surface total: `{case['one_x']['surface_total']:+.2f} W/m2`",
            f"- Column residual: `{case['one_x']['column_residual']:+.2f} W/m2`",
            f"- Precip: `{case['one_x']['precip_mm_day']:.2f} mm/day`",
            f"- Late Ts drift: `{case['one_x'].get('late_ts_drift', 0.0):.3f} K/window`",
            f"- Late |TOA net|: `{case['one_x'].get('late_toa_abs', 0.0):.2f} W/m2`",
            f"- Equilibrium: `{'PASS' if case['one_x']['equilibrium'] else 'NO'}`",
            "",
        ])
        if 'two_x' in case:
            lines.extend([
                "### 2x Summary",
                "",
                f"- Ts: `{case['two_x']['ts']:.2f} K`",
                f"- TOA net: `{case['two_x']['toa_net']:+.2f} W/m2`",
                f"- Surface total: `{case['two_x']['surface_total']:+.2f} W/m2`",
                f"- Column residual: `{case['two_x']['column_residual']:+.2f} W/m2`",
                f"- Precip: `{case['two_x']['precip_mm_day']:.2f} mm/day`",
                f"- Late Ts drift: `{case['two_x'].get('late_ts_drift', 0.0):.3f} K/window`",
                f"- Late |TOA net|: `{case['two_x'].get('late_toa_abs', 0.0):.2f} W/m2`",
                f"- Equilibrium: `{'PASS' if case['two_x']['equilibrium'] else 'NO'}`",
                "",
                "### Sensitivity",
                "",
                f"- ECS: `{case['sensitivity']['ecs']:.2f} K`",
                f"- Delta precip: `{case['sensitivity']['delta_precip']:.3f} mm/day`",
                f"- Hydro sensitivity: `{case['sensitivity']['hydro_sensitivity']:.2f} %/K`",
                "",
            ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--suite',
        type=str,
        default='scm/configs/benchmark_suite.toml',
        help='path to benchmark suite TOML',
    )
    parser.add_argument('--device', type=str, default=None, help='force device (cpu/cuda/mps)')
    args = parser.parse_args()

    suite_path = Path(args.suite)
    with suite_path.open('rb') as f:
        suite_cfg = tomllib.load(f)

    suite = suite_cfg.get('suite', {})
    label = suite.get('label', 'benchmark')
    base_config_path = _resolve_path(suite.get('base_config', 'scm/configs/mf_baseline_v1.toml'), suite_path)
    base_config = load_run_config(base_config_path)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device_name = suite.get('device', 'auto')
        device = pick_device() if device_name == 'auto' else torch.device(device_name)

    print(f"using device: {device}")
    print(f"benchmark suite: {label}")
    print(f"baseline config: {base_config_path}")

    case_results = []
    for case_name, case_cfg in suite_cfg.get('cases', {}).items():
        print(f"\nrunning case: {case_name}")
        case_results.append(run_benchmark_case(case_name, case_cfg, base_config, device))

    results_path = Path(f"scm_benchmark_{label}_results.pt")
    report_path = Path(f"scm_benchmark_{label}_report.md")
    torch.save({'label': label, 'base_config': str(base_config_path), 'cases': case_results}, results_path)
    report_path.write_text(render_markdown_report(label, base_config_path, case_results))

    print(f"\nsaved benchmark results to {results_path}")
    print(f"saved benchmark report to {report_path}")


if __name__ == '__main__':
    main()
