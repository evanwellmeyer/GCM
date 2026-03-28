import argparse
import time
import tomllib
from pathlib import Path

import torch

from scm.column_model import initial_state, run
from scm.configuration import deep_merge, extract_param_overrides, load_run_config
from scm.diagnostics import check_equilibrium, climate_sensitivity, equilibrium_metrics, equilibrium_stats
from scm.ensemble import make_fixed_ensemble_params
from scm.experiment import (
    apply_param_overrides, cpu_tensors, load_restart_bundle, move_tensors
)
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


def _clone_for_device(obj, device):
    """Deep-clone nested tensors and move them to the requested device."""

    return move_tensors(cpu_tensors(obj), device)


def evaluate_thresholds(actual, thresholds):
    """Evaluate min/max/equality threshold checks for a metrics dict."""

    checks = []
    passed = True
    for key, expected in thresholds.items():
        if key.endswith('_min'):
            metric = key[:-4]
            condition = f">= {expected}"
            actual_value = actual.get(metric)
            ok = actual_value is not None and actual_value >= expected
        elif key.endswith('_max'):
            metric = key[:-4]
            condition = f"<= {expected}"
            actual_value = actual.get(metric)
            ok = actual_value is not None and actual_value <= expected
        else:
            metric = key
            condition = f"== {expected}"
            actual_value = actual.get(metric)
            ok = actual_value is not None and actual_value == expected

        checks.append({
            'metric': metric,
            'actual': actual_value,
            'condition': condition,
            'passed': bool(ok),
        })
        passed = passed and bool(ok)

    return {'passed': passed, 'checks': checks}


def evaluate_case_thresholds(case_result, thresholds_cfg):
    """Evaluate thresholds for one_x, two_x, and sensitivity sections."""

    sections = {}
    overall = True
    for section_name in ['one_x', 'two_x', 'sensitivity']:
        section_thresholds = thresholds_cfg.get(section_name, {})
        if not section_thresholds or section_name not in case_result:
            continue
        section_result = evaluate_thresholds(case_result[section_name], section_thresholds)
        sections[section_name] = section_result
        overall = overall and section_result['passed']

    return {'passed': overall, 'sections': sections}


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
    if 'tau_cape_eff_mean' in stats:
        summary['tau_cape_eff'] = float(stats['tau_cape_eff_mean'].mean().item())
    if eq_metrics is not None:
        summary.update({
            'late_ts_drift': float(eq_metrics.get('max_ts_window_drift', 0.0)),
            'late_toa_abs': float(eq_metrics.get('max_toa_imbalance', 0.0)),
            'late_surface_total_abs': float(eq_metrics.get('max_surface_total_imbalance', 0.0)),
            'late_column_residual_abs': float(eq_metrics.get('max_column_residual', 0.0)),
        })
    return summary


def _print_case_summary(case_name, phase_name, summary):
    tau_str = f"  tau={summary['tau_cape_eff']:.0f} s" if 'tau_cape_eff' in summary else ""
    print(
        f"  {case_name} {phase_name}: "
        f"Ts={summary['ts']:.2f} K  "
        f"TOA={summary['toa_net']:+.2f} W/m2  "
        f"P={summary['precip_mm_day']:.2f} mm/day"
        f"{tau_str}  "
        f"eq={'PASS' if summary['equilibrium'] else 'NO'}"
    )


def _print_case_status(case_name, evaluation):
    status = 'PASS' if evaluation['passed'] else 'FAIL'
    print(f"  {case_name} benchmark status: {status}")
    if evaluation['passed']:
        return
    for section_name, section_eval in evaluation['sections'].items():
        for check in section_eval['checks']:
            if check['passed']:
                continue
            actual = check['actual']
            actual_str = 'missing' if actual is None else f"{actual}"
            print(f"    {section_name}.{check['metric']}: {actual_str} {check['condition']}")


def run_benchmark_case(case_name, case_cfg, base_config, device, suite_path=None, spinup_cache=None):
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
    reuse_spinup_from = case_cfg.get('reuse_spinup_from')
    restart_from = case_cfg.get('restart_from')
    thresholds_cfg = case_cfg.get('thresholds', {})

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
    elapsed_1x = 0.0
    elapsed_2x = 0.0
    history_2x = []
    source = 'fresh'

    if restart_from:
        if suite_path is None:
            raise ValueError("suite_path is required when using restart_from")
        restart_path = _resolve_path(restart_from, suite_path)
        bundle = load_restart_bundle(restart_path, device=device)
        if bundle.get('kind') != 'scm_restart':
            raise ValueError(f"{restart_path} is not an SCM restart bundle")
        grid = make_grid(nlevels=int(bundle.get('nlevels', numerics_cfg.get('nlevels', 20))), device=device)
        params = bundle['params']
        state = bundle['state']
        history_1x = bundle.get('history_1x', [])
        stats_1x = bundle.get('stats_1x')
        eq_metrics_1x = bundle.get('eq_metrics_1x')
        eq_1x = bool(bundle.get('eq_1x', False))
        history_2x = bundle.get('history_2x', [])
        source = f"restart:{Path(restart_path).name}"
    elif reuse_spinup_from:
        if spinup_cache is None or reuse_spinup_from not in spinup_cache:
            raise ValueError(f"benchmark case {case_name} requested unknown spinup cache '{reuse_spinup_from}'")
        cached = _clone_for_device(spinup_cache[reuse_spinup_from], device)
        if cached['scheme'] != scheme or bool(cached['fixed_sst']) != fixed_sst:
            raise ValueError(
                f"benchmark case {case_name} cannot reuse {reuse_spinup_from}: "
                f"scheme/fixed_sst mismatch"
            )
        grid = make_grid(nlevels=int(cached['nlevels']), device=device)
        params = cached['params']
        state = cached['state']
        history_1x = cached['history_1x']
        stats_1x = cached['stats_1x']
        eq_metrics_1x = cached['eq_metrics_1x']
        eq_1x = cached['eq_1x']
        source = f"cache:{reuse_spinup_from}"
    else:
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

    if stats_1x is None and history_1x:
        stats_1x = equilibrium_stats(history_1x, last_n=50)
    if eq_metrics_1x is None and history_1x:
        eq_metrics_1x = equilibrium_metrics(history_1x)
    if history_1x and stats_1x is not None and eq_metrics_1x is not None:
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
        'source': source,
        'one_x': summary_1x,
    }

    if spinup_cache is not None and history_1x:
        spinup_cache[case_name] = cpu_tensors({
            'nlevels': int(grid['sigma_full'].numel()),
            'scheme': scheme,
            'fixed_sst': fixed_sst,
            'params': params,
            'state': state,
            'history_1x': history_1x,
            'stats_1x': stats_1x,
            'eq_metrics_1x': eq_metrics_1x,
            'eq_1x': bool(eq_1x),
        })

    if perturb_days <= 0:
        case_result['evaluation'] = evaluate_case_thresholds(case_result, thresholds_cfg)
        return case_result

    params = _clone_for_device(params, device)
    state = _clone_for_device(state, device)

    if source.startswith('restart:') and history_2x:
        # continue an in-progress 2x branch from the loaded restart
        pass
    else:
        params['co2'] = float(forcing_cfg.get('co2_2x', 800.0))
        history_2x = []

    t0 = time.time()
    state, new_history_2x = run(
        state, grid, params, perturb_steps,
        rad_interval=rad_every, diag_interval=diag_every,
        callback=None,
    )
    elapsed_2x = time.time() - t0
    history_2x = history_2x + new_history_2x

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
    case_result['evaluation'] = evaluate_case_thresholds(case_result, thresholds_cfg)
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
            f"- Source: `{case.get('source', 'fresh')}`",
            f"- 1x spinup days: `{case['spinup_days']}`",
            f"- 2x branch days: `{case['perturb_days']}`",
            f"- Benchmark status: `{'PASS' if case.get('evaluation', {}).get('passed', True) else 'FAIL'}`",
            "",
            "### 1x Summary",
            "",
            f"- Ts: `{case['one_x']['ts']:.2f} K`",
            f"- TOA net: `{case['one_x']['toa_net']:+.2f} W/m2`",
            f"- Surface total: `{case['one_x']['surface_total']:+.2f} W/m2`",
            f"- Column residual: `{case['one_x']['column_residual']:+.2f} W/m2`",
            f"- Precip: `{case['one_x']['precip_mm_day']:.2f} mm/day`",
            f"- Tau CAPE eff: `{case['one_x'].get('tau_cape_eff', 0.0):.0f} s`" if 'tau_cape_eff' in case['one_x'] else "- Tau CAPE eff: `n/a`",
            f"- Late Ts drift: `{case['one_x'].get('late_ts_drift', 0.0):.3f} K/window`",
            f"- Late |TOA net|: `{case['one_x'].get('late_toa_abs', 0.0):.2f} W/m2`",
            f"- Equilibrium: `{'PASS' if case['one_x']['equilibrium'] else 'NO'}`",
            "",
        ])
        if case.get('evaluation', {}).get('sections', {}).get('one_x'):
            lines.extend([
                "",
                "### 1x Threshold Checks",
                "",
            ])
            for check in case['evaluation']['sections']['one_x']['checks']:
                actual = 'missing' if check['actual'] is None else check['actual']
                lines.append(
                    f"- `{'PASS' if check['passed'] else 'FAIL'}` "
                    f"`{check['metric']}` actual=`{actual}` target=`{check['condition']}`"
                )
            lines.append("")
        if 'two_x' in case:
            lines.extend([
                "### 2x Summary",
                "",
                f"- Ts: `{case['two_x']['ts']:.2f} K`",
                f"- TOA net: `{case['two_x']['toa_net']:+.2f} W/m2`",
                f"- Surface total: `{case['two_x']['surface_total']:+.2f} W/m2`",
                f"- Column residual: `{case['two_x']['column_residual']:+.2f} W/m2`",
                f"- Precip: `{case['two_x']['precip_mm_day']:.2f} mm/day`",
                f"- Tau CAPE eff: `{case['two_x'].get('tau_cape_eff', 0.0):.0f} s`" if 'tau_cape_eff' in case['two_x'] else "- Tau CAPE eff: `n/a`",
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
            if case.get('evaluation', {}).get('sections', {}).get('two_x'):
                lines.extend([
                    "### 2x Threshold Checks",
                    "",
                ])
                for check in case['evaluation']['sections']['two_x']['checks']:
                    actual = 'missing' if check['actual'] is None else check['actual']
                    lines.append(
                        f"- `{'PASS' if check['passed'] else 'FAIL'}` "
                        f"`{check['metric']}` actual=`{actual}` target=`{check['condition']}`"
                    )
                lines.append("")
            if case.get('evaluation', {}).get('sections', {}).get('sensitivity'):
                lines.extend([
                    "### Sensitivity Threshold Checks",
                    "",
                ])
                for check in case['evaluation']['sections']['sensitivity']['checks']:
                    actual = 'missing' if check['actual'] is None else check['actual']
                    lines.append(
                        f"- `{'PASS' if check['passed'] else 'FAIL'}` "
                        f"`{check['metric']}` actual=`{actual}` target=`{check['condition']}`"
                    )
                lines.append("")
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
    spinup_cache = {}
    for case_name, case_cfg in suite_cfg.get('cases', {}).items():
        print(f"\nrunning case: {case_name}")
        case_result = run_benchmark_case(
            case_name, case_cfg, base_config, device,
            suite_path=suite_path, spinup_cache=spinup_cache,
        )
        _print_case_status(case_name, case_result.get('evaluation', {'passed': True, 'sections': {}}))
        case_results.append(case_result)

    results_path = Path(f"scm_benchmark_{label}_results.pt")
    report_path = Path(f"scm_benchmark_{label}_report.md")
    torch.save({'label': label, 'base_config': str(base_config_path), 'cases': case_results}, results_path)
    report_path.write_text(render_markdown_report(label, base_config_path, case_results))

    print(f"\nsaved benchmark results to {results_path}")
    print(f"saved benchmark report to {report_path}")


if __name__ == '__main__':
    main()
