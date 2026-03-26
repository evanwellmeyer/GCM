import itertools
import time

import torch

from scm.column_model import initial_state, run
from scm.configuration import extract_param_overrides
from scm.diagnostics import equilibrium_stats
from scm.ensemble import make_fixed_ensemble_params
from scm.experiment import (
    apply_param_overrides,
    build_calibration_output_stem,
    member_counts,
)
from scm.thermo import make_grid


DEFAULT_TARGETS = {
    'ts': 300.0,
    'olr': 240.0,
    'asr': 240.0,
    'toa_net': 0.0,
    'surface_net_flux': 0.0,
}

DEFAULT_SCALES = {
    'ts': 5.0,
    'olr': 20.0,
    'asr': 20.0,
    'toa_net': 5.0,
    'surface_net_flux': 10.0,
}


def iter_parameter_grid(parameter_grid):
    """Yield candidate override dicts from a TOML parameter grid."""

    if not parameter_grid:
        yield {}
        return

    names = list(parameter_grid.keys())
    values = []
    for name in names:
        candidates = parameter_grid[name]
        if not isinstance(candidates, list) or len(candidates) == 0:
            raise ValueError(f"calibration.parameter_grid.{name} must be a non-empty list")
        values.append(candidates)

    for combo in itertools.product(*values):
        yield dict(zip(names, combo))


def calibration_score(metrics, targets=None, scales=None):
    """Score a calibration candidate. Lower is better."""

    targets = {**DEFAULT_TARGETS, **(targets or {})}
    scales = {**DEFAULT_SCALES, **(scales or {})}
    score = 0.0

    for key in ['toa_net', 'surface_net_flux', 'olr', 'asr', 'ts']:
        if key not in metrics:
            continue
        scale = max(float(scales.get(key, 1.0)), 1e-6)
        target = float(targets.get(key, 0.0))
        score += abs(float(metrics[key]) - target) / scale

    return score


def _format_scalar(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return repr(value)


def format_candidate(candidate):
    if not candidate:
        return 'baseline'
    return ', '.join(f"{key}={_format_scalar(value)}" for key, value in candidate.items())


def candidate_toml_snippet(candidate):
    """Return a TOML snippet for the best candidate overrides."""

    sections = {
        'radiation.longwave': [],
        'radiation.shortwave': [],
        'radiation.trace_gases': [],
        'radiation.clouds': [],
        'params': [],
    }

    for key, value in candidate.items():
        if key in {'f_window', 'kappa_wv', 'co2_log_factor', 'co2_base_tau', 'o3_lw_tau', 'other_ghg_tau'}:
            section = 'radiation.longwave'
        elif key in {'sw_kappa_wv', 'albedo', 'o3_sw_tau'}:
            section = 'radiation.shortwave'
        elif key.startswith('cloud_'):
            section = 'radiation.clouds'
        elif key.startswith('ch4') or key.startswith('n2o'):
            section = 'radiation.trace_gases'
        else:
            section = 'params'
        sections[section].append(f"{key} = {_format_scalar(value)}")

    lines = []
    for section, entries in sections.items():
        if not entries:
            continue
        lines.append(f"[{section}]")
        lines.extend(entries)
        lines.append("")

    return '\n'.join(lines).rstrip()


def _equilibrium_flags(diag_history, window=50, ts_threshold=0.1, toa_threshold=1.0):
    """Per-member equilibrium flags over the requested window."""

    batch = diag_history[-1]['ts'].shape[0]
    if window < 2 or len(diag_history) < window:
        return torch.zeros(batch, dtype=torch.bool)

    recent = diag_history[-window:]
    recent_ts = torch.stack([d['ts'] for d in recent])
    x = torch.arange(window, dtype=recent_ts.dtype, device=recent_ts.device)
    x = x - x.mean()
    slopes = (recent_ts * x.unsqueeze(1)).sum(dim=0) / (x * x).sum()
    ts_ok = slopes.abs() < ts_threshold

    if 'toa_net' not in recent[-1]:
        return ts_ok.cpu()

    recent_toa = torch.stack([d['toa_net'] for d in recent])
    toa_ok = recent_toa.mean(dim=0).abs() < toa_threshold
    return (ts_ok & toa_ok).cpu()


def _candidate_member_indices(n_candidates, n_bm_per_candidate, n_mf_per_candidate):
    """Return member indices for each candidate within a batched chunk."""

    member_indices = []
    bm_block = n_candidates * n_bm_per_candidate

    for idx in range(n_candidates):
        indices = []
        if n_bm_per_candidate:
            start = idx * n_bm_per_candidate
            indices.extend(range(start, start + n_bm_per_candidate))
        if n_mf_per_candidate:
            start = bm_block + idx * n_mf_per_candidate
            indices.extend(range(start, start + n_mf_per_candidate))
        member_indices.append(indices)

    return member_indices


def _candidate_overrides_for_chunk(candidates, n_bm_per_candidate, n_mf_per_candidate):
    """Expand candidate values into per-member override lists for a chunk."""

    if not candidates or not candidates[0]:
        return {}

    override_keys = list(candidates[0].keys())
    overrides = {}

    for key in override_keys:
        values = []
        if n_bm_per_candidate:
            for candidate in candidates:
                values.extend([candidate[key]] * n_bm_per_candidate)
        if n_mf_per_candidate:
            for candidate in candidates:
                values.extend([candidate[key]] * n_mf_per_candidate)
        overrides[key] = values

    return overrides


def run_radiation_calibration(config, device):
    """Run short fixed-parameter control integrations over a radiation grid."""

    run_cfg = config.get('run', {})
    numerics_cfg = config.get('numerics', {})
    initial_cfg = config.get('initial', {})
    forcing_cfg = config.get('forcing', {})
    calibration_cfg = config.get('calibration', {})
    param_overrides = extract_param_overrides(config)

    scheme = calibration_cfg.get('scheme', run_cfg.get('scheme', 'mf'))
    fixed_sst = bool(calibration_cfg.get('fixed_sst', run_cfg.get('fixed_sst', False)))
    preserve_shape = bool(calibration_cfg.get('preserve_ensemble_shape', False))
    spinup_days = int(calibration_cfg.get('spinup_days', run_cfg.get('spinup_days', 200)))
    last_n = int(calibration_cfg.get('last_n', 20))
    eq_window = int(calibration_cfg.get('equilibrium_window', last_n))
    top_n = int(calibration_cfg.get('top_n', 5))
    diag_interval_days = int(
        calibration_cfg.get('diag_interval_days', numerics_cfg.get('diag_interval_days', 10))
    )
    rad_interval_steps = int(
        calibration_cfg.get('rad_interval_steps', numerics_cfg.get('rad_interval_steps', 8))
    )
    ts_threshold = float(calibration_cfg.get('ts_threshold', 0.1))
    toa_threshold = float(calibration_cfg.get('toa_threshold', 1.0))
    label = calibration_cfg.get('label', run_cfg.get('label', ''))
    batch_candidates = bool(calibration_cfg.get('batch_candidates', True))
    batch_size = int(calibration_cfg.get('batch_size', 16))
    if not batch_candidates:
        batch_size = 1
    batch_size = max(1, batch_size)

    n_bm_per_candidate, n_mf_per_candidate = member_counts(
        'full', scheme, fixed_params=True, preserve_ensemble_shape=preserve_shape
    )
    dt = float(numerics_cfg.get('dt', 900.0))
    steps_per_day = int(86400 / dt)
    spinup_steps = spinup_days * steps_per_day
    diag_every = steps_per_day * diag_interval_days

    grid = make_grid(nlevels=int(numerics_cfg.get('nlevels', 20)), device=device)

    base_params = {
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

    targets = {**DEFAULT_TARGETS, **calibration_cfg.get('targets', {})}
    scales = {**DEFAULT_SCALES, **calibration_cfg.get('scales', {})}
    candidates = list(iter_parameter_grid(calibration_cfg.get('parameter_grid', {})))

    print(
        f"\nradiation calibration: {len(candidates)} candidates, "
        f"scheme={scheme}, {spinup_days}-day control runs, "
        f"batch_size={batch_size}"
    )

    results = []
    t0_all = time.time()

    for chunk_start in range(0, len(candidates), batch_size):
        chunk = candidates[chunk_start:chunk_start + batch_size]
        chunk_n_bm = len(chunk) * n_bm_per_candidate
        chunk_n_mf = len(chunk) * n_mf_per_candidate
        chunk_n_total = chunk_n_bm + chunk_n_mf

        params = make_fixed_ensemble_params(
            chunk_n_bm, chunk_n_mf, base_params=base_params, device=device
        )
        apply_param_overrides(params, param_overrides, chunk_n_total, device)
        apply_param_overrides(
            params,
            _candidate_overrides_for_chunk(chunk, n_bm_per_candidate, n_mf_per_candidate),
            chunk_n_total,
            device,
        )
        params['use_slab_ocean'] = not fixed_sst

        state = initial_state(chunk_n_total, grid, params, device=device)
        t0 = time.time()
        state, history = run(
            state, grid, params, spinup_steps,
            rad_interval=rad_interval_steps,
            diag_interval=diag_every,
            callback=None,
        )
        elapsed = time.time() - t0
        stats = equilibrium_stats(history, last_n=last_n)
        eq_flags = _equilibrium_flags(
            history,
            window=min(eq_window, len(history)),
            ts_threshold=ts_threshold,
            toa_threshold=toa_threshold,
        )
        member_groups = _candidate_member_indices(
            len(chunk), n_bm_per_candidate, n_mf_per_candidate
        )

        for local_idx, candidate in enumerate(chunk):
            member_idx = member_groups[local_idx]
            idx = chunk_start + local_idx + 1
            metrics = {
                'ts': float(stats['ts_mean'][member_idx].mean().item()),
                'olr': float(stats['olr_mean'][member_idx].mean().item()),
                'asr': float(stats['asr_mean'][member_idx].mean().item()),
                'toa_net': float(stats['toa_net_mean'][member_idx].mean().item()),
                'surface_net_flux': float(stats['surface_net_flux_mean'][member_idx].mean().item()),
                'precip': float((stats['precip_total_mean'][member_idx] * 86400.0).mean().item()),
            }
            eq = bool(eq_flags[member_idx].all().item())
            score = calibration_score(metrics, targets=targets, scales=scales)

            result = {
                'index': idx,
                'candidate': candidate,
                'metrics': metrics,
                'score': score,
                'equilibrium': eq,
                'elapsed_s': elapsed / max(1, len(chunk)),
            }
            results.append(result)

            print(
                f"  [{idx:02d}/{len(candidates):02d}] score={score:5.2f}  "
                f"TOA={metrics['toa_net']:+6.2f}  SFC={metrics['surface_net_flux']:+6.2f}  "
                f"Ts={metrics['ts']:6.2f}  OLR={metrics['olr']:6.1f}  "
                f"ASR={metrics['asr']:6.1f}  P={metrics['precip']:4.2f}  "
                f"eq={'PASS' if eq else 'NO'}  {format_candidate(candidate)}"
            )

    ranked = sorted(results, key=lambda item: item['score'])
    elapsed_all = time.time() - t0_all

    print(f"\ncalibration ranking (top {min(top_n, len(ranked))}):")
    for rank, result in enumerate(ranked[:top_n], start=1):
        metrics = result['metrics']
        print(
            f"  {rank}. score={result['score']:.2f}  TOA={metrics['toa_net']:+.2f}  "
            f"SFC={metrics['surface_net_flux']:+.2f}  Ts={metrics['ts']:.2f}  "
            f"OLR={metrics['olr']:.1f}  ASR={metrics['asr']:.1f}  "
            f"eq={'PASS' if result['equilibrium'] else 'NO'}"
        )
        print(f"     {format_candidate(result['candidate'])}")

    best = ranked[0]
    best_snippet = candidate_toml_snippet(best['candidate'])
    if best_snippet:
        print("\nbest candidate TOML overrides:")
        print(best_snippet)

    output_stem = build_calibration_output_stem(
        scheme, fixed_sst, spinup_days, len(candidates), label=label
    )
    output_path = f"{output_stem}_results.pt"
    torch.save(
        {
            'config': config,
            'scheme': scheme,
            'targets': targets,
            'scales': scales,
            'batch_candidates': batch_candidates,
            'batch_size': batch_size,
            'results': ranked,
            'best': best,
            'best_toml_snippet': best_snippet,
            'elapsed_s': elapsed_all,
        },
        output_path,
    )

    print(f"\ncalibration done in {elapsed_all:.1f} s")
    print(f"results saved to {output_path}")

    return ranked
