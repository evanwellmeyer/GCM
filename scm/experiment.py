from pathlib import Path

import torch


def build_output_stem(mode, scheme, sampling, fixed_sst, spinup_days,
                      perturb_days, label=''):
    sst_mode = 'fixedsst' if fixed_sst else 'slabocean'
    stem = (
        f"scm_{mode}_{scheme}_{sampling}_{sst_mode}_"
        f"spin{spinup_days}d_pert{perturb_days}d"
    )
    if label:
        stem = f"{stem}_{label}"
    return stem


def apply_param_overrides(params, overrides, n_total, device):
    """Apply scalar or per-member parameter overrides from config."""

    for key, value in overrides.items():
        if key in params and isinstance(params[key], torch.Tensor):
            target = params[key]
            if target.dim() == 1 and target.shape[0] == n_total:
                if isinstance(value, list):
                    tensor = torch.tensor(value, dtype=target.dtype, device=device)
                    if tensor.shape != target.shape:
                        raise ValueError(
                            f"override for {key} must have shape {tuple(target.shape)}, "
                            f"got {tuple(tensor.shape)}"
                        )
                    params[key] = tensor
                else:
                    params[key] = torch.full(
                        target.shape, value, dtype=target.dtype, device=device
                    )
                continue
        params[key] = value


def member_counts(mode, scheme, fixed_params=False, preserve_ensemble_shape=False):
    if mode == 'full' and fixed_params and not preserve_ensemble_shape:
        if scheme == 'mixed':
            return 1, 1
        if scheme == 'bm':
            return 1, 0
        return 0, 1

    if scheme == 'mixed':
        return (5, 5) if mode == 'demo' else (50, 50)
    if scheme == 'bm':
        return (10, 0) if mode == 'demo' else (100, 0)
    return (0, 10) if mode == 'demo' else (0, 100)


def build_calibration_output_stem(scheme, fixed_sst, spinup_days, ncases, label=''):
    sst_mode = 'fixedsst' if fixed_sst else 'slabocean'
    stem = f"scm_radiation_calibration_{scheme}_{sst_mode}_spin{spinup_days}d_{ncases}cases"
    if label:
        stem = f"{stem}_{label}"
    return stem


def move_tensors(obj, device):
    """Recursively move tensors in nested containers to a device."""

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_tensors(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_tensors(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_tensors(v, device) for v in obj)
    return obj


def cpu_tensors(obj):
    """Recursively detach tensors to CPU for serialization."""

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: cpu_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tensors(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tensors(v) for v in obj)
    return obj


def build_restart_path(output_stem, phase):
    """Return the restart filename for a run phase."""

    if phase not in ('1x', '2x'):
        raise ValueError(f"restart phase must be '1x' or '2x', got {phase}")
    return Path(f"{output_stem}_{phase}_restart.pt")


def save_restart_bundle(path, bundle):
    """Save a restart bundle with tensors moved to CPU."""

    save_path = Path(path)
    torch.save(cpu_tensors(bundle), save_path)
    return save_path


def load_restart_bundle(path, device):
    """Load a restart bundle and move tensors to the requested device."""

    bundle = torch.load(Path(path), map_location='cpu', weights_only=False)
    return move_tensors(bundle, device)
