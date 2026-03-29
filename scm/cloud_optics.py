import torch


def as_batch_tensor(x, batch, device, dtype):
    """Broadcast scalar or batch-like input to shape (batch,)."""

    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.dim() == 0:
        return t.expand(batch)
    if t.dim() == 1:
        if t.shape[0] == 1:
            return t.expand(batch)
        if t.shape[0] == batch:
            return t
    raise ValueError(f"cannot broadcast value with shape {tuple(t.shape)} to batch={batch}")


def available_cloud_optics_schemes():
    return [
        "auto",
        "microphysics",
        "microphysics_linear",
        "prescribed",
        "prescribed_gaussian",
        "clear_sky",
    ]


def clouds_enabled(params):
    mode = params.get("radiation_mode", "semi_gray")
    return bool(params.get("cloud_microphysics_enabled", False)) or bool(
        params.get("cloud_radiative_effects_enabled", False)
    ) or (
        mode == "semi_gray_plus_clouds"
        or mode == "semi_gray_plus_trace_gases_clouds"
    )


def cloud_layer_weights(grid, batch, device, dtype, params):
    """Return normalized cloud-layer weights on model full levels."""

    sigma = grid["sigma_full"].to(device=device, dtype=dtype)
    top = as_batch_tensor(params.get("cloud_top_sigma", 0.65), batch, device, dtype)
    bottom = as_batch_tensor(params.get("cloud_bottom_sigma", 0.95), batch, device, dtype)
    sigma_2d = sigma.unsqueeze(0).expand(batch, -1)
    mask = ((sigma_2d >= top.unsqueeze(1)) & (sigma_2d <= bottom.unsqueeze(1))).to(dtype)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return mask / counts


def _microphysics_cloud_optics(state, params, batch, dtype, linear=False):
    device = state["t"].device
    nlevels = state["t"].shape[1]

    cf = state.get("cloud_fraction", torch.zeros(batch, nlevels, device=device, dtype=dtype))
    sw_tau_layer = state.get("cloud_sw_tau_layer", torch.zeros(batch, nlevels, device=device, dtype=dtype))
    lw_tau_layer = state.get("cloud_lw_tau_layer", torch.zeros(batch, nlevels, device=device, dtype=dtype))
    cf = cf.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)
    sw_tau_layer = sw_tau_layer.to(device=device, dtype=dtype)
    lw_tau_layer = lw_tau_layer.to(device=device, dtype=dtype)

    cloud_cover = 1.0 - torch.prod(1.0 - cf, dim=1)
    scatter_eff = as_batch_tensor(
        params.get("cloud_sw_scattering_efficiency", 0.18),
        batch, device, dtype
    )
    total_sw_tau = sw_tau_layer.sum(dim=1)
    if linear:
        reflectivity = cloud_cover * scatter_eff * total_sw_tau
    else:
        reflectivity = cloud_cover * (1.0 - torch.exp(-scatter_eff * total_sw_tau))
    abs_frac = as_batch_tensor(
        params.get("cloud_sw_absorption_fraction", 0.15),
        batch, device, dtype
    )
    return reflectivity.clamp(min=0.0, max=0.95), abs_frac.unsqueeze(1) * sw_tau_layer, lw_tau_layer


def _prescribed_cloud_optics(grid, params, batch, dtype, gaussian=False):
    device = grid["sigma_full"].device
    cloud_reflectivity = (
        as_batch_tensor(params.get("cloud_fraction", 0.0), batch, device, dtype).clamp(min=0.0, max=1.0)
        * as_batch_tensor(params.get("cloud_sw_reflectivity", 0.0), batch, device, dtype)
    ).clamp(min=0.0, max=0.95)
    cloud_sw_tau_total = (
        as_batch_tensor(params.get("cloud_fraction", 0.0), batch, device, dtype).clamp(min=0.0, max=1.0)
        * as_batch_tensor(params.get("cloud_sw_tau", 0.0), batch, device, dtype).clamp(min=0.0)
    )
    cloud_lw_tau_total = (
        as_batch_tensor(params.get("cloud_fraction", 0.0), batch, device, dtype).clamp(min=0.0, max=1.0)
        * as_batch_tensor(params.get("cloud_lw_tau", 0.0), batch, device, dtype).clamp(min=0.0)
    )
    if gaussian:
        sigma = grid["sigma_full"].to(device=device, dtype=dtype)
        top = as_batch_tensor(params.get("cloud_top_sigma", 0.65), batch, device, dtype)
        bottom = as_batch_tensor(params.get("cloud_bottom_sigma", 0.95), batch, device, dtype)
        center = 0.5 * (top + bottom)
        width = (0.35 * (bottom - top)).clamp(min=0.03)
        sigma_2d = sigma.unsqueeze(0).expand(batch, -1)
        mask = ((sigma_2d >= top.unsqueeze(1)) & (sigma_2d <= bottom.unsqueeze(1))).to(dtype)
        weights = torch.exp(-0.5 * ((sigma_2d - center.unsqueeze(1)) / width.unsqueeze(1)) ** 2) * mask
        fallback = weights.sum(dim=1, keepdim=True) <= 0.0
        if fallback.any():
            uniform = cloud_layer_weights(grid, batch, device, dtype, params)
            weights = torch.where(fallback.expand_as(weights), uniform, weights)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0e-8)
    else:
        weights = cloud_layer_weights(grid, batch, device, dtype, params)
    return (
        cloud_reflectivity,
        cloud_sw_tau_total.unsqueeze(1) * weights,
        cloud_lw_tau_total.unsqueeze(1) * weights,
    )


def cloud_optical_properties(state, grid, params, batch, dtype, force_clear_sky=False):
    """Return cloud SW reflectivity, SW absorption tau layer, and LW tau layer."""

    if force_clear_sky:
        scheme = "clear_sky"
    else:
        scheme = str(params.get("cloud_optics_scheme", "auto"))

    if scheme == "auto":
        if params.get("cloud_microphysics_enabled", False):
            scheme = "microphysics"
        elif clouds_enabled(params):
            scheme = "prescribed"
        else:
            scheme = "clear_sky"

    if scheme == "clear_sky":
        device = state["t"].device
        nlevels = state["t"].shape[1]
        zeros_layer = torch.zeros(batch, nlevels, device=device, dtype=dtype)
        zeros_col = torch.zeros(batch, device=device, dtype=dtype)
        return zeros_col, zeros_layer, zeros_layer
    if scheme == "microphysics":
        return _microphysics_cloud_optics(state, params, batch, dtype)
    if scheme == "microphysics_linear":
        return _microphysics_cloud_optics(state, params, batch, dtype, linear=True)
    if scheme == "prescribed":
        return _prescribed_cloud_optics(grid, params, batch, dtype)
    if scheme == "prescribed_gaussian":
        return _prescribed_cloud_optics(grid, params, batch, dtype, gaussian=True)

    valid = ", ".join(available_cloud_optics_schemes())
    raise ValueError(f"unknown cloud optics scheme: {scheme} (valid: {valid})")
