import torch

from scm.thermo import sigma_sb, g, cp


# diffusivity factor for converting vertical optical depth to effective
# path length through the atmosphere (accounts for non-vertical photon paths)
mu_diff = 1.66


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


def trace_gases_enabled(params):
    mode = params.get("radiation_mode", "semi_gray")
    return bool(params.get("trace_gases_enabled", False)) or mode == "semi_gray_plus_trace_gases"


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


def forward_flux_sweep(transmissivity, source, boundary):
    """Vectorized solution of y[k+1] = y[k] * transmissivity[k] + source[k]."""

    batch = transmissivity.shape[0]
    one = torch.ones(batch, 1, device=transmissivity.device, dtype=transmissivity.dtype)
    zero = torch.zeros(batch, 1, device=transmissivity.device, dtype=transmissivity.dtype)
    prefix = torch.cat([one, torch.cumprod(transmissivity, dim=1)], dim=1)
    scaled_source = source / prefix[:, 1:].clamp(min=1.0e-12)
    accum = torch.cumsum(scaled_source, dim=1)
    return prefix * (boundary.unsqueeze(1) + torch.cat([zero, accum], dim=1))


def band_vector(values, default, device, dtype):
    v = torch.as_tensor(default if values is None else values, device=device, dtype=dtype)
    if v.dim() == 0:
        return v.unsqueeze(0)
    if v.dim() != 1:
        raise ValueError(f"band parameters must be 1D, got shape {tuple(v.shape)}")
    return v


def cloud_radiative_properties(state, grid, params, batch, dtype):
    device = state["t"].device
    nlevels = state["t"].shape[1]

    if params.get("cloud_microphysics_enabled", False):
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
        reflectivity = cloud_cover * (1.0 - torch.exp(-scatter_eff * sw_tau_layer.sum(dim=1)))
        abs_frac = as_batch_tensor(
            params.get("cloud_sw_absorption_fraction", 0.15),
            batch, device, dtype
        )
        return reflectivity.clamp(min=0.0, max=0.95), abs_frac.unsqueeze(1) * sw_tau_layer, lw_tau_layer

    if not clouds_enabled(params):
        zeros_layer = torch.zeros(batch, nlevels, device=device, dtype=dtype)
        zeros_col = torch.zeros(batch, device=device, dtype=dtype)
        return zeros_col, zeros_layer, zeros_layer

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
    weights = cloud_layer_weights(grid, batch, device, dtype, params)
    return (
        cloud_reflectivity,
        cloud_sw_tau_total.unsqueeze(1) * weights,
        cloud_lw_tau_total.unsqueeze(1) * weights,
    )


def trace_total_tau(batch, device, dtype, params):
    if not trace_gases_enabled(params):
        return torch.zeros(batch, 1, device=device, dtype=dtype)

    def to_col(x):
        return as_batch_tensor(x, batch, device, dtype).unsqueeze(1)

    ch4_ratio = to_col(params.get("ch4", 1.8)) / to_col(params.get("ch4_ref", 1.8)).clamp(min=1.0e-6)
    n2o_ratio = to_col(params.get("n2o", 0.332)) / to_col(params.get("n2o_ref", 0.332)).clamp(min=1.0e-6)

    ch4_total_tau = (
        to_col(params.get("ch4_base_tau", 0.0))
        + to_col(params.get("ch4_log_factor", 0.0))
        * torch.log(ch4_ratio.clamp(min=0.01))
    )
    n2o_total_tau = (
        to_col(params.get("n2o_base_tau", 0.0))
        + to_col(params.get("n2o_log_factor", 0.0))
        * torch.log(n2o_ratio.clamp(min=0.01))
    )
    o3_lw_tau = to_col(params.get("o3_lw_tau", 0.0))
    other_ghg_tau = to_col(params.get("other_ghg_tau", 0.0))
    return ch4_total_tau + n2o_total_tau + o3_lw_tau + other_ghg_tau


__all__ = [
    "as_batch_tensor",
    "band_vector",
    "cloud_radiative_properties",
    "clouds_enabled",
    "cp",
    "forward_flux_sweep",
    "g",
    "mu_diff",
    "sigma_sb",
    "trace_gases_enabled",
    "trace_total_tau",
]
