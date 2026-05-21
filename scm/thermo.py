# thermodynamic functions and vertical grid for the single column model.
# everything operates on batched tensors, first dimension is ensemble members.

import torch

# physical constants
g = 9.81
cp = 1004.0
Rd = 287.04
Rv = 461.5
Lv = 2.5e6
sigma_sb = 5.67e-8
p0 = 1e5
kappa = Rd / cp
eps = Rd / Rv  # ~0.622
rho_water = 1000.0
c_water = 4218.0


def make_grid(nlevels=20, device='cpu', dtype=None, p_top=0.0):
    """Build the standalone SCM vertical grid.

    Levels are indexed top (0) to bottom (nlevels-1).  The historical grid is
    a pure sigma grid with ``p_top=0``.  Supplying ``p_top`` turns the same
    level distribution into a simple hybrid-pressure grid:

        p_interface = p_top * (1 - sigma_half) + sigma_half * ps

    Coupled dycores should normally use ``grid_from_hybrid_coefficients`` or
    ``grid_from_pressure_interfaces`` instead of asking the SCM to invent the
    vertical coordinate.
    """

    nlevels = int(nlevels)
    if nlevels <= 0:
        raise ValueError("nlevels must be positive")

    sigma_half_reference = torch.tensor([
        0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.27, 0.34,
        0.42, 0.5, 0.58, 0.65, 0.72, 0.78, 0.84, 0.89,
        0.93, 0.96, 0.98, 0.995, 1.0
    ], device=device, dtype=dtype)

    if nlevels == sigma_half_reference.numel() - 1:
        sigma_half = sigma_half_reference
    else:
        # Coupled dycores may request their native vertical level count.
        # Resampling the reference sigma coordinate preserves the existing
        # 20-level grid exactly while avoiding a hidden hard-coded level count.
        position = torch.linspace(
            0.0,
            float(sigma_half_reference.numel() - 1),
            nlevels + 1,
            device=device,
            dtype=sigma_half_reference.dtype,
        )
        index_lo = torch.floor(position).to(torch.long).clamp(0, sigma_half_reference.numel() - 1)
        index_hi = torch.ceil(position).to(torch.long).clamp(0, sigma_half_reference.numel() - 1)
        weight = position - index_lo.to(dtype=sigma_half_reference.dtype)
        sigma_half = (1.0 - weight) * sigma_half_reference[index_lo] + weight * sigma_half_reference[index_hi]
        sigma_half[0] = 0.0
        sigma_half[-1] = 1.0

    p_top_tensor = torch.as_tensor(float(p_top), device=device, dtype=sigma_half.dtype)
    a_half = p_top_tensor * (1.0 - sigma_half)
    b_half = sigma_half
    grid = grid_from_hybrid_coefficients(a_half, b_half, device=device, dtype=sigma_half.dtype)
    grid['p_top'] = p_top_tensor
    return grid


def grid_from_hybrid_coefficients(a_half, b_half, device='cpu', dtype=None):
    """Build a dycore-supplied hybrid grid.

    Interface pressure is defined as ``p_interface = A + B * ps``.  ``A`` and
    ``B`` may be 1-D arrays shared by all columns, or 2-D arrays with a leading
    batch/column dimension.  The returned dict keeps ``sigma_half`` and
    ``sigma_full`` as the dimensionless B coordinate so older parameterizations
    can still use level masks while pressure and layer mass come from the
    dycore-owned vertical coordinate.
    """

    a_half = torch.as_tensor(a_half, device=device, dtype=dtype)
    b_half = torch.as_tensor(b_half, device=device, dtype=dtype)
    if a_half.shape != b_half.shape:
        raise ValueError("hybrid A and B coefficients must have the same shape")
    if a_half.ndim not in (1, 2):
        raise ValueError("hybrid coefficients must be 1-D or 2-D")
    if a_half.shape[-1] < 2:
        raise ValueError("hybrid coefficients need at least two interfaces")

    a_full = 0.5 * (a_half[..., :-1] + a_half[..., 1:])
    b_full = 0.5 * (b_half[..., :-1] + b_half[..., 1:])
    dsigma = b_half[..., 1:] - b_half[..., :-1]
    return {
        'hybrid_a_half': a_half,
        'hybrid_b_half': b_half,
        'hybrid_a_full': a_full,
        'hybrid_b_full': b_full,
        'sigma_half': b_half,
        'sigma_full': b_full,
        'dsigma': dsigma,
        'nlevels': int(a_half.shape[-1] - 1),
    }


def grid_from_pressure_interfaces(p_interface, device='cpu', dtype=None):
    """Build a grid from externally supplied pressure interfaces.

    This is useful when the host model already knows the pressure at every
    interface for each column.  ``p_interface`` may be shape ``(nlevels + 1,)``
    or ``(batch, nlevels + 1)`` and must be ordered top to bottom.
    """

    p_half = torch.as_tensor(p_interface, device=device, dtype=dtype)
    if p_half.ndim not in (1, 2):
        raise ValueError("pressure interfaces must be 1-D or 2-D")
    if p_half.shape[-1] < 2:
        raise ValueError("pressure interfaces need at least two levels")

    p_full = 0.5 * (p_half[..., :-1] + p_half[..., 1:])
    dp = p_half[..., 1:] - p_half[..., :-1]
    span = (p_half[..., -1:] - p_half[..., :1]).clamp(min=torch.finfo(p_half.dtype).tiny)
    sigma_half = (p_half - p_half[..., :1]) / span
    sigma_full = 0.5 * (sigma_half[..., :-1] + sigma_half[..., 1:])
    return {
        'p_interface': p_half,
        'p_full': p_full,
        'dp': dp,
        'sigma_half': sigma_half,
        'sigma_full': sigma_full,
        'dsigma': sigma_half[..., 1:] - sigma_half[..., :-1],
        'nlevels': int(p_half.shape[-1] - 1),
    }


def _batch_size_from_ps(ps):
    if ps is None:
        return None
    return int(ps.reshape(-1).shape[0])


def _grid_tensor(value, *, ps=None, batch=None, device=None, dtype=None, name='grid tensor'):
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if ps is not None:
        ps = ps.reshape(-1)
        batch = _batch_size_from_ps(ps)
        device = ps.device
        dtype = ps.dtype
        tensor = tensor.to(device=device, dtype=dtype)
    elif device is not None or dtype is not None:
        tensor = tensor.to(device=device or tensor.device, dtype=dtype or tensor.dtype)

    if tensor.ndim == 1:
        return tensor if batch is None else tensor.unsqueeze(0).expand(batch, -1)
    if tensor.ndim == 2:
        if batch is None:
            return tensor
        if tensor.shape[0] == batch:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch, -1)
    raise ValueError(f"{name} with shape {tuple(tensor.shape)} cannot broadcast to batch={batch}")


def pressure_at_full(grid, ps):
    """pressure at full levels. ps is (batch,), returns (batch, nlevels)."""
    if 'p_full' in grid:
        return _grid_tensor(grid['p_full'], ps=ps, name='p_full')
    if 'hybrid_a_full' in grid and 'hybrid_b_full' in grid:
        a = _grid_tensor(grid['hybrid_a_full'], ps=ps, name='hybrid_a_full')
        b = _grid_tensor(grid['hybrid_b_full'], ps=ps, name='hybrid_b_full')
        return a + b * ps.reshape(-1, 1)
    half = pressure_at_half(grid, ps)
    return 0.5 * (half[:, :-1] + half[:, 1:])


def pressure_at_half(grid, ps):
    """pressure at half levels (interfaces). returns (batch, nlevels+1)."""
    if 'p_interface' in grid:
        return _grid_tensor(grid['p_interface'], ps=ps, name='p_interface')
    if 'hybrid_a_half' in grid and 'hybrid_b_half' in grid:
        a = _grid_tensor(grid['hybrid_a_half'], ps=ps, name='hybrid_a_half')
        b = _grid_tensor(grid['hybrid_b_half'], ps=ps, name='hybrid_b_half')
        return a + b * ps.reshape(-1, 1)
    sigma_half = _grid_tensor(grid['sigma_half'], ps=ps, name='sigma_half')
    return sigma_half * ps.reshape(-1, 1)


def dp_from_ps(grid, ps):
    """pressure thickness of each layer. returns (batch, nlevels)."""
    if 'dp' in grid:
        return _grid_tensor(grid['dp'], ps=ps, name='dp')
    half = pressure_at_half(grid, ps)
    return half[:, 1:] - half[:, :-1]


def full_level_coordinate(grid, *, state=None, ps=None, batch=None, device=None, dtype=None):
    """Return the dimensionless full-level coordinate as ``(batch, nlevels)``.

    Older physics closures use sigma-like level masks.  This helper keeps those
    masks working for standalone sigma grids, hybrid dycore grids, and explicit
    pressure-interface grids.
    """

    if state is not None:
        ref = state['t']
        batch = int(ref.shape[0])
        device = ref.device
        dtype = ref.dtype
        if ps is None and 'ps' in state:
            ps = state['ps']
    return _grid_tensor(grid['sigma_full'], ps=ps, batch=batch, device=device, dtype=dtype, name='sigma_full')


def half_level_coordinate(grid, *, state=None, ps=None, batch=None, device=None, dtype=None):
    """Return the dimensionless interface coordinate as ``(batch, nlevels + 1)``."""

    if state is not None:
        ref = state['t']
        batch = int(ref.shape[0])
        device = ref.device
        dtype = ref.dtype
        if ps is None and 'ps' in state:
            ps = state['ps']
    return _grid_tensor(grid['sigma_half'], ps=ps, batch=batch, device=device, dtype=dtype, name='sigma_half')


def saturation_vapor_pressure(t):
    """bolton formula for saturation vapor pressure in Pa."""
    tc = t - 273.15
    return 611.2 * torch.exp(17.67 * tc / (tc + 243.5))


def saturation_specific_humidity(t, p):
    """saturation specific humidity in kg/kg."""
    es = saturation_vapor_pressure(t)
    return eps * es / (p - (1.0 - eps) * es).clamp(min=1.0)


def relative_humidity(q, t, p):
    """relative humidity as a fraction."""
    qs = saturation_specific_humidity(t, p)
    return q / qs.clamp(min=1e-10)


def virtual_temperature(t, q):
    return t * (1.0 + (1.0 / eps - 1.0) * q.clamp(min=0.0))


def moist_adiabatic_lapse_rate(t, p):
    """dT/dp along a moist adiabat, in K/Pa. positive means T decreases
    as p decreases (going up)."""
    qs = saturation_specific_humidity(t, p)
    num = (Rd * t / p) * (1.0 + Lv * qs / (Rd * t))
    den = cp + Lv * Lv * qs / (Rv * t * t)
    return num / den


def moist_adiabat_profile(t_base, p_base, p_levels):
    """integrate a moist adiabat upward from (t_base, p_base) to each p_level.
    t_base and p_base are (batch,), p_levels is (batch, nlevels).
    returns (batch, nlevels). assumes levels are ordered top to bottom."""

    batch = t_base.shape[0]
    nlevels = p_levels.shape[1]
    t_adiabat = torch.zeros_like(p_levels)

    # start at the bottom and integrate upward
    t_current = t_base.clone()
    p_current = p_base.clone()

    for k in range(nlevels - 1, -1, -1):
        p_target = p_levels[:, k]
        dp = p_target - p_current

        # midpoint method for accuracy
        gamma = moist_adiabatic_lapse_rate(t_current, p_current)
        t_mid = t_current + gamma * dp * 0.5
        p_mid = 0.5 * (p_current + p_target)
        gamma_mid = moist_adiabatic_lapse_rate(t_mid, p_mid.clamp(min=100.0))
        t_current = t_current + gamma_mid * dp
        p_current = p_target

        t_adiabat[:, k] = t_current

    return t_adiabat


def cape(t, q, p, grid):
    """undilute CAPE from the lowest model level. returns (batch,) in J/kg.
    uses virtual temperature for buoyancy."""

    nlevels = t.shape[1]

    # lift parcel from the lowest level
    t_parcel = t[:, -1].clone()
    q_parcel = q[:, -1].clone()
    p_parcel = p[:, -1].clone()

    cape_val = torch.zeros(t.shape[0], device=t.device)

    for k in range(nlevels - 2, -1, -1):
        p_target = p[:, k]
        dp = p_target - p_parcel

        qs_parcel = saturation_specific_humidity(t_parcel, p_parcel)
        saturated = (q_parcel >= qs_parcel).float()

        gamma_dry = Rd * t_parcel / (cp * p_parcel)
        gamma_moist = moist_adiabatic_lapse_rate(t_parcel, p_parcel)
        gamma = (1.0 - saturated) * gamma_dry + saturated * gamma_moist

        t_parcel = t_parcel + gamma * dp
        p_parcel = p_target

        # condense excess moisture
        qs_new = saturation_specific_humidity(t_parcel, p_target)
        q_parcel = torch.min(q_parcel, qs_new)

        # virtual temperature buoyancy
        tv_parcel = virtual_temperature(t_parcel, q_parcel)
        tv_env = virtual_temperature(t[:, k], q[:, k])
        buoyancy = torch.clamp((tv_parcel - tv_env) / tv_env, min=0.0)

        dlnp = torch.log(p[:, k + 1] / p[:, k].clamp(min=1.0))
        cape_val = cape_val + Rd * tv_env * buoyancy * dlnp

    return cape_val


def geopotential(t, q, p, grid):
    """geopotential height at each full level by integrating the hypsometric
    equation upward from the surface. returns (batch, nlevels) in meters."""

    nlevels = t.shape[1]
    tv = virtual_temperature(t, q)
    z = torch.zeros_like(t)

    for k in range(nlevels - 2, -1, -1):
        dlnp = torch.log(p[:, k + 1] / p[:, k].clamp(min=1.0))
        tv_mean = 0.5 * (tv[:, k] + tv[:, k + 1])
        z[:, k] = z[:, k + 1] + Rd * tv_mean * dlnp / g

    return z
