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


def make_grid(nlevels=20, device='cpu'):
    """sigma coordinate grid with levels indexed top (0) to bottom (nlevels-1).
    returns half levels (interfaces), full levels (midpoints), and layer thicknesses.
    boundary layer gets extra resolution."""

    sigma_half = torch.tensor([
        0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.27, 0.34,
        0.42, 0.5, 0.58, 0.65, 0.72, 0.78, 0.84, 0.89,
        0.93, 0.96, 0.98, 0.995, 1.0
    ], device=device)

    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    dsigma = sigma_half[1:] - sigma_half[:-1]

    return {
        'sigma_half': sigma_half,
        'sigma_full': sigma_full,
        'dsigma': dsigma,
        'nlevels': nlevels,
    }


def pressure_at_full(grid, ps):
    """pressure at full levels. ps is (batch,), returns (batch, nlevels)."""
    return grid['sigma_full'].unsqueeze(0) * ps.unsqueeze(1)


def pressure_at_half(grid, ps):
    """pressure at half levels (interfaces). returns (batch, nlevels+1)."""
    return grid['sigma_half'].unsqueeze(0) * ps.unsqueeze(1)


def dp_from_ps(grid, ps):
    """pressure thickness of each layer. returns (batch, nlevels)."""
    return grid['dsigma'].unsqueeze(0) * ps.unsqueeze(1)


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
