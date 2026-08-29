import torch

from scm.boundary_layer import _as_batch_tensor, diagnose_boundary_layer_depth, tridiag_solve
from scm.thermo import Lv, Rd, cp, g, geopotential, kappa, p0, virtual_temperature


def tke_boundary_layer(state, grid, params):
    """One-equation TKE closure with flux-form scalar transport."""

    t = state['t']
    q = state['q']
    qc = state.get('qc', torch.zeros_like(q))
    p = state['p']
    dp = state['dp']
    u = state.get('u', torch.zeros_like(t))
    v = state.get('v', torch.zeros_like(t))
    timestep = float(params.get('dt', 60.0))
    batch, levels = t.shape
    mass = dp / g
    height = geopotential(t, q, p, grid)

    tke = state.get('tke', torch.full_like(t, float(params.get('tke_initial', 0.1))))
    tke = tke.clamp(min=float(params.get('tke_min', 1.0e-4)))
    mixing_length = tke_mixing_length(height, params)
    diffusivity, production = tke_diffusivity(
        t, q, u, v, p, height, tke, mixing_length, params
    )

    sensible_flux = _as_batch_tensor(
        params.get('_surface_sensible_heat_flux', 0.0), batch, t.device, t.dtype
    )
    moisture_flux = _as_batch_tensor(
        params.get('_surface_moisture_flux', 0.0), batch, t.device, t.dtype
    )
    density = p[:, -1] / (Rd * t[:, -1].clamp(min=150.0))
    heat_kinematic = sensible_flux / (density * cp).clamp(min=1.0e-8)
    moisture_kinematic = moisture_flux / density.clamp(min=1.0e-8)
    surface_buoyancy = (
        g / t[:, -1].clamp(min=150.0)
        * (heat_kinematic + 0.61 * t[:, -1] * moisture_kinematic)
    )
    bottom_depth = (height[:, -2] - height[:, -1]).clamp(min=10.0)
    production[:, -1] = production[:, -1] + surface_buoyancy / bottom_depth

    dissipation_constant = float(params.get('tke_dissipation_constant', 0.7))
    dissipation = dissipation_constant * tke.pow(1.5) / mixing_length.clamp(min=1.0)
    if params.get('tke_time_integration', 'explicit') == 'semi_implicit':
        produced = torch.clamp(
            tke + timestep * production,
            min=float(params.get('tke_min', 1.0e-4)),
        )
        damping = (
            timestep
            * dissipation_constant
            * torch.sqrt(produced)
            / mixing_length.clamp(min=1.0)
        )
        tke_new = produced / (1.0 + damping)
    else:
        tke_new = tke + timestep * (production - dissipation)
    tke_new = tke_new.clamp(
        min=float(params.get('tke_min', 1.0e-4)),
        max=float(params.get('tke_max', 10.0)),
    )

    new_diffusivity, _ = tke_diffusivity(
        t, q, u, v, p, height, tke_new, mixing_length, params
    )
    interface_diffusivity = 0.5 * (diffusivity + new_diffusivity)
    coefficients = pressure_diffusion_coefficients(
        interface_diffusivity, t, p, dp, timestep
    )

    total_water = q + qc
    water_rhs = total_water.clone()
    water_rhs[:, -1] = water_rhs[:, -1] + timestep * moisture_flux / mass[:, -1]
    total_water_new = solve_scalar(total_water, water_rhs, coefficients)
    qc_new = solve_scalar(qc, qc, coefficients)
    q_new = torch.clamp(total_water_new - qc_new, min=0.0)

    mse = cp * t + Lv * q + g * height
    energy_flux = sensible_flux + Lv * moisture_flux
    mse_rhs = mse.clone()
    mse_rhs[:, -1] = mse_rhs[:, -1] + timestep * energy_flux / mass[:, -1]
    mse_new = solve_scalar(mse, mse_rhs, coefficients)
    t_new = (mse_new - Lv * q_new - g * height) / cp
    if params.get('tke_diagnose_boundary_layer_depth', False):
        boundary_depth = tke_boundary_layer_depth(height, tke_new, params)
    else:
        wind2 = u[:, -1].pow(2) + v[:, -1].pow(2) + 1.0
        boundary_depth = diagnose_boundary_layer_depth(
            height,
            theta_v=t_new * (p0 / p.clamp(min=1.0)) ** kappa * (1.0 + 0.61 * q_new),
            wind2=wind2,
            ri_crit=torch.full_like(wind2, float(params.get('ri_crit', 0.25))),
            params=params,
        )

    return {
        'dt': (t_new - t) / timestep,
        'dq': (q_new - q) / timestep,
        'dqc': (qc_new - qc) / timestep,
        'tke': tke_new,
        'diffusivity': interface_diffusivity,
        'boundary_layer_depth_m': boundary_depth,
    }


def tke_boundary_layer_depth(height, tke, params):
    threshold = float(params.get('tke_boundary_layer_threshold_m2s2', 0.01))
    minimum = float(params.get('bl_min_depth_m', 100.0))
    maximum = float(
        params.get(
            'tke_boundary_layer_max_m',
            params.get('bl_max_depth_m', 1500.0),
        )
    )
    active = tke >= threshold
    active_height = torch.where(active, height, torch.zeros_like(height))
    return torch.max(active_height, dim=1).values.clamp(min=minimum, max=maximum)


def tke_mixing_length(height, params):
    vonkarman = 0.4
    maximum = float(params.get('tke_mixing_length_max_m', 250.0))
    distance = height.clamp(min=5.0)
    return 1.0 / (1.0 / (vonkarman * distance) + 1.0 / maximum)


def tke_diffusivity(t, q, u, v, p, height, tke, mixing_length, params):
    theta_v = virtual_temperature(t, q) * (p0 / p.clamp(min=1.0)) ** kappa
    depth = (height[:, :-1] - height[:, 1:]).clamp(min=1.0)
    theta_mean = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:]).clamp(min=150.0)
    stability = g / theta_mean * (theta_v[:, :-1] - theta_v[:, 1:]) / depth
    shear = (
        (u[:, :-1] - u[:, 1:]).pow(2)
        + (v[:, :-1] - v[:, 1:]).pow(2)
    ) / depth.pow(2)

    interface_tke = 0.5 * (tke[:, :-1] + tke[:, 1:])
    interface_length = 0.5 * (mixing_length[:, :-1] + mixing_length[:, 1:])
    coefficient = float(params.get('tke_diffusivity_constant', 0.18))
    neutral = coefficient * interface_length * torch.sqrt(interface_tke.clamp(min=0.0))
    richardson = stability / shear.clamp(min=1.0e-6)
    stable_factor = 1.0 / (1.0 + 5.0 * richardson.clamp(min=0.0))
    unstable_factor = torch.sqrt(1.0 + 8.0 * (-richardson).clamp(min=0.0))
    factor = torch.where(richardson >= 0.0, stable_factor, unstable_factor)
    factor = factor.clamp(
        max=float(params.get('tke_stability_factor_max', float('inf')))
    )
    maximum = float(params.get('tke_diffusivity_max_m2s', 100.0))
    interface_diffusivity = (neutral * factor).clamp(min=0.0, max=maximum)

    buoyancy = -interface_diffusivity * stability
    shear_production = interface_diffusivity * shear
    interface_production = buoyancy + shear_production
    production = torch.zeros_like(tke)
    production[:, 0] = interface_production[:, 0]
    production[:, -1] = interface_production[:, -1]
    if tke.shape[1] > 2:
        production[:, 1:-1] = 0.5 * (
            interface_production[:, :-1] + interface_production[:, 1:]
        )
    return interface_diffusivity, production


def pressure_diffusion_coefficients(diffusivity, t, p, dp, timestep):
    batch, levels = t.shape
    mass = dp / g
    upper_pressure = p[:, :-1].clamp(min=1.0)
    lower_pressure = p[:, 1:].clamp(min=1.0)
    pressure_depth = (lower_pressure - upper_pressure).clamp(min=100.0)
    interface_temperature = 0.5 * (t[:, :-1] + t[:, 1:]).clamp(min=150.0)
    interface_pressure = 0.5 * (upper_pressure + lower_pressure)
    density = interface_pressure / (Rd * interface_temperature)
    conductance = diffusivity * density * density * g / pressure_depth

    lower = torch.zeros(batch, levels, device=t.device, dtype=t.dtype)
    diagonal = torch.ones(batch, levels, device=t.device, dtype=t.dtype)
    upper = torch.zeros(batch, levels, device=t.device, dtype=t.dtype)
    upper[:, :-1] = -timestep * conductance / mass[:, :-1]
    lower[:, 1:] = -timestep * conductance / mass[:, 1:]
    diagonal[:, :-1] = diagonal[:, :-1] - upper[:, :-1]
    diagonal[:, 1:] = diagonal[:, 1:] - lower[:, 1:]
    return lower, diagonal, upper


def solve_scalar(original, rhs, coefficients):
    lower, diagonal, upper = coefficients
    return tridiag_solve(
        lower.clone(), diagonal.clone(), upper.clone(), rhs, 0, original.shape[1]
    )
