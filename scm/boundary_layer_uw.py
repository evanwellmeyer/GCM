"""PyTorch implementation of the UW moist-turbulence core.

This first increment treats surface-connected convective layers and local
stable turbulence. Elevated and cloud-top radiatively driven layers will be
added after these paths pass their component cases.
"""

import torch

from scm.boundary_layer import _as_batch_tensor, diagnose_boundary_layer_depth
from scm.boundary_layer_tke_v2 import pressure_diffusion_coefficients, solve_scalar
from scm.shallow_plume_v2 import partition_mse
from scm.thermo import Lv, Rd, cp, g, geopotential, kappa, p0, virtual_temperature


def uw_moist_turbulence(state, grid, params):
    """Return conservative tendencies from the UW diagnostic-TKE closure."""

    t = state["t"]
    q = state["q"]
    qc = state.get("qc", torch.zeros_like(q))
    u = state.get("u", torch.zeros_like(t))
    v = state.get("v", torch.zeros_like(t))
    p = state["p"]
    dp = state["dp"]
    timestep = float(params.get("dt", 900.0))
    batch = t.shape[0]
    mass = dp / g
    height = geopotential(t, q, p, grid)

    sensible = _as_batch_tensor(
        params.get("_surface_sensible_heat_flux", 0.0),
        batch,
        t.device,
        t.dtype,
    )
    moisture = _as_batch_tensor(
        params.get("_surface_moisture_flux", 0.0),
        batch,
        t.device,
        t.dtype,
    )
    surface_buoyancy = surface_buoyancy_flux(t, q, p, sensible, moisture)
    theta_v = virtual_temperature(t, q) * (p0 / p.clamp(min=1.0)) ** kappa
    wind2 = u[:, -1].square() + v[:, -1].square()
    wind2 = wind2 + float(params.get("uw_surface_shear_floor_ms", 1.0)) ** 2
    boundary_depth = diagnose_boundary_layer_depth(
        height,
        theta_v,
        wind2,
        torch.full_like(wind2, float(params.get("uw_critical_ri", 0.19))),
        {
            "bl_min_depth_m": params.get("bl_min_depth_m", 50.0),
            "bl_max_depth_m": params.get("bl_max_depth_m", 3000.0),
        },
    )

    heat_diffusivity, momentum_diffusivity, interface_tke, entrainment = uw_diffusivity(
        t,
        q,
        u,
        v,
        p,
        height,
        theta_v,
        boundary_depth,
        surface_buoyancy,
        params,
    )

    heat_coefficients = pressure_diffusion_coefficients(
        heat_diffusivity,
        t,
        p,
        dp,
        timestep,
    )
    momentum_coefficients = pressure_diffusion_coefficients(
        momentum_diffusivity,
        t,
        p,
        dp,
        timestep,
    )

    total_water = q + qc
    water_rhs = total_water.clone()
    water_rhs[:, -1] = water_rhs[:, -1] + timestep * moisture / mass[:, -1]
    total_water_new = solve_scalar(total_water, water_rhs, heat_coefficients)

    liquid_static_energy = cp * t + g * height - Lv * qc
    energy_rhs = liquid_static_energy.clone()
    energy_rhs[:, -1] = energy_rhs[:, -1] + timestep * sensible / mass[:, -1]
    liquid_static_energy_new = solve_scalar(
        liquid_static_energy,
        energy_rhs,
        heat_coefficients,
    )
    moist_static_energy_new = liquid_static_energy_new + Lv * total_water_new
    partitioned_t, partitioned_q, partitioned_qc = partition_mse(
        total_water_new,
        moist_static_energy_new,
        height,
        p,
    )
    mixed_layer = interface_to_layer(heat_diffusivity) > 0.0
    surface_source = (sensible.abs() + Lv * moisture.abs()) > 0.0
    mixed_layer[:, -1] = mixed_layer[:, -1] | surface_source
    t_new = torch.where(mixed_layer, partitioned_t, t)
    q_new = torch.where(mixed_layer, partitioned_q, q)
    qc_new = torch.where(mixed_layer, partitioned_qc, qc)

    u_new = solve_scalar(u, u, momentum_coefficients)
    v_new = solve_scalar(v, v, momentum_coefficients)
    layer_tke = interface_to_layer(interface_tke)
    water_before = torch.sum(total_water * mass, dim=1)
    water_after = torch.sum(total_water_new * mass, dim=1)
    energy_before = torch.sum((liquid_static_energy + Lv * total_water) * mass, dim=1)
    energy_after = torch.sum(moist_static_energy_new * mass, dim=1)

    return {
        "dt": (t_new - t) / timestep,
        "dq": (q_new - q) / timestep,
        "dqc": (qc_new - qc) / timestep,
        "du": (u_new - u) / timestep,
        "dv": (v_new - v) / timestep,
        "tke": layer_tke,
        "heat_diffusivity": heat_diffusivity,
        "momentum_diffusivity": momentum_diffusivity,
        "boundary_layer_depth_m": boundary_depth,
        "entrainment_velocity_ms": entrainment,
        "surface_buoyancy_flux_m2s3": surface_buoyancy,
        "water_residual": (water_after - water_before) / timestep - moisture,
        "energy_residual": (
            (energy_after - energy_before) / timestep
            - sensible
            - Lv * moisture
        ),
    }


def surface_buoyancy_flux(t, q, p, sensible, moisture):
    density = p[:, -1] / (Rd * t[:, -1].clamp(min=150.0))
    heat_kinematic = sensible / (density * cp).clamp(min=1.0e-8)
    moisture_kinematic = moisture / density.clamp(min=1.0e-8)
    return (
        g
        / t[:, -1].clamp(min=150.0)
        * (heat_kinematic + 0.61 * t[:, -1] * moisture_kinematic)
    )


def uw_diffusivity(
    t,
    q,
    u,
    v,
    p,
    height,
    theta_v,
    boundary_depth,
    surface_buoyancy,
    params,
):
    """Diagnose UW heat and momentum diffusivities at layer interfaces."""

    depth = (height[:, :-1] - height[:, 1:]).clamp(min=1.0)
    interface_height = 0.5 * (height[:, :-1] + height[:, 1:])
    theta_mean = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:]).clamp(min=150.0)
    stability = g / theta_mean * (theta_v[:, :-1] - theta_v[:, 1:]) / depth
    shear = (
        (u[:, :-1] - u[:, 1:]).square()
        + (v[:, :-1] - v[:, 1:]).square()
    ) / depth.square()
    shear = shear + float(params.get("uw_shear_floor_s2", 1.0e-8))
    richardson = stability / shear

    mixing_length = uw_mixing_length(interface_height, boundary_depth, params)
    sh, sm = galperin_functions(richardson, params)
    b1 = float(params.get("uw_dissipation_constant", 5.8))
    local_tke = b1 * mixing_length.square() * (-sh * stability + sm * shear)
    local_tke = local_tke.clamp(min=0.0)

    wstar_cubed = (surface_buoyancy * boundary_depth).clamp(min=0.0)
    convective_tke = (
        float(params.get("uw_convective_tke_ratio", 0.30))
        * wstar_cubed.pow(2.0 / 3.0)
    )
    connected = (
        (surface_buoyancy > 0.0).unsqueeze(1)
        & (interface_height <= boundary_depth.unsqueeze(1))
    )
    tke = torch.where(
        connected,
        torch.maximum(local_tke, convective_tke.unsqueeze(1)),
        local_tke,
    )
    tke = tke.clamp(max=float(params.get("uw_tke_max_m2s2", 20.0)))

    heat = mixing_length * torch.sqrt(tke) * sh
    momentum = mixing_length * torch.sqrt(tke) * sm
    heat, momentum, entrainment = add_convective_top_entrainment(
        heat,
        momentum,
        height,
        theta_v,
        boundary_depth,
        wstar_cubed,
        params,
    )
    maximum_height = float(params.get("uw_maximum_turbulent_height_m", 5000.0))
    allowed = interface_height <= maximum_height
    heat = torch.where(allowed, heat, torch.zeros_like(heat))
    momentum = torch.where(allowed, momentum, torch.zeros_like(momentum))
    maximum = float(params.get("uw_diffusivity_max_m2s", 200.0))
    return heat.clamp(0.0, maximum), momentum.clamp(0.0, maximum), tke, entrainment


def uw_mixing_length(interface_height, boundary_depth, params):
    vonkarman = float(params.get("uw_von_karman", 0.40))
    asymptotic = float(params.get("uw_asymptotic_length_fraction", 0.085))
    power = float(params.get("uw_mixing_length_power", 3.0))
    wall = (vonkarman * interface_height.clamp(min=1.0)).clamp(min=1.0)
    outer = (asymptotic * boundary_depth).clamp(min=1.0).unsqueeze(1)
    return (wall.pow(-power) + outer.pow(-power)).pow(-1.0 / power)


def galperin_functions(richardson, params):
    """Evaluate the UW form of the Galperin stability functions."""

    alph1 = float(params.get("uw_alph1", 0.5562))
    alph2 = float(params.get("uw_alph2", -4.3640))
    alph3 = float(params.get("uw_alph3", -34.6764))
    alph4 = float(params.get("uw_alph4", -6.1272))
    alph5 = float(params.get("uw_alph5", 0.6986))
    b1 = float(params.get("uw_dissipation_constant", 5.8))
    critical = float(params.get("uw_critical_ri", 0.19))

    ri = richardson.clamp(max=critical)
    a = alph3 * alph4 * ri + 2.0 * b1 * (alph2 - alph4 * alph5 * ri)
    b = (alph3 + alph4) * ri + 2.0 * b1 * (-alph5 * ri + alph1)
    discriminant = (b.square() - 4.0 * a * ri).clamp(min=0.0)
    safe_a = torch.where(a.abs() < 1.0e-12, torch.full_like(a, -1.0e-12), a)
    gh = (-b + torch.sqrt(discriminant)) / (2.0 * safe_a)
    gh = gh.clamp(min=-3.5334, max=0.0233)
    sh = (alph5 / (1.0 + alph3 * gh)).clamp(min=0.0)
    sm = (
        (alph1 + alph2 * gh)
        / (1.0 + alph3 * gh)
        / (1.0 + alph4 * gh)
    ).clamp(min=0.0)
    active = richardson < critical
    return torch.where(active, sh, torch.zeros_like(sh)), torch.where(
        active,
        sm,
        torch.zeros_like(sm),
    )


def add_convective_top_entrainment(
    heat,
    momentum,
    height,
    theta_v,
    boundary_depth,
    wstar_cubed,
    params,
):
    """Apply the dry UW entrainment closure at the surface-layer top."""

    interface_height = 0.5 * (height[:, :-1] + height[:, 1:])
    top_index = torch.argmin(
        torch.abs(interface_height - boundary_depth.unsqueeze(1)),
        dim=1,
    )
    columns = torch.arange(height.shape[0], device=height.device)
    upper = theta_v[columns, top_index]
    lower = theta_v[columns, top_index + 1]
    theta_mean = 0.5 * (upper + lower).clamp(min=150.0)
    buoyancy_jump = (g * (upper - lower) / theta_mean).clamp(
        min=float(params.get("uw_minimum_buoyancy_jump_ms2", 1.0e-3))
    )
    efficiency = float(params.get("uw_dry_entrainment_efficiency", 0.10))
    entrainment = efficiency * wstar_cubed / (
        boundary_depth.clamp(min=1.0) * buoyancy_jump
    )
    entrainment = entrainment.clamp(
        min=0.0,
        max=float(params.get("uw_entrainment_velocity_max_ms", 0.05)),
    )
    layer_depth = (
        height[columns, top_index] - height[columns, top_index + 1]
    ).clamp(min=1.0)
    entrainment_diffusivity = entrainment * layer_depth
    heat = heat.clone()
    momentum = momentum.clone()
    heat[columns, top_index] = torch.maximum(
        heat[columns, top_index],
        entrainment_diffusivity,
    )
    momentum[columns, top_index] = torch.maximum(
        momentum[columns, top_index],
        entrainment_diffusivity,
    )
    return heat, momentum, entrainment


def interface_to_layer(values):
    layers = values.shape[1] + 1
    result = torch.zeros(
        values.shape[0],
        layers,
        device=values.device,
        dtype=values.dtype,
    )
    result[:, 0] = values[:, 0]
    result[:, -1] = values[:, -1]
    if layers > 2:
        result[:, 1:-1] = 0.5 * (values[:, :-1] + values[:, 1:])
    return result
