"""PyTorch implementation of UW shallow-convection physics."""

import math
import torch

from scm.boundary_layer import diagnose_boundary_layer_depth
from scm.shallow_plume_v2 import partition_mse, partition_plume
from scm.thermo import Lv, Rd, cp, g, geopotential, kappa, p0, virtual_temperature


def lateral_mixing_rate(height, density, efficiency=8.0):
    """Return the UW environmental mixing rate in inverse pascals."""

    height = torch.as_tensor(height)
    density = torch.as_tensor(density, device=height.device, dtype=height.dtype)
    return float(efficiency) / (
        density.clamp(min=1.0e-4) * g * height.clamp(min=50.0)
    )


def implicit_cin_factor(cin_change, tke, iterations=16):
    """Solve the UW long-timestep relation ``a = exp(-a dcin / tke)``."""

    cin_change = torch.as_tensor(cin_change)
    tke = torch.as_tensor(tke, device=cin_change.device, dtype=cin_change.dtype)
    ratio = cin_change.clamp(min=0.0) / tke.clamp(min=1.0e-6)
    lower = torch.zeros_like(ratio)
    upper = torch.ones_like(ratio)
    for _ in range(int(iterations)):
        middle = 0.5 * (lower + upper)
        residual = middle - torch.exp(-middle * ratio)
        upper = torch.where(residual > 0.0, middle, upper)
        lower = torch.where(residual > 0.0, lower, middle)
    factor = 0.5 * (lower + upper)
    return torch.where(cin_change > 0.0, factor, torch.ones_like(factor))


def cloud_base_mass_flux(density, tke, cin, cin_change=None, coefficient=0.4):
    """Evaluate the UW CIN closure, optionally with its implicit correction."""

    density = torch.as_tensor(density)
    tke = torch.as_tensor(tke, device=density.device, dtype=density.dtype)
    cin = torch.as_tensor(cin, device=density.device, dtype=density.dtype)
    mass_flux = (
        float(coefficient)
        * density
        * torch.sqrt(tke.clamp(min=0.0))
        * torch.exp(-cin.clamp(min=0.0) / tke.clamp(min=1.0e-6))
    )
    if cin_change is not None:
        mass_flux = mass_flux * implicit_cin_factor(cin_change, tke)
    return mass_flux


def uw_shallow_convection(state, grid, params):
    """Transport conserved scalars with a CIN-closed entraining plume."""

    t = state["t"]
    q = state["q"]
    qc = state.get("qc", torch.zeros_like(q))
    u = state.get("u", torch.zeros_like(t))
    v = state.get("v", torch.zeros_like(t))
    p = state["p"]
    dp = state["dp"]
    timestep = float(params.get("dt", 900.0))
    batch, levels = t.shape
    mass = dp / g
    height = geopotential(t, q, p, grid)
    exner = (p / p0).clamp(min=1.0e-6).pow(kappa)
    theta_liquid = t / exner - Lv * qc / (cp * exner)
    total_water = q + qc
    mse = cp * t + Lv * q + g * height
    boundary_depth = state.get("boundary_layer_depth_m")
    if boundary_depth is None:
        theta_v = virtual_temperature(t, q) * (p0 / p.clamp(min=1.0)).pow(kappa)
        wind2 = u[:, -1].square() + v[:, -1].square() + 1.0
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

    outputs = _integrate_columns(
        t,
        q,
        qc,
        u,
        v,
        p,
        dp,
        height,
        theta_liquid,
        total_water,
        mse,
        state.get("tke", torch.full_like(t, 0.1)),
        boundary_depth,
        params,
    )
    outputs = apply_implicit_cin_correction(
        outputs,
        t,
        q,
        qc,
        p,
        dp,
        height,
        mse,
        state.get("tke", torch.full_like(t, 0.1)),
        boundary_depth,
        timestep,
    )
    water_tendency = (outputs["water_flux"][:, 1:] - outputs["water_flux"][:, :-1]) / mass
    mse_tendency = (outputs["mse_flux"][:, 1:] - outputs["mse_flux"][:, :-1]) / mass
    u_tendency = (outputs["u_flux"][:, 1:] - outputs["u_flux"][:, :-1]) / mass
    v_tendency = (outputs["v_flux"][:, 1:] - outputs["v_flux"][:, :-1]) / mass
    evaporation, precipitation = precipitation_evaporation(
        outputs["precipitation_source"],
        q,
        t,
        p,
        mass,
        params,
    )
    water_tendency = water_tendency - outputs["precipitation_source"] / mass + evaporation / mass
    limiter = conservative_positivity_factor(total_water, water_tendency, timestep)
    water_tendency = water_tendency * limiter.unsqueeze(1)
    mse_tendency = mse_tendency * limiter.unsqueeze(1)
    u_tendency = u_tendency * limiter.unsqueeze(1)
    v_tendency = v_tendency * limiter.unsqueeze(1)
    evaporation = evaporation * limiter.unsqueeze(1)
    precipitation = precipitation * limiter
    for name in (
        "mass_flux_profile",
        "plume_condensate",
        "cloud_fraction",
        "precipitation_source",
        "mse_flux",
        "water_flux",
        "u_flux",
        "v_flux",
        "liquid_flux",
    ):
        outputs[name] = outputs[name] * limiter.unsqueeze(1)
    outputs["cloud_base_mass_flux"] = outputs["cloud_base_mass_flux"] * limiter
    water_new = total_water + timestep * water_tendency
    mse_new = mse + timestep * mse_tendency
    if bool(params.get("uw_shallow_layer_mean_saturation", False)):
        t_new, q_new, qc_new = partition_layer_mean(
            water_new,
            mse_new,
            height,
            p,
            dp,
        )
    else:
        t_new, q_new, qc_new = partition_mse(water_new, mse_new, height, p)

    active = height <= float(params.get("uw_shallow_maximum_height_m", 4000.0))
    t_new = torch.where(active, t_new, t)
    q_new = torch.where(active, q_new, q)
    qc_new = torch.where(active, qc_new, qc)
    u_tendency = torch.where(active, u_tendency, torch.zeros_like(u_tendency))
    v_tendency = torch.where(active, v_tendency, torch.zeros_like(v_tendency))

    water_residual = (
        torch.sum((q_new + qc_new - q - qc) * mass, dim=1) / timestep
        + precipitation
    )
    energy_residual = torch.sum(
        (cp * (t_new - t) + Lv * (q_new - q)) * mass,
        dim=1,
    ) / timestep
    condensate_detrainment = torch.clamp(
        (outputs["liquid_flux"][:, 1:] - outputs["liquid_flux"][:, :-1]) / mass,
        min=0.0,
    )
    in_cloud_condensate = float(params.get("uw_shallow_in_cloud_condensate_kgkg", 3.0e-3))
    environment_cloud = (qc_new / in_cloud_condensate).clamp(min=0.0, max=1.0)
    cloud_fraction = torch.maximum(outputs["cloud_fraction"], environment_cloud)
    cloud_fraction = torch.where(active, cloud_fraction, torch.zeros_like(cloud_fraction))
    return {
        "dt": (t_new - t) / timestep,
        "dq": (q_new - q) / timestep,
        "dqc": (qc_new - qc) / timestep,
        "du": u_tendency,
        "dv": v_tendency,
        "precip": precipitation,
        "cloud_base_mass_flux": outputs["cloud_base_mass_flux"],
        "cloud_fraction": cloud_fraction,
        "plume_condensate": outputs["plume_condensate"],
        "plume_mass_flux_profile": outputs["mass_flux_profile"],
        "condensate_detrainment": condensate_detrainment,
        "plume_top_height_m": outputs["plume_top_height"],
        "plume_cloud_base_height_m": outputs["cloud_base_height"],
        "maximum_plume_condensate_kgkg": outputs["maximum_condensate"],
        "cin_jkg": outputs["cin"],
        "implicit_cin_factor": outputs["implicit_cin_factor"],
        "water_residual": water_residual,
        "energy_residual": energy_residual,
        "mse_residual": energy_residual,
        "precipitation_evaporation": evaporation / mass,
    }


def conservative_positivity_factor(total_water, tendency, timestep, minimum=1.0e-8):
    """Scale a column update so no layer crosses the total-water floor."""

    available = (total_water - float(minimum)).clamp(min=0.0)
    required = (-float(timestep) * tendency).clamp(min=0.0)
    layer_factor = torch.where(
        required > 0.0,
        available / required.clamp(min=1.0e-20),
        torch.ones_like(required),
    ).clamp(min=0.0, max=1.0)
    return torch.min(layer_factor, dim=1).values


def limited_pressure_slope(value, pressure):
    """Reconstruct a monotone pressure slope at each model level."""

    slope = torch.zeros_like(value)
    downward = (value[:, 1:-1] - value[:, :-2]) / (
        pressure[:, 1:-1] - pressure[:, :-2]
    ).clamp(min=1.0)
    upward = (value[:, 2:] - value[:, 1:-1]) / (
        pressure[:, 2:] - pressure[:, 1:-1]
    ).clamp(min=1.0)
    same_sign = downward * upward > 0.0
    magnitude = torch.minimum(downward.abs(), upward.abs())
    slope[:, 1:-1] = torch.where(
        same_sign,
        torch.sign(downward) * magnitude,
        torch.zeros_like(magnitude),
    )
    return slope


def partition_layer_mean(total_water, mse, height, pressure, dp):
    """Saturation-adjust reconstructed sublayer states and average them.

    Two symmetric pressure points represent each layer. Their conserved-state
    means equal the supplied layer means, avoiding dependence on saturation at
    one full-level sample while preserving total water and moist static energy.
    """

    water_slope = limited_pressure_slope(total_water, pressure)
    mse_slope = limited_pressure_slope(mse, pressure)
    height_slope = limited_pressure_slope(height, pressure)
    quarter_dp = 0.25 * dp

    water_top = torch.clamp(total_water - water_slope * quarter_dp, min=1.0e-8)
    water_bottom = torch.clamp(total_water + water_slope * quarter_dp, min=1.0e-8)
    water_correction = total_water - 0.5 * (water_top + water_bottom)
    water_top = water_top + water_correction
    water_bottom = water_bottom + water_correction

    mse_top = mse - mse_slope * quarter_dp
    mse_bottom = mse + mse_slope * quarter_dp
    height_top = height - height_slope * quarter_dp
    height_bottom = height + height_slope * quarter_dp
    pressure_top = (pressure - quarter_dp).clamp(min=1.0)
    pressure_bottom = pressure + quarter_dp

    top = partition_mse(water_top, mse_top, height_top, pressure_top)
    bottom = partition_mse(
        water_bottom,
        mse_bottom,
        height_bottom,
        pressure_bottom,
    )
    return tuple(0.5 * (top_value + bottom_value) for top_value, bottom_value in zip(top, bottom))


def precipitation_evaporation(source, q, t, p, mass, params):
    """Evaporate falling shallow-cumulus precipitation into unsaturated air."""

    from scm.thermo import saturation_specific_humidity

    relative_humidity = q / saturation_specific_humidity(t, p).clamp(min=1.0e-8)
    coefficient = float(params.get("uw_shallow_evaporation_coefficient", 2.0e-6))
    batch, levels = source.shape
    evaporation = torch.zeros_like(source)
    surface = torch.zeros(batch, device=q.device, dtype=q.dtype)
    for column in range(batch):
        falling = torch.zeros((), device=q.device, dtype=q.dtype)
        for layer in range(levels):
            falling = falling + source[column, layer]
            capacity = (
                coefficient
                * (1.0 - relative_humidity[column, layer]).clamp(min=0.0)
                * torch.sqrt(falling.clamp(min=0.0))
                * mass[column, layer]
            )
            evaporated = torch.minimum(falling, capacity)
            evaporation[column, layer] = evaporated
            falling = falling - evaporated
        surface[column] = falling
    return evaporation, surface


def apply_implicit_cin_correction(
    outputs,
    t,
    q,
    qc,
    p,
    dp,
    height,
    mse,
    tke,
    boundary_depth,
    timestep,
):
    """Scale normalized plume tendencies using predicted end-step CIN."""

    mass = dp / g
    water = q + qc
    water_tendency = (outputs["water_flux"][:, 1:] - outputs["water_flux"][:, :-1]) / mass
    mse_tendency = (outputs["mse_flux"][:, 1:] - outputs["mse_flux"][:, :-1]) / mass
    predicted_water = torch.clamp(water + timestep * water_tendency, min=1.0e-8)
    predicted_mse = mse + timestep * mse_tendency
    predicted_t, predicted_q, predicted_qc = partition_mse(
        predicted_water,
        predicted_mse,
        height,
        p,
    )
    predicted_exner = (p / p0).clamp(min=1.0e-6).pow(kappa)
    predicted_theta = (
        predicted_t / predicted_exner
        - Lv * predicted_qc / (cp * predicted_exner)
    )
    factor = torch.ones_like(outputs["cin"])
    for column in range(t.shape[0]):
        inside = torch.nonzero(
            height[column] <= boundary_depth[column], as_tuple=False
        ).flatten()
        if inside.numel() == 0 or outputs["cloud_base_mass_flux"][column] <= 0.0:
            continue
        source = int(inside[0])
        source_theta = torch.min(predicted_theta[column, inside])
        source_water = predicted_water[column, -1]
        predicted_cin, _ = undilute_cin(
            source_theta,
            source_water,
            source,
            predicted_t[column],
            predicted_q[column],
            predicted_qc[column],
            p[column],
            height[column],
        )
        change = predicted_cin - outputs["cin"][column]
        factor[column] = implicit_cin_factor(change, tke[column, source])

    for name in ("mse_flux", "water_flux", "u_flux", "v_flux", "liquid_flux"):
        outputs[name] = outputs[name] * factor.unsqueeze(1)
    for name in ("mass_flux_profile", "cloud_fraction"):
        outputs[name] = outputs[name] * factor.unsqueeze(1)
    outputs["cloud_base_mass_flux"] = outputs["cloud_base_mass_flux"] * factor
    outputs["precipitation_source"] = (
        outputs["precipitation_source"] * factor.unsqueeze(1)
    )
    outputs["implicit_cin_factor"] = factor
    return outputs


def _integrate_columns(
    t,
    q,
    qc,
    u,
    v,
    p,
    dp,
    height,
    theta_liquid,
    total_water,
    mse,
    tke,
    boundary_depth,
    params,
):
    batch, levels = t.shape
    shape = (batch, levels + 1)
    result = {
        "mse_flux": torch.zeros(shape, device=t.device, dtype=t.dtype),
        "water_flux": torch.zeros(shape, device=t.device, dtype=t.dtype),
        "u_flux": torch.zeros(shape, device=t.device, dtype=t.dtype),
        "v_flux": torch.zeros(shape, device=t.device, dtype=t.dtype),
        "liquid_flux": torch.zeros(shape, device=t.device, dtype=t.dtype),
        "cloud_fraction": torch.zeros_like(t),
        "plume_condensate": torch.zeros_like(t),
        "mass_flux_profile": torch.zeros_like(t),
        "cloud_base_mass_flux": torch.zeros(batch, device=t.device, dtype=t.dtype),
        "plume_top_height": torch.zeros(batch, device=t.device, dtype=t.dtype),
        "cloud_base_height": torch.zeros(batch, device=t.device, dtype=t.dtype),
        "maximum_condensate": torch.zeros(batch, device=t.device, dtype=t.dtype),
        "cin": torch.zeros(batch, device=t.device, dtype=t.dtype),
        "precipitation_source": torch.zeros_like(t),
    }
    for column in range(batch):
        _integrate_one_column(
            column,
            result,
            t,
            q,
            qc,
            u,
            v,
            p,
            dp,
            height,
            theta_liquid,
            total_water,
            mse,
            tke,
            boundary_depth,
            params,
        )
    return result


def _integrate_one_column(
    column,
    result,
    t,
    q,
    qc,
    u,
    v,
    p,
    dp,
    height,
    theta_liquid,
    total_water,
    mse,
    tke,
    boundary_depth,
    params,
):
    levels = t.shape[1]
    inside = torch.nonzero(height[column] <= boundary_depth[column], as_tuple=False).flatten()
    if inside.numel() == 0:
        return
    source = int(inside[0])
    source_layers = inside
    plume_theta = torch.min(theta_liquid[column, source_layers])
    plume_water = total_water[column, -1]
    plume_u = u[column, source]
    plume_v = v[column, source]
    cin, reaches_lfc = undilute_cin(
        plume_theta,
        plume_water,
        source,
        t[column],
        q[column],
        qc[column],
        p[column],
        height[column],
    )
    result["cin"][column] = cin
    if not reaches_lfc:
        return

    source_density = p[column, source] / (Rd * t[column, source].clamp(min=150.0))
    source_tke = tke[column, source].clamp(min=1.0e-4)
    mass_flux = cloud_base_mass_flux(source_density, source_tke, cin)
    velocity_squared = (2.0 * source_tke - 2.0 * cin).clamp(min=0.05)
    velocity = torch.sqrt(velocity_squared)
    area_max = float(params.get("uw_shallow_core_area_max", 0.10))
    mass_flux = torch.minimum(mass_flux, area_max * source_density * velocity)
    if mass_flux <= 1.0e-10:
        return
    result["cloud_base_mass_flux"][column] = mass_flux

    maximum_height = float(params.get("uw_shallow_maximum_height_m", 4000.0))
    vertical_step = float(params.get("uw_shallow_vertical_step_m", 50.0))
    mixing_efficiency = float(params.get("uw_shallow_mixing_efficiency", 8.0))
    drag = float(params.get("uw_shallow_velocity_drag", 1.0))
    cloud_multiplier = float(params.get("uw_shallow_cloud_area_multiplier", 2.0))
    condensate_maximum = float(params.get("uw_shallow_condensate_maximum_kgkg", 1.0e-3))
    for lower in range(source, 0, -1):
        upper = lower - 1
        layer_depth = (height[column, upper] - height[column, lower]).clamp(min=1.0)
        if height[column, upper] > maximum_height:
            break
        steps = max(1, math.ceil(float(layer_depth) / vertical_step))
        step_height = layer_depth / steps
        active = True
        plume_temperature = t[column, lower]
        plume_vapor = q[column, lower]
        plume_liquid = qc[column, lower]
        for step in range(steps):
            fraction = (step + 1.0) / steps
            subheight = height[column, lower] + fraction * layer_depth
            subpressure = p[column, lower] + fraction * (p[column, upper] - p[column, lower])
            environment_theta = theta_liquid[column, lower] + fraction * (
                theta_liquid[column, upper] - theta_liquid[column, lower]
            )
            environment_water = total_water[column, lower] + fraction * (
                total_water[column, upper] - total_water[column, lower]
            )
            environment_temperature = t[column, lower] + fraction * (
                t[column, upper] - t[column, lower]
            )
            environment_vapor = q[column, lower] + fraction * (
                q[column, upper] - q[column, lower]
            )
            environment_liquid = qc[column, lower] + fraction * (
                qc[column, upper] - qc[column, lower]
            )
            density = subpressure / (Rd * environment_temperature.clamp(min=150.0))
            pressure_step = density * g * step_height
            rate = lateral_mixing_rate(subheight, density, mixing_efficiency)
            mixed_fraction = 1.0 - torch.exp(-rate * pressure_step)
            candidate_theta = plume_theta + mixed_fraction * (environment_theta - plume_theta)
            candidate_water = plume_water + mixed_fraction * (environment_water - plume_water)
            candidate_temperature, candidate_vapor, candidate_liquid = partition_plume(
                candidate_theta,
                candidate_water,
                subpressure,
            )
            environment_virtual = environment_temperature * (
                1.0 + 0.61 * environment_vapor - environment_liquid
            )
            candidate_virtual = candidate_temperature * (
                1.0 + 0.61 * candidate_vapor - candidate_liquid
            )
            buoyancy = g * (
                candidate_virtual - environment_virtual
            ) / environment_virtual.clamp(min=150.0)
            can_mix = buoyancy + velocity_squared / (2.0 * step_height) > 0.0
            accepted = torch.where(can_mix, mixed_fraction, 0.25 * mixed_fraction)
            detrain = torch.where(can_mix, torch.zeros_like(accepted), 0.75 * mixed_fraction)
            plume_theta = plume_theta + accepted * (environment_theta - plume_theta)
            plume_water = plume_water + accepted * (environment_water - plume_water)
            plume_u = plume_u + accepted * (u[column, upper] - plume_u)
            plume_v = plume_v + accepted * (v[column, upper] - plume_v)
            mass_flux = mass_flux * torch.exp((accepted - detrain).clamp(-0.9, 0.9))
            plume_temperature, plume_vapor, plume_liquid = partition_plume(
                plume_theta,
                plume_water,
                subpressure,
            )
            excess_condensate = (plume_liquid - condensate_maximum).clamp(min=0.0)
            if excess_condensate > 0.0:
                result["precipitation_source"][column, upper] = (
                    result["precipitation_source"][column, upper]
                    + mass_flux * excess_condensate
                )
                plume_water = plume_water - excess_condensate
                plume_temperature, plume_vapor, plume_liquid = partition_plume(
                    plume_theta,
                    plume_water,
                    subpressure,
                )
            plume_virtual = plume_temperature * (
                1.0 + 0.61 * plume_vapor - plume_liquid
            )
            buoyancy = g * (
                plume_virtual - environment_virtual
            ) / environment_virtual.clamp(min=150.0)
            velocity_squared = velocity_squared + 2.0 * step_height * (
                buoyancy - drag * accepted * velocity_squared / step_height
            )
            if velocity_squared <= 0.0:
                active = False
                break
        if not active:
            break

        velocity = torch.sqrt(velocity_squared.clamp(min=1.0e-4))
        plume_density = p[column, upper] / (Rd * plume_temperature.clamp(min=150.0))
        mass_flux = torch.minimum(mass_flux, area_max * plume_density * velocity)
        environment_mse = 0.5 * (mse[column, lower] + mse[column, upper])
        environment_water = 0.5 * (
            total_water[column, lower] + total_water[column, upper]
        )
        environment_u = 0.5 * (u[column, lower] + u[column, upper])
        environment_v = 0.5 * (v[column, lower] + v[column, upper])
        interface_height = 0.5 * (height[column, lower] + height[column, upper])
        plume_mse = cp * plume_temperature + Lv * plume_vapor + g * interface_height
        result["mse_flux"][column, lower] = mass_flux * (plume_mse - environment_mse)
        result["water_flux"][column, lower] = mass_flux * (plume_water - environment_water)
        result["u_flux"][column, lower] = mass_flux * (plume_u - environment_u)
        result["v_flux"][column, lower] = mass_flux * (plume_v - environment_v)
        result["liquid_flux"][column, lower] = mass_flux * plume_liquid
        result["mass_flux_profile"][column, upper] = mass_flux
        result["plume_condensate"][column, upper] = plume_liquid
        result["plume_top_height"][column] = height[column, upper]
        result["maximum_condensate"][column] = torch.maximum(
            result["maximum_condensate"][column], plume_liquid
        )
        if plume_liquid > 0.0:
            if result["cloud_base_height"][column] == 0.0:
                result["cloud_base_height"][column] = height[column, upper]
            core_area = mass_flux / (plume_density * velocity).clamp(min=1.0e-8)
            result["cloud_fraction"][column, upper] = (
                cloud_multiplier * core_area
            ).clamp(min=0.0, max=2.0 * area_max)


def undilute_cin(theta_liquid, total_water, source, t, q, qc, p, height):
    """Integrate negative undilute-parcel buoyancy from PBL top to LFC."""

    cin = torch.zeros((), device=t.device, dtype=t.dtype)
    reaches_lfc = False
    for lower in range(source, 0, -1):
        upper = lower - 1
        pressure = 0.5 * (p[lower] + p[upper])
        parcel_t, parcel_q, parcel_qc = partition_plume(
            theta_liquid,
            total_water,
            pressure,
        )
        environment_t = 0.5 * (t[lower] + t[upper])
        environment_q = 0.5 * (q[lower] + q[upper])
        environment_qc = 0.5 * (qc[lower] + qc[upper])
        parcel_virtual = parcel_t * (1.0 + 0.61 * parcel_q - parcel_qc)
        environment_virtual = environment_t * (
            1.0 + 0.61 * environment_q - environment_qc
        )
        buoyancy = g * (
            parcel_virtual - environment_virtual
        ) / environment_virtual.clamp(min=150.0)
        layer_depth = (height[upper] - height[lower]).clamp(min=1.0)
        if buoyancy > 0.0:
            reaches_lfc = True
            break
        cin = cin + (-buoyancy).clamp(min=0.0) * layer_depth
    return cin, reaches_lfc
