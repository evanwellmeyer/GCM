import math

import torch

from scm.thermo import Lv, cp, g, geopotential, kappa, p0, saturation_specific_humidity


def partition_plume(theta_liquid, total_water, pressure):
    exner = (pressure / p0).clamp(min=1.0e-6) ** kappa
    def residual(vapor):
        condensate = torch.clamp(total_water - vapor, min=0.0)
        temperature = (theta_liquid + Lv * condensate / (cp * exner)) * exner
        return vapor - saturation_specific_humidity(temperature, pressure)

    saturated = residual(total_water) > 0.0
    lower = torch.zeros_like(total_water)
    upper = total_water
    for _ in range(20):
        middle = 0.5 * (lower + upper)
        above = residual(middle) > 0.0
        upper = torch.where(above, middle, upper)
        lower = torch.where(above, lower, middle)
    vapor = torch.where(saturated, 0.5 * (lower + upper), total_water)
    condensate = torch.clamp(total_water - vapor, min=0.0)
    temperature = (theta_liquid + Lv * condensate / (cp * exner)) * exner
    return temperature, vapor, condensate


def partition_mse(total_water, mse, height, pressure):
    def residual(vapor):
        temperature = (mse - Lv * vapor - g * height) / cp
        return vapor - saturation_specific_humidity(temperature, pressure)

    saturated = residual(total_water) > 0.0
    lower = torch.zeros_like(total_water)
    upper = total_water
    for _ in range(20):
        middle = 0.5 * (lower + upper)
        above = residual(middle) > 0.0
        upper = torch.where(above, middle, upper)
        lower = torch.where(above, lower, middle)
    vapor = torch.where(saturated, 0.5 * (lower + upper), total_water)
    temperature = (mse - Lv * vapor - g * height) / cp
    condensate = torch.clamp(total_water - vapor, min=0.0)
    return temperature, vapor, condensate


def shallow_plume(state, grid, params):
    """Entraining shallow plume transporting theta-l and total water."""

    t = state['t']
    q = state['q']
    qc = state.get('qc', torch.zeros_like(q))
    p = state['p']
    dp = state['dp']
    timestep = float(params.get('dt', 60.0))
    batch, levels = t.shape
    mass = dp / g
    height = geopotential(t, q, p, grid)
    exner = (p / p0).clamp(min=1.0e-6) ** kappa
    theta_liquid = t / exner - Lv * qc / (cp * exner)
    total_water = q + qc

    mse = cp * t + Lv * q + g * height
    mse_flux = torch.zeros(batch, levels + 1, device=t.device, dtype=t.dtype)
    water_flux = torch.zeros_like(mse_flux)
    liquid_flux = torch.zeros_like(mse_flux)
    plume_cloud_fraction = torch.zeros_like(t)
    plume_condensate_profile = torch.zeros_like(t)
    plume_mass_flux_profile = torch.zeros_like(t)
    cloud_mass_flux = torch.zeros(batch, device=t.device, dtype=t.dtype)
    plume_top_height = torch.zeros(batch, device=t.device, dtype=t.dtype)
    plume_cloud_base_height = torch.zeros_like(plume_top_height)
    maximum_plume_condensate = torch.zeros_like(plume_top_height)
    maximum_height = float(params.get('shallow_plume_top_m', 2500.0))
    temperature_excess = float(params.get('shallow_plume_temperature_excess_k', 0.15))
    updraft_area = float(params.get('shallow_plume_updraft_area', 0.13))
    entrainment_constant = float(params.get('shallow_plume_entrainment_constant', 0.4))
    velocity_drag = float(params.get('shallow_plume_velocity_drag', 2.0))
    velocity_buoyancy = float(params.get('shallow_plume_velocity_buoyancy', 4.0))
    vertical_step = float(params.get('shallow_plume_vertical_step_m', 50.0))
    surface_flux_fraction = float(params.get('shallow_plume_surface_flux_fraction', 0.25))
    detrainment_depth = float(params.get('shallow_plume_detrainment_depth_m', 500.0))
    detrainment_strength = float(params.get('shallow_plume_detrainment_strength', 0.0))
    buoyancy_detrainment = float(
        params.get('shallow_plume_buoyancy_detrainment_constant', 0.0)
    )
    tke = state.get('tke', torch.full_like(t, 0.1))

    for column in range(batch):
        source = levels - 1
        source_density = p[column, source] / (
            287.0 * t[column, source].clamp(min=150.0)
        )
        velocity_squared = torch.clamp(
            (2.0 / 3.0) * tke[column, source], min=0.01
        )
        area_fraction = torch.as_tensor(
            updraft_area, device=t.device, dtype=t.dtype
        ).clamp(min=0.0, max=float(params.get('shallow_plume_area_max', 0.20)))
        source_mass_flux = source_density * area_fraction * torch.sqrt(velocity_squared)
        plume_mass_flux = source_mass_flux
        sensible_flux = params.get('_surface_sensible_heat_flux', None)
        moisture_flux = params.get('_surface_moisture_flux', None)
        if sensible_flux is None:
            plume_theta = theta_liquid[column, source] + temperature_excess
        else:
            plume_theta = theta_liquid[column, source] + (
                surface_flux_fraction * sensible_flux[column]
                / (cp * exner[column, source] * source_mass_flux)
            )
        if moisture_flux is None:
            plume_water = total_water[column, source]
        else:
            plume_water = (
                total_water[column, source]
                + surface_flux_fraction * moisture_flux[column] / source_mass_flux
            )
        cloud_mass_flux[column] = source_mass_flux

        for lower in range(levels - 1, 0, -1):
            upper = lower - 1
            interface_height = 0.5 * (height[column, lower] + height[column, upper])
            if interface_height > maximum_height:
                break
            depth = (height[column, upper] - height[column, lower]).clamp(min=1.0)
            steps = max(1, math.ceil(float(depth) / vertical_step))
            step_depth = depth / steps
            active = True

            for step in range(steps):
                fraction = (step + 1.0) / steps
                subheight = height[column, lower] + fraction * depth
                distance_from_surface = subheight.clamp(min=1.0)
                distance_from_top = (maximum_height - subheight).clamp(min=1.0)
                entrainment = entrainment_constant * (
                    1.0 / (distance_from_surface + step_depth)
                    + 1.0 / (distance_from_top + step_depth)
                )
                environment_theta = (
                    (1.0 - fraction) * theta_liquid[column, lower]
                    + fraction * theta_liquid[column, upper]
                )
                environment_water = (
                    (1.0 - fraction) * total_water[column, lower]
                    + fraction * total_water[column, upper]
                )
                subpressure = (
                    (1.0 - fraction) * p[column, lower]
                    + fraction * p[column, upper]
                )
                environment_temperature = (
                    (1.0 - fraction) * t[column, lower]
                    + fraction * t[column, upper]
                )
                environment_vapor = (
                    (1.0 - fraction) * q[column, lower]
                    + fraction * q[column, upper]
                )
                environment_liquid = (
                    (1.0 - fraction) * qc[column, lower]
                    + fraction * qc[column, upper]
                )
                retained = torch.exp(-entrainment * step_depth)
                plume_mass_flux = plume_mass_flux / retained.clamp(min=1.0e-6)
                plume_theta = retained * plume_theta + (1.0 - retained) * environment_theta
                plume_water = retained * plume_water + (1.0 - retained) * environment_water
                plume_temperature, plume_vapor, plume_liquid = partition_plume(
                    plume_theta, plume_water, subpressure
                )
                plume_top_height[column] = torch.maximum(
                    plume_top_height[column], subheight
                )
                maximum_plume_condensate[column] = torch.maximum(
                    maximum_plume_condensate[column], plume_liquid
                )
                if plume_liquid > 0.0 and plume_cloud_base_height[column] == 0.0:
                    plume_cloud_base_height[column] = subheight
                environment_virtual = environment_temperature * (
                    1.0 + 0.61 * environment_vapor - environment_liquid
                )
                plume_virtual = plume_temperature * (
                    1.0 + 0.61 * plume_vapor - plume_liquid
                )
                buoyancy = (plume_virtual - environment_virtual) / environment_virtual.clamp(min=150.0)
                detrainment_start = maximum_height - detrainment_depth
                terminal_fraction = torch.clamp(
                    (subheight - detrainment_start) / max(detrainment_depth, 1.0),
                    min=0.0,
                    max=1.0,
                )
                terminal_detrainment = (
                    detrainment_strength * terminal_fraction / max(detrainment_depth, 1.0)
                )
                negative_buoyancy_detrainment = (
                    buoyancy_detrainment
                    * torch.clamp(-g * buoyancy, min=0.0)
                    / velocity_squared.clamp(min=0.01)
                )
                detrainment = terminal_detrainment + negative_buoyancy_detrainment
                plume_mass_flux = plume_mass_flux * torch.exp(-detrainment * step_depth)
                damping_rate = velocity_drag * entrainment
                velocity_retained = torch.exp(-damping_rate * step_depth)
                buoyancy_equilibrium = (
                    velocity_buoyancy * g * buoyancy / damping_rate.clamp(min=1.0e-8)
                )
                velocity_squared = (
                    velocity_retained * velocity_squared
                    + (1.0 - velocity_retained) * buoyancy_equilibrium
                )
                if velocity_squared <= 0.0:
                    active = False
                    break

            if not active:
                break

            plume_density = p[column, upper] / (
                287.0 * plume_temperature.clamp(min=150.0)
            )
            plume_velocity = torch.sqrt(velocity_squared.clamp(min=0.0))
            maximum_mass_flux = (
                plume_density
                * float(params.get('shallow_plume_area_max', 0.20))
                * plume_velocity
            )
            mass_flux = torch.minimum(plume_mass_flux, maximum_mass_flux)
            plume_mass_flux = mass_flux
            plume_mass_flux_profile[column, upper] = mass_flux
            area_fraction = (
                mass_flux / (plume_density * plume_velocity).clamp(min=1.0e-8)
            ).clamp(min=0.0, max=float(params.get('shallow_plume_area_max', 0.20)))
            if plume_liquid > 0.0:
                plume_cloud_fraction[column, upper] = area_fraction
                plume_condensate_profile[column, upper] = plume_liquid

            environment_water = 0.5 * (
                total_water[column, lower] + total_water[column, upper]
            )
            environment_mse = 0.5 * (mse[column, lower] + mse[column, upper])
            plume_mse = (
                cp * plume_temperature
                + Lv * plume_vapor
                + g * interface_height
            )
            mse_flux[column, lower] = mass_flux * (plume_mse - environment_mse)
            water_flux[column, lower] = mass_flux * (plume_water - environment_water)
            liquid_flux[column, lower] = mass_flux * plume_liquid

    mse_tendency = (mse_flux[:, 1:] - mse_flux[:, :-1]) / mass
    water_tendency = (water_flux[:, 1:] - water_flux[:, :-1]) / mass
    condensate_detrainment = torch.clamp(
        (liquid_flux[:, 1:] - liquid_flux[:, :-1]) / mass,
        min=0.0,
    )
    mse_new = mse + timestep * mse_tendency
    water_new = torch.clamp(total_water + timestep * water_tendency, min=1.0e-8)
    if params.get('shallow_plume_grid_saturation_adjustment', True):
        t_new, q_new, qc_new = partition_mse(water_new, mse_new, height, p)
    else:
        qc_new = torch.minimum(
            qc + timestep * condensate_detrainment,
            water_new,
        )
        q_new = water_new - qc_new
        t_new = (mse_new - Lv * q_new - g * height) / cp
    in_cloud_condensate = float(
        params.get('shallow_plume_in_cloud_condensate_kgkg', 3.0e-3)
    )
    environment_cloud_fraction = (
        qc_new / max(in_cloud_condensate, 1.0e-8)
    ).clamp(min=0.0, max=1.0)
    diagnosed_cloud_fraction = torch.maximum(
        plume_cloud_fraction,
        environment_cloud_fraction,
    ).clamp(max=float(params.get('shallow_plume_cloud_fraction_max', 0.30)))
    active_layer = height <= maximum_height
    t_new = torch.where(active_layer, t_new, t)
    q_new = torch.where(active_layer, q_new, q)
    qc_new = torch.where(active_layer, qc_new, qc)
    diagnosed_cloud_fraction = torch.where(
        active_layer, diagnosed_cloud_fraction, torch.zeros_like(diagnosed_cloud_fraction)
    )
    energy_error = torch.sum(
        (cp * (t_new - t) + Lv * (q_new - q)) * mass, dim=1
    )
    temperature_correction = energy_error / (cp * mass[:, -1].clamp(min=1.0e-8))
    t_new[:, -1] = t_new[:, -1] - temperature_correction

    return {
        'dt': (t_new - t) / timestep,
        'dq': (q_new - q) / timestep,
        'dqc': (qc_new - qc) / timestep,
        'cloud_base_mass_flux': cloud_mass_flux,
        'cloud_fraction': diagnosed_cloud_fraction,
        'plume_condensate': plume_condensate_profile,
        'plume_mass_flux_profile': plume_mass_flux_profile,
        'condensate_detrainment': condensate_detrainment,
        'plume_top_height_m': plume_top_height,
        'plume_cloud_base_height_m': plume_cloud_base_height,
        'maximum_plume_condensate_kgkg': maximum_plume_condensate,
        'water_residual': torch.sum(
            (q_new + qc_new - q - qc) * mass, dim=1
        ) / timestep,
        'energy_residual': torch.sum(
            (cp * (t_new - t) + Lv * (q_new - q)) * mass, dim=1
        ) / timestep,
    }
