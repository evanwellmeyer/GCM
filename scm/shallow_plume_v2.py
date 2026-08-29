import torch

from scm.thermo import Lv, cp, g, geopotential, kappa, p0, saturation_specific_humidity


def partition_plume(theta_liquid, total_water, pressure):
    exner = (pressure / p0).clamp(min=1.0e-6) ** kappa
    temperature = theta_liquid * exner
    vapor = total_water
    for _ in range(6):
        saturation = saturation_specific_humidity(temperature, pressure)
        vapor = torch.minimum(total_water, saturation)
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
    cloud_mass_flux = torch.zeros(batch, device=t.device, dtype=t.dtype)
    entrainment = float(params.get('shallow_plume_entrainment_m1', 1.5e-3))
    maximum_height = float(params.get('shallow_plume_top_m', 2500.0))
    base_mass_flux = float(params.get('shallow_plume_mass_flux_kgm2s', 0.03))
    temperature_excess = float(params.get('shallow_plume_temperature_excess_k', 0.15))

    for column in range(batch):
        source = levels - 1
        plume_theta = theta_liquid[column, source] + temperature_excess
        plume_water = total_water[column, source]
        mass_flux = torch.as_tensor(base_mass_flux, device=t.device, dtype=t.dtype)

        for lower in range(levels - 1, 0, -1):
            upper = lower - 1
            interface_height = 0.5 * (height[column, lower] + height[column, upper])
            if interface_height > maximum_height:
                break
            depth = (height[column, upper] - height[column, lower]).clamp(min=1.0)
            retained = torch.exp(torch.as_tensor(-entrainment, device=t.device) * depth)
            plume_theta = retained * plume_theta + (1.0 - retained) * theta_liquid[column, lower]
            plume_water = retained * plume_water + (1.0 - retained) * total_water[column, lower]
            plume_temperature, plume_vapor, plume_condensate = partition_plume(
                plume_theta, plume_water, p[column, upper]
            )
            environment_virtual = t[column, upper] * (
                1.0 + 0.61 * q[column, upper] - qc[column, upper]
            )
            plume_virtual = plume_temperature * (
                1.0 + 0.61 * plume_vapor - plume_condensate
            )
            buoyancy = (plume_virtual - environment_virtual) / environment_virtual.clamp(min=150.0)
            if buoyancy <= 0.0:
                break

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
            cloud_mass_flux[column] = mass_flux

    mse_tendency = (mse_flux[:, 1:] - mse_flux[:, :-1]) / mass
    water_tendency = (water_flux[:, 1:] - water_flux[:, :-1]) / mass
    mse_new = mse + timestep * mse_tendency
    water_new = torch.clamp(total_water + timestep * water_tendency, min=1.0e-8)
    t_new, q_new, qc_new = partition_mse(water_new, mse_new, height, p)
    active_layer = height <= maximum_height
    t_new = torch.where(active_layer, t_new, t)
    q_new = torch.where(active_layer, q_new, q)
    qc_new = torch.where(active_layer, qc_new, qc)
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
        'water_residual': torch.sum(
            (q_new + qc_new - q - qc) * mass, dim=1
        ) / timestep,
        'energy_residual': torch.sum(
            (cp * (t_new - t) + Lv * (q_new - q)) * mass, dim=1
        ) / timestep,
    }
