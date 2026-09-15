"""PyTorch implementation of UW shallow-convection physics."""

import math
import torch

from scm.boundary_layer import diagnose_boundary_layer_depth
from scm.phase_partition import partition_water
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
    """Advance shallow physics with optional conservative internal substeps.

    The inversion displacement is a finite-time flux, not an instantaneous
    tendency. Reconstruct it from the evolving state on each internal step.
    A zero maximum retains the legacy single-step path for comparison.
    """
    timestep = float(params.get('dt', 900.))
    maximum = float(params.get('uw_shallow_maximum_timestep_s', 0.))
    if not math.isfinite(timestep) or timestep <= 0.:
        raise ValueError('shallow timestep must be finite and positive')
    if not math.isfinite(maximum) or maximum < 0.:
        raise ValueError('maximum shallow timestep must be finite and nonnegative')
    if maximum == 0. or timestep <= maximum:
        return shallow_step(state, grid, params)

    count = math.ceil(timestep / maximum)
    step = timestep / count
    settings = dict(params, dt=step)
    evolved = dict(state)
    for name in ('t', 'q', 'qc', 'u', 'v'):
        evolved[name] = state.get(name, torch.zeros_like(state['t'])).clone()
    initial = {name: evolved[name].clone() for name in ('t', 'q', 'qc', 'u', 'v')}
    averaged = {}
    for _ in range(count):
        result = shallow_step(evolved, grid, settings)
        for name, value in result.items():
            # Rate and profile diagnostics are interval means. Keep the named
            # condensate maximum as a maximum over the internal steps.
            if name in ('maximum_plume_condensate_kgkg', 'plume_sorting_scale_relative_error'):
                averaged[name] = torch.maximum(averaged[name], value) if name in averaged else value
            else:
                averaged[name] = averaged.get(name, 0.) + value / count
        for name in initial:
            evolved[name] = evolved[name] + step * result['d' + name]

    # Return endpoint tendencies so the caller applies exactly this update.
    for name in initial:
        averaged['d' + name] = (evolved[name] - initial[name]) / timestep
    mass = state['dp'].double() / g
    water = (evolved['q'].double() + evolved['qc'].double()
             - initial['q'].double() - initial['qc'].double())
    energy = (cp * (evolved['t'].double() - initial['t'].double())
              + Lv * (evolved['q'].double() - initial['q'].double()))
    averaged['water_residual'] = (water * mass).sum(dim=1) / timestep + averaged['precip']
    averaged['energy_residual'] = (energy * mass).sum(dim=1) / timestep
    averaged['mse_residual'] = averaged['energy_residual']
    return averaged


def shallow_step(state, grid, params):
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
        state.get('tke_interfaces'),
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
    critical = float(params.get("condensation_rh_crit", 1.0))
    if bool(params.get("uw_shallow_layer_mean_saturation", False)):
        t_new, q_new, qc_new = partition_layer_mean(
            water_new,
            mse_new,
            height,
            p,
            dp,
        )
    elif critical < 1.0:
        # Use the partial-cloud partition that condensation and UW turbulence use.
        # With full saturation here, each step evaporated the cloud water that
        # condensation had just made above the critical RH, and condensation made
        # it again: a 46 K/day loop at 475 m in BOMEX (11 Sep 2026).
        t_new, q_new, qc_new, _ = partition_water(water_new, mse_new - g * height, p, critical)
    else:
        t_new, q_new, qc_new = partition_mse(water_new, mse_new, height, p)

    active = height <= float(params.get("uw_shallow_maximum_height_m", 4000.0))
    # A plume may enter only the lower part of a cell whose centre is above
    # the height ceiling. Do not discard that cell's conservative flux update.
    active = active | (water_tendency != 0.0) | (mse_tendency != 0.0)
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
        "plume_sorting_distance_m": outputs["sorting_distance"],
        "plume_sorting_scale_relative_error": outputs["sorting_scale_error"],
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


def source_slope(value, pressure):
    """CAM's pressure reconstruction, with host top-to-bottom ordering."""
    field, levels = value.flip(0), pressure.flip(0)
    slope = torch.zeros_like(field)
    below = (field[1] - field[0]) / (levels[1] - levels[0])
    for k in range(1, len(field)):
        above = (field[k] - field[k - 1]) / (levels[k] - levels[k - 1])
        slope[k - 1] = torch.where(above * below > 0., torch.sign(above) *
                                  torch.minimum(above.abs(), below.abs()), above * 0.)
        below = above
    slope[-1] = slope[-2]
    return slope.flip(0)


def source_properties(pressure, thickness, height, theta, water, u, v, tke, depth, interfaces=None):
    """Reconstruct source air and average TKE over its subcloud support."""
    faces = torch.cat((pressure[-1:] + thickness[-1:] / 2. - thickness.sum(),
                       pressure[-1:] + thickness[-1:] / 2. - thickness.sum() + thickness.cumsum(0)))
    heights = torch.empty_like(faces)
    for k in range(len(faces)):
        lower = max(1, min(k, len(pressure) - 1))
        share = (faces[k] - pressure[lower - 1]) / (pressure[lower] - pressure[lower - 1])
        heights[k] = height[lower - 1] + share * (height[lower] - height[lower - 1])
    candidates = torch.nonzero((heights[:-1] > depth + 5.) & (heights[1:] <= depth + 5.)).flatten()
    if candidates.numel() == 0 or int(candidates[0]) >= len(pressure) - 1:
        return None
    source = int(candidates[0]) + 1
    thetagradient, watergradient = source_slope(theta, pressure), source_slope(water, pressure)
    virtual = []
    for face in (faces[source:-1], faces[source + 1:]):
        offset = face - pressure[source:]
        virtual.append((theta[source:] + thetagradient[source:] * offset) *
                       (1. + .608 * (water[source:] + watergradient[source:] * offset)))
    sourcetheta = torch.cat(virtual).min() / (1. + .608 * water[-1])
    # Native interior TKE is retained. The turbulence module has no surface
    # prognostic value; use its nearest-interior zero-gradient boundary value.
    # Old checkpoints without interfaces use an explicitly approximate mapping.
    if interfaces is None:
        interfaces = .5 * (tke[:-1] + tke[1:])
    native = torch.cat((interfaces[:1], interfaces, interfaces[-1:]))
    weights = torch.cat((pressure[source:source + 1] - faces[source:source + 1],
                         pressure[source + 1:] - pressure[source:-1],
                         faces[-1:] - pressure[-1:]))
    average = (native[source:] * weights).sum() / weights.sum()
    wind = u[source] + source_slope(u, pressure)[source] * (faces[source] - pressure[source])
    crosswind = v[source] + source_slope(v, pressure)[source] * (faces[source] - pressure[source])
    return dict(index=source, faces=faces, heights=heights, theta=sourcetheta,
                water=water[-1], u=wind, v=crosswind, tke=average)


def subcloud_flux(massflux, faces, source, timestep, value, mean, top, bottom):
    """Pressure-linear CAM subcloud flux including inversion displacement."""
    thickness = faces[source] - faces[source - 1]
    contrast = bottom - top
    denominator = torch.where(contrast >= 0., contrast.clamp(min=1e-20), contrast.clamp(max=-1e-20))
    position = ((mean - top) / denominator).clamp(0., 1.)
    original = contrast
    bottom = torch.where((position == 0.) | (position == 1.), mean, bottom)
    fraction = massflux * g * timestep / thickness
    inversion = faces[source] - position * thickness
    flux = massflux * (value - bottom) * (faces[-1] - faces[source:]) / (faces[-1] - inversion)
    flux[0] += (1. - position / fraction).clamp(min=0.) * massflux * original
    return flux


def subcloud_transport(result, column, launch, massflux, theta, water, u, v, pressure, timestep):
    source, faces = launch['index'], launch['faces']
    fluxes = {}
    for name, field in [('theta', theta), ('water', water), ('u', u), ('v', v)]:
        slope = source_slope(field, pressure)
        bottom = field[source] + slope[source] * (faces[source] - pressure[source])
        upper = max(0, source - 2)
        top = field[upper] + slope[upper] * (faces[source - 1] - pressure[upper])
        fluxes[name] = subcloud_flux(massflux, faces, source, timestep, launch[name],
                                     field[source - 1], top, bottom)
    result['water_flux'][column, source:] = fluxes['water']
    # Convert theta_l transport to liquid static energy at the same face,
    # then add latent total-water transport for the host's MSE convention.
    result['mse_flux'][column, source:] = cp * (faces[source:] / p0) ** kappa * fluxes['theta'] + Lv * fluxes['water']
    result['u_flux'][column, source:] = fluxes['u']
    result['v_flux'][column, source:] = fluxes['v']


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
        launch = source_properties(p[column], dp[column], height[column], predicted_theta[column],
                                   predicted_water[column], torch.zeros_like(p[column]),
                                   torch.zeros_like(p[column]), tke[column], boundary_depth[column])
        if launch is None or outputs["cloud_base_mass_flux"][column] <= 0.0:
            continue
        source = launch['index']
        source_theta, source_water = launch['theta'], launch['water']
        predicted_cin, _ = undilute_cin(
            source_theta,
            source_water,
            source,
            predicted_t[column],
            predicted_q[column],
            predicted_qc[column],
            p[column],
            height[column],
            launch['heights'][source],
        )
        change = predicted_cin - outputs["cin"][column]
        factor[column] = implicit_cin_factor(change, outputs['source_tke'][column])

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
    interfaces=None,
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
        "source_tke": torch.zeros(batch, device=t.device, dtype=t.dtype),
        "sorting_distance": torch.zeros(batch, device=t.device, dtype=t.dtype),
        "sorting_scale_error": torch.zeros(batch, device=t.device, dtype=t.dtype),
        "precipitation_source": torch.zeros_like(t),
    }
    for column in range(batch):
        diagnosed = bool(params.get('uw_shallow_diagnose_sorting_distance', False))
        iterations = int(params.get('uw_shallow_sorting_iterations', 6)) if diagnosed else 1
        fraction = float(params.get('uw_shallow_sorting_fraction', .1))
        if iterations < 1 or not 0. < fraction <= 1.:
            raise ValueError('invalid sorting-scale iteration settings')
        local = dict(params)
        for _ in range(iterations):
            # Each trial is a fresh plume, not an additional physical update.
            for value in result.values():
                value[column] = 0.
            _integrate_one_column(column, result, t, q, qc, u, v, p, dp, height,
                                  theta_liquid, total_water, mse, tke, boundary_depth,
                                  local, interfaces)
            top = float(result['plume_top_height'][column])
            if not diagnosed or top <= 0.:
                break
            target = max(1., fraction * top)
            used = float(result['sorting_distance'][column])
            result['sorting_scale_error'][column] = abs(target - used) / target
            if abs(target - used) <= .01 * target:
                break
            local['uw_shallow_sorting_distance_m'] = target
    return result


def conserved_environment(pressure, center, values, slopes, height, critical=1.):
    """Reconstruct one layer's conserved state before diagnosing its phases."""
    theta, water, wind, crosswind = [value + slope * (pressure - center)
                                   for value, slope in zip(values, slopes)]
    if critical < 1.:
        energy = cp * (pressure / p0) ** kappa * theta + Lv * water
        temperature, vapor, liquid, _ = partition_water(water, energy, pressure, critical)
    else:
        temperature, vapor, liquid = partition_plume(theta, water, pressure, iterations=32)
    energy = cp * temperature + Lv * vapor + g * height
    return pressure, temperature, vapor, liquid, theta, water, wind, crosswind, energy


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
    interfaces=None,
):
    levels = t.shape[1]
    launch = source_properties(p[column], dp[column], height[column],
                               theta_liquid[column], total_water[column],
                               u[column], v[column], tke[column], boundary_depth[column],
                               None if interfaces is None else interfaces[column])
    if launch is None:
        return
    source = launch['index']
    plume_theta, plume_water = launch['theta'], launch['water']
    plume_u, plume_v = launch['u'], launch['v']
    cin, reaches_lfc = undilute_cin(
        plume_theta,
        plume_water,
        source,
        t[column],
        q[column],
        qc[column],
        p[column],
        height[column],
        launch['heights'][source],
    )
    result["cin"][column] = cin
    if not reaches_lfc:
        return

    source_density = p[column, source] / (Rd * t[column, source].clamp(min=150.0))
    source_tke = launch['tke'].clamp(min=1.0e-4)
    result['source_tke'][column] = source_tke
    mass_flux = cloud_base_mass_flux(source_density, source_tke, cin)
    velocity_squared = (2.0 * source_tke - 2.0 * cin).clamp(min=0.05)
    velocity = torch.sqrt(velocity_squared)
    area_max = float(params.get("uw_shallow_core_area_max", 0.10))
    mass_flux = torch.minimum(mass_flux, area_max * source_density * velocity)
    if mass_flux <= 1.0e-10:
        return
    result["cloud_base_mass_flux"][column] = mass_flux
    subcloud_transport(result, column, launch, mass_flux, theta_liquid[column],
                       total_water[column], u[column], v[column], p[column],
                       float(params.get('dt', 900.)))

    maximum = float(params.get("uw_shallow_maximum_height_m", 4000.0))
    spacing = float(params.get("uw_shallow_vertical_step_m", 50.0))
    if spacing <= 0.0:
        raise ValueError("uw_shallow_vertical_step_m must be positive")
    efficiency = float(params.get("uw_shallow_mixing_efficiency", 8.0))
    drag = float(params.get("uw_shallow_velocity_drag", 1.0))
    distance = float(params.get("uw_shallow_sorting_distance_m", 100.0))
    if distance <= 0.0:
        raise ValueError("uw_shallow_sorting_distance_m must be positive")
    result['sorting_distance'][column] = distance
    cloudmultiplier = float(params.get("uw_shallow_cloud_area_multiplier", 2.0))
    condensatemaximum = float(params.get("uw_shallow_condensate_maximum_kgkg", 1.0e-3))
    conserved = bool(params.get('uw_shallow_conserved_environment', False))
    environmentpartition = params.get('uw_shallow_environment_partition', 'host')
    if environmentpartition not in ('host', 'saturation'):
        raise ValueError('environment partition must be host or saturation')
    environmentcritical = float(params.get('condensation_rh_crit', 1.)) if environmentpartition == 'host' else 1.
    if conserved:
        slopes = [source_slope(field[column], p[column]).double()
                  for field in (theta_liquid, total_water, u, v)]

    # Float64 internal states keep adaptive error estimates meaningful even
    # when the host column stores float32. Output tensors retain the host dtype.
    values = torch.stack((plume_theta, plume_water, plume_u, plume_v,
                          torch.log(mass_flux), velocity_squared)).double()
    tolerance = values.new_tensor([4e-6, 4e-10, 2e-5, 2e-5, 2e-5, 4e-6])
    position = float(launch['heights'][source])
    stopped = False
    crossed = []

    for lower in range(source, 0, -1):
        upper = lower - 1
        bottom = float(height[column, lower])
        top = float(height[column, upper])
        if bottom >= maximum:
            break
        interface = float(launch['heights'][lower])
        environmentcache = {}

        def environment(z, cell=None):
            share = (z - bottom) / (top - bottom)
            if conserved:
                index = layer if cell is None else cell
                key = (index, float(z))
                if key in environmentcache:
                    return environmentcache[key]
                pressure = p[column, lower].double() + share * (p[column, upper].double() - p[column, lower].double())
                fields = (theta_liquid, total_water, u, v)
                reconstructed = conserved_environment(pressure, p[column, index].double(),
                                             [field[column, index].double() for field in fields],
                                             [slope[index] for slope in slopes], z, environmentcritical)
                environmentcache[key] = reconstructed
                return reconstructed
            fields = (p, t, q, qc, theta_liquid, total_water, u, v, mse)
            return tuple(field[column, lower].double() + share *
                         (field[column, upper].double() - field[column, lower].double())
                         for field in fields)

        def rates(z, value):
            pressure, temperature, vapor, liquid, theta, water, wind, crosswind, energy = environment(z)
            parcel = partition_plume(value[0], value[1], pressure, iterations=32)
            virtual = temperature * (1.0 + .61 * vapor - liquid)
            parcelvirtual = parcel[0] * (1.0 + .61 * parcel[1] - parcel[2])
            buoyancy = g * (parcelvirtual - virtual) / virtual
            fraction = critical_mixing_fraction(
                value[0], value[1], theta, water, pressure, virtual,
                value[5].clamp(min=0.0), distance)
            # CAM's sorting rates: entrainment = rei*x^2,
            # detrainment = rei*(1-x)^2. These are rates per metre, not
            # fractions per numerical step.
            rate = efficiency / max(z, 50.0)
            entrainment = rate * fraction.square()
            detrainment = rate * (1.0 - fraction).square()
            return torch.stack((
                entrainment * (theta - value[0]),
                entrainment * (water - value[1]),
                entrainment * (wind - value[2]),
                entrainment * (crosswind - value[3]),
                entrainment - detrainment,
                2.0 * buoyancy - 2.0 * drag * entrainment * value[5],
            ))

        def advance(z, value, step):
            first = rates(z, value)
            middle = value + .5 * step * first
            updated = value + step * rates(z + .5 * step, middle)
            pressure = environment(z + step)[0]
            temperature, vapor, liquid = partition_plume(updated[0], updated[1], pressure, iterations=32)
            removed = (liquid - condensatemaximum).clamp(min=0.0)
            # Removing liquid leaves temperature and vapor unchanged, so
            # theta_l increases; keeping theta_l fixed would spuriously cool.
            correction = torch.stack((
                Lv * removed / (cp * (pressure / p0) ** kappa),
                -removed, removed * 0, removed * 0, removed * 0, removed * 0))
            return updated + correction, torch.exp(updated[4]) * removed

        # End a step exactly at each transport interface and model centre.
        # No scalar or velocity from a different height is reused for a flux.
        for target, layer in ((interface, lower), (top, upper)):
            target = min(target, maximum)
            if target <= position:
                continue
            step = min(spacing, target - position)
            attempts = 0
            while target - position > 1e-7:
                attempts += 1
                if attempts > 20000:
                    raise RuntimeError("UW plume integration did not converge")
                step = min(step, target - position)
                whole, _ = advance(position, values, step)
                half, rainfirst = advance(position, values, .5 * step)
                refined, rainsecond = advance(position + .5 * step, half, .5 * step)
                error = float(torch.max(torch.abs(refined - whole) / tolerance))
                if not math.isfinite(error):
                    raise RuntimeError("nonfinite UW plume integration error")
                if error > 1.0:
                    if step <= 1e-4:
                        raise RuntimeError("UW plume integration reached its minimum step")
                    step *= max(.1, .8 * error ** (-1.0 / 3.0))
                    continue
                if refined[5] <= 0.0:
                    # Locate the stopping event, without exporting into a cell
                    # whose interface was never crossed.
                    share = (values[5] / (values[5] - refined[5]).clamp(min=1e-12)).clamp(0.0, 1.0)
                    position += float(share) * step
                    values = values + share * (refined - values)
                    result["precipitation_source"][column, layer] += (rainfirst + rainsecond) * share
                    stopped = True
                    break
                position += step
                values = refined
                result["precipitation_source"][column, layer] += rainfirst + rainsecond
                step = min(spacing, step * min(2.0, .8 * max(error, 1e-8) ** (-1.0 / 3.0)))
            result["plume_top_height"][column] = position
            if stopped:
                # Only the already-entered stopping cell gets fractional
                # cloud coverage. Keeping crossed flux is now unconditional.
                if crossed and crossed[-1] == layer:
                    base = float(launch['heights'][layer + 1])
                    ceiling = float(launch['heights'][layer])
                    share = max(0.0, min(1.0, (position - base) / (ceiling - base)))
                    result["cloud_fraction"][column, layer] *= .5 * share
                break
            if target == interface and interface < maximum:
                pressure, temperature, vapor, liquid, theta, water, wind, crosswind, energy = environment(interface, upper)
                parceltemperature, parcelvapor, parcelwater = partition_plume(values[0], values[1], pressure, iterations=32)
                velocity = torch.sqrt(values[5].clamp(min=1e-12))
                density = pressure / (Rd * parceltemperature)
                flux = torch.minimum(torch.exp(values[4]), area_max * density * velocity)
                values = torch.cat((values[:4], torch.log(flux.clamp(min=1e-30)).reshape(1), values[5:]))
                parcelenergy = cp * parceltemperature + Lv * parcelvapor + g * interface
                result["water_flux"][column, lower] = flux * (values[1] - water)
                result["mse_flux"][column, lower] = flux * (parcelenergy - energy)
                result["u_flux"][column, lower] = flux * (values[2] - wind)
                result["v_flux"][column, lower] = flux * (values[3] - crosswind)
                result["liquid_flux"][column, lower] = flux * parcelwater
                result["mass_flux_profile"][column, upper] = flux
                result["plume_condensate"][column, upper] = parcelwater
                result["maximum_condensate"][column] = torch.maximum(
                    result["maximum_condensate"][column], parcelwater.to(t.dtype))
                if parcelwater > 0:
                    if result["cloud_base_height"][column] == 0:
                        result["cloud_base_height"][column] = interface
                    area = flux / (density * velocity)
                    result["cloud_fraction"][column, upper] = (cloudmultiplier * area).clamp(max=2 * area_max)
                crossed.append(upper)
        if stopped or position >= maximum:
            break


def critical_mixing_fraction(theta, water, environmenttheta, environmentwater,
                             pressure, environmentvirtual, velocity, distance):
    """UW-style buoyancy sorting over a physical stopping distance.

    Virtual temperature is piecewise linear across the saturation point, as
    in CAM's sorting construction. The first downward root of mixed-parcel
    kinetic energy plus buoyancy work bounds the entrained fraction.
    """
    from scm.thermo import saturation_specific_humidity

    exner = (pressure / p0) ** kappa
    excess = water - saturation_specific_humidity(theta * exner, pressure)
    environmentexcess = environmentwater - saturation_specific_humidity(environmenttheta * exner, pressure)
    opposite = bool(excess * environmentexcess < 0)
    saturation = (excess / (excess - environmentexcess)).clamp(0., 1.) if opposite else torch.ones_like(excess)
    points = torch.stack((torch.zeros_like(saturation), saturation, torch.ones_like(saturation)))
    mixedtheta = theta + points * (environmenttheta - theta)
    mixedwater = water + points * (environmentwater - water)
    temperature, vapor, liquid = partition_plume(mixedtheta, mixedwater, pressure, iterations=32)
    virtual = temperature * (1 + .61 * vapor - liquid)
    # The endpoint is the supplied mean environment, not a separately
    # saturation-adjusted copy of it.
    middle = virtual[1] if opposite else environmentvirtual
    virtual = torch.stack((virtual[0], middle, environmentvirtual))
    buoyancy = g * (virtual - environmentvirtual) / environmentvirtual
    energy = velocity.clamp(min=1e-12)
    if energy + 2 * distance * buoyancy[0] <= 0:
        return torch.zeros_like(energy)
    for index in range(2):
        left, right = points[index], points[index + 1]
        if right - left < 1e-8:
            continue
        slope = (buoyancy[index + 1] - buoyancy[index]) / (right - left)
        intercept = buoyancy[index] - slope * left
        linear = -2 * energy + 2 * distance * slope
        constant = energy + 2 * distance * intercept
        discriminant = linear.square() - 4 * energy * constant
        if discriminant > 0:
            root = (-linear - torch.sqrt(discriminant)) / (2 * energy)
            if left <= root and root < right - 1e-8:
                return root.clamp(0., 1.)
    return torch.ones_like(energy)


def undilute_cin(theta_liquid, total_water, source, t, q, qc, p, height, launchheight=None):
    """Integrate negative undilute-parcel buoyancy from PBL top to LFC."""

    cin = torch.zeros((), device=t.device, dtype=t.dtype)
    reaches_lfc = False
    for lower in range(source, 0, -1):
        upper = lower - 1
        bottom = height[lower] if launchheight is None else torch.maximum(height[lower], launchheight)
        if bottom >= height[upper]:
            continue
        share = (.5 * (bottom + height[upper]) - height[lower]) / (height[upper] - height[lower])
        pressure = p[lower] + share * (p[upper] - p[lower])
        parcel_t, parcel_q, parcel_qc = partition_plume(
            theta_liquid,
            total_water,
            pressure,
        )
        environment_t = t[lower] + share * (t[upper] - t[lower])
        environment_q = q[lower] + share * (q[upper] - q[lower])
        environment_qc = qc[lower] + share * (qc[upper] - qc[lower])
        parcel_virtual = parcel_t * (1.0 + 0.61 * parcel_q - parcel_qc)
        environment_virtual = environment_t * (
            1.0 + 0.61 * environment_q - environment_qc
        )
        buoyancy = g * (
            parcel_virtual - environment_virtual
        ) / environment_virtual.clamp(min=150.0)
        layer_depth = (height[upper] - bottom).clamp(min=0.0)
        if buoyancy > 0.0:
            reaches_lfc = True
            break
        cin = cin + (-buoyancy).clamp(min=0.0) * layer_depth
    return cin, reaches_lfc
