import torch

from scm.boundary_layer import _as_batch_tensor, diagnose_boundary_layer_depth, tridiag_solve
from scm.boundary_layer_edmf import edmf_updraft
from scm.thermo import (
    Lv, Rd, cp, g, geopotential, kappa, p0,
    saturation_specific_humidity, virtual_temperature,
)


def tke_edmf_boundary_layer(state, grid, params):
    """Advance one-equation TKE, local diffusion, and the shallow updraft."""

    dt = float(params.get('dt', 900.0))
    tkeoutput = advance_tke(state, grid, params)
    local = diffuse_conservative_variables(
        state, grid, params, tkeoutput['diffusivity'], tkeoutput['boundary_depth']
    )

    mixed = dict(state)
    mixed['t'] = state['t'] + dt * local['dt']
    mixed['q'] = state['q'] + dt * local['dq']
    mixed['qc'] = state.get('qc', torch.zeros_like(state['q'])) + dt * local['dqc']
    velocity = torch.sqrt(2.0 * tkeoutput['surface_tke'].clamp(min=1.0e-4))
    areafraction = float(params.get('tke_updraft_area_fraction', 0.05))
    areafraction = min(max(areafraction, 1.0e-3), 1.0)
    turnover = tkeoutput['boundary_depth'] / (areafraction * velocity.clamp(min=0.05))
    turnover = turnover.clamp(
        min=float(params.get('tke_updraft_tau_min_s', 3600.0)),
        max=float(params.get('tke_updraft_tau_max_s', 86400.0)),
    )
    plumeparams = dict(params)
    plumeparams['edmf_updraft_tau_s'] = turnover
    updraft = edmf_updraft(
        mixed, grid, plumeparams, tkeoutput['boundary_depth']
    )

    return {
        'dt': local['dt'] + updraft['dt'],
        'dq': local['dq'] + updraft['dq'],
        'dqc': local['dqc'] + updraft['dqc'],
        'dtke': tkeoutput['dtke'],
        'boundary_layer_depth_m': tkeoutput['boundary_depth'],
        'edmf_activity': updraft['activity'],
        'edmf_mass_flux_kgm2s': updraft['mass_flux'],
        'edmf_condensate_kgm2s': updraft['condensate'],
        'tke_mean_m2s2': tkeoutput['mean_tke'],
        'tke_max_diffusivity_m2s': tkeoutput['max_diffusivity'],
    }


def advance_tke(state, grid, params):
    t = state['t']
    q = state['q']
    p = state['p']
    u = state.get('u', torch.zeros_like(t))
    v = state.get('v', torch.zeros_like(t))
    batch, nlevels = t.shape
    dtype = t.dtype
    device = t.device
    dt = float(params.get('dt', 900.0))
    tke = state.get(
        'tke',
        torch.full_like(t, float(params.get('tke_initial_m2s2', 0.05))),
    )

    height = geopotential(t, q, p, grid)
    tv = virtual_temperature(t, q)
    theta = tv * (p0 / p.clamp(min=1.0)) ** kappa
    wind = _surface_wind(state, params)
    shear_floor = _as_batch_tensor(params.get('bl_shear_floor', 1.0), batch, device, dtype)
    wind2 = wind * wind + shear_floor * shear_floor
    ricrit = _as_batch_tensor(params.get('ri_crit', 0.25), batch, device, dtype)
    boundarydepth = diagnose_boundary_layer_depth(height, theta, wind2, ricrit, params)

    length = mixing_length(height, boundarydepth, params)
    interfacelength = 0.5 * (length[:, :-1] + length[:, 1:])
    interfacetke = 0.5 * (tke[:, :-1] + tke[:, 1:]).clamp(min=0.0)
    coefficient = float(params.get('tke_diffusivity_coefficient', 0.20))
    diffusivity = coefficient * interfacelength * torch.sqrt(interfacetke)

    dz = (height[:, :-1] - height[:, 1:]).clamp(min=1.0)
    thetamean = 0.5 * (theta[:, :-1] + theta[:, 1:]).clamp(min=150.0)
    stability = g * (theta[:, :-1] - theta[:, 1:]) / (thetamean * dz)
    shear = ((u[:, :-1] - u[:, 1:]) / dz) ** 2
    shear = shear + ((v[:, :-1] - v[:, 1:]) / dz) ** 2

    productioninterface = diffusivity * (
        shear - stability / float(params.get('tke_prandtl_number', 1.0))
    )
    production = torch.zeros_like(tke)
    production[:, :-1] = production[:, :-1] + 0.5 * productioninterface
    production[:, 1:] = production[:, 1:] + 0.5 * productioninterface

    drag = _as_batch_tensor(params.get('cd', 1.2e-3), batch, device, dtype)
    frictionvelocity = torch.sqrt(drag.clamp(min=0.0)) * wind
    sourcedepth = max(float(params.get('tke_surface_source_depth_m', 250.0)), 1.0)
    sourceweight = (height <= sourcedepth).to(dtype)
    sourcecount = sourceweight.sum(dim=1).clamp(min=1.0)
    surfacesource = float(params.get('tke_surface_source_coefficient', 1.0))
    surfacesource = surfacesource * frictionvelocity ** 3 / sourcedepth
    production = production + sourceweight * (surfacesource / sourcecount).unsqueeze(1)

    dissipationcoefficient = float(params.get('tke_dissipation_coefficient', 0.70))
    dissipation = dissipationcoefficient * tke.clamp(min=0.0) ** 1.5 / length.clamp(min=1.0)
    tkemin = float(params.get('tke_min_m2s2', 1.0e-4))
    tkemax = float(params.get('tke_max_m2s2', 20.0))
    tkenew = torch.clamp(tke + dt * (production - dissipation), min=tkemin, max=tkemax)

    meantke = torch.sum(tkenew * state['dp'] / g, dim=1) / torch.sum(state['dp'] / g, dim=1)
    return {
        'dtke': (tkenew - tke) / dt,
        'diffusivity': diffusivity,
        'boundary_depth': boundarydepth,
        'surface_tke': tkenew[:, -1],
        'mean_tke': meantke,
        'max_diffusivity': torch.amax(diffusivity, dim=1),
    }


def mixing_length(height, boundarydepth, params):
    vonkarman = float(params.get('tke_von_karman', 0.40))
    minimum = float(params.get('tke_min_mixing_length_m', 10.0))
    maximum = float(params.get('tke_max_mixing_length_m', 150.0))
    surfaceheight = float(params.get('tke_surface_height_m', 10.0))
    distance = height + surfaceheight
    walllength = vonkarman * distance
    depthlimit = 0.30 * boundarydepth.unsqueeze(1)
    return torch.minimum(
        torch.maximum(walllength, torch.full_like(height, minimum)),
        torch.minimum(torch.full_like(height, maximum), depthlimit),
    )


def diffuse_conservative_variables(state, grid, params, diffusivity, boundarydepth):
    t = state['t']
    q = state['q']
    qc = state.get('qc', torch.zeros_like(q))
    p = state['p']
    dp = state['dp']
    batch, nlevels = t.shape
    dt = float(params.get('dt', 900.0))
    height = geopotential(t, q, p, grid)
    interfaceheight = 0.5 * (height[:, :-1] + height[:, 1:])
    depthfactor = (
        (boundarydepth.unsqueeze(1) - interfaceheight)
        / boundarydepth.unsqueeze(1).clamp(min=1.0)
    ).clamp(min=0.0, max=1.0) ** 2
    diffusivity = diffusivity * depthfactor
    maximum = float(params.get('tke_max_diffusivity_m2s', 100.0))
    diffusivity = diffusivity.clamp(min=0.0, max=maximum)

    pressureupper = p[:, :-1].clamp(min=1.0)
    pressurelower = p[:, 1:].clamp(min=1.0)
    pressuredepth = (pressurelower - pressureupper).clamp(min=100.0)
    density = pressureupper / (Rd * t[:, :-1].clamp(min=150.0))
    pressurecoefficient = diffusivity * g * density * density / pressuredepth
    mass = dp / g

    a = torch.zeros(batch, nlevels, device=t.device, dtype=t.dtype)
    b = torch.ones(batch, nlevels, device=t.device, dtype=t.dtype)
    c = torch.zeros(batch, nlevels, device=t.device, dtype=t.dtype)
    below = dt * g * pressurecoefficient / mass[:, :-1]
    above = dt * g * pressurecoefficient / mass[:, 1:]
    c[:, :-1] = -below
    b[:, :-1] = b[:, :-1] + below
    a[:, 1:] = -above
    b[:, 1:] = b[:, 1:] + above

    totalwater = q + qc
    totalwaternew = tridiag_solve(a, b, c, totalwater, 0, nlevels)
    liquidenergy = cp * t + Lv * q + g * height
    liquidenergynew = tridiag_solve(a, b, c, liquidenergy, 0, nlevels)
    tnew, qnew, qcnew = partition_moist_state(
        liquidenergynew, totalwaternew, height, p
    )
    return {
        'dt': (tnew - t) / dt,
        'dq': (qnew - q) / dt,
        'dqc': (qcnew - qc) / dt,
    }


def partition_moist_state(liquidenergy, totalwater, height, pressure):
    """Recover temperature, vapor, and condensate from conserved moist variables."""

    unsaturatedtemperature = (liquidenergy - Lv * totalwater - g * height) / cp
    unsaturatedlimit = saturation_specific_humidity(unsaturatedtemperature, pressure)
    saturated = totalwater > unsaturatedlimit

    saturatedtemperature = unsaturatedtemperature.clone()
    for _ in range(5):
        saturation = saturation_specific_humidity(saturatedtemperature, pressure)
        residual = cp * saturatedtemperature + Lv * saturation + g * height - liquidenergy
        slope = cp + Lv * Lv * saturation / (461.5 * saturatedtemperature ** 2)
        saturatedtemperature = saturatedtemperature - residual / slope.clamp(min=cp)

    temperature = torch.where(saturated, saturatedtemperature, unsaturatedtemperature)
    vapor = torch.where(
        saturated,
        saturation_specific_humidity(temperature, pressure),
        totalwater,
    )
    condensate = torch.clamp(totalwater - vapor, min=0.0)
    return temperature, vapor, condensate


def _surface_wind(state, params):
    batch = state['t'].shape[0]
    device = state['t'].device
    dtype = state['t'].dtype
    value = params.get(
        'relative_wind_speed_cell',
        params.get('relative_wind_speed', params.get('surface_wind_speed', params.get('wind_speed', 5.0))),
    )
    return _as_batch_tensor(value, batch, device, dtype)
