"""Conservative bulk-updraft transport, with explicit pseudoadiabatic rain.

Interface fluxes are positive upward; arrays run from top to bottom.
This is a development operator, not a port of an established closure.
"""

import torch

from scm.thermo import cp, Lv, g, eps, saturation_specific_humidity


def transport(energy, water, mass, flow, plumeenergy, plumewater, rain):
    """Return tendencies and fluxes for a closed plume/environment circulation.

Each internal interface has upward plume mass and equal downward environmental
mass. The descending donor is the layer above the interface. Transport dry
static energy, not temperature, so displacement includes gravitational work.
Rain is an explicit layer vapor sink in kg m-2 s-1; its latent heating cancels
in the moist-energy budget. Geopotential is fixed during this process step.
"""
    heatflux = torch.zeros_like(flow)
    waterflux = torch.zeros_like(flow)
    donorenergy = torch.where(flow[:, 1:-1] >= 0, energy[:, :-1], energy[:, 1:])
    donorwater = torch.where(flow[:, 1:-1] >= 0, water[:, :-1], water[:, 1:])
    heatflux[:, 1:-1] = flow[:, 1:-1] * (plumeenergy[:, 1:-1] - donorenergy)
    waterflux[:, 1:-1] = flow[:, 1:-1] * (plumewater[:, 1:-1] - donorwater)
    heating = (heatflux[:, 1:] - heatflux[:, :-1] + Lv * rain) / mass
    moistening = (waterflux[:, 1:] - waterflux[:, :-1] - rain) / mass
    return heating / cp, moistening, heatflux, waterflux


def updraft(t, q, height, pressure, thickness, entrainment, detrainment,
            decay, buoyancyweight, floor=0.):
    """Build a mass-continuous entraining plume per unit cloud-base mass flux.

Mix conserved scalars at the destination height, then condense at fixed moist
enthalpy. All new condensate falls out. Lost mass, including plume termination,
detrains locally. Detrainment is already represented in the flux divergence;
adding another local replacement tendency would count it twice.
"""
    energy = cp * t + g * height
    mass = thickness / g
    shape = (t.shape[0], t.shape[1] + 1)
    flow = t.new_zeros(shape)
    plumeenergy = t.new_zeros(shape)
    plumewater = t.new_zeros(shape)
    rain = torch.zeros_like(t)
    exchange = torch.zeros_like(t)
    current = torch.ones_like(t[:, -1])
    sensible = energy[:, -1].clone()
    vapor = q[:, -1].clone()
    flow[:, -2] = current
    plumeenergy[:, -2] = sensible
    plumewater[:, -2] = vapor
    exchange[:, -1] = -current

    for level in range(t.shape[1] - 2, -1, -1):
        span = pressure[:, level + 1] - pressure[:, level]
        incoming = current
        expanded = incoming * torch.exp((entrainment * span).clamp(max=5.0))
        added = expanded - incoming
        sensible = (incoming * sensible + added * energy[:, level]) / expanded.clamp(min=1e-30)
        vapor = (incoming * vapor + added * q[:, level]) / expanded.clamp(min=1e-30)
        dry = (sensible - g * height[:, level]) / cp
        lower = torch.zeros_like(vapor)
        # Respect the host vapor floor inside phase conversion, rather than
        # letting a later host clamp add back water that was counted as rain.
        upper = (vapor - floor).clamp(min=0.0)
        for iteration in range(40):
            liquid = (lower + upper) / 2
            saturated = saturation_specific_humidity(dry + Lv / cp * liquid, pressure[:, level])
            excess = vapor - liquid - saturated
            lower = torch.where(excess > 0, liquid, lower)
            upper = torch.where(excess > 0, upper, liquid)
        liquid = (lower + upper) / 2
        liquid = torch.where(vapor > saturation_specific_humidity(dry, pressure[:, level]),
                             liquid, torch.zeros_like(liquid))
        sensible = sensible + Lv * liquid
        vapor = vapor - liquid
        rain[:, level] = expanded * liquid
        temperature = (sensible - g * height[:, level]) / cp
        plumevirtual = temperature * (1 + (1 / eps - 1) * vapor)
        virtual = t[:, level] * (1 + (1 / eps - 1) * q[:, level])
        buoyant = torch.sigmoid((plumevirtual - virtual) * 5)
        loss = detrainment * (1 - buoyancyweight * buoyant) + decay * (1 - buoyant)
        current = expanded * torch.exp(-(loss * span).clamp(max=5.0))
        if level == 0:
            current = torch.zeros_like(current)
        exchange[:, level] = incoming - current
        flow[:, level] = current
        plumeenergy[:, level] = sensible
        plumewater[:, level] = vapor

    heating, moistening, heatflux, waterflux = transport(
        energy, q, mass, flow, plumeenergy, plumewater, rain)
    return {'dt': heating, 'dq': moistening, 'rain': rain,
            'massflux': flow, 'heatflux': heatflux, 'waterflux': waterflux,
            'exchange': exchange}


def downdraft(t, q, height, pressure, thickness, sigma, rain, fraction, params):
    """Rain-fed descending plume with compensating upward environmental flow.

Rain is supplied explicitly by the updraft, never inferred from advective
drying. Evaporation solves the saturation deficit including latent cooling.
Entrainment and detrainment rates are per pascal, not per model layer.
"""
    energy = cp * t + g * height
    shape = (t.shape[0], t.shape[1] + 1)
    flow = t.new_zeros(shape)
    sensibleflux = t.new_zeros(shape)
    vaporflux = t.new_zeros(shape)
    evaporation = torch.zeros_like(t)
    start = (sigma - float(params.get('mf_downdraft_start_sigma', .60))).abs().argmin(dim=1)
    entrainment = float(params.get('mf_downdraft_entrainment_pa', 1e-5))
    detrainment = float(params.get('mf_downdraft_detrainment_pa', 1e-5))
    release = float(params.get('mf_downdraft_release_pa', 1.2e-4))
    share = min(max(float(params.get('mf_downdraft_rain_share', .5)), 0.), 1.)
    current = torch.zeros_like(t[:, 0])
    sensible = energy[:, 0].clone()
    vapor = q[:, 0].clone()
    available = torch.zeros_like(current)
    for level in range(t.shape[1]):
        available = available + share * rain[:, level]
        launching = (start == level) & (available > 0)
        if level:
            span = pressure[:, level] - pressure[:, level - 1]
            expanded = current * torch.exp((entrainment * span).clamp(max=5))
            added = expanded - current
            sensible = (current * sensible + added * energy[:, level]) / expanded.clamp(min=1e-30)
            vapor = (current * vapor + added * q[:, level]) / expanded.clamp(min=1e-30)
            current = expanded
        current = torch.where(launching, fraction.clamp(min=0), current)
        sensible = torch.where(launching, energy[:, level], sensible)
        vapor = torch.where(launching, q[:, level], vapor)
        temperature = (sensible - g * height[:, level]) / cp
        lower = torch.zeros_like(current)
        temperature = torch.where(current > 0, temperature, t[:, level])
        # Bound the solver before evaluating saturation at unphysical temperatures.
        upper = torch.minimum(available / current.clamp(min=1e-30),
                              ((temperature - 150) * cp / Lv).clamp(min=0))
        upper = torch.where(current > 0, upper, torch.zeros_like(upper))
        for iteration in range(40):
            uptake = (lower + upper) / 2
            deficit = saturation_specific_humidity(
                temperature - Lv / cp * uptake, pressure[:, level]) - vapor - uptake
            lower = torch.where(deficit > 0, uptake, lower)
            upper = torch.where(deficit > 0, upper, uptake)
        uptake = (lower + upper) / 2
        evaporation[:, level] = current * uptake
        available = (available - evaporation[:, level]).clamp(min=0)
        sensible = sensible - Lv * uptake
        vapor = vapor + uptake
        if level < t.shape[1] - 1:
            span = pressure[:, level + 1] - pressure[:, level]
            rate = torch.where(sigma[:, level] >= float(params.get('mf_downdraft_release_sigma', .90)),
                               release, detrainment)
            current = current * torch.exp(-(rate * span).clamp(max=5))
            flow[:, level + 1] = -current
            sensibleflux[:, level + 1] = sensible
            vaporflux[:, level + 1] = vapor
    heating, moistening, heatflux, waterflux = transport(
        energy, q, thickness / g, flow, sensibleflux, vaporflux, -evaporation)
    return {'dt': heating, 'dq': moistening, 'evaporation': evaporation,
            'massflux': flow, 'heatflux': heatflux, 'waterflux': waterflux}
