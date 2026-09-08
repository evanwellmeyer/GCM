"""Moist stability and connected-layer closure for the UW development scheme.

Uses the layer organization and TKE relaxation of Bretherton and Park (2009).
Surface production and entrainment remain a reduced discretization; this is
not a complete reproduction of CAM UWMT's radiative entrainment treatment.
"""
import torch

from scm.thermo import cp, Lv, g, Rd, saturation_specific_humidity


def moist_stability(t, q, liquid, fraction, p, height):
    """Linearized buoyancy response to liquid static energy and total water."""
    water = q + liquid
    energy = cp * t + g * height - Lv * liquid
    middle = (t[:, :-1] + t[:, 1:]) / 2
    pressure = (p[:, :-1] + p[:, 1:]) / 2
    total = (water[:, :-1] + water[:, 1:]) / 2
    vapor = (q[:, :-1] + q[:, 1:]) / 2
    condensate = (liquid[:, :-1] + liquid[:, 1:]) / 2
    cover = torch.minimum(fraction[:, :-1], (fraction[:, :-1] + fraction[:, 1:]) / 2).clamp(0, 1)
    saturation = saturation_specific_humidity(middle, pressure)
    # Differentiate the same saturation formula used by phase partitioning.
    derivative = (saturation_specific_humidity(middle + 0.01, pressure)
                  - saturation_specific_humidity(middle - 0.01, pressure)) / 0.02
    dryheat = (1 + 0.608 * total) / cp
    drywater = 0.608 * middle
    response = 1 + 1.608 * saturation - total + 1.608 * middle * derivative
    wetheat = response / (cp + Lv * derivative)
    wetwater = Lv * wetheat - middle
    heat = (1 - cover) * dryheat + cover * wetheat
    moisture = (1 - cover) * drywater + cover * wetwater
    virtual = middle * (1 + 0.608 * vapor - condensate)
    spacing = (height[:, :-1] - height[:, 1:]).clamp(min=1)
    return g / virtual * (heat * (energy[:, :-1] - energy[:, 1:])
                          + moisture * (water[:, :-1] - water[:, 1:])) / spacing


def layer_diffusivity(t, q, liquid, fraction, u, v, p, dp, height, buoyancy, params):
    from scm.boundary_layer_uw import galperin_functions

    stability = moist_stability(t, q, liquid, fraction, p, height)
    spacing = (height[:, :-1] - height[:, 1:]).clamp(min=1)
    interface = (height[:, :-1] + height[:, 1:]) / 2
    shear = ((u[:, :-1] - u[:, 1:]).square()
             + (v[:, :-1] - v[:, 1:]).square()) / spacing.square()
    shear = shear + float(params.get('uw_shear_floor_s2', 1e-8))
    weights = (dp[:, :-1] + dp[:, 1:]) / (2 * g)
    heat = torch.zeros_like(stability)
    momentum = torch.zeros_like(stability)
    tke = torch.zeros_like(stability)
    labels = torch.full_like(stability, -1, dtype=torch.long)
    boundary = torch.zeros_like(buoyancy)
    entrainment = torch.zeros_like(buoyancy)
    b1 = float(params.get('uw_dissipation_constant', 5.8))
    limit = float(params.get('uw_diffusivity_max_m2s', 200))
    critical = float(params.get('uw_critical_ri', 0.19))
    ratio = float(params.get('uw_layer_extension_ratio', 0.04))

    for column in range(t.shape[0]):
        count = stability.shape[1]
        regions = []
        index = count - 1
        while index >= 0:
            surface = index == count - 1 and buoyancy[column] > 0
            if stability[column, index] >= 0 and not surface:
                index -= 1
                continue
            bottom = index
            top = index
            # Extend only while the stable work is a small fraction of driving.
            # The surface supplies a seed even for a neutral initial column.
            source = buoyancy[column].clamp(min=0) if surface else buoyancy[column] * 0
            while top > 0:
                trial = slice(top - 1, bottom + 1)
                thickness = (interface[column, top - 1] -
                             (interface[column, bottom + 1] if bottom + 1 < count else 0)).clamp(min=1)
                length = 0.085 * thickness
                work = -length.square() * stability[column, trial]
                mass = weights[column, trial]
                driving = (work.clamp(min=0) * mass).sum()
                driving = driving + 0.30 * (source * thickness).pow(2 / 3) * mass.sum() / b1
                cost = ((-work).clamp(min=0) * mass).sum()
                if bool(cost > ratio * driving):
                    break
                top -= 1
            regions.append((top, bottom, bool(surface)))
            index = top - 1

        # Stable shear layers receive their own contiguous-layer length scale.
        index = count - 1
        while index >= 0:
            if any(top <= index <= bottom for top, bottom, surface in regions):
                index -= 1
                continue
            if stability[column, index] / shear[column, index] >= critical:
                index -= 1
                continue
            bottom = index
            while index > 0 and 0 <= stability[column, index - 1] / shear[column, index - 1] < critical:
                index -= 1
            regions.append((index, bottom, False))
            index -= 1

        for label, (top, bottom, surface) in enumerate(regions):
            connected = bottom == count - 1
            selected = slice(top, bottom + 1)
            base = interface[column, bottom + 1] if bottom + 1 < count else interface[column, bottom] * 0
            roof = interface[column, top - 1] if top > 0 else height[column, 0]
            thickness = (roof - base).clamp(min=1)
            mass = weights[column, selected]
            normalized = mass / mass.sum()
            localstability = stability[column, selected]
            localshear = shear[column, selected]
            convective = surface or bool(torch.any(localstability < 0))
            outer = 0.085 * thickness
            wall = (0.4 * (interface[column, selected] - base)).clamp(min=1)
            length = outer / (1 + outer / wall) if connected else torch.ones_like(wall) * outer
            richardson = (localstability * normalized).sum() / (localshear * normalized).sum()
            sh, sm = galperin_functions(richardson if convective else localstability / localshear, params)
            production = length.square() * (-sh * localstability + sm * localshear)
            surfaceenergy = (0.30 * (buoyancy[column].clamp(min=0) * thickness).pow(2 / 3)
                             if surface else production.sum() * 0)
            mean = (b1 * (production * normalized).sum() + surfaceenergy).clamp(min=0)
            if convective:
                # Equation 28 with relaxation rate one, within this layer only.
                energy = (b1 * production + b1 * mean + surfaceenergy) / (1 + b1)
            else:
                energy = b1 * production
            energy = energy.clamp(0, float(params.get('uw_tke_max_m2s2', 20)))
            heat[column, selected] = length * energy.sqrt() * sh
            momentum[column, selected] = length * energy.sqrt() * sm
            tke[column, selected] = energy
            labels[column, selected] = label
            if connected:
                boundary[column] = roof
            if convective:
                # Entrainment spends a fraction of the layer buoyancy driving.
                available = (-heat[column, selected] * localstability * spacing[column, selected]).sum()
                available = available + (buoyancy[column].clamp(min=0) * thickness / 2 if surface else 0)
                edges = [edge for edge in (top - 1, bottom + 1) if 0 <= edge < count]
                for edge in edges:
                    if labels[column, edge] >= 0 or stability[column, edge] <= 0:
                        continue
                    jump = (stability[column, edge] * spacing[column, edge]).clamp(min=1e-3)
                    speed = (0.2 * available.clamp(min=0) / (thickness * jump * max(1, len(edges)))).clamp(max=0.05)
                    exchange = speed * spacing[column, edge]
                    heat[column, edge] = torch.maximum(heat[column, edge], exchange)
                    momentum[column, edge] = torch.maximum(momentum[column, edge], exchange)
                    if surface and edge == top - 1:
                        entrainment[column] = speed

    return {'heat': heat.clamp(0, limit), 'momentum': momentum.clamp(0, limit),
            'tke': tke, 'depth': boundary, 'entrainment': entrainment,
            'stability': stability, 'labels': labels}
