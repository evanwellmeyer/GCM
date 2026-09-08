"""Nonprecipitating total-water partition at fixed moist enthalpy."""
import torch

from scm.thermo import cp, Lv, saturation_specific_humidity


def partition_water(water, enthalpy, pressure, critical=1.0):
    critical = min(max(float(critical), 0.5), 1.0)

    def distribution(liquid):
        temperature = (enthalpy - Lv * (water - liquid)) / cp
        saturation = saturation_specific_humidity(temperature, pressure)
        width = ((1 - critical) * saturation).clamp(min=1e-12)
        above = water + width - saturation
        fraction = (above / (2 * width)).clamp(0, 1)
        target = torch.where(above >= 2 * width, water - saturation,
                             above.clamp(min=0).square() / (4 * width)).clamp(min=0)
        return target, fraction

    lower = torch.zeros_like(water)
    upper = water.clone()
    for iteration in range(48):
        middle = (lower + upper) / 2
        target, fraction = distribution(middle)
        upper = torch.where(middle > target, middle, upper)
        lower = torch.where(middle > target, lower, middle)
    liquid = (lower + upper) / 2
    vapor = water - liquid
    temperature = (enthalpy - Lv * vapor) / cp
    target, fraction = distribution(liquid)
    return temperature, vapor, liquid, fraction
