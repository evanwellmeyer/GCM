import torch

from scm.convection_mf import parcel_ascent
from scm.thermo import cp, Lv, Rd, Rv, saturation_specific_humidity


def test_dry_parcel_follows_poisson_law():
    temperature = torch.tensor([290.0], dtype=torch.float64)
    vapor = torch.tensor([0.00001], dtype=torch.float64)
    lower = torch.tensor([100000.0], dtype=torch.float64)
    upper = torch.tensor([90000.0], dtype=torch.float64)
    warmed, remaining, condensed = parcel_ascent(temperature, vapor, lower, upper)
    torch.testing.assert_close(warmed, temperature * (upper / lower) ** (Rd / cp))
    torch.testing.assert_close(remaining, vapor)
    assert condensed.item() == 0


def test_saturated_parcel_conserves_enthalpy_after_pressure_work():
    temperature = torch.tensor([270.0, 290.0, 305.0], dtype=torch.float64)
    lower = torch.full_like(temperature, 90000.0)
    upper = torch.full_like(temperature, 85000.0)
    vapor = saturation_specific_humidity(temperature, lower)
    warmed, remaining, condensed = parcel_ascent(temperature, vapor, lower, upper)
    dry = temperature * (upper / lower) ** (Rd / cp)
    torch.testing.assert_close(cp * warmed + Lv * remaining, cp * dry + Lv * vapor)
    torch.testing.assert_close(remaining + condensed, vapor)
    torch.testing.assert_close(remaining, saturation_specific_humidity(warmed, upper), atol=1e-9, rtol=1e-6)
    assert torch.all(warmed > dry) and torch.all(warmed < temperature)


def test_small_saturated_step_matches_moist_lapse_rate():
    temperature = torch.tensor([290.0], dtype=torch.float64)
    lower = torch.tensor([90000.0], dtype=torch.float64)
    vapor = saturation_specific_humidity(temperature, lower)
    warmed, _, _ = parcel_ascent(temperature, vapor, lower, lower - 1)
    expected = (Rd * temperature / (cp * lower)) * (
        1 + Lv * vapor / (Rd * temperature)) / (
        1 + Lv * Lv * vapor / (cp * Rv * temperature ** 2))
    # The saturation fit and specific-humidity approximation differ slightly
    # from the analytic dilute-water lapse formula.
    torch.testing.assert_close(temperature - warmed, expected, rtol=0.04, atol=1e-6)
