import torch

from scm.boundary_layer import boundary_layer_mixing
from scm.thermo import g, Rd, cp, Lv


def test_diffusion_matches_two_layer_mass_exchange():
    temperature = torch.tensor([[280.0, 280.0]], dtype=torch.float64)
    pressure = torch.tensor([[80000.0, 90000.0]], dtype=torch.float64)
    thickness = torch.tensor([[8000.0, 12000.0]], dtype=torch.float64)
    vapor = torch.tensor([[0.001, 0.005]], dtype=torch.float64)
    state = {'t': temperature, 'p': pressure, 'dp': thickness, 'q': vapor}
    timestep = 900.0
    diffusivity = 2.0
    result = boundary_layer_mixing(state, {'nlevels': 2}, {
        'boundary_layer_scheme': 'constant', 'bl_mix_levels': 2,
        'dt': timestep, 'k_diff': diffusivity,
    })
    density = pressure[0, 0] / (Rd * temperature[0, 0])
    distance = (pressure[0, 1] - pressure[0, 0]) / (density * g)
    conductance = density * diffusivity / distance
    mass = thickness[0] / g
    difference = (vapor[0, 1] - vapor[0, 0]) / (
        1 + timestep * conductance * (1 / mass[0] + 1 / mass[1]))
    exchange = timestep * conductance * difference
    expected = vapor + torch.stack((exchange / mass[0], -exchange / mass[1]))
    actual = vapor + timestep * result['dq']
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close((actual * mass).sum(), (vapor * mass).sum())


def test_conserved_mixing_preserves_water_and_column_enthalpy():
    temperature = torch.tensor([[270.0, 285.0]], dtype=torch.float64)
    vapor = torch.tensor([[0.003, 0.008]], dtype=torch.float64)
    liquid = torch.tensor([[0.001, 0.0]], dtype=torch.float64)
    thickness = torch.tensor([[8000.0, 12000.0]], dtype=torch.float64)
    state = {'t': temperature, 'q': vapor, 'qc': liquid, 'dp': thickness,
             'p': torch.tensor([[80000.0, 90000.0]], dtype=torch.float64)}
    result = boundary_layer_mixing(state, {'nlevels': 2}, {
        'boundary_layer_scheme': 'constant', 'bl_mix_levels': 2,
        'dt': 900.0, 'k_diff': 2.0,
        'bl_mix_total_water': True, 'bl_mix_moist_static_energy': True,
    })
    mass = thickness / g
    torch.testing.assert_close(((result['dq'] + result['dqc']) * mass).sum(),
                               torch.tensor(0.0, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(((cp * result['dt'] + Lv * result['dq']) * mass).sum(),
                               torch.tensor(0.0, dtype=torch.float64), atol=1e-8, rtol=0)
