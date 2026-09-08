import torch

from scm.radiation_schemes.multiband import planck_band_fractions


edges = [10, 350, 500, 630, 700, 820, 980, 1080, 1180,
         1390, 1480, 1800, 2080, 2250, 2380, 2600, 3250]


def test_planck_band_fractions_sum_to_one():
    temperature = torch.tensor([[200., 250., 300.]], dtype=torch.float64)
    fractions = planck_band_fractions(temperature, edges)
    torch.testing.assert_close(fractions.sum(dim=-1), torch.ones_like(temperature))
    assert torch.all(fractions >= 0)


def test_warmer_emission_moves_to_higher_wavenumber():
    temperature = torch.tensor([220., 300.], dtype=torch.float64)
    fractions = planck_band_fractions(temperature, edges)
    centers = (torch.tensor(edges[:-1]) + torch.tensor(edges[1:])) / 2
    means = torch.sum(fractions * centers, dim=-1)
    assert means[1] > means[0]
