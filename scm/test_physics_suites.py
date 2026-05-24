"""Smoke tests for the SCM physics suite registry."""

import torch


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_registry(device):
    from scm.column_model import initial_state, update_derived
    from scm.ensemble import default_params
    from scm.physics_suites import (
        apply_physics_suite_defaults,
        available_physics_schemes,
        available_physics_suites,
        physics_suite_components,
        run_physics_scheme,
    )
    from scm.thermo import make_grid

    schemes = available_physics_schemes()
    assert "mass_flux" in schemes["convection"]
    assert "betts_miller" in schemes["convection"]
    assert "none" in schemes["convection"]
    assert "mass_flux_default" in available_physics_suites()
    assert "legacy_betts_miller" in available_physics_suites()
    assert physics_suite_components("legacy_betts_miller")["convection_scheme"] == "betts_miller"

    params = apply_physics_suite_defaults({"physics_suite": "mass_flux_default"})
    assert params["convection_scheme"] == "mass_flux"
    assert params["boundary_layer_scheme"] == "richardson"

    override = apply_physics_suite_defaults({
        "physics_suite": "mass_flux_default",
        "convection_scheme": "none",
    })
    assert override["convection_scheme"] == "none"
    assert default_params(device=device)["convection_scheme"] == "mass_flux"

    grid = make_grid(nlevels=6, device=device)
    state = initial_state(1, grid, default_params(device=device), device=device)
    state = update_derived(state, grid)
    out = run_physics_scheme("convection", "none", state, grid, params)
    assert torch.allclose(out["dt"], torch.zeros_like(state["t"]))
    assert torch.allclose(out["dq"], torch.zeros_like(state["q"]))
    assert torch.allclose(out["precip"], torch.zeros_like(out["precip"]))


def main():
    test_registry(pick_device())
    print("physics suite registry: PASS")


if __name__ == "__main__":
    main()

