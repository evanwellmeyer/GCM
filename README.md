# GCM / SCM Column Model

This repository currently contains a single-column atmospheric model built as a research sandbox for tropical radiative-convective equilibrium experiments.

Despite the directory name, the codebase is not yet a full general circulation model. It is better described as a **single-column model (SCM)** with modular physics packages and an ensemble driver. The intent is to keep the model small, inspectable, and easy to couple to a future dynamical core.

## What It Does

- Integrates a vertical atmospheric column forward in time
- Applies parameterized physics in sequence:
  - radiation
  - surface fluxes
  - boundary layer mixing
  - convection
  - large-scale condensation
  - optional slab ocean coupling
- Supports batched execution for ensembles
- Supports two convection schemes:
  - Betts-Miller adjustment
  - simplified mass-flux convection
- Supports mixed structural ensembles where different members use different convection schemes

## What It Does Not Do Yet

- Horizontal dynamics
- Momentum equations
- Advection
- A dynamical core
- Multi-column coupling

So, if the future goal is a GCM, this code is currently the **physics column component** rather than the full dynamical system.

## Repository Layout

- `scm/column_model.py` - time stepping, state updates, and physics dispatch
- `scm/thermo.py` - thermodynamics, vertical grid, CAPE, and moist adiabat utilities
- `scm/radiation.py` - semi-gray shortwave and longwave radiation
- `scm/surface.py` - surface fluxes and slab ocean update
- `scm/boundary_layer.py` - implicit boundary layer mixing
- `scm/condensation.py` - large-scale saturation adjustment
- `scm/convection_bm.py` - Betts-Miller convective adjustment
- `scm/convection_mf.py` - mass-flux convection with entraining-plume closure
- `scm/ensemble.py` - parameter sampling and mixed structural ensembles
- `scm/diagnostics.py` - equilibrium and climate sensitivity diagnostics
- `scm/run_scm.py` - experiment driver for spinup and CO2 perturbation runs
- `scm/test_components.py` - standalone component checks

## Model Architecture

The model state is batched, so the first tensor dimension is the ensemble member dimension.

Typical prognostic variables:

- `t`: atmospheric temperature profile
- `q`: atmospheric specific humidity profile
- `ts`: sea surface temperature or slab ocean temperature
- `ps`: surface pressure

Derived fields such as `p` and `dp` are recomputed from `ps` as needed.

The core timestep is managed by `scm/column_model.py`. In simplified terms:

```text
state -> radiation -> surface fluxes -> boundary layer -> convection -> condensation -> slab ocean
```

Each physics package returns tendencies or flux diagnostics, and the column model applies them sequentially.

## Convection Schemes

The code supports two alternate convective closures:

- **Betts-Miller**: relaxes temperature and moisture toward a reference moist-adiabatic state
- **Mass-flux**: uses an entraining plume, a dilute-CAPE closure, detrainment moistening, and compensating subsidence warming

This makes it easy to compare sensitivity to different structural assumptions.

### Current mass-flux design

The mass-flux scheme is no longer just a local plume-environment mixing model. Its current design is closer to a simplified moist-convection closure with three explicit ideas:

- **Dilute CAPE closure**: convection is triggered and scaled using CAPE from an entraining parcel rather than undilute CAPE
- **Detrainment moistening**: when plume buoyancy weakens, the scheme detrains near-saturated air into the free troposphere
- **Subsidence warming**: the thermal tendency includes compensating subsidence associated with the plume mass budget

This matters because the vertical heating and moistening structure is now intended to be more sensitive to free-tropospheric humidity and warming than the earlier version.

## Ensemble Mode

The ensemble driver in `scm/ensemble.py` samples uncertain parameters using Latin hypercube sampling.

The current experiment design is:

- half of the members use Betts-Miller convection
- half use mass-flux convection
- each member gets a sampled parameter set within configured ranges

This is useful for exploring how parameter uncertainty and structural uncertainty interact.

## How To Run

### Quick component tests

```bash
python -m scm.test_components
```

This runs isolated checks for the thermodynamics, radiation, surface, condensation, and convection components.

### Ensemble SCM experiment

```bash
python -m scm.run_scm --demo
```

Demo mode runs a small ensemble for a shorter spinup and perturbation period.

### Full experiment

```bash
python -m scm.run_scm
```

By default this runs a larger ensemble, spins up under 1xCO2, then branches to 2xCO2.

## Output

The main driver saves:

- `scm_ensemble_results.pt`
- `scm_ensemble_diagnostics.png` if plotting is available

These contain equilibrium statistics, sensitivity estimates, parameter fields, and history snapshots.

## Future GCM Connection

This SCM is already structured in a way that could be coupled to a future dynamical core.

The likely coupling pattern would be:

- the dycore advances resolved dynamics
- this SCM supplies column physics tendencies
- a coupling layer exchanges state and tendencies between the two

That means the current code is best thought of as the physics package layer of a future GCM, not the dycore itself.

## Requirements

The code is written for PyTorch and uses batched tensor operations.

Typical use expects:

- Python 3
- `torch`
- optional `matplotlib` for plotting

## Notes

- The code is a work in progress.
- Several modules include defensive clamping and NaN checks to keep long integrations stable.
- Some paths in the driver are still development-oriented and may be cleaned up later.
