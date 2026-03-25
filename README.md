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

## Radiation And Greenhouse Gases

The radiation in this SCM is intentionally simple. It is a **semi-gray** scheme in [`scm/radiation.py`](scm/radiation.py):

- Longwave is split into:
  - a transparent window fraction
  - a single absorbing band
- Shortwave is represented as one band with water-vapor absorption

In the current code, the longwave optical depth is made of two explicit pieces:

- **Water vapor**: prognostic, through the model humidity field `q`
- **CO2**: prescribed, through `co2` relative to `co2_ref`

Concretely:

- longwave water-vapor optical depth scales like `kappa_wv * q * dp / g`
- longwave CO2 optical depth uses a simple logarithmic dependence on concentration, through `co2_base_tau + co2_log_factor * log(co2 / co2_ref)`
- shortwave absorption also depends on water vapor through `sw_kappa_wv`

This means the model currently treats:

- `H2O` as an explicit radiatively active gas
- `CO2` as an explicit prescribed greenhouse gas

And it does **not** yet explicitly represent:

- methane (`CH4`)
- nitrous oxide (`N2O`)
- ozone (`O3`)
- aerosols
- spectrally resolved clouds

So if you are describing the current SCM to someone else, the right summary is:

- it has a simplified radiative transfer model
- it includes explicit water-vapor and CO2 effects
- all other greenhouse-gas and cloud effects are either absent or implicitly folded into tuning parameters such as `f_window`, `co2_base_tau`, and `albedo`

This is appropriate for a compact research SCM, but it is not yet a full GCM-grade radiation package.

### Optional trace-gas extension

The code now also supports an **optional bulk trace-gas extension** while keeping the same semi-gray solver.

If enabled, the radiation can add simple extra optical-depth terms for:

- methane (`CH4`)
- nitrous oxide (`N2O`)
- ozone longwave absorption (`o3_lw_tau`)
- ozone shortwave absorption (`o3_sw_tau`)
- a catch-all minor-greenhouse-gas term (`other_ghg_tau`)

This is still **not** a spectrally resolved radiation package. It is a way to add configurable trace-gas effects without replacing the current simplified radiation model.

### Optional cloud-radiative extension

The same semi-gray radiation can also include a simple optional cloud-radiative layer.

The current cloud hooks are bulk and configurable:

- `cloud_fraction`
- `cloud_sw_reflectivity`
- `cloud_sw_tau`
- `cloud_lw_tau`
- `cloud_top_sigma`
- `cloud_bottom_sigma`

In this form, clouds are still a simplified radiative parameterization, not a full cloud microphysics or cloud overlap scheme.

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

### Config-driven runs

The driver now supports TOML configuration files:

```bash
python -m scm.run_scm --config scm/configs/default.toml
```

Two example configs are included:

- `scm/configs/default.toml` - baseline run settings
- `scm/configs/trace_gases_example.toml` - example of the optional trace-gas radiation mode
- `scm/configs/clouds_example.toml` - example of the optional cloud-radiative mode
- `scm/configs/radiation_calibration.toml` - short-run sweep for tuning the semi-gray radiation baseline

The config file is the preferred place for persistent run setup. Existing CLI flags still work and act as overrides.

The radiation settings are now structured into sections like:

- `[radiation]`
- `[radiation.longwave]`
- `[radiation.shortwave]`
- `[radiation.trace_gases]`
- `[radiation.clouds]`

### Quick component tests

```bash
python -m scm.test_components
```

This runs isolated checks for the thermodynamics, radiation, surface, condensation, and convection components.

### Ensemble SCM experiment

```bash
python -m scm.run_scm --demo
```

Demo mode now defaults to:

- 10 members
- fixed parameters
- 500-day 1xCO2 spinup
- 500-day 2xCO2 branch

You can also isolate one convection scheme:

```bash
python -m scm.run_scm --demo --scheme mf
python -m scm.run_scm --demo --scheme bm
python -m scm.run_scm --demo --scheme mixed
```

For controlled debugging runs, the most useful option is usually:

```bash
python -m scm.run_scm --scheme mf --fixed-params --spinup-days 2000 --perturb-days 2000 --device cpu --no-plot
```

Or the equivalent config-driven path:

```bash
python -m scm.run_scm --config scm/configs/default.toml --scheme mf --fixed-params --device cpu --no-plot
```

### Radiation calibration workflow

The driver can also run a short, config-driven radiation calibration sweep. This is meant for tuning the current semi-gray baseline before adding more physics complexity.

Example:

```bash
python -m scm.run_scm --config scm/configs/radiation_calibration.toml --device cpu --no-plot
```

When `[calibration].enabled = true`, the driver runs short fixed-parameter control integrations over the parameter combinations listed in `[calibration.parameter_grid]`, then ranks them by a simple score based on:

- TOA net flux
- surface net flux
- OLR
- ASR
- surface temperature

By default, the calibration runner batches candidates into chunks instead of running them strictly one by one. The main controls are:

- `[calibration].batch_candidates = true`
- `[calibration].batch_size = 16`

This is the preferred way to make calibration runs use CPU threads or MPS more effectively.

The calibration output prints the top candidates and saves a results file with the ranked sweep summary plus a TOML snippet for the best candidate.

For full-mode fixed-parameter debugging runs, the driver now defaults to **one deterministic member** instead of creating 100 identical copies. If you want the old ensemble-shaped output, set:

```toml
[run]
preserve_ensemble_shape = true
```

### Full experiment

```bash
python -m scm.run_scm
```

By default this runs a larger ensemble, spins up under 1xCO2, then branches to 2xCO2.

Useful driver flags:

- `--config PATH`: load a TOML config file
- `--scheme {mixed,bm,mf}`: choose mixed, Betts-Miller-only, or mass-flux-only runs
- `--fixed-params`: use default parameters instead of sampling
- `--spinup-days N`: override the 1xCO2 run length
- `--perturb-days N`: override the 2xCO2 run length
- `--fixed-sst`: disable slab-ocean evolution for faster debugging
- `--device {cpu,cuda,mps}`: choose the execution device
- `--no-plot`: skip figure generation

## Diagnostics

The SCM now carries an explicit top-of-atmosphere energy-balance diagnostic:

- `ASR` = absorbed shortwave radiation
- `OLR` = outgoing longwave radiation
- `TOA net` = `ASR - OLR`
- `surface net` = net radiative plus turbulent flux into the slab ocean / surface reservoir

This is useful for checking whether a control or perturbed run is actually near radiative-convective equilibrium.

## Output

The main driver saves:

- a results file named like `scm_demo_mf_fixed_slabocean_spin500d_pert500d_results.pt`
- a diagnostics figure with a matching stem if plotting is available

These contain equilibrium statistics, sensitivity estimates, parameter fields, and history snapshots. The filenames now encode mode, scheme, sampling mode, SST mode, and run lengths so runs do not overwrite each other as easily.

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
