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
- `scm/radiation.py` - semi-gray and multiband radiation solvers
- `scm/surface.py` - surface fluxes and slab ocean update
- `scm/boundary_layer.py` - implicit boundary layer mixing
- `scm/condensation.py` - large-scale saturation adjustment
- `scm/cloud_microphysics.py` - prognostic cloud condensate and cloud-radiative properties
- `scm/convection_bm.py` - Betts-Miller convective adjustment
- `scm/convection_mf.py` - mass-flux convection with entraining-plume closure
- `scm/ensemble.py` - parameter sampling and mixed structural ensembles
- `scm/diagnostics.py` - equilibrium and climate sensitivity diagnostics
- `scm/experiment.py` - output stem builder and per-member parameter override utilities
- `scm/calibration.py` - radiation calibration sweep runner
- `scm/configuration.py` - TOML config loader and param resolution
- `scm/plotting.py` - diagnostics figure generation
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

The radiation in this SCM now has two selectable modes in [`scm/radiation.py`](scm/radiation.py):

- **`multiband`**: the current default
- **`semi_gray`**: the simplified fallback

The default `multiband` path is still lightweight compared with a full GCM radiation package, but it is more realistic than the original single-band solver:

- longwave is split across multiple gray bands with different water-vapor and CO2 optical depths
- shortwave is split across multiple bands with configurable water-vapor and ozone absorption
- optional bulk trace-gas terms can be added for `CH4`, `N2O`, `O3`, and a catch-all minor-greenhouse-gas optical depth
- cloud-radiative properties can be taken either from prescribed bulk cloud parameters or from prognostic cloud condensate produced by `scm/cloud_microphysics.py`

The simplified `semi_gray` path is still available for fast debugging and calibration. In that mode:

- longwave is split into a transparent window fraction and a single absorbing band
- shortwave is represented as one band with water-vapor absorption
- CO2 forcing enters through `co2_base_tau + co2_log_factor * log(co2 / co2_ref)`

Across both modes, the model currently treats:

- `H2O` as an explicit radiatively active gas
- `CO2` as an explicit prescribed greenhouse gas
- `CH4`, `N2O`, and `O3` as optional bulk radiative extensions

It still does **not** yet provide:

- aerosols
- line-by-line or correlated-k spectroscopy
- a production-quality cloud overlap scheme

So if you are describing the current SCM to someone else, the right summary is:

- it has a modular radiation system with a richer default multiband mode and a simpler semi-gray fallback
- it includes explicit water-vapor and prescribed-CO2 effects
- it can add bulk trace-gas and cloud-radiative effects without switching to a full spectral package

This is appropriate for a compact research SCM, but it is not yet a full GCM-grade radiation package.

### Optional trace-gas extension

Both radiation modes support an **optional bulk trace-gas extension**.

If enabled, the radiation can add simple extra optical-depth terms for:

- methane (`CH4`)
- nitrous oxide (`N2O`)
- ozone longwave absorption (`o3_lw_tau`)
- ozone shortwave absorption (`o3_sw_tau`)
- a catch-all minor-greenhouse-gas term (`other_ghg_tau`)

This is still **not** a spectrally resolved radiation package. It is a way to add configurable trace-gas effects without replacing the lightweight SCM solver.

### Optional cloud-radiative extension

Cloud-radiative effects can now be supplied in two ways:

- **prescribed bulk clouds** through `[radiation.clouds]`
- **microphysics-coupled clouds** through `[cloud_microphysics]`, where large-scale condensation and convective detrainment feed a prognostic condensate field `qc`

The microphysics-coupled path diagnoses:

- cloud fraction
- shortwave cloud optical depth
- longwave cloud optical depth
- liquid and ice water paths
- cloud precipitation from thresholded quadratic autoconversion

This is still a deliberately simple cloud-radiative model, not a full microphysics package with separate hydrometeor classes and overlap assumptions. The important point is that the default radiation can now respond to internally generated condensate rather than only to prescribed bulk cloud parameters.

## Differences From CESM2 and GFDL Single-Column Models

This SCM makes several deliberate design choices that differ from or improve upon the physics in CESM2 (CAM6 SCM) and GFDL (AM4 SCM).

### Convection

**Dilute-CAPE closure instead of undilute CAPE.**
CESM2's Zhang-McFarlane scheme triggers and scales convection using CAPE computed from an undilute boundary-layer parcel — one that does not mix with environmental air as it rises. This can fire convection even in a dry free troposphere where a real plume would lose buoyancy quickly. This SCM computes CAPE from an entraining parcel that progressively mixes with environmental air layer-by-layer (`dilute_cape` in `convection_mf.py`). Convective strength therefore depends on free-tropospheric humidity, which is a known driver of tropical climate sensitivity. GFDL's Donner deep convection scheme also uses a dilute parcel, but through a more complex two-moment plume model; this SCM achieves the same physical behavior more directly and transparently.

**Explicit RH-capped detrainment.**
When the mass-flux plume loses buoyancy and detaches, it deposits air capped at a configurable fraction of saturation (`mf_detrain_rh`, default 0.70). Standard mass-flux schemes including Zhang-McFarlane detrain plume air at whatever humidity it carries, which can lead to spurious free-tropospheric oversaturation over long integrations. The explicit RH cap prevents this and makes the free-tropospheric moistening behavior transparent and tunable.

**Explicit subsidence warming term.**
The heating tendency includes a separate term for compensating environmental subsidence driven by mass-flux divergence at each level. Many simplified schemes lump this into an implicit heating-cooling balance or omit it. Having it as an explicit term makes the vertical heating structure easier to diagnose and modify independently of the detrainment term.

**Condensate-loaded plume buoyancy.**
The mass-flux plume and the dilute-CAPE closure now carry a simple condensate reservoir, with configurable retention and fallout. Buoyancy is computed from vapor plus condensate loading rather than from vapor alone, so deep plumes lose buoyancy more realistically once they condense heavily.

**Structural ensembles across convection schemes.**
The ensemble driver can run a mixed ensemble where half the members use Betts-Miller adjustment and half use mass-flux convection, within a single batched experiment. This directly samples structural uncertainty alongside parametric uncertainty. Neither the CESM2 SCM framework nor the GFDL AM4 SCM supports this natively — they fix one scheme per experiment.

**Betts-Miller with enforced column-drying constraint.**
The Betts-Miller implementation explicitly scales the moistening tendency so that column moistening cannot exceed 0.8× the column drying before applying the adjustment. This prevents moisture non-conservation issues that appeared in early BM implementations in GFDL models, where the adjustment could add net moisture to the column under certain humidity profiles.

### Radiation

**Multi-band gray radiation with per-band gas weighting.**
The default `multiband` radiation solver splits the longwave across four spectral bands (window, near-infrared, CO2 band, water-vapor continuum) and the shortwave across three bands, each with independently specified water-vapor absorption, CO2 sensitivity, and trace-gas scaling. CESM2 and GFDL use full correlated-k solvers (RRTMG: 16 LW + 14 SW bands) where gas forcing emerges from line-by-line databases. The multiband scheme is simpler than RRTMG but richer than a single-band gray model: the CO2 forcing is distributed realistically across bands, water-vapor absorption varies with spectral region, and ozone absorption is placed in the correct shortwave bands — all without requiring a full spectral database.

**Explicit logarithmic CO2 sensitivity.**
Across both radiation modes, CO2 forcing enters as `co2_base_tau + co2_log_factor * ln(CO2/CO2_ref)` — a directly inspectable and tunable parameter. This correctly captures the logarithmic saturation physics of CO2 (doubling CO2 adds a fixed optical depth increment) while making the CO2 climate sensitivity an explicit quantity rather than an emergent result of a spectral calculation. In the multiband mode, this logarithmic increment is distributed per-band via `lw_band_co2_log_factor`, giving per-band CO2 forcing control.

**Prognostic cloud condensate coupled to radiation.**
The `cloud_microphysics.py` module carries a prognostic condensate field `qc` that is fed by large-scale condensation and convective detrainment, and partitioned into liquid and ice fractions by temperature. Shortwave and longwave cloud optical depths are diagnosed from liquid water path (LWP) and ice water path (IWP) using configurable mass-extinction coefficients (`cloud_k_liq_sw`, `cloud_k_ice_lw`, etc.), cloud fraction is diagnosed jointly from relative humidity and condensate amount, and cloud water is removed by thresholded quadratic autoconversion so thick clouds precipitate more efficiently than thin clouds. This closes the cloud-radiation-convection feedback loop without requiring prescribed bulk cloud parameters. CESM2 and GFDL SCMs couple radiation to cloud microphysics too, but through much more complex multi-moment schemes; this model provides the same internal coupling at a cost appropriate for a research SCM.

**Incremental physics complexity via additive extensions.**
RRTMG and GFDL-RAD require full atmospheric composition profiles (temperature, humidity, O3, CO2, CH4, N2O, aerosols) even for simple runs. This SCM requires only temperature, humidity, and CO2 at minimum. Trace gases (CH4, N2O, O3), clouds, and microphysics are additive extensions that each add one config section without replacing the core solver. This makes it straightforward to isolate individual feedbacks by enabling or disabling components.

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

- `scm/configs/default.toml` - richer default run settings: multiband radiation, optional trace gases, and microphysics-coupled cloud radiation
- `scm/configs/simplified_physics.toml` - simplified fallback: semi-gray radiation and no cloud microphysics
- `scm/configs/trace_gases_example.toml` - example of the optional trace-gas radiation mode
- `scm/configs/clouds_example.toml` - example of the optional cloud-radiative mode
- `scm/configs/radiation_calibration.toml` - short-run sweep for tuning the semi-gray baseline without the richer cloud path

The config file is the preferred place for persistent run setup. Existing CLI flags still work and act as overrides.

The radiation settings are now structured into sections like:

- `[radiation]`
- `[radiation.longwave]`
- `[radiation.shortwave]`
- `[radiation.multiband]`
- `[radiation.trace_gases]`
- `[radiation.clouds]`
- `[cloud_microphysics]`
- `[shallow_convection]`
- `[params]` for generic physics overrides that do not yet have their own dedicated section

Two config knobs are especially relevant for the richer default path:

- `[numerics].rad_interval_microphysics_steps` reduces the radiation cadence when prognostic cloud condensate is enabled
- `[params].boundary_layer_scheme` selects either the richer Richardson-based BL mixing or the legacy constant-diffusion mode
- `[shallow_convection].enforce_mse_conservation` keeps the capped shallow-convection tendencies energetically closed

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

If you want the older simpler physics path instead, use:

```bash
python -m scm.run_scm --config scm/configs/simplified_physics.toml --scheme mf --fixed-params --device cpu --no-plot
```

The richer default config also uses a shallower surface-coupling stencil through `[params]`:

- `boundary_layer_scheme = "richardson"`
- `surface_heat_levels = 2`
- `surface_moisture_levels = 1`
- `include_precip_enthalpy_flux = true`

Its cloud-microphysics defaults are also intentionally stricter than the earlier prototype:

- `cloud_ls_precip_fraction = 0.90`
- `cloud_rh_min = 0.90`
- `cloud_qc_ref = 8.0e-4`
- `cloud_autoconv_qc_thresh = 2.0e-4`
- `cloud_autoconv_power = 2.0`
- `conv_cloud_efficiency = 0.0008`
- `cloud_cf_rh_power = 2.0`
- `cloud_cf_qc_power = 1.5`
- `cloud_k_liq_lw = 4.0`
- `cloud_k_ice_lw = 2.0`
- `cloud_sw_scattering_efficiency = 0.05`

And the richer default shortwave settings keep a modestly higher reflective baseline than the simplified path:

- `albedo = 0.32`

The richer default path also enables a conservative shallow-convection section:

- `top_sigma = 0.72`
- `base_sigma = 0.90`
- `tau = 14400 s`
- no shallow-convective precipitation by default

And the richer default MF plume keeps a modest amount of condensate loading through `[params]`:

- `mf_condensate_retention = 0.25`
- `mf_condensate_fallout = 0.45`

Those overrides are there to keep the lowest atmospheric level better coupled to the slab surface under the multiband plus cloud-microphysics configuration.

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
- `--restart-from PATH`: resume from a saved `1x` or `2x` restart bundle
- `--fixed-sst`: disable slab-ocean evolution for faster debugging
- `--device {cpu,cuda,mps}`: choose the execution device
- `--no-plot`: skip figure generation

The driver now saves restart bundles after both phases by default:

- `..._1x_restart.pt`: spun-up control state plus `1x` diagnostics/history
- `..._2x_restart.pt`: current `2x` branch state plus the stored `1x` baseline

This lets you continue long runs without rerunning the whole control branch. For example:

```bash
python -m scm.run_scm --scheme mf --fixed-params --spinup-days 2000 --perturb-days 2000 --device cpu --no-plot
python -m scm.run_scm --restart-from scm_full_mf_fixed_slabocean_spin2000d_pert2000d_2x_restart.pt --perturb-days 4000 --device cpu --no-plot
```

The first command produces the restart bundle. The second extends the existing `2xCO2`
branch by another `4000` days and reuses the saved `1xCO2` control statistics.

## Diagnostics

The SCM now carries an explicit column energy-budget diagnostic set:

- `ASR` = absorbed shortwave radiation
- `OLR` = outgoing longwave radiation
- `TOA net` = `ASR - OLR`
- `surface net` = net radiative plus turbulent flux into the slab ocean / surface reservoir
- `precip heat` = precipitation enthalpy exchange `c_w P (T_precip - T_s)` at the surface
- `surface total` = `surface net + precip heat`
- `atmos flux convergence` = `TOA net - surface total`
- `atmos energy tendency` = diagnosed storage tendency of atmospheric moist enthalpy `cp*T + Lv*q`
- `column tendency` = atmospheric plus slab storage tendency
- `column residual` = `TOA net - column tendency`
- `column mse residual` = the same residual but using moist static energy `cp*T + Lv*q+gz` as a secondary thermodynamic diagnostic

This is useful for separating a true radiative imbalance from missing energy bookkeeping. In the richer default path the precipitation enthalpy term matters enough that `TOA net` alone is no longer the whole story.

The main closure diagnostic now uses moist enthalpy rather than moist static energy.
The old moist-static-energy form is still retained as a secondary diagnostic because it
is useful thermodynamically, but it is not the right primary closure metric for this
fixed-mass SCM.

The printed `equilibrium check` now uses the late diagnostic window rather than just
the broad branch mean. A branch only passes if the late-window surface-temperature
trend, mean `TOA net`, mean `surface total`, and mean `column residual` are all small.
This avoids false positives where the slab tendency happens to be small even though the
column is still drifting or the TOA budget is still open.

The mass-flux convection path also applies a column moist-enthalpy correction so the
combined convective `T` and `q` tendencies stay energetically closed after the
profile caps are applied.

The slab ocean is also carried internally as a heat-content anomaly rather than by
adding tiny `dTs` increments directly to `Ts`. This avoids the float32 precision floor
that can otherwise freeze long integrations while nonzero surface fluxes remain.

## Output

The main driver saves:

- a results file named like `scm_demo_mf_fixed_slabocean_spin500d_pert500d_results.pt`
- restart bundles named like `scm_demo_mf_fixed_slabocean_spin500d_pert500d_1x_restart.pt`
  and `scm_demo_mf_fixed_slabocean_spin500d_pert500d_2x_restart.pt`
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
