# Column audit and recovery plan — September 7, 2026

This is the current interpretation of the working tree, not a declaration that the model is validated. Earlier experiments and their original interpretations remain in [column_open_problems.md](column_open_problems.md). The initial audit changed documentation only; the implementation follow-up below adds a gated transport candidate. Existing uncommitted work was preserved. No default configuration, checkpoint, or notebook was changed by this follow-up.

## Close-out status: conservative flux configuration accepted

`scm/configs/atm407_flux_v1.toml` passed evaluation and its settings are now the accepted `atm407.toml` and repository default. It selects the conservative flux-form convection, uses matched entrainment and detrainment rates of 5e-6 Pa^-1, a six-hour CAPE closure, the ozone-profile radiation option, critical RH 0.90, and diagnosed moist-static-energy boundary-layer mixing capped at 900 m. The older experiments remain evidence, not competing recommendations.

The isolated convection response now passes its eight gates at 900 and 300 s timesteps: cooling strengthens mass flux, rain, and CAPE removal; CAPE falls in every case; the common limiter remains inactive; and energy and water errors remain below 1e-4 W/m2 and 1e-10 kg/m2/s. The active-path focused suite passes 20 tests. The complete SCM suite passes 146 tests.

The 20-level column completed 500 adjustment days with a 5 m slab. Correctly time-averaged final 50-day means are surface temperature 287.41 K, TOA net +0.93 W/m2, surface flux +0.41 W/m2, CAPE 777 J/kg, deep rain 1.57 mm/day, large-scale rain 0.88 mm/day, and zero mass at RH at or above 95%. Water closure is 4.1e-11 kg/m2/s, the column-energy and MSE residuals are 0.003 W/m2, and no convection caps are active. Temperature drift is 0.038 K over the window. The state passes every equilibrium gate and is now the canonical notebook reference.

The generator formerly sampled flux diagnostics once per day at one fixed point in the physics cycle. Those instantaneous samples did not represent the flux integrated by the slab and produced false surface and MSE failures. It now samples every two hours and evaluates all means and trends over the same 50 days. Clear-sky radiation obeys the configured two-hour cadence; rapid radiation updates are reserved for active cloud optics. The MSE diagnostic also now includes hydrostatic pressure work associated with vertical redistribution; without this term it mislabeled changing geopotential as an external energy source. The corrected MSE and primary energy residuals agree.

To reproduce or continue the accepted 5 m state for one diagnostic window, rerun the following command. The script resumes the named output automatically and resets the slab-energy reference consistently.

```bash
cd /Users/evanwellmeyer/Documents/GCM
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python scripts/make_atm407_reference.py \
  --config scm/configs/atm407_flux_v1.toml \
  --output-label flux_profile_v1_5m \
  --adjustment-days 50 \
  --adjustment-ocean-depth 5 \
  --final-dt 900
```

The notebook reference intentionally remains the equilibrated 5 m state. Individual exercises may reset the slab and use 50 m when ocean mixed-layer response time is part of the lesson; a second 50 m equilibrium is not required to initialize the atmosphere.

The simple multiband radiation remains a known scientific limitation; its matched-grid RRTMG comparison has an OLR error of about -16 W/m2 and roughly 1 K/day heating-rate RMS error. A fitted 16-band variant failed its independent-column acceptance test and was rejected. Replacing this with a faithful correlated-k implementation is a separate development project, not a coefficient tweak needed for the accepted teaching checkpoint.

## Latest follow-up: downdraft and artificial export corrected in the candidate

The `flux` candidate now uses a descending plume with equal compensating upward environmental mass at each interface. The upward environmental donor is the lower layer, unlike the downward donor in the updraft circulation. It transports dry static energy and vapor together. Rain supplied by the updraft is accumulated downward, and the downdraft can evaporate only its allocated share of that explicit reservoir. Saturation deficit is solved with the corresponding latent cooling, rather than evaluating saturation once before cooling. No-rain cases do not launch this rain-fed draft.

Entrainment/detrainment use explicit rates per pascal: `mf_downdraft_entrainment_pa = 1e-5`, `mf_downdraft_detrainment_pa = 1e-5`, and `mf_downdraft_release_pa = 1.2e-4`. These are initial development coefficients, not a validated calibration; the old per-level downdraft coefficients remain confined to `legacy`. The candidate retains the existing launch fraction and launch/release sigma thresholds. Launch-depth sampling and the dynamical closure still require validation.

Artificial subcloud export is disabled in `flux`, because vapor-to-rain conversion is now explicit in the updraft. The global moist-energy correction is also disabled, regardless of its legacy setting. A single column scale limits the coupled heat, vapor and rain tendencies, including the rain ceiling, rather than clipping them independently. The candidate rejects the separate legacy rain-evaporation option, which inferred rain from negative vapor tendencies and would count transport drying as a rain source.

An initial screening attempt exposed a limiter interaction with the host's vapor floor: vanishing upper-level drying could suppress the whole column. Phase conversion now respects that floor, and the limiter allows only a machine-precision tolerance around it. A regression checks that the floor does not disable an active plume. In the final six-hour and one-day screens the common limiter remains **1**, so the small residual is not achieved by shutting down convection.

**Verification:** 137 tests passed in 70.39 seconds, including 13 transport tests. New tests check local downdraft heat/water flux divergence, finite rain supply, zero-rain inactivity, explicit production-minus-evaporation precipitation, and conservation with active limiting. A test makes the global energy-correction function raise if called; the complete candidate still passes. `git diff --check` passed. No defaults or checkpoint files were changed.

The final reports are `outputs/column/diagnostics/convective_downdraft_audit_6hour.json` and `convective_downdraft_audit_24hour.json`. Reproduce the latter using `scripts/check_convective_transport.py --hours 24 --output outputs/column/diagnostics/convective_downdraft_audit_24hour.json` with the gcm Python environment.

| One-day mean | Legacy, 900 s | Complete flux candidate, 900 s | Complete flux candidate, 300 s |
|---|---:|---:|---:|
| Raw convection energy residual, W/m² | −20.58 | +1.6e-9 | −6.0e-8 |
| Artificial export contribution, W/m² | −36.57 | 0 | 0 |
| Whole-column energy residual, W/m² | −0.010 | −0.007 | +0.028 |
| Deep rain, mm/day | 1.720 | 0.261 | 0.272 |
| Large-scale rain, mm/day | 0.456 | 1.179 | 1.188 |
| CAPE, J/kg | 1278 | 1417 | 1427 |
| TOA net, W/m² | −0.142 | +0.970 | +1.006 |
| Surface total flux, W/m² | −0.497 | +7.714 | +9.214 |

All runs remain finite and have zero reported deep temperature/moisture cap fractions. Mean water residual magnitudes remain below 1.3e-10 kg/m²/s in these one-day screens. These are transient adjustments from a legacy equilibrium, not new equilibrium states. Surface flux differences and changed precipitation partition prevent promotion based on these checks alone.

The legacy export sink is equivalent to **1.264 mm/day**, about 73% of its diagnosed deep rain in this screen. It removes vapor independently of explicit plume condensation and the net drying is subsequently labeled rain. Therefore restoring legacy precipitation by increasing an efficiency parameter would not be an appropriate validation target. The next scientific check is the explicit condensation/evaporation and CAPE response under controlled forcing, followed by the matched-column radiation reference already planned. Do not add back the export sink or global energy repair to recover the old equilibrium.

## Previous increment: conservative updraft only

`scm/convective_transport.py` now provides one closed interface circulation for dry static energy and vapor, selected through `mf_transport_form = 'flux'`. The default remains `legacy` pending validation of the complete convection scheme. This is an original simplified finite-volume implementation, not a CESM/IFS port. The mass-flux use of dry static energy and specific humidity follows the [ECMWF formulation](https://www.ecmwf.int/sites/default/files/elibrary/2019/81139-ifs-documentation-cy46r1-part-ii-data-assimilation_1.pdf), equation 3.49; our discrete operator uses paired plume/environment fluxes instead of independently implemented subsidence terms.

The updraft transports `s = cp*T + g*z` and vapor through the same interfaces, with equal upward plume and downward environmental mass flux. Flux divergence includes entrainment/detrainment implicitly; no extra local replacement tendency is added. Condensation removes vapor explicitly and releases the matching latent heat. Plume termination closes the upper boundary. The candidate is pseudoadiabatic (immediate fallout), including its CAPE parcel; retained convective condensate is explicitly rejected. Geopotential is held fixed within this process operator. This establishes its moist-enthalpy budget, not exact conservation of every subsequent hydrostatic readjustment.

Eight new regression tests cover uniform-scalar preservation, paired local exchanges and donor direction, unequal layer masses at 10/20/40 levels, explicit rain/water/energy accounting, plume mass continuity, dry neutrality, and the host convection call without an energy correction. These are discrete contract tests, not proof of full closure convergence or observed-profile agreement. The final complete suite passed **132 tests in 68.59 seconds**. `git diff --check` also passed.

The reproducible six-hour screen is `scripts/check_convective_transport.py`. It loads the same promoted 20-level checkpoint into each run, uses a 5 m slab, evaluates radiation every step, and retains ATM407's existing downdraft and moisture-export terms. It records source/checkpoint hashes and final profiles in `outputs/column/diagnostics/convective_transport_audit_6hour.json`. Run it with the gcm Python environment and `--output outputs/column/diagnostics/convective_transport_audit_6hour.json`.

| Six-hour mean, W/m² unless stated | Legacy, 900 s | Flux candidate, 900 s | Flux candidate, 300 s |
|---|---:|---:|---:|
| Updraft energy residual before correction | +26.05 | approximately zero | approximately zero |
| Additional downdraft residual | −10.17 | −4.05 | −4.30 |
| Additional moisture-export residual | −36.27 | −14.69 | −15.63 |
| Combined raw convection residual | −20.40 | −18.74 | −19.93 |
| Final whole-column energy residual | +0.005 | +0.006 | −0.027 |
| TOA net flux | −0.166 | +0.014 | +0.012 |
| Surface total flux | −1.334 | +0.131 | −0.485 |
| CAPE, J/kg | 1268 | 1295 | 1267 |

All three screens remain finite, with no reported deep temperature/moisture tendency caps. The candidate's pre-correction updraft residual is below 2e-7 W/m² in these means. The existing global energy adjustment conceals substantial residuals from the other terms. The screen **does not establish a better equilibrium**, and the 300/900 s precipitation and surface-flux differences still need attribution.

The next required correction is now explicit: account for downdraft mass/rain/energy exchanges and retire or physically replace the artificial subcloud vapor-export sink. The latter removes vapor without corresponding local latent heating and is already being counted as rain downstream. Simply adding an arbitrary compensating heating profile would preserve the existing failure mode. Independently clipped tendencies, rain caps, and the remaining global energy correction must also be checked before promoting the complete candidate. Do not run another long equilibrium to diagnose these local source contracts.

## Scope of the broader code review

A staged review of all active SCM paths is warranted before GCM integration. Review equations, units and sign conventions; then interfaces, tendency-versus-increment handling, timestep limits, phase/rain reservoirs and budget corrections; then forcing, surface exchange, radiation, and checkpoint/restart provenance. Each finding needs an isolated regression or independent benchmark, not code inspection alone. Review inactive schemes before enabling them rather than treating every experimental module as equally urgent. The convection work above is the first stage, not a certification of the remaining code.

## What we actually have

The notebook configuration is `scm/configs/atm407.toml`, layered over `default.toml`; it is not interchangeable with the repository default alone. It uses 20 levels, mass-flux deep convection, Richardson mixing, partial condensation at critical RH 0.95, prognostic condensate, and multiband radiation with cloud radiative effects disabled. The UW/connected-layer experiments are not this baseline.

The promoted `notebooks/data/atm407_equilibrium_20level.json` reports a **5 m** slab checkpoint, surface temperature 285.45 K, TOA −0.24 W/m², surface flux −0.49 W/m², precipitation 2.39 mm/day, CAPE 1275 J/kg, and RH95 mass fraction 0.11. Its 0.105 K window drift fails its 0.05 K gate: `equilibrium_passed` is false. These are saved metadata, not a new integration under today's working tree. They demonstrate a near-balanced teaching reference, not validated GCM physics.

Real fixes have landed, including regression coverage for the radiation switch, diffusion units, parcel ascent, absorber weighting, and partial condensation. Successful conservation tests matter, but do not establish realistic vertical transport or cloud profiles.

## Corrections to the previous brief

- **Shared partitioning is not shown to be broken by the old reproducer.** At critical RH below one, UW and partial condensation now call `phase_partition.partition_water`. `diagnose_partition_handoff.py` still deliberately alternates the old full-saturation `partition_mse` with partial condensation. It reproduces the old mismatch, not the repaired path. `test_phase_partition.py` tests stationarity of the shared contract. UW's full-saturation branch remains different and needs separate coverage.
- **There is no established precipitation shortfall to tune away.** The brief explicitly replaces the 1.87 versus 2.37 mm/day snapshot with nearly balanced ten-day means, then incorrectly reuses the snapshot to rank precipitation efficiency first. Furthermore, `precip_efficiency` only changes the retained-condensate branch in `convection_mf.py`; ATM407 disables that branch. Turning this parameter cannot address the baseline problem.
- **Two similar 20/40-level ten-day runs are limited sensitivity evidence, not proof of convergence.** They support persistence of the moist band in those tests, not grid independence of all processes or equilibria. Full multi-resolution equilibria are not needed for the next component audit.
- **Reduced dry-adjustment activity does not establish correct transport.** Reassigning tendencies between adjustment and mixing without removing the band weakens the claim that adjustment alone causes it. It does not prove that either transport operator is physically correct. Global water balance also cannot rule out compensating local errors.
- **The RRTMG result is a lead, not a validated calibration.** The brief admits different pressure grids. Both layer heating and OLR require a matched atmospheric column: interfaces, layer masses, temperature, humidity, gases, and surface boundary conditions. No reusable RRTMG comparison script was found in `scripts/` or `scm/`; installed climlab is not a reproducible harness. The quoted 14 W/m² difference and spectral explanation remain provisional. A colder equilibrium or lower CAPE than the old baseline does not alone disqualify a radiation correction.

## Code problems identified in this audit

1. **Active deep convection does not use a consistent environmental transport operator.** In `scm/convection_mf.py`, the subsidence temperature term uses `t[k+1] - t[k]`, borrowing the temperature below without adiabatic pressure work. Moisture instead moves downward between adjacent layers using a conservative flux. These are not the matched heat/water pair claimed by the comment. The later column-integrated energy correction does not validate their vertical structure. This is the first active-physics correction to derive and test; it is not yet proven to cause the saturated band.
2. **Active shortwave ozone optical depth is divided equally by level count.** `radiation_schemes/multiband.py` uses `band_o3_tau[band] / nlevels` unless a profile is supplied. Unequal layers therefore receive equal absorber optical depth, not a specified physical ozone distribution. Switching to the existing profile scheme is a candidate correction, not an already validated replacement. Related count weighting remains in the optional semi-gray scheme.
3. **Experimental sedimentation has a units error.** `cloud_microphysics.py` computes its removed fraction as `w * dt * g / (dp/g)`. With `w` in m/s, this is not dimensionless. A finite-volume settling fraction instead involves `rho * w * dt / (dp/g)` or a consistent `w * dt / dz`, with a defined interface flux and stable time integration. Redistribution can conserve water despite an incorrect fall timescale. The reported sedimentation experiments cannot validate settling physics. The option defaults to zero and is inactive in ATM407.
4. **The UW cloud-fraction handoff is incomplete.** UW returns `condensation_cloud_fraction`, but `case_benchmarks.apply_boundary_layer` only consumes `cloud_fraction`. The connected-layer subcycling loop updates temperature, vapor, condensate, and winds but not the returned fraction, although the next substep's stability uses it. Thus the claimed benchmark/substep repair is incomplete. The full-column `maximum` merge with a fraction diagnosed before later physics also needs a consistency test after evaporation; this last concern is not yet a demonstrated failure.

The experimental UW closure remains a reduced implementation, with a documented moist-case failure. It should not replace Richardson merely because it suppresses dry adjustment. Neither a passing global budget nor a scheme name establishes equivalence to CESM/UW.

## One recovery path

Keep the current notebook reference unchanged as a reproducible comparison case. Stop adding or calibrating schemes against this single equilibrium profile.

First, derive and regression-test the **active deep-convection transport contract**: plume/environment mass continuity, one consistent interface transport for energy and water, the appropriate pressure-work treatment, explicit detrainment and precipitation exchanges, and diagnostics before any global energy correction. Use isolated dry-neutral and moist transport cases, one-step budget checks, and short timestep/layer-splitting checks. Do not require a chosen RH or CAPE as the answer and do not hide a local error with an integral correction.

Then establish a reproducible, matched-column radiation comparison before changing absorption coefficients. [Climlab's RRTMG interface](https://climlab.readthedocs.io/en/stable/api/climlab.radiation.rrtm.RRTMG.html) exposes both pressure centers and interfaces; the comparison must verify them, not merely use the same number of levels.

Repair the experimental sedimentation and cloud-fraction contracts before relying on their results. Turbulence/shallow-cloud acceptance should use independently specified forcing and reference profiles, such as the [BOMEX intercomparison](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%282003%2960%3C1201%3AALESIS%3E2.0.CO%3B2), rather than treating permissive bounds or agreement with another local approximation as validation.

Only after component checks pass, use a short 20-level coupled screening run with stage-resolved heat/water/cloud budgets. One 20-level slab equilibrium follows if it passes; do not repeatedly run long integrations to diagnose operator errors. Promotion requires conservation, stable timestepping, justified profiles and precipitation partition, TOA/surface balance, and sufficient checkpoint/configuration/code provenance. Surface drift alone is insufficient. GCM coupling additionally needs forcing, momentum, and restart-contract tests.

## Verification from this audit

Ran `/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python -m pytest scm -q`: **124 passed in 71.06 seconds**, including the shared-partition tests. These tests do not cover all defects listed above and do not overturn the documented experimental moist-case failure. No new long integration or notebook execution is claimed. Results of older scratch experiments without saved inputs and executable drivers are treated as historical reports, not independently reproduced evidence.
