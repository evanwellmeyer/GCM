# BOMEX moisture-transport audit, 14 September 2026

## Diagnosed sorting scale, 15 September 2026

The CAM source rule `qtsrc = qt0(1)` matches the host; no unimplemented source humidity excess was found. A separate omission is present in buoyancy sorting: CAM uses `cridis = 0.1*scaleh`, then updates `scaleh` from the diagnosed cumulus top (approximately lines 2289–2299, 2474 and 2943). The host used a fixed 100 m stopping distance regardless of plume depth. Thus the earlier transport mismatch is not evidence for arbitrarily raising the source humidity or reducing launch mass flux.

An opt-in `uw_shallow_diagnose_sorting_distance` path now repeats fresh diagnostic plumes, setting stopping distance to 10% of cloud-top height. Iteration stops at 1% relative consistency, with at most six trials; `plume_sorting_scale_relative_error` exposes any remaining error. Trials reset their output arrays and do not accumulate heat, water or rain. The fraction is the reference value, not fitted to these profiles. The legacy fixed-distance path remains unchanged. Two tests verify per-column convergence/resetting and rejection of an invalid iteration budget.

With host-compatible conserved reconstruction and unchanged launch/mixing coefficients, hour-3 drying becomes **−1.534868 g/kg/day** (original approximately −3.799); top height is 1891.75 m, used sorting distance 188.814 m, relative discrepancy 0.00190. At hour 5.75 drying becomes **−0.338157** (original −3.407603); top height is 2050.60 m, distance 204.034 m, relative discrepancy 0.00500. Float64 water residuals are below 3e-18 kg/m²/s and energy residuals below 3e-10 W/m². These are frozen-state tests, not a demonstrated six-hour cure. Earlier three-pass diagnostic results were replaced by the converged runs in the same diagnostic output files.

```bash
python scripts/check_uw_conserved_environment.py --hour 3 --only-conserved --double --diagnose-sorting
python scripts/check_uw_conserved_environment.py --hour 5.75 --only-conserved --double --diagnose-sorting
python scripts/screen_uw_conserved_environment.py --hours 1 --partition host --diagnose-sorting
```

Outputs have a `_diagnosed` suffix. The one-hour screen is explicitly not six-hour validation. Neither correction is promoted to the teaching default or its checkpoint.

**Combined one-hour result:** maximum absolute theta_l change 0.15753 K, maximum absolute water change 0.21538 g/kg, and rain during minutes 30–60 0.13054 mm/day. Reconstructing the same first-hour baseline from its four archived endpoint tendencies gives 0.14854 K and 0.23037 g/kg; baseline rain for the corresponding two steps averages 0.15421 mm/day. At 825 m water changes −0.23037 to approximately −0.163 g/kg; at 1823 m it changes +0.14866 to +0.21538. This is mixed redistribution, not demonstrated elimination of the bias. The comparison includes substeps and both corrections together, not an isolated causal comparison of either correction. Short-window checks must not be interpreted as six-hour acceptance. File: `bomex_conserved_environment_1h_host_diagnosed.json`. The next unresolved issue is the vertical distribution of cloud-layer transport; no production setting is promoted on these data.

## Conserved environmental reconstruction, 15 September 2026

Reference comparison found a concrete difference in the ascent, not just its final flux evaluation. CAM reconstructs layer-local theta_l and qt in pressure, partitions that conserved state, and obtains environmental buoyancy from it (`uwshcu.F90`, approximately lines 2360–2430). It also uses the upper-cell reconstruction for compensating subsidence at an interface (approximately lines 3285–3325). The host previously interpolated T, qv, qc, theta_l and qt separately between centres, so buoyancy and the entrained conserved state could be thermodynamically inconsistent. An hour-3 cloud-base sample differs by 0.092 K in virtual temperature even before changing the spatial reconstruction.

The opt-in `uw_shallow_conserved_environment` path now reconstructs theta_l, qt and winds within each pressure layer, partitions the reconstructed theta_l/qt together, and uses the upper-side state for interface fluxes. Adaptive ascent steps still land on cell faces and centres. Launch closure, mass-flux limits, mixing coefficients and condensation thresholds are unchanged. This corrects the environment contract; it is not a full port of CAM.

**Host-compatibility qualification:** the initial trials below used CAM-style full saturation partitioning. The production opt-in path now defaults to `uw_shallow_environment_partition = "host"`, using the host's `condensation_rh_crit` and partial-cloud partition. Otherwise even a reconstruction at an unchanged layer centre could spuriously evaporate partial cloud. A regression test verifies that the host mode reproduces T/qv/qc at the centre. The explicit `"saturation"` option retains the reference-comparison variant only. The host-mode hour-5.75 float64 trial retains −1.589684 g/kg/day drying and near-zero energy residual. Repeated evaluations of the same layer/height environment within ascent are cached without changing the equations.

Paired 225 s hour-3 trials reduce local drying from −3.798723 to −2.896428 g/kg/day. Entering water flux stays 2.546664 kg/m²/day; export drops from 4.131999 to 3.754146. The float32 instantaneous energy residual rises to 0.134592 W/m²; a float64 replay preserves the drying result (−2.896063) and closes energy to near machine precision, identifying endpoint quantization rather than a physical source. No energy correction or looser conservation threshold was introduced.

At hour 5.75, paired float64 trials give −3.407603 versus −1.589684 g/kg/day (53% less drying). A float32 900 s call with four internal 225 s steps gives −1.133323 g/kg/day, compared with the matched baseline interval's approximately −2.94247 (61% less drying); its energy residual is −0.014699 W/m². These short tests are not a claim of six-hour BOMEX validation.

Six dedicated tests check linear conserved reconstruction, wet/dry phase consistency, constant conserved profiles, unchanged partial-cloud centre state, and frozen-ascent convergence at 50/25 m internal vertical steps. Thirty-nine distinct focused tests pass including source/configuration/substep and diagnostic tests. Commands and artifacts:

```bash
python scripts/check_uw_conserved_environment.py --hour 3 --partition saturation
python scripts/check_uw_conserved_environment.py --hour 3 --only-conserved --double --partition saturation
python scripts/check_uw_conserved_environment.py --hour 5.75 --double --partition saturation
python scripts/check_uw_conserved_environment.py --hour 5.75 --only-conserved --timestep 900 --internal-step 225 --partition saturation
python scripts/check_uw_conserved_environment.py --hour 5.75 --only-conserved --double --partition host
python scripts/screen_uw_conserved_environment.py --partition host
```

Results use `bomex_conserved_environment_*` under `outputs/column/diagnostics`. The six-hour screen retains the previous full-physics BOMEX forcing and configuration overrides, adding only conserved-environment reconstruction and internal shallow substeps. Production defaults and notebook checkpoints remain unchanged.

**Saturation-control six-hour result: rejected for promotion.** Maximum theta_l change is −0.76834 K and total-water change +0.98013 g/kg, both at 1822.6 m; late rain is 0.06652 mm/day. Only rain passes the unchanged gates. At 825.4 m the column still dries by 0.912 g/kg (versus 1.007 previously), while upper-cloud-layer moistening increases. The reference-like environmental representation plus substeps is therefore not a complete remedy. The initial saved-state reductions must not be presented as an improvement of the entire coupled solution. This control ran the saturation-only implementation identified by its recorded hash; the separate host-compatible screen tests the partial-cloud contract explicitly.

**Host-compatible six-hour result: also rejected for promotion.** Theta_l change −0.76776 K, water change +0.97970 g/kg, and late rain 0.06716 mm/day; only rain passes. This establishes that the failed profile result is not remedied by simply switching between the saturation and host partial-cloud partition. File: `bomex_conserved_environment_6h_host.json`. Both six-hour screens omit the subsequent diagnosed-sorting correction; their recorded source hashes identify the executed versions.

## Matched launch comparison, 15 September 2026

One archived hour-5.75 interval was repeated with both original and Gaussian launch closures using the same four internal 225 s updates. At 825 m, baseline total-water change is −0.030650757 g/kg and Gaussian change is −0.041592866 g/kg (35.7% more drying). Baseline water/energy residuals are 3.39e-10 kg/m²/s and 0.0069294 W/m²; Gaussian residuals are 1.36e-9 kg/m²/s and 0.0037538 W/m². The Gaussian result reproduces the preceding internal-substep replay.

This removes the unequal time-integration ambiguity: joint Gaussian launch is not a cure for the local water bias. The mechanism remains excessive upward flux divergence, but these data do not uniquely identify a wrong physical coefficient. Retain the baseline closure while auditing the plume/environment scalar contrast and entrainment/detrainment terms. No physical parameter was tuned to reduce the drift, and no claim of a solved BOMEX bias is made.

Reproduction: `scripts/check_bomex_inversion_substeps.py --substeps 900 --internal-step 225 --output outputs/column/diagnostics/bomex_matched_launch_225.json`. As before, TKE and boundary depth are frozen inputs, and raw captured launch diagnostics belong to the last internal step rather than an interval average.

## Internal time integration correction, 15 September 2026

The host `subcloud_flux` displacement formula was rechecked against the saved CAM `fluxbelowinv` implementation (lines 4969–5034). The finite-horizon threshold is present in the reference too; removing it or declaring it a transcription bug would not be justified. Instead, `uw_shallow_convection` now accepts `uw_shallow_maximum_timestep_s`: when positive and shorter than the host timestep, the complete shallow operator is internally substepped. Source properties, plume ascent, inversion reconstruction, CIN response, rain handling and phase partition are recomputed from the evolving T/q/qc/u/v state. TKE and diagnosed boundary depth remain inputs held fixed over this shallow call. No persistent subcell inversion state was invented.

The original computation remains in `shallow_step`. The wrapper returns endpoint tendencies and interval-mean rate/profile diagnostics (the named maximum condensate retains its temporal maximum), and recomputes water/energy residuals from endpoint storage and mean rain. It neither clips condensate afterward nor applies an energy correction. Inputs and caller parameters are not mutated. The option defaults to zero, retaining the old single-step behavior; no production configuration or checkpoint is promoted. This is a numerical integration facility, not a new physical closure or a complete cure for BOMEX drying.

Nine new wrapper tests cover evolving state, unequal layer masses, two-column batching, rain averaging, endpoint conservation, uneven subdivisions, input immutability, legacy dispatch and invalid time settings. Together with the prior diagnostic and source-reference tests, 28 tests pass.

```bash
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python scripts/check_bomex_inversion_substeps.py --substeps 900 --internal-step 225 --joint-only --output outputs/column/diagnostics/bomex_internal_substeps_225.json
```

The Gaussian launch replacement remains diagnostic-only. In an internally substepped replay, captured raw launch values describe the last internal call, while returned tendencies and precipitation cover the whole interval; do not treat the captured raw flux as an interval mean.

**Replay passed:** one 900 s call with maximum internal step 225 s matches the prior four externally applied 225 s steps exactly in total-water and theta_l endpoint profiles. Water change at 825 m is −0.041592866 g/kg; integrated water residual is 1.35954e-9 kg/m²/s and energy residual 0.00375379 W/m². Integrated rain differs by only 7.67e-11 kg/m² from floating-point averaging. Assertions require profile agreement within 2e-6 g/kg and the unchanged 2e-8 kg/m²/s and 0.1 W/m² conservation tolerances. Output: `outputs/column/diagnostics/bomex_internal_substeps_225.json`. This verifies implementation equivalence to the refined update, not elimination of physical drying. A matched internally refined baseline comparison remains the next bounded comparison; the Gaussian closure is not promoted.

## Evolving-state inversion check, 15 September 2026

The saved hour-5.75 state was advanced through one 900 s interval using shallow physics alone. Each substep updates temperature, vapor, condensate and winds, and re-diagnoses the source and plume. Pressure, TKE, native interface TKE and boundary-layer depth remain fixed. This isolates the shallow time update; it does not establish full coupled-model stability.

| launch closure | substep | total-water change at 825 m |
|---|---:|---:|
| baseline | 900 s | −0.035496 g/kg |
| joint Gaussian | 900 s | +0.026335 g/kg |
| joint Gaussian | 450 s | −0.044034 g/kg |
| joint Gaussian | 225 s | −0.041593 g/kg |
| joint Gaussian | 112.5 s | −0.040401 g/kg |

The apparent joint-closure moistening reverses under substepping. At 450 and 225 s the initial tendencies are −4.720867 and −4.720688 g/kg/day, matching the earlier algebraic estimate without the coarse-step inversion-displacement term. Subsequent tendencies relax as the state evolves. The maximum 450-to-225 s profile differences are 0.002441 g/kg in total water and 0.001031 K in liquid-water potential temperature. A refined baseline comparison was not run: this experiment tests joint-update consistency, not the relative accuracy of two converged closures.

Integrated budgets use fixed layer mass and endpoint storage, with precipitation included in total water. Thermodynamic energy uses the host's cp*T + Lv*qv convention, not a separately added hydrostatic potential-energy storage term. Maximum water residual is 1.36e-9 kg/m²/s and maximum absolute energy residual is 0.030 W/m². All sources remain saturated. Two new driver tests verify evolving-state reuse and conservative exchange arithmetic and reject substeps that do not cover the interval; twelve tests pass including the existing launch and budget tests.

**Decision:** do not interpret the 900 s moistening as a physical improvement or promote the launch closure. The next numerical target is the moving-inversion reconstruction's consistency with evolving layer means and swept subcell mass. Conservation passes but time accuracy does not follow from it. No new equilibrium, production-physics edit, default change or checkpoint promotion was made.

The final 112.5 s refinement changes the water profile by at most 0.001193 g/kg and theta_l by 0.000531 K relative to 225 s. These differences approximately halve with timestep, consistent with first-order convergence toward drying. Its integrated water residual is −4.88e-10 kg/m²/s and energy residual +0.02215 W/m². This supports rejection of the coarse-step sign reversal as a robust improvement, not physical validation of the refined solution.

```bash
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python scripts/check_bomex_inversion_substeps.py
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python scripts/check_bomex_inversion_substeps.py --substeps 112.5 --output outputs/column/diagnostics/bomex_inversion_substeps_fine.json
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python -m pytest scm/test_inversion_substep_diagnostic.py scm/test_joint_launch_diagnostic.py scm/test_bomex_launch_budget.py -q
```

Output: `outputs/column/diagnostics/bomex_inversion_substeps.json`. The runner reuses the diagnostic in-memory closure replacement, now optionally returning the evolved host state. The production plume hash remains unchanged from the joint-launch experiment below.

## Joint launch experiment, 15 September 2026

`scripts/test_bomex_launch_closure.py` uses archived plume inputs from hours 0, 3 and 5.75, retaining source properties, density, CIN and the existing ascent. It creates a guarded in-memory diagnostic copy of the ascent function that replaces only the launch mass/velocity pair; production source files are not modified. The Gaussian variant derives mass flux, conditional velocity and area together, with the 10% area bound, source-layer transported-mass bound, and TKE variance offset from [CAM cam6_3_000](https://raw.githubusercontent.com/ESCOMP/CAM/cam6_3_000/src/physics/cam/uwshcu.F90), around lines 2094–2167. Exact Gaussian constants and the inverse erfc area threshold replace the rounded literals. All tested parcels are already saturated at launch and CIN is zero; this test does not implement or validate CAM's additional unsaturated-LCL constraint or its complete plume model.

| saved hour | baseline water tendency at 825 m | joint Gaussian | baseline / joint mass flux | baseline / joint velocity |
|---|---:|---:|---:|---:|
| 0 | −5.97360 g/kg/day | −7.99717 g/kg/day | 0.075778 / 0.094142 kg/m²/s | 0.66972 / 0.83202 m/s |
| 3 | −3.79863 g/kg/day | −5.26688 g/kg/day | 0.078716 / 0.097784 kg/m²/s | 0.69599 / 0.86459 m/s |
| 5.75 | −3.40766 g/kg/day | +2.52816 g/kg/day | 0.079012 / 0.098151 kg/m²/s | 0.69884 / 0.86812 m/s |

The hour-3 velocity-only diagnostic retains the baseline mass flux, gives source area 0.08050, and produces −4.23744 g/kg/day. It is a causal decomposition, not a proposed complete closure. The joint closure increases both mass flux and velocity by about 24%; it worsens drying in the first two states.

The final-state reversal is not evidence of a stable cure. The moving-inversion reconstruction uses the ratio of transported mass fraction to reconstructed inversion fraction. At hour 5.75 the baseline transported fraction is 0.171822, below the reconstructed 0.194949; the joint value is 0.213442, activating the displacement term. Linear subcloud inflow alone would be 3.332679 kg/m²/day, but the term raises it to 6.332798; above-layer export is 5.305110. The added 3.000120 kg/m²/day contributes +7.249057 g/kg/day to this cell. Subtracting this identifiable term gives a counterfactual tendency of −4.72090 g/kg/day. This is an algebraic decomposition of one frozen evaluation, not a separately evolved solution. The first two states do not cross this threshold.

All baseline raw-flux replays have zero discrepancy. The seven trial water residuals remain below 1.32e-9 kg/m²/s, and absolute energy residuals below 0.02942 W/m². Source liquid water is 0.000247–0.000294 kg/kg, verifying the saturated-source restriction. Six analytic tests cover half-normal moments, area bounds, mass/velocity/area identity, the thin-layer mass bound, and strong-CIN shutdown. Together with existing diagnostic-budget tests, ten tests pass. No new full-column integration was run.

**Decision:** do not promote this joint closure on these results. The next bounded test is the joint-source/moving-inversion update over one evolving-state interval with shorter substeps. Finite-step fluxes legitimately depend on the time horizon; verify integrated-budget consistency rather than calling threshold activation itself a bug. This experiment establishes that source-cap activity alone is not enough to explain or cure the excessive drying.

```bash
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python scripts/test_bomex_launch_closure.py
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python -m pytest scm/test_joint_launch_diagnostic.py scm/test_bomex_launch_budget.py -q
```

Output: `outputs/column/diagnostics/bomex_joint_launch_trials.json`, including full tendency profiles, conservation residuals and input/physics hashes. The production plume SHA256 remains `eb1c1c4320da64b3f20b704fd5822ff1611d64b9b34c1aea8d9d0ca622b1e3dd`.

## Launch-layer budget follow-up

`scripts/audit_bomex_launch_budget.py` repeats the corrected six-hour, 20-level, 900 s screening without changing its tendencies. Every step stores per-process total-water and theta_l rates, raw/applied shallow fluxes, the subcloud linear and inversion-displacement pieces, precipitation terms, and replayable plume input including native interface TKE. It also saves the final state. `--summarize` produces full-run and late-window means; the final endpoint and late rain reproduce the preceding screening exactly.

At the approximately 825 m full level:

| process | six-hour total-water change (g/kg) | hours 3–6 water tendency (g/kg/day) | hours 3–6 theta_l tendency (K/day) |
|---|---:|---:|---:|
| boundary-layer transport, including surface supply | +0.506862 | +2.192162 | −1.145343 |
| prescribed forcing | −0.485174 | −2.005325 | −0.863889 |
| shallow convection | −1.028933 | −3.574520 | +3.777433 |
| deep/dry/condensation/cloud terms | approximately 0 | approximately 0 | approximately 0 |
| actual storage | −1.007247 | −3.387682 | +1.768463 |

The shallow contribution further separates into +6.334024 g/kg/day from the subcloud linear flux, −9.943637 from above-source transport, zero moving-inversion correction, and +0.035090 from precipitation evaporation minus plume precipitation production. The entering and outgoing shallow fluxes are respectively 2.621421 and 4.115308 kg/m²/day. Their imbalance explains the drying; the tiny precipitation term opposes it. The source stays at host index 16, below the affected index-15 cell. Subcloud/above-source attribution is an algebraic split of a shared conservative flux field, not two independent parameterizations or two separately added launch fluxes.

The complete water budget closes within 0.0001141 g/kg/day over all levels/steps; theta_l within 0.002151 K/day. Shallow flux-plus-precipitation reconstruction agrees with its returned total-water tendency within 0.0002211 g/kg/day. Accumulated actual rates reproduce endpoint water within 5.6e-8 g/kg. Small residuals include float32 update/partition roundoff. An initial recorder logged the deep scheme at both registry and dispatch; the summary explicitly removes that alias before evaluating closure. The raw file explains this and notes that its recorder-script hash was captured after diagnostic code development, while the physics hashes match the preceding screening. Future runs capture hashes before integration.

`scripts/check_bomex_launch_flux.py` replays hour 3.0 with **zero raw water-flux discrepancy**. At the first overlying face (approximately 1020 m), upward water flux is 4.1320 kg/m²/day; plume total water is 16.1913 g/kg versus environmental 12.9927. Mass flux has fallen from 0.078716 at launch to 0.014951 kg/m²/s: this is not runaway mass-flux growth. CAM's upper-cell reconstruction gives the same environment at this face, so changing that sample alone does not fix its flux. At the next face it would increase export, a diagnostic counterfactual rather than a validated correction.

The frozen source requests 0.222643 kg/m²/s and its area cap reduces that to 0.078716. The two subsequent area caps are inactive. Overall implicit-CIN/positivity scaling is 1 throughout the column replay. These are distinct limiters and must not be conflated. CAM's source calculation diagnoses mass flux, velocity and area jointly from a truncated velocity distribution; our existing source instead independently clips mass flux while retaining its prescribed TKE-derived velocity. See [CAM tagged source](https://raw.githubusercontent.com/ESCOMP/CAM/cam6_3_000/src/physics/cam/uwshcu.F90), source closure around lines 2094–2167 and above-inversion fluxes around 3287–3323. This verified difference motivates the next **frozen-input joint-closure test**, not an assumption that lifting the cap improves the column.

Reproduction:

```bash
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python scripts/audit_bomex_launch_budget.py --output outputs/column/diagnostics/bomex_launch_budget_20_6h.json
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python scripts/audit_bomex_launch_budget.py --summarize outputs/column/diagnostics/bomex_launch_budget_20_6h.json --output outputs/column/diagnostics/bomex_launch_budget_20_6h_summary.json
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python scripts/check_bomex_launch_flux.py --input outputs/column/diagnostics/bomex_launch_budget_20_6h.json --hour 3 --output outputs/column/diagnostics/bomex_launch_flux_hour3_caps.json
```

Four new tests cover unequal-layer flux telescoping, conserved theta_l under condensation, diagnostic alias handling, and immutable tensor serialization. No model physics, default, notebook or checkpoint was modified by this audit.

## Source/subcloud implementation follow-up

The experimental UW source/subcloud correction is implemented. Earlier statements below describing missing transport or unchanged physics refer to their pre-correction audit stage.

- Source support is selected from pressure-mass interfaces and their heights, with CAM's 5 m inversion-search offset. Face heights use the same piecewise-linear pressure/height interpolation as the plume environment. Ascent and flux evaluation now stop at those pressure faces, not arithmetic height midpoints.
- Source total water remains the lowest full-level value. Source theta_l comes from the minimum reconstructed virtual-liquid potential temperature divided by the source virtual correction. Source winds are reconstructed just below the launch interface.
- The boundary-layer output retains native interior interface TKE. The shallow closure pressure-weights it over the source support. There is no separate surface TKE in this turbulence implementation: a stated nearest-interior, zero-gradient boundary value is used. Old states without native interfaces use a layer-to-interface interpolation fallback, not a claimed exact CAM reconstruction.
- The same source thermodynamic and geometric diagnosis is used in the predicted-CIN calculation; its closure factor uses the launch's averaged TKE. The existing mass-flux functional form and density approximation are otherwise retained.
- CAM-style pressure-linear subcloud fluxes transport total water, theta_l and momentum, including the moving-inversion correction. Their surface flux is zero. The launch-face flux is retained when ascent begins there, rather than overwritten by a second source-face formula. Theta_l flux is converted with face Exner and combined with latent total-water flux to match the host MSE convention. Flux divergences telescope across the complete column.
- The optional refined-physics-grid adapter interpolates the intensive interface TKE diagnostic in pressure. It neither treats interface values as layer means nor passes arrays belonging to another grid through unchanged. Switching away from an interface-producing turbulence scheme clears stale interface state.

Validation so far: **42 distinct targeted tests pass** (six ascent/conservation tests, seven source-reference tests, seven lightweight convection tests, ten turbulence/benchmark-input tests, and twelve grid-remapping tests). The source tests include analytic inversion crossing, uniform scalars, an explicit native-interface TKE quadrature, and agreement with the independent reference translation. Three frozen states retain the 50/10 m ascent convergence gates; the full and partially entered top-cell water/energy tests pass unchanged. This is not a full-suite pass: the older expensive multi-resolution convection tests were excluded, and the previously documented corrected-benchmark regressions in other configurations remain unresolved.

One six-hour 20-level screening run uses:

```bash
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python scripts/bomex_observed_steady_state.py --config scm/configs/uw_candidate_v1.toml --levels 20 --hours 6 --dt 900 --set uw_layer_closure=true uw_shallow_keep_crossed_interface=true cloud_ls_precip_fraction=0 entrainment_rate=0.00005 --output outputs/column/diagnostics/bomex_source_subcloud_20_6h.json
```

The completed run gives maximum absolute theta_l change **0.579742 K**, maximum absolute total-water change **1.007247 g/kg**, and late rain **0.111152 mm/day**. All three fixed gates fail. Compared with the prior corrected-ascent run (0.510040 K, 0.598713 g/kg, 0.365205 mm/day), precipitation improves but the moisture-profile error worsens. The largest drying is at approximately 825 m, with moistening below (498 m) and aloft (1823 m). Cloud water path is 0.034731 kg/m², maximum cloud fraction 0.258718, and boundary-layer depth 662.126 m. The endpoint alone does not distinguish excessive plume export from changed boundary-layer supply or phase changes; the next diagnostic must separate their flux divergences across the launch/inversion layer. The corrected candidate is **not ready for promotion**.

Tested source SHA256 hashes (the working tree also contains unrelated earlier edits):

```text
eb1c1c4320da64b3f20b704fd5822ff1611d64b9b34c1aea8d9d0ca622b1e3dd  scm/convection_uw.py
11f49af4ccddf07704835d8011f6d6fbb712a5ef912ae656867534cd8af4af42  scm/boundary_layer_uw.py
243b7f634ba69e2e4d915f0e83b1bb80bf3b189b11ec5defe347067464e7068d  scm/column_model.py
af71b357a797964a0e2ce1c468c784001d2d277a895dfa2d99a4c0343a6043bc  scm/case_benchmarks.py
```

Defaults, student notebooks and checkpoints were not changed by this follow-up.

## Decision

Do not reject UW on the strength of the September 12 stop rule. Our implementation is not equivalent to the reference scheme. Surface moisture enters, but the plume often stops before its first transport interface. Positive launch mass flux does not establish actual water export. Repair the benchmark and launch-to-interface contracts before tuning another closure.

No production physics, configuration, notebook, or checkpoint was changed. The added audit script records budgets and performs diagnostic-only comparisons.

## Reproduction

`scripts/audit_bomex_transport.py` runs 20 levels for six hours using `uw_candidate_v1.toml` with the latest documented UW experiment's explicit settings: `uw_layer_closure=true`, `uw_shallow_keep_crossed_interface=true`, `cloud_ls_precip_fraction=0`, and transported-plume `entrainment_rate=5e-5`. This is **not** the promoted ATM407 configuration, which dilutes only the CAPE parcel. Other physics remains active; this is the existing screening experiment, not isolated shallow convection or executable CAM.

```bash
cd /Users/evanwellmeyer/Documents/GCM
/Users/evanwellmeyer/miniconda3/envs/atm407/bin/python scripts/audit_bomex_transport.py --output outputs/column/diagnostics/bomex_transport_audit_900.json
/Users/evanwellmeyer/miniconda3/envs/atm407/bin/python scripts/audit_bomex_transport.py --dt 300 --output outputs/column/diagnostics/bomex_transport_audit_300.json
/Users/evanwellmeyer/miniconda3/envs/atm407/bin/python scripts/audit_bomex_transport.py --liquid-forcing --output outputs/column/diagnostics/bomex_transport_audit_liquid_900.json
```

JSON outputs record resolved parameters, every step's water budget, raw/applied fluxes, first-ascent trajectories, frozen-state vertical-step replays at hour 3.5, and key source hashes. HEAD was `a208fddf649966d10e3168323a02b089d654bd51` with pre-existing uncommitted changes; HEAD alone does not identify the tested code. Outputs remain gitignored. The local CAM source SHA256 is `a7ec372f33c6b6fdbee1fe4c4a99ada23f0128206e792ea17a6462c76d67ce2f`.

## Results

| Six-hour run | Maximum water change below 2.5 km, g/kg | Rain during hours 3–6, mm/day | Late samples without shallow water flux |
|---|---:|---:|---:|
| Existing forcing, 900 s | 1.643 | 0.781 | 9/12 |
| Existing forcing, 300 s | 1.600 | 0.627 | 21/36 |
| Liquid-potential-temperature forcing, 900 s | 1.610 | 0.977 | 11/12 |

Water changes are relative to the initial state, not an LES envelope. All fail the existing local 0.5 g/kg gate. Neither timestep reduction nor the temperature-forcing change independently restores transport.

The 900 s baseline establishes:

- Surface evaporation averages 5.279 kg/m²/day. Integrated boundary-layer moistening matches the supplied surface flux within `8.2e-9 kg/m²/s` on every step.
- Summed process tendencies agree with actual layer water storage within `0.000210 g/kg/day`. Condensation/cloud budgets include condensate, not just vapor.
- At approximately 1.023 km, balancing the supplied forcing below the interface in a non-raining steady column would require upward transport of 3.915 kg/m²/day. Boundary-layer transport supplies 0.276 and shallow transport 0.867. This is a calculated forcing-balance requirement, not downloaded LES flux; transient storage and precipitation explain the difference.
- Near 805 m, boundary-layer moistening is +12.513 g/kg/day, shallow transport −2.095, forcing −1.956, and condensation plus clouds −1.886. The remaining +6.576 accumulates. Near 1242 m, water instead declines by 2.368 g/kg/day.
- Mean launch mass flux is 0.04265 kg/m²/s. The late implicit CIN factor is always 1; shallow precipitation is zero. These do not explain the missing transport. Positive launch mass flux persists on steps with entirely zero applied water flux.
- At hour 3.5 the source is near 805 m. Squared plume velocity becomes negative before the roughly 1023 m interface. The retained-crossing switch cannot help a plume that never crossed. There is no condensate-flux handoff on that step.

**Frozen-state numerical failure:** replaying that exact hour-3.5 baseline input with internal vertical steps of 50, 25, and 10 m produces maximum upward water fluxes of **0, 1.640, and 4.732 kg/m²/day**, respectively. Launch mass flux stays exactly 0.057806 kg/m²/s. These are raw plume replays, not different atmospheric equilibria. The 300 s trajectory's frozen state responds non-monotonically (2.932, 0, 0.703), so simply choosing 10 m is not a demonstrated fix. The saved input arrays permit the ascent/interface operator to be tested independently of subsequent column evolution. This is a direct numerical acceptance failure, stronger evidence than comparing equilibrium profiles.

## Benchmark mismatches

The [published specification](https://cdn.knmi.nl/system/data_center_publications/files/000/066/548/original/siebesma_04_jas_copy1.pdf?1495620557=) uses liquid-water potential temperature and total specific water, includes prescribed momentum flux and large-scale momentum forcing, and compares six-hour LES statistics. Our pointwise ±0.5 gates are local screening choices. The harness forces dry potential temperature, omits momentum forcing, and uses a different sounding/cooling-taper variant. It is not yet a faithful reproduction.

Water was copied from the [CLUBB sounding](https://raw.githubusercontent.com/larson-group/clubb_release/master/input/case_setups/bomex_sounding.in), whose `rt` denotes mixing ratio ([definitions](https://github.com/larson-group/clubb_release)), directly into SCM specific humidity. For example, `r=0.01729` means `q=r/(1+r)=0.016996`, not `q=0.01729`: about 0.294 g/kg extra near-surface specific water. Its effect was not isolated here. The height helper also omits the surface-to-lowest-center offset.

The liquid-forcing experiment changes only the subsidence temperature variable; condensate stays unchanged during forcing, so `dT = Exner * dtheta_l` applies exactly to that step. It is not a complete benchmark correction. The alternative forcing differs by up to 0.491 K/day along the baseline trajectory. Surface heat-flux conversion also omits Exner; surface water delivery itself is consistent with the harness's chosen density.

## Reference-code gaps

Comparison with local `outputs/cam_reference/uwshcu.F90` and the [tagged CAM source](https://raw.githubusercontent.com/ESCOMP/CAM/cam6_3_000/src/physics/cam/uwshcu.F90) establishes:

1. CAM uses pressure-weighted boundary-layer interface TKE and virtual-liquid-potential-temperature source construction. We use one full-level TKE and minimum liquid potential temperature. Source TKE here averages 0.128 m²/s²; a layer-mean full-level proxy is 0.208. That proxy is not an exact CAM calculation.
2. CAM reconstructs below-inversion flux in `fluxbelowinv`. Our plume leaves fluxes below its source at zero, despite sourcing humidity from the lowest level. Boundary-layer diffusion does not establish equivalence to that reconstruction.
3. CAM uses buoyancy sorting with a physical stopping-distance scale. Our binary rule tests `buoyancy + velocity_squared / (2 * step_height) > 0`, where `step_height` is the numerical integration increment. Changing that increment therefore changes the mixing decision, not just integration accuracy. This is an implementation-level source of the frozen-state sensitivity.
4. We generally place upper-full-level plume properties into midpoint-interface fluxes; retained crossings also use substep-end rather than consistently interpolated scalar properties.

The first missing transport component relative to CAM is below-source flux reconstruction. Failed export also involves launch/first ascent. This does not prove which individual correction will recover BOMEX, but it invalidates attributing failure to UW's established physics.

## Next bounded task

Test and correct benchmark moisture units and conserved-variable forcing first. Then implement one coherent reference-based launch/first-interface contract: source thermodynamics, TKE sampling, below-source flux reconstruction, and interface evaluation. Start from the frozen failing states and require convergence of the inner vertical integration, water/energy closure, and actual flux, not merely positive diagnosed mass flux. In particular, the ascent and interface evaluation must not turn transport on/off simply because an internal step was halved. Do not manufacture flux across an interface the plume never reaches. No long equilibrium or default promotion yet.

Validation: `scm/test_convection_uw.py`, `scm/test_boundary_layer_uw.py`, and `scm/test_uw_layers.py`: **18 passed** in the `gcm` environment. The `atm407` environment ran the integrations but lacks pytest. These tests do not establish BOMEX acceptance. No executable CAM comparison or LES-data fit was performed.

## Implementation follow-up, 14 September 2026

The numerical correction is now implemented in the experimental UW path. The benchmark now uses the published Table B1 specific-water and liquid-potential-temperature sounding directly, the 3 km cooling taper, geostrophic/subsidence momentum forcing, prescribed surface stress, and the surface Exner factor for sensible heat. A benchmark-only surface height datum includes the lowest cell's surface-to-centre offset; other grids retain their old datum. The upper extension is subsaturated and matches the sounding-top RH. Existing production configurations and notebook checkpoints are unchanged.

The plume now separates physical mixing from numerical integration. Buoyancy sorting uses a fixed physical stopping distance (100 m by default), not the integration increment. Entrainment and detrainment are rates per metre derived from the sorted fraction. An adaptive midpoint solver with step doubling controls error and stops exactly at transport interfaces. Internal thermodynamics use float64 and a more accurate saturation solve. Interface fluxes use co-located pressure, parcel scalars, and velocity. Crossed-interface transport is always retained; the former `uw_shallow_keep_crossed_interface` switch no longer selects a broken path. Removing excess liquid adjusts liquid potential temperature consistently, and partially entered top cells keep their conservative flux updates.

This remains an approximate UW-inspired scheme, not a validated CAM port. The fixed physical stopping distance is not CAM's evolving cloud-depth calculation. Source thermodynamics/TKE averaging and below-source flux reconstruction identified above remain unresolved; this correction did not silently retune them.

All three archived frozen states pass the preselected convergence gates across **50, 25, and 10 m** internal steps. Maximum changes across those runs are **0.00941 kg/m²/day** for water flux, **0.205 W/m²** for MSE flux, and **0.00757 m** for plume-top height. Gate limits were 0.05 kg/m²/day, 2 W/m², and 2 m. This establishes numerical convergence on these cases, not realism of the resulting flux magnitudes. The frozen states are tracked in `scm/testdata/uw_launch_frozen_20260914.json`; unlike the old ignored outputs, they remain available in a fresh clone.

Reproduce the numerical check with:

```bash
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python scripts/check_uw_launch_convergence.py
```

Validation so far: 26 targeted tests passed (four benchmark-input tests, six ascent/conservation tests, and sixteen existing component tests). Two older multi-resolution six-hour composite tests were not completed in this pass. The separate legacy benchmark suite has **14 passes and three failures** under the corrected inputs, with thresholds unchanged: TKE/plume mass-flux convergence (spread 0.103 kg/m²/s, limit 0.015), EDMF depth convergence (spread 76 m, limit 75), and EDMF detrained-water-path convergence (spread 0.136 kg/m², limit 0.1). These other closures were not changed by the ascent correction and are not validated under the corrected benchmark.

The final 20-level, six-hour screening run, with the tighter numerical tolerances that pass all frozen-state gates, gives maximum liquid-potential-temperature change **0.5100 K**, maximum water change **0.5987 g/kg**, and late rain **0.3652 mm/day**. All three screening gates still fail. Cloud water path is 0.04549 kg/m², maximum cloud fraction 0.2639, and boundary-layer depth 662.3 m. This is a smaller profile error than the old experiment, but both the benchmark and plume changed, so it is not a clean attribution experiment. The looser preliminary integration and final integration agree to the quoted two-decimal profile precision.

```bash
/Users/evanwellmeyer/miniconda3/envs/gcm/bin/python scripts/bomex_observed_steady_state.py \
  --config scm/configs/uw_candidate_v1.toml --levels 20 --hours 6 --dt 900 \
  --set uw_layer_closure=true uw_shallow_keep_crossed_interface=true cloud_ls_precip_fraction=0 entrainment_rate=0.00005 \
  --output outputs/column/diagnostics/bomex_corrected_ascent_20_6h.json
```

Next is source-state/TKE sampling and below-source flux reconstruction against the reference code, not precipitation retuning or a long equilibrium. The production defaults and notebook checkpoint remain unchanged.
# Source and subcloud reference follow-up

Compared `CAM/cam6_3_000/src/physics/cam/uwshcu.F90`, specifically source reconstruction (1270–1310, 1460–1528), scalar flux call sites (3248–3273), pressure slopes (4772–4802), and `fluxbelowinv` (4969–5034). Reference: https://raw.githubusercontent.com/ESCOMP/CAM/cam6_3_000/src/physics/cam/uwshcu.F90. Local reference SHA256: `a7ec372f33c6b6fdbee1fe4c4a99ada23f0128206e792ea17a6462c76d67ce2f`.

`scripts/check_uw_source_contract.py` is an isolated diagnostic translation, not production physics. It holds the current launch mass flux and source-layer support fixed. It reproduces CAM's pressure-slope and below-inversion scalar algebra and compares three archived states. Four manufactured tests cover linear reconstruction, uniform scalars, inversion crossing, zero surface flux and telescoping column conservation. These tests do not constitute execution of CAM itself.

| frozen state | source theta_l, current / reconstructed (K) | source TKE / layer-weighted proxy (m²/s²) | reconstructed water flux at highest compared subcloud face (kg/m²/day) |
|---|---:|---:|---:|
| 900 | 299.057 / 298.979 | 0.139 / 0.219 | 4.928 |
| 300 | 299.103 / 299.019 | 0.110 / 0.187 | 3.221 |
| liquid_900 | 299.074 / 298.984 | 0.117 / 0.207 | 3.108 |

All three have four interior subcloud interfaces with **zero current shallow flux**. With identical launch mass flux, the reconstructed flux increases through those interfaces; the highest of the four is 3.136, 2.050 and 1.978 kg/m²/day respectively. Current transport instead starts above the source full level, producing 22.661, 16.510 and 16.762 kg/m²/day at the next indexed face. This is evidence of a concentrated source-layer divergence, not evidence that the CAM reconstruction alone will fix the integrated column. Both schemes have zero internal plume flux at the physical surface; prescribed surface evaporation must not be added twice.

Limitations: CAM selects its inversion cell from interface heights (including a 5 m offset), not the current full-level cutoff. This comparison deliberately fixes the source-layer set rather than reproducing that independent diagnosis. Current ascent uses midpoints of full-level heights/pressures, whereas reference reconstruction uses pressure faces from layer mass; the highest-face magnitudes are therefore not a fully co-located CAM solution. Source virtual correction uses a declared 0.608 coefficient. Native CAM averaging uses interface TKE and includes the surface; the fixture only stores full-level TKE. The layer average above is explicitly a proxy, not an exact CAM closure prediction. Source water agrees in definition (lowest full-level total water), but source winds and minimum virtual-liquid temperature require interface reconstruction. The current implicit CIN update repeats the same source-level TKE/minimum-theta approximations.

Recommended correction: pass native interface TKE and consistent faces through the boundary-layer/shallow contract; diagnose the source and predicted source identically; reconstruct subcloud total-water and thermodynamic fluxes and join them conservatively to the plume. CAM converts theta_l flux to liquid-static-energy flux using interface Exner; our MSE convention additionally requires the matching latent total-water term. Do not paste a theta_l tendency directly into temperature or MSE. Require column conservation and interface continuity first, then one six-hour 20-level BOMEX comparison. No long equilibrium, default promotion, or production physics edit was performed by this audit.
