# ATM407 single column: state and open problems

> September 7 audit: start with [the reconciled status and recovery plan](column_audit_2026-09-07.md). The notes below are preserved development history, not a consistent current specification. In particular, their claims of grid convergence, failed shared partitioning, completed UW cloud-fraction handoff, and a precipitation-efficiency shortfall are not established by the current evidence. Do not promote a configuration on those claims.

> Current reference (10 Sep): `atm407.toml`, label `atm407_plume_stop` (boundary-layer `k_diff = 40`, plume stops at neutral buoyancy). 800 days on a 5 m slab after the stop was switched on: TOA +0.46 W/m2, surface -0.23 W/m2, surface temperature 289.0 K, CAPE 566 J/kg, theta_v drop 0.47 K. Passes every equilibrium gate. The notebooks use it. The `atm407_flux_v1` baseline in the next note is superseded.

> [Superseded 10 Sep.] Reproducible baseline: the conservative `atm407_flux_v1` settings are now the ATM407 and repository defaults. After 500 days with a 5 m slab, correctly sampled 50-day means pass the bulk gates: TOA +0.93 W/m2, surface +0.41 W/m2, drift 0.038 K, CAPE 777 J/kg, no RH95 layer, and deep rain exceeding large-scale rain. The canonical checkpoint and both student notebooks use it. This establishes a reproducible numerical baseline, not scientific acceptance of its vertical structure; the September 8 findings below must be resolved before that claim is made.

## Tried and ruled out: do not retry without new evidence

Each row was tested and failed, or was withdrawn. Search this file for the name
in the first column to find the details. Updated 10 Sep.

| tried | result |
|---|---|
| Height offset fix (lowest level at 0 m, not about 21 m) | barely moves the column, breaks 4 tests; reverted |
| MSE-conservation correction as the cause of stratospheric cooling | tendency unchanged when moved or turned off |
| CAPE parcel launched from a 20 or 100 hPa mean | refuted; the 100 hPa mean kills CAPE on the smooth test sounding |
| Plume source air from the lowest 20 hPa | refuted |
| Boundary-layer source layer (`mf_source_layer`) | withdrawn; the divergence it fixed came from interpolated soundings, and it kills CAPE on a smooth native sounding |
| Plume overshoot as implemented (`mf_plume_overshoot`) | leaves a 205 K layer at 235 hPa; not promoted |
| Plume spectrum (`mf_plume_count` 3 or 5, spread 3 or 10) | lifts the plume top one level, to 305 hPa, and adds a cooling spike there; the source air is the limit |
| UW partial-layer fix, as implemented (11 Sep) | kept 3 of 24 plumes at 20 levels and broke a cloud-fraction test; reverted |
| GFDL mixing rate (3 per km) alone in our UW plume | deeper plumes, but 2.5-3.5x the cloud-water limit |
| Legacy mass-flux transport (`mf_transport_form = 'legacy'`) | loses 122.6 W/m2 of column MSE; removed 10 Sep |
| `k_diff_cap_factor` / `unstable_diffusion_boost` as the limiter | not binding |
| Raising the boundary-layer depth ceiling | does not fix the near-surface instability |
| Stronger shallow-convection cap | pumps band moisture downward; wrong design for this |
| Autoconversion threshold 0.2 -> 0 g/kg | inert; band unmoved |
| Surface moisture stencil 0.005 -> 0.05 | band unmoved |
| `cloud_ls_precip_fraction` 0.95 -> 0.05 | humidity worsens |
| Condensate sedimentation | monotonically worse |
| Time step 900 -> 100 s | instability unchanged; it is physical, not numerical |
| RRTMG band calibration of the longwave | ts 285.45 -> 279.36 K, CAPE -39%; reverted |
| `surface_heat_sigma_depth = 0.05` (wider heat stencil) | band relocated, not removed; reverted |
| Betts-Miller at `rhbm = 0.7` | slow runaway, 289.7 -> 300.0 K over 1000 days |
| Old partition pair (`partition_mse` with `partial_condensation`) | recycles condensate; only inactive schemes use it |

Adopted 10 Sep: stopping the plume at neutral buoyancy
(`mf_plume_stop_at_neutral_buoyancy`). See "Why the surface cools with the stop".

## Cleanup, 10 Sep

Removed because nothing used them. Git history keeps every tracked file.

- 27 old checkpoints (54 files) from `notebooks/data`. Only
  `atm407_equilibrium_20level` stays. Sections below still name some of them.
- `scm/configs/bm_conservative_v1.toml` and `bm_conservative_v2.toml`.
  Betts-Miller is ruled out.
- `scripts/diagnose_partition_handoff.py`. `scripts/diagnose_partition_contract.py`
  tests the same pair and the production pair.
- `scripts/inspect_partial_cloud_restart.py` and `scripts/trace_partial_cloud.py`.
  One-offs for removed checkpoints.
- The source-layer and plume-overshoot options, with their tests. They were never
  committed. A copy of the code before removal is in `outputs/removed_2026-09-10/`,
  which is local and not in git.
- This session's four test runs moved to `outputs/atm407_tests/`, also local:
  `kdiff15_5m`, `kdiff40_5m`, `plume_top_5m`, `overshoot_5m`.

- 22 stale development configs, `mf_profile_v4` to `mf_edmf_internal_launch_v26`.
  None ran as built, and two no longer ran at all.
- The legacy mass-flux transport: its plume loop, its downdraft, the
  boundary-layer moisture export, the column energy correction and the separate
  rain evaporation. All were off or unreachable on the flux path.
  `mf_transport_form` now accepts only `'flux'`, and `'legacy'` raises an error.
  Output is bit-identical on 185 checked arrays (three configurations and eight
  coupled steps). `scm/convection_mf.py` went from 635 to 397 lines. The copy
  from before is in `outputs/removed_2026-09-10/`.
- Config keys that fed only the legacy path: 21 lines across `atm407.toml`,
  `default.toml` and four example configs, the 8 matching loader mappings in
  `scm/configuration.py`, and the `--detrain-rh` option of
  `scripts/compare_atm407_resolutions.py`. Loaded parameters differ only by those
  keys, and output stays bit-identical. `downdraft_release_sigma` stays, because
  the flux downdraft uses it.

Full test suite after the cleanup: 147 passed, 1 expected-fail, 0 failures. The two
overshoot tests left with their code.
After the legacy and config-key removal: 148 passed, 1 expected-fail, 0 failures.
One test was added: asking for the legacy transport now raises an error.

## September 10: systematic source audit

Goal: trace every open problem to its source, look for links, then decide which
fixes are macro (one change fixes several problems) and which are micro (one
local defect). Method: per-scheme heating and moistening budgets from one 2-day
run of the promoted column, plus direct checks of each known code defect. No
physics changed during the audit.

### Code defects, re-checked 10 Sep

**Phase-partition recycling: fixed in production. The old test checks the wrong
pair.** [Corrected 10 Sep.] `scripts/diagnose_partition_handoff.py` (removed 10 Sep) alternates
`partition_mse` with `partial_condensation`, and that pair does flip condensate
between 0 and 0.0245 g/kg. But production never calls `partition_mse`. The UW
boundary layer uses `partition_water` at the configured critical humidity, the
same contract as condensation. Tested with the pair production actually uses,
condensate is stable on every cycle with zero water and energy error. New test:
`scripts/diagnose_partition_contract.py`, which fails if the production pair ever
recycles. The older pair is still inconsistent. That is a gap only in the
inactive UW convection and shallow-plume-v2 schemes, which call `partition_mse`
unconditionally. [Rechecked 10 Sep: the script first used RH 0.95, but production
has used 0.90 since commit 22fb0fe (7 Sep). It now reads the value from
`atm407.toml`. At 0.90 the production pair is still stable.]

**Shortwave ozone count-weighting: real in the code, but production does not use it.**
[Corrected 10 Sep.] `atm407.toml` selects `multiband_ozone_profile`, and production
ozone heating matches the profile numbers below exactly (0.347 K/day at 10 hPa,
-0.020 at 997.5 hPa). The measurement that follows called the multiband function
with its default `ozone_profile=False`, which is not the production path. The
count-weighting at `multiband.py:198` only affects configs that select plain
`multiband` or `semi_gray`.
`multiband.py:198` spreads ozone evenly over the 20 layers instead of placing it
in the stratosphere. Ozone shortwave heating, K/day:

| p hPa | now (count-weighted) | correct (profile) |
|---|---|---|
| 10 | 0.187 | 0.347 |
| 75 | 0.075 | 0.623 |
| 615 | 0.051 | -0.002 |
| 997.5 | **0.672** | -0.020 |

The column total is identical (ASR 247.12 both ways), so the radiation
calibration is unaffected. But the lowest 5 hPa layer, right at the ocean, gets
0.67 K/day of spurious heating, and the stratosphere is under-heated by a factor
of 2-8. The fix already exists: the `ozone_profile` path. Micro fix, but it
touches the near-surface layer, so re-check the surface-air temperature gap after
applying it.

**Silent defaults on the active physics path: 30.** Down from 149 across the whole
code, because most belong to schemes the lab does not run. The 30 that matter
include `ri_crit` (0.25), `unstable_diffusion_boost` (4.0), `k_diff_min` (0.05),
`bl_shear_floor` (1.0), `surface_flux_coupling` ('distributed'),
`condensation_scheme` ('large_scale'), `cloud_evaporation_scheme`
('relative_humidity'), `cloud_autoconversion_scheme` ('local'),
`mf_subsidence_drying` (True) and the downdraft settings. `k_diff` and
`k_diff_cap_factor` are now explicit in `atm407.toml`. Micro fix: write the rest
in with their current values. Zero behaviour change, and it stops a future
`bl_diagnose_depth`-style surprise.
**[Done 10 Sep.]** A fuller scan, including `_column_param` accessors, found 26
physics settings; the other 4 of the 30 are runtime flags such as `dtype` and
`profile_diagnostics`. All 26 are now written into `atm407.toml` with their
previous values. Verified: resolved values equal the old code defaults, no other
parameter changed, and two physics steps are bit-identical before and after.
Loader gap found in passing: `[numerics] rad_interval_microphysics_steps` is
never read by the physics. It equals the code default, so it has no effect; the
`[params]` copy is the one that counts.

**Height offset: lowest level placed at 0 m.** `thermo.geopotential` starts its
upward integration at the lowest *full* level and sets it to z = 0. But that level
is at 997.5 hPa, 2.5 hPa above the surface, so hypsometrically it is about 21 m
up. Every height in the column is therefore about 21 m too low. Users:
`boundary_layer.py` (lines 86, 123, 228: the bulk-Richardson depth diagnosis and
the depth factor) plus the inactive EDMF and TKE schemes. Differences between
levels are unaffected, so local Richardson numbers are fine; absolute heights
used for the boundary-layer depth are not. Low impact near a 1500 m depth, but
about 50% at the lowest interface (42 m reported, about 63 m true). Micro fix:
start the integration from the surface pressure.
**[Tried and reverted 10 Sep.]** Changing `thermo.geopotential` to start from the
surface pressure (lowest level at 21.1 m instead of 0) is correct in principle and
barely moves the promoted column. Boundary-layer depth stays at its 1500 m ceiling,
conductance falls 3-8% through the boundary layer, temperature moves at most
0.003 K over two steps, and TOA goes 1.0430 -> 1.0426. But it breaks four tests that
the first, targeted test run did not cover:
`test_components.py::test_energy_budget_diagnostics`,
`test_case_benchmarks.py::test_unified_edmf_bomex_depth_is_resolution_convergent`,
`test_case_benchmarks.py::test_edmf_detrainment_hands_cloud_water_to_the_grid`, and
`test_case_benchmarks.py::test_dry_mixed_layer_conserves_surface_energy`. The energy-budget failure is small but real. With the lowest level above
the ground, the potential energy of the bottom half-layer changes as that air warms
and cools, so the MSE residual and the primary energy residual no longer match
exactly (both -0.0131, differing in the fifth decimal). The old code left that term
out, and the test assumed it was zero. The EDMF tests depend on absolute heights.
Under the project rule of not changing what works without strong evidence, 21 m of
height accuracy with negligible effect on the column does not justify four test
changes. Reverted; `thermo.py` matches HEAD. Revisit together with those tests, or
with the vertical-coordinate change.

**Deep convection heats high but dries low.** From the full budget: deep
convection's heating is centred at 645 hPa, its drying at 944 hPa. This puts
numbers on the professor's mass-flux structure point.

**Deep convection cools the stratosphere.** In equilibrium, radiation heats
10-235 hPa by +0.13 to +0.68 K/day, and deep convection cools it by exactly the
same amount. Convection should not act that high. Ruled out as the cause: the
mass-flux MSE-conservation correction. Moving its reach to sigma 0.3, or turning
it off, leaves the deep tendency at 10-235 hPa unchanged to three decimals, and
the MSE residual is zero.

**Cause, verified:** the plume in the production flux path
(`scm/convective_transport.py`, `updraft()`) does not stop at its level of
neutral buoyancy. Past that level its mass only decays gradually, and it is forced
to zero only at the top model layer. 4.5% of the cloud-base mass flux gets above
305 hPa and 0.08% reaches 35 hPa. That air sits on the plume's adiabat, far colder
than the stratosphere, and mixing it in cools every layer above the cloud top.
Clean test in a scratch copy: stopping the plume where it first loses buoyancy
(all remaining mass detrains there) removes the cooling at 10-305 hPa completely
(-0.595 -> 0.000 K/day at 10 hPa), leaves every level from 460 hPa down unchanged,
and changes rain only from 2.178 to 2.152 mm/day.
**[Implemented 10 Sep, off by default.]** Setting `mf_plume_stop_at_neutral_buoyancy`
in `scm/convective_transport.py` (`updraft()`) and `scm/convection_mf.py`. Verified:
with the setting off, output is bit-identical to before; with it on, it matches the
scratch test exactly (10 hPa -0.595 -> 0.000 K/day, 380 hPa 0.290 -> -0.373, rain
2.178 -> 2.152 mm/day). This is the CESM2 behaviour. Not yet enabled in
`atm407.toml`. The 400-day test passed its criteria but moved the tropopause
down to 305 hPa, so it was not promoted (see "Plume-top fix alone" below).
Regression tests: `test_plume_stops_at_neutral_buoyancy_when_asked` and
`test_mass_flux_scheme_passes_plume_top_switch` in `scm/test_convective_transport.py`.

**Which mass-flux code actually runs.** [10 Sep: only the flux path exists now. The
legacy loop was removed.] `atm407.toml` sets
`mf_transport_form = 'flux'`. In that mode `mass_flux_convection` calls a separate
`updraft()` routine for each plume and skips the older plume loop entirely
(`convection_mf.py` from about line 264). So the 'legacy' loop, including its
compensating-subsidence expression at line 331 (the one described elsewhere in
this file as using temperature from below without the pressure-work term), does
**not** run in production. Earlier entries that diagnose production behaviour
from that loop should be read with this in mind. A patch to the legacy loop was
tested on 10 Sep and changed nothing, which is how this was found.

**The production convection closure is resolution-dependent.** Found by switching
the mass-flux code default to 'flux'.
`scm/test_vertical_resolution.py::test_mass_flux_cape_response_converges_across_teaching_grids`
builds its column from `default_params()`, so it had been testing the unused legacy
path. On the production flux path the CAPE response per unit mass flux differs by
about 61% across 10, 20 and 40 levels, against the test's 15% limit. The legacy
path passes. The test is now marked expected-to-fail (strict), with this reason,
so the suite stays green and it will flag when fixed. This matters most for the
long-term goal of a high-resolution column: a closure whose strength depends on
the grid gives a different climate as resolution changes. Source not yet traced.
The test uses a default initial sounding and default settings, not the ATM407
equilibrium.

**Not linked to the plume-top problem (measured 10 Sep).** Stopping the plume at its
natural top leaves the spread unchanged at 60.6%. Per grid (10 / 20 / 40 levels):
CAPE 122.8 / 146.9 / 150.1 J/kg, CAPE response per unit mass flux 821 / 1115 / 603.
The response is non-monotonic in resolution, with 20 levels the highest. That
points to a discrete grid effect, not slow convergence. The mass-flux limit is
identical on all three grids. Candidates to measure first: which levels fall inside
the source layer (`mf_source_top_sigma = 0.90`), the CAPE integration step
(`cape_max_pressure_step`) against the level spacing, and the source-layer mass
used by `mf_available_mass_fraction`. Fixing it will change the 20-level convection
strength, so the lab reference would need a new equilibrium.

**Where the parcel starts (10 Sep).** `dilute_cape` launches its parcel from the
single lowest model level (`t[:, -1]`, `q[:, -1]`), and the flux-path `updraft()`
does the same. That level's thickness depends on the grid: 20 / 5 / 2.5 hPa at
10 / 20 / 40 levels. The trial plume's drying of it jumps around between grids:
-0.061 / -0.075 / -0.038 g/kg. This was the leading hypothesis for the
non-monotonic response. It is refuted below. Ruled out as a fix: launching
the parcel from the mean of the lowest 100 hPa. On the test sounding CAPE
collapses to 8-21 J/kg and convection switches off on every grid, so the spread
gets worse, not better.

**Parcel launch refuted as the cause (10 Sep).** Launching the CAPE parcel from the
mean of the lowest 20 hPa, which is exactly the same air on 10, 20 and 40 levels
(each grid has a level boundary at 980 hPa), does not remove the spread: 49% on
the test sounding and 93% on the ATM407 equilibrium, against 61% and 48% shipped.

**The dependence is real on the production column.** On the ATM407 equilibrium
interpolated to 10/20/40 levels, CAPE itself agrees to 6% (669 / 628 / 630 J/kg).
But the CAPE response per unit mass flux is 1951 / 2351 / 3143, a 48% spread that
grows with resolution. So the problem is not CAPE. It is how much the trial plume's
tendencies change CAPE.

**Where the response comes from (measured).** Splitting the trial plume's CAPE change
into the part from the parcel's starting level and the part from the environment
above. The two add up to the full response within 2%.

| | 10 levels | 20 levels | 40 levels | spread |
|---|---|---|---|---|
| equilibrium, starting level only | 1555 | 2077 | 2972 | 64% |
| equilibrium, environment only | 436 | 266 | 181 | 87% |
| test sounding, starting level only | 752 | 1024 | 519 | 66% |
| test sounding, environment only | 68 | 94 | 84 | 32% |

80-95% of the response comes from the trial plume changing the single level the
parcel starts from. On the real column that part grows steadily as the lowest layer
thins (20, 5, 2.5 hPa). This fits the code: the plume draws all of its source air
from the lowest layer (`exchange[:, -1] = -current` in `updraft()`), so a fixed
trial mass flux changes a thinner layer more, and the parcel starts from that same
layer. Averaging only the parcel's start did not help, because the plume's source
stayed in one layer. **Refuted (10 Sep):** drawing the plume's source air from the lowest 20 hPa in
proportion to mass, with or without starting the parcel from the same layer, makes
the spread worse on both soundings. Equilibrium: 48% shipped, 58% source only, 75%
with the parcel too. Test sounding: 61%, 162%, 78%. The 10-level grid is unchanged
by that patch, because its lowest layer already is 20 hPa. The 20- and 40-level
responses fall below it, so the dependence flips direction instead of
disappearing. That is two mechanisms refuted in a row, both built from a plausible
reading of partial measurements. Next: a straight refinement test (10 to 160
levels) to see whether the response converges at all, before proposing another
cause. On the real column, convective rain also differs 122% across the three
grids.

**Refinement test (10 Sep): the response does not converge. It diverges.** ATM407
equilibrium interpolated to finer grids, shipped code, one call from the same
state:

| levels | lowest layer hPa | CAPE | response | starting-level part | cloud-base mass flux | rain mm/day |
|---|---|---|---|---|---|---|
| 10 | 20.00 | 669 | 1951 | 1555 | 0.0130 | 4.39 |
| 20 | 5.00 | 628 | 2351 | 2077 | 0.0100 | 2.18 |
| 40 | 2.50 | 630 | 3143 | 2972 | 0.0075 | 1.22 |
| 80 | 1.25 | 622 | 6409 | 6260 | 0.0037 | 0.54 |
| 160 | 0.62 | 637 | 12445 | 12319 | 0.0019 | 0.27 |

From 40 levels on, the response doubles each time the lowest layer halves: it
scales as one over that layer's thickness. CAPE stays at 620-670. So the closure
divides CAPE by a number that grows without limit as the grid is refined. From the
same state, the mass flux and the rain halve with each refinement; a coupled run
would have to build far more CAPE on a fine grid to rain the same amount. **The
closure is resolution-dependent by construction, not through a coarse-grid error.**
Almost all of the response comes from the trial plume changing the single lowest
level, and that level's mass shrinks toward zero with refinement. This blocks the
high-resolution goal outright. The lab's 20-level equilibrium is a property of
that particular grid.

**A formulation that converges (10 Sep, scratch test).** Starting the CAPE parcel
from a fixed-depth layer (the lowest 20 hPa, mass-weighted mean potential
temperature and humidity) *and* drawing the plume's source air from that same
layer in proportion to mass:

| levels | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|
| response, shipped | 1951 | 2351 | 3143 | 6409 | 12445 |
| response, plume source 20 hPa only | 1951 | 2445 | 3487 | 7003 | 12977 |
| response, plume source + parcel 20 hPa | 1951 | 1123 | 943 | 995 | 995 |
| cloud-base mass flux, both | 0.0130 | 0.0211 | 0.0254 | 0.0240 | 0.0247 |

With both changes the response settles from 40 levels on (within 5%), 20 levels is
within about 15% of the converged value, and 10 levels is too coarse to resolve a
20 hPa layer against its 50 hPa neighbour. Changing only the plume's source does
nothing, so both halves are needed. CAPE itself is unchanged.

The scratch patch is **not yet a correct implementation**: from the same state its
rain per unit mass flux halves with each refinement (171, 74, 35, 17), while the
shipped code's converges near 140. It does not leak water: vapour removed equals rain exactly, residual zero, in both
the shipped code and the patch, on 20, 40 and 80 levels. **Answer: the tendency limiter binds.** In the patch the lowest level hits the
-20 K/day cooling cap on every grid (one capped level out of 20, 40 and 80). In the
flux path the limiter scales the whole scheme by that single worst level, rain
included, and the patch's uncapped cooling there grows with refinement. So the
patch carries a transport inconsistency at the bottom of its source layer. The
shipped code's lowest-level cooling falls with refinement (-14, -8.8, -6.9 K/day)
and never caps.

**Status of the trace: cause identified, fix not implemented.** The closure's CAPE
parcel and the plume's source are both tied to the single lowest model level, and
that level's mass shrinks toward zero as the grid is refined. Evidence: the
response diverges as one over that layer's thickness; 80-95% of it comes through
that level; and moving both the parcel and the source to a fixed-depth layer makes
the response converge (40, 80 and 160 levels within 5%). Still to do: a correct
implementation. The scratch patch's transport at the bottom of the source layer
must be rewritten so the whole layer is drawn down evenly. The source depth is a
design choice: 20 hPa works here; the 100 hPa mixed layer tried earlier removed
nearly all CAPE on the test sounding. At 20 levels the shipped response is about
2.4 times the value the 20 hPa version converges to. So for a given CAPE the lab
closure gives about 2.4 times too little mass flux, and the lab equilibrium makes
up for it by carrying more CAPE.

**Correction, later on 10 Sep: the divergence was partly an artifact.** The finer-grid
soundings above were built by interpolating the 20-level equilibrium, which leaves
the lowest levels of the 40-, 80- and 160-level columns clamped to the 997.5 hPa
values. On the unit test's own smooth sounding, built natively on each grid, the
shipped closure converges: response 603 / 610 / 614 on 40 / 80 / 160 levels (2%
spread). What holds on both soundings is that 10 and 20 levels are far from
converged: 821 and 1115 against about 610. So at the lab's 20 levels the closure is
roughly 1.8 times its high-resolution sensitivity on that sounding. The closure is
**under-resolved at 20 levels, not ill-posed**. The earlier claims that it is
"resolution-dependent by construction" and "blocks the high-resolution goal
outright" were too strong.

The boundary-layer source layer also fails on the smooth sounding. With a 500 m
layer CAPE collapses to 13-31 J/kg and the response is erratic (29% spread); with
1000 m convection switches off entirely. Its good convergence was on the
interpolated soundings only, so it is **not adopted**. The code was removed in the 10 Sep cleanup. It was off by
default (`mf_source_layer = "lowest_level"`), with the verification recorded
above. A decisive resolution test needs native equilibria at 20, 40 and 80 levels
rather than interpolated soundings, or the CESM2 hybrid: parcel from the launch
level, with its change spread over the sub-cloud layer (`dsubcld`). The plume-top
fix is independent of all this and stands.

### Surface-air temperature gap: traced to the surface energy budget

Promoted column, mean of steps 5-8 after restart:

| quantity | model | observed tropical ocean |
|---|---|---|
| ocean minus lowest air | 4.15 K | about 1 K |
| sensible heat | 29.9 W/m2 | 10-15 |
| latent heat | 99.0 W/m2 | about 100 |
| Bowen ratio | 0.30 | about 0.1 |
| sunlight absorbed at surface | 212.4 W/m2 | about 170 with clouds |

The surface budget closes (+0.9 W/m2). Latent heat is right; sensible heat is 2-3
times too big, and it is sensible heat that sets the gap: with `C_H = 1.2e-3` and
5 m/s wind, 30 W/m2 needs a 4.15 K gap.

The likely link: clouds are radiatively off, so the surface absorbs the full
clear-sky 212 W/m2 of sunlight and must get rid of it. Evaporation is limited by
the humidity deficit (5.47 g/kg), so sensible heat carries the remainder. **This is
a hypothesis linking the gap to the cloud problem, not yet tested.** A second
contributor is the 5 m/s prescribed wind, low for the trades (about 7 m/s).


### How CESM2 and GFDL handle the same choices, 10 Sep

Checked before committing to the convection fixes. Sources: CAM's
[`zm_conv.F90`](https://docs.cesm.ucar.edu/models/cesm2/config/old/cesmBbrowser/html_code/cam/zm_conv.F90.html)
(CESM2 release), the [E3SM ZM technical guide](https://docs.e3sm.org/E3SM/EAM/tech-guide/zm/),
the [GFDL AM4 page](https://www.gfdl.noaa.gov/am4/), and
[Chu et al. (2022, J. Climate)](https://journals.ametsoc.org/view/journals/clim/35/2/JCLI-D-21-0267.1.xml),
which describes the UW, GFDL double-plume and Zhang-McFarlane schemes side by side.

| choice | CESM2 (ZM deep) | GFDL AM4 (double plume) | this column |
|---|---|---|---|
| where the parcel starts | level of maximum moist static energy inside the PBL (one level) | one layer's properties (surface layer or maximum-MSE layer), leaving from the PBL top as in the UW scheme | the lowest model level |
| closure | Mb = (CAPE - CAPE0) / (tau F), tau = 1 h, CAPE0 = 70 J/kg | deep: cloud work function relaxed to 10 J/kg over 8 h, only if mean RH > 0.4; shallow: convective inhibition and TKE | same form as ZM, tau = 6 h, threshold 50.0 J/kg |
| where the plume stops | level of neutral buoyancy; mass flux set to zero above | vertical-velocity equation; overshoots into the stable layer until the velocity reaches zero | decays slowly to the model top (the stratospheric-cooling bug) |
| layer below cloud | carries `dsubcld`, the pressure depth from the launch level to the surface, into the closure | - | not used |
| ozone | prescribed, full 3-D field from WACCM6 | prescribed | already a stratospheric profile (`multiband_ozone_profile`) |
| model top | [32 levels, 3.6 hPa](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019MS001916) | [33 levels, 1 hPa](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017MS001208) | `p_top = 0` |

What this teaches:

1. **Neither model launches from "the lowest model level".** ZM picks a physically
   chosen level; GFDL uses boundary-layer properties. Chu et al. go further and
   average the source air over the whole PBL, for two stated reasons: to avoid false
   convection when the surface layer heats in the morning, and to include turbulent
   mixing. That is the published version of our fixed-depth fix: tie the source to
   a physical layer, not to the grid. It suggests using the diagnosed PBL, or the
   layer below cloud, instead of an arbitrary 20 hPa. Not yet tested here. A 100 hPa
   mean removed nearly all CAPE on the unit-test sounding, so it must be checked on
   the real column before adopting it.
2. **Two accepted ways to end the plume.** Stop at neutral buoyancy (ZM), or let it
   overshoot under a vertical-velocity equation (UW, GFDL; Chu et al. call this
   more consistent with observations). Our slow decay to the model top matches
   neither. Stopping at neutral buoyancy is the simple, CESM2-consistent choice and
   is already verified in a scratch test.
3. **Both prescribe ozone with a real vertical profile, and both have a model top
   near 1-4 hPa.** Confirms the ozone-profile fix and the `p_top` item.
4. **Our CAPE timescale is 6 h, against 1 h in CESM2 and 8 h in GFDL.** Worth
   knowing before any retuning.
5. **Verified from the CESM2 source** (`zm_conv.F90`, `cam_cesm2_1_rel`, subroutine
   `closure`): the launch air's change is the cloud-base flux divergence divided by
   `dsubcld`, the whole sub-cloud layer from the launch level to the surface, never
   one grid level:
   `dtbdt(i) = (1._r8/dsubcld(i))* (mu(i,mx(i))*(shat(i,mx(i))-su(i,mx(i)))+ ...`.
   The closure is `mb(i) = max(dltaa/tau/dadt(i),0._r8)` with
   `dltaa = -(cape - capelmt)` and `dadt` the CAPE change per unit mass flux. That is
   why ZM does not have our resolution problem.

### Source layer for the convection: tested options, 10 Sep

The lowest levels are merged into one level of the chosen depth (rounded to the
nearest whole level; exact splitting left thin sliver levels that made rain jump
by factors of 3-5), the unchanged scheme runs on that column, and the merged
tendency is spread evenly over the layer. ATM407 equilibrium on 10-160 levels:

| source layer | response spread, 40-160 levels | rain spread, 40-160 | CAPE |
|---|---|---|---|
| lowest level (shipped) | diverges (x2 per refinement) | diverges | ~630 |
| 20 hPa | 3% | 36% | ~630 |
| 50 hPa | 3% | 46% | ~610 |
| 100 hPa | 12% | 20% | ~550 |
| **boundary layer (~160 hPa)** | **7%** | **2%** | ~430 |

Only the boundary-layer source converges in both the closure response and the
rain, so it is the choice. **[Withdrawn later on 10 Sep: these results are from interpolated
soundings; on a smooth native sounding the boundary-layer source removes nearly
all CAPE. See the correction under the resolution trace.]** It matches CESM2's sub-cloud-layer treatment and the PBL
mean of Chu et al. (2022). CAPE for a given state falls (~630 -> ~430) because the
parcel now starts from boundary-layer-mean air; the equilibrium will re-adjust.
Implemented as `mf_source_layer = "boundary_layer"` (default `"lowest_level"`,
i.e. unchanged).
Verified: with the default, output is bit-identical to before; with the setting on,
it reproduces the scratch test exactly on 20, 40 and 80 levels (CAPE 441/432/431,
response 852/711/707, rain 5.38/6.16/6.21 mm/day), and the water budget closes to
about 1e-11 kg/m2/s. It needs `state['boundary_layer_depth_m']`, which the
boundary-layer scheme sets before convection runs; it raises an error rather than
fall back silently if that is missing. Code: `_source_layer_depth`,
`_merge_source_layer` and the `mass_flux_convection` wrapper in `scm/convection_mf.py`;
the original scheme is `_mass_flux_convection_core`, unchanged. [All removed in the
10 Sep cleanup; `mass_flux_convection` is the original scheme again.]
Full test suite with both new settings in the code (off by default): 145 passed, 1
expected-fail, 0 failures.

### Combined convection fix: equilibrium test, 10 Sep

Both CESM2-style changes together, `mf_plume_stop_at_neutral_buoyancy = true` and
`mf_source_layer = "boundary_layer"`, on top of `atm407.toml`. 400 days on a 5 m
slab from the promoted reference (label `combined_conv_5m`). Criteria fixed before
the run; keep only if all hold at the end:

1. deep-convection heating above the plume top about zero
2. theta_v drop from 998 to 865 hPa below 1.0 K
3. |TOA| <= 1.1 W/m2 (the accepted baseline is 1.06)
4. |surface| <= 1.0 W/m2
5. drift <= 0.005 K/day
6. CAPE >= 300 J/kg (the new parcel lowers CAPE by design)
7. deep rain > large-scale rain
8. mass at RH >= 95% <= 0.15

Smoke test before launch, 4 coupled steps: all fields finite, deep heating above
305 hPa exactly zero, CAPE 417 J/kg, TOA +1.18 W/m2. **Stopped before finishing**,
once the source layer was withdrawn. Replaced by the plume-top test below.

### Plume-top fix alone: 400-day test, 10 Sep. Not promoted.

[Later on 10 Sep: promoted, after the surface cooling was traced. See "Promoting the
stop".]

Setup: `atm407.toml` plus `mf_plume_stop_at_neutral_buoyancy = true`, nothing
else changed. 400 days on a 5 m slab from the promoted reference. Checkpoint:
`outputs/atm407_tests/atm407_equilibrium_20level_plume_top_5m` (local, not in git). Same criteria as the
combined test, except CAPE must stay within 450-850 J/kg.

All 8 criteria pass:

| criterion | result |
|---|---|
| 1 deep heating above the plume top | exactly 0 above 380 hPa |
| 2 theta_v drop 998-865 hPa < 1.0 K | 0.45 K |
| 3 abs(TOA) <= 1.1 W/m2 | +0.08 |
| 4 abs(surface) <= 1.0 W/m2 | -0.51 |
| 5 drift <= 0.005 K/day | 0.002 |
| 6 CAPE 450-850 J/kg | 591 |
| 7 deep rain > large-scale rain | 2.02 vs 1.14 mm/day |
| 8 mass at RH >= 95% <= 0.15 | 0.09 |

It was not promoted anyway, because of a side effect the criteria did not cover.
The upper troposphere got colder and the tropopause moved down.

| hPa | 10 | 35 | 75 | 125 | 175 | 235 | 305 | 380 |
|---|---|---|---|---|---|---|---|---|
| T before (K) | 193.9 | 224.3 | 227.8 | 221.4 | 212.3 | 214.0 | 222.0 | 232.5 |
| T after (K) | 222.8 | 230.2 | 233.1 | 226.7 | 218.3 | 219.1 | 217.0 | 228.7 |
| US Standard Atmosphere (K) | 227.7 | 219.5 | 216.7 | 216.7 | 216.7 | 218.2 | 229.3 | 239.1 |

- The 10 hPa error drops from 34 K to 5 K. That was the target.
- The cold point moves from about 175 hPa to 305 hPa. 305-380 hPa cools 4-5 K.
- 35-125 hPa is now 10-16 K warmer than the standard atmosphere (5-11 K before).
  Cause not traced.
- Below 400 hPa the column is about 2 K colder. It follows the surface, which
  goes from 291.0 to 289.5 K.
- The run has not fully settled. The 50-day drift is 0.10 K against the 0.05 K
  gate in `check_equilibrium`.

Why the upper troposphere cooled (measured). The plume's real top is 380 hPa. It
is buoyant from 945 to 460 hPa and negatively buoyant at every level from 380 hPa
up (-0.75 K at 380, -3.9 K at 305, -178 K at 10 hPa, in the old state). There is
only one crossing, so this is not a thin stable layer. Before the fix, this
non-buoyant plume kept rising with decaying mass. That caused the cooling at
10-235 hPa. Removing it warmed 175-235 hPa by 5-6 K. Part of the old ~200 hPa
tropopause therefore came from that spurious cooling. With the stop, nothing
heats 305 hPa and above. [Later on 10 Sep: this is only part of it. Most of the
cooling follows colder surface air. See the overshoot section below.]

So above 400 hPa the old profile looked closer to the standard atmosphere partly
because errors cancelled. The fix removes one error and exposes others: the
plume top is low, and the lower stratosphere runs warm.

How CESM2 picks the top (`zm_conv.F90`, `buoyan_dilute`). It records up to five
levels where buoyancy goes from positive below to non-positive above. It keeps
the one with the largest CAPE. Ours stops at the first. Here there is only one
crossing, so both rules give 380 hPa. The CESM2 rule is worth copying later for
soundings with a stable layer inside the cloud.

Possible next step: let the plume overshoot its neutral level by a limited
distance, as GFDL AM4 does with a vertical-velocity equation. Then repeat this
test. The switch stays in the code, off by default.

### Plume overshoot: 400-day test, 10 Sep. Not promoted.

Change: the plume keeps rising past its neutral level until its kinetic energy
runs out. It then detrains all remaining mass in that level. This is the GFDL
AM4 and UW approach (Bretherton et al. 2004): 1/2 dw2/dz = a B - b e w2, where e
is the entrainment rate. Work below the first buoyant level is ignored, because
the closure decides whether the plume starts. Settings: `mf_plume_overshoot =
true` and `mf_overshoot_drag` (b, default 2). Off by default and bit-identical
when off. Code: `updraft()` in `scm/convective_transport.py`. Regression tests:
`test_plume_overshoots_neutral_level_then_stops_below_the_top` and
`test_mass_flux_scheme_overshoot_reaches_higher_than_neutral_stop`. Full test suite
with this code: 149 passed, 1 expected-fail, 0 failures. [Code and tests removed
in the 10 Sep cleanup. A copy is in `outputs/removed_2026-09-10/`.]

Constants. CAM's `uwshcu.F90` sets `rbuoy = 1.0` (this is a) and `rdrag = 1.0`.
Both verified in the source. That the code turns this into b = 1 + rdrag = 2 is
from memory, not verified. It does not matter here: b = 1 and b = 2 give the
same plume top.

Smoke test on the reference state, one call, deep-convection heating in K/day:

| hPa | production | neutral stop | overshoot |
|---|---|---|---|
| 10-175 | -0.60 to -0.13 | 0 | 0 |
| 235 | -0.120 | 0 | -0.651 |
| 305 | -0.018 | 0 | -0.018 |
| 380 | +0.290 | -0.373 | +0.290 |

The plume top is 235 hPa. Heating from 305 hPa down is the same as production.
What production spread over 10-235 hPa now all lands in the 235 hPa layer. Rain
and cloud-base mass flux are unchanged. Four coupled steps stay finite.

Criteria, fixed before the run. The first eight are the plume-top test's.

1. deep-convection heating at 10 and 35 hPa exactly zero
2. theta_v drop from 998 to 865 hPa below 1.0 K
3. abs(TOA) <= 1.1 W/m2
4. abs(surface) <= 1.0 W/m2
5. drift <= 0.005 K/day
6. CAPE 450-850 J/kg
7. deep rain > large-scale rain
8. mass at RH >= 95% <= 0.15
9. T at 10 hPa >= 217.7 K, within 10 K of the US Standard Atmosphere. The
   reference has 193.9 K.
10. T at 305 hPa >= 221.0 K and at 380 hPa >= 231.5 K, at most 1 K colder than
    the reference.
11. The coldest level between 50 and 500 hPa is at 235 hPa or higher.

Decision rule: turn it on in `atm407.toml` only if all 11 pass. Replace the lab
reference only once the run also passes the model's own 0.05 K drift gate,
continuing the run if needed. Label `overshoot_5m`.

**Result.** Criteria 1-9 pass. Criterion 10 fails. Criterion 11 passes: the
coldest level is the 235 hPa level itself. The scoring script first marked 11 as
failed, because in floating point that level sits a hair above 235.000 hPa. That
is fixed. With 10 failing, the overshoot is not promoted.

| T (K) at hPa | 10 | 235 | 305 | 380 | lowest level |
|---|---|---|---|---|---|
| reference | 193.9 | 214.0 | 222.0 | 232.5 | 286.9 |
| neutral stop | 222.8 | 219.1 | 217.0 | 228.7 | 284.9 |
| overshoot | 222.8 | 205.4 | 218.1 | 229.9 | 285.4 |
| US Standard Atmosphere | 227.7 | 218.2 | 229.3 | 239.1 | 287.3 |

Other results: TOA +0.11 and surface -0.44 W/m2, CAPE 591 J/kg, surface
temperature 289.9 K (reference 291.0). The stratosphere is the same as with the
neutral stop. The new problem is the 235 hPa layer. It is 205.4 K, which is 8.6 K
colder than the reference and 12.8 K colder than the standard atmosphere. That is
where the overshooting plume dumps its cold air. So the overshoot is worse than
the neutral stop. Do not use it as implemented.

**Why the upper troposphere cools in both runs (measured).** Both plume-top
changes make the air at the lowest level colder and drier: 286.9 -> 284.9 K and
285.4 K, and q 7.35 -> 6.46 and 6.67 g/kg. That air feeds the plume, so the plume
itself is colder at 305/380 hPa: by 3.1/3.2 K with the stop and 2.3/2.4 K with the
overshoot. The environment there cools by 5.0/3.8 K and 3.8/2.6 K. So most of the
upper-troposphere cooling follows the colder source air. The plume-top rule
itself adds only 0.2-1.9 K. Why the surface air gets colder is not traced. Both
runs also bring TOA from +1.06 to about +0.1 W/m2.

What this means. Criterion 10 was meant to catch the plume-top rule cooling the
upper troposphere. It mostly measured the colder surface air instead. The neutral
stop was rejected on that basis and on its low cold point. So whether to turn the
neutral stop on is an open decision for the maintainer. These runs do not settle
it.

The lever for the low tropopause is a plume that reaches higher. The code already
has a plume spectrum (`mf_plume_count`, `mf_plume_entrainment_spread`), like ZM's
range of entrainment rates. Weakly entraining members would reach higher. [Tested
11 Sep: they do not. See "Low plume top".]

### Why the surface cools with the stop: trace, 10 Sep

Hypothesis: the production plume carries heat from the stratosphere down into the
troposphere. The stop turns that off. The stratosphere then warms, and the column
below must cool until the energy balance closes again.

**Step 1, measured.** Deep-convection heating in W/m2, one call on each saved
equilibrium:

| state | above 270 hPa | below 270 hPa | latent heat released |
|---|---|---|---|
| reference, production plume | -5.39 | +68.42 | 63.0 |
| stop equilibrium, stop on | 0.00 | +53.54 | 53.5 |
| stop equilibrium, production plume | -16.07 | +70.56 | 54.5 |

In the reference the plume takes 5.4 W/m2 out of the layers above 270 hPa and adds
it below. The heating below exceeds the latent heat released by exactly that
amount. With the stop, the heating below equals the latent heat. On the warmer
stop state the production plume would take out 16 W/m2: the warmer the
stratosphere is relative to the plume, the harder the plume pumps.

**Step 2, measured.** Radiation split on the two saved states. Each part of the
stop state was swapped into the reference one at a time. The four parts add up to
the total within about 0.1 W/m2.

- Above 270 hPa, radiation heats the reference by +6.08 W/m2 and the stop state by
  +0.01. Once the plume stops cooling the stratosphere, it sits in radiative
  equilibrium.
- The radiation entering the troposphere and surface through 270 hPa goes from
  -5.04 to 0.00 W/m2. That +5.04 replaces the 5.4 W/m2 the plume used to deliver.

| part swapped in | into troposphere at 270 hPa | TOA | surface radiation |
|---|---|---|---|
| warmer air above 270 hPa | +2.68 | -2.83 | +0.02 |
| colder air below 270 hPa | +8.05 | +6.58 | -9.08 |
| colder surface | +1.03 | +1.02 | +8.85 |
| less water vapour | -6.68 | -5.77 | -2.61 |
| all four together | +5.04 | -1.03 | -2.70 |

All in W/m2. The warmer stratosphere radiates the pumped heat about half to space
(OLR +2.83) and half back down (+2.68). The troposphere must make up the rest,
plus what the drier air lets escape, so it cools. The surface follows: the colder,
drier air above sends it 11.7 W/m2 less radiation, and it cools until its own
emission drops by 8.85 W/m2. Drier air also absorbs less sunlight (ASR -1.29).

**Verdict: traced.** The surface cools because the production plume was carrying
5.4 W/m2 of heat from the stratosphere into the troposphere, and the stop removes
that. The colder surface and troposphere are the column's honest response, not a
flaw in the stop. The old 291.0 K surface was partly held up by that spurious
heat pump. The low plume top (380 hPa) is a separate problem.

### Promoting the stop, 10 Sep. Promoted.

`atm407.toml` now sets `mf_plume_stop_at_neutral_buoyancy = true`, label
`atm407_plume_stop`. The 400-day stop run had not settled (50-day drift 0.10 K
against the model's 0.05 K gate), so it continues 400 more days from its end state
(label `plume_stop_5m`). Checks fixed before the run: the lab reference is replaced
only if the continuation passes the model's own equilibrium check and still passes
criteria 1-8 of the plume-top test.

**Result.** Both checks pass. After 800 days: TOA +0.46 W/m2, surface -0.23 W/m2, 50-day drift 0.035 K, surface temperature 289.03 K, CAPE 566 J/kg, theta_v drop 0.47 K. It is now the lab reference. The old k_diff = 40 reference is in git history and in `outputs/atm407_tests/`. The overshoot test's criteria 10 and 11 (upper troposphere and cold point
against the old reference) fail, as expected. They were not promotion checks; the stop's
upper-troposphere cooling is traced above.

Notebook 1's CO2 demonstration still shows opposite signs with the new reference.
At 800 against 400 ppm, the lower troposphere cools 0.108 K/day less and the
stratosphere 0.139 K/day more. A headless run at the default slider (400 ppm) prints
zeros, because it compares 400 with 400.

Both notebooks run headless against the new reference with 0 failures. Full test suite after the promotion: 148 passed, 1 expected-fail, 0 failures.

### Low plume top: multi-plume tested, 11 Sep. Not the fix.

With the stop in production, the plume ends at 380 hPa and the coldest level is
305 hPa.

**Multi-plume, one call on the reference.** Members use the entrainment rate
scaled from 1/spread to spread.

| plumes x spread | plume top | rain mm/day |
|---|---|---|
| 1 x 3 (production) | 380 hPa | 1.94 |
| 3 x 3 | 305 hPa | 1.93 |
| 5 x 3 | 305 hPa | 1.89 |
| 3 x 10 | 305 hPa | 2.25 |
| 5 x 10 | 305 hPa | 2.13 |

Every setting lifts the top one level at most. Each one cools 305 hPa by 0.6-0.8
K/day, where its least-mixing member ends. 3 x 10 also puts -2.1 K/day at 615 hPa
next to +6.0 K/day at 685 hPa. Not run to equilibrium.

**Why: the source air limits the top (measured).** Parcels from the lowest level,
which is 4.8 K colder than the surface:

| source air | last buoyant level | buoyancy at 305 / 235 hPa |
|---|---|---|
| production plume, as is | 460 hPa | -3.7 / -20.4 K |
| no mixing, as is | 380 hPa | -2.5 / -19.9 K |
| no mixing, gap half closed | 305 hPa | +2.4 / -14.9 K |
| no mixing, gap closed | 305 hPa | +7.7 / -9.5 K |
| production plume, gap closed | 305 hPa | +3.6 / -13.3 K |

"Gap closed" means the lowest level warmed to the surface temperature at the same
relative humidity. Even air that mixes with nothing stops at 380 hPa, so mixing is
not the limit. The cool, dry source air is. That ties the low top to the
surface-air gap (problem 2). These are what-ifs on the source air alone; in
equilibrium the whole troposphere would change too.

Nothing reaches 235 hPa even with the gap closed. Whether a convective top near
300 hPa is wrong for a column with a 289 K surface is not settled. The standard
atmosphere may be the wrong yardstick for a radiative-convective column.

Next: trace the surface-air gap. One lead fits this result. On the previous
reference, deep convection cooled the 5 hPa lowest layer by 14 K/day, and that
layer is the plume's source air.

### Surface-air gap: trace, 11 Sep. Partly traced.

The lowest model level (998 hPa) is 4.81 K colder than the surface.

**Step 1, measured.** The bulk formula gives sensible heat = rho cp C_H U (Ts - Ta),
with a fixed wind U = 5 m/s. One-day means from the reference: sensible heat 34.5
W/m2, latent heat 91.9 W/m2. So rho cp C_H U = 34.5 / 4.81 = 7.2 W/m2/K, which is
C_H = 1.2e-3, a normal ocean value. The gap is large because the sensible heat
flux is large, not because the exchange is weak. 5 m/s is at the low end of ocean
winds, but even at 7 m/s the gap would be 3.4 K.

Surface budget in W/m2, model against the Trenberth, Fasullo and Kiehl (2009)
global mean:

| term | model | observed |
|---|---|---|
| sunlight absorbed | 214.6 | 161 |
| longwave down | 306.5 | 333 |
| longwave up | 395.5 | 396 |
| net radiation | 125.6 | 98 |
| latent heat | 91.9 | 80 |
| sensible heat | 34.5 | 17 |

With cloud radiation off, the surface gets about 54 W/m2 more sunlight and 27 W/m2
less longwave from above. Its net radiation is about 28 W/m2 higher. Sensible heat
carries 17.5 of the extra and latent heat 12.

**Step 2, measured.** Per-scheme heating, 2-day means from the reference, W/m2:

| scheme | 998 hPa (5 hPa thick) | 988 hPa | all below 865 hPa |
|---|---|---|---|
| surface flux | +8.5 | +25.5 | +34.1 |
| boundary-layer mixing | -1.1 | -22.1 | +1.0 |
| deep convection | -7.3 | -1.5 | -15.9 |
| shallow convection | -0.9 | -2.7 | -17.5 |
| radiation | +0.8 | +0.8 | -11.2 |
| condensation | 0 | 0 | +10.6 |

Below 865 hPa, shallow and deep convection carry the surface heat out (17.5 and
15.9), radiation cools by 11.2, and condensation in the cloud deck returns 10.6.
At the lowest level, deep convection removes 7.3 of the 8.5 W/m2 the surface puts
in.

The gap is not a thin-layer effect. Potential temperature is nearly uniform through
the mixed layer (284.4 K at 998 hPa, 284.5 K at 988 hPa, 284.4 K at 970 hPa, 284.3 K at 945 hPa, 284.3 K at 910 hPa, 284.3 K at 865 hPa), and the surface is 289.0 K. The whole layer sits
4.5-4.8 K below the surface. So the lead "deep convection cools the thin lowest
layer" is ruled out as the cause of the gap.

**Verdict: partly traced.** The gap comes from a large sensible heat flux through a
normal exchange coefficient. The flux is large for two reasons. The surface gets
about 28 W/m2 more net radiation than the observed Earth, because cloud radiation
is off. And sensible heat takes a large share of the turbulent flux: Bowen ratio
0.37, against about 0.2 in the observed global mean. How much of the gap realistic
cloud shading would remove is not tested. That makes the gap part of problem 1,
the cloud deck.

### Cloud deck: design options, 11 Sep. Decision needed.

**The problem.** The boundary layer carries about 3 g/kg/day of water up to its top
(865-910 hPa). Deep convection removes about half. The rest can only condense, so
the top saturates into a deck with 89% cover and about 245 g/m2 of cloud water.
Cloud radiation is off, so the deck is invisible. Switched on, it would reflect
about 160 W/m2, over three times the observed global cloud effect. The surface-air
gap and the low plume top both trace back to this.

**What a real atmosphere does.** Shallow cumulus carries moisture up and out of the
boundary-layer top. Entrainment at the inversion mixes in drier, warmer air from
above, which thins decks. CESM2 does both in one scheme (CLUBB). GFDL AM4 uses a
separate shallow plume (Bretherton et al. 2004) next to its boundary-layer scheme.

| option | what it is | cost | evidence so far |
|---|---|---|---|
| 1. Leave it | keep cloud radiation off and document the deck | none | fine for the lab, not for goal 1 |
| 2. Clouds on as is | turn cloud radiation on, re-derive the albedo | one equilibrium | the deck reflects about 160 W/m2, so the surface would need to be made unrealistically dark to compensate |
| 3. Shallow plume in the current boundary layer | `uw_shallow` with Richardson mixing, the GFDL-style route | moderate | smoke test below: the plume does not rise |
| 4. Finish the UW suite | UW turbulence plus UW shallow, with moist cloud-top entrainment | substantial | the partial version took the band off saturation but choked deep convection, and it fails BOMEX |
| 5. Assumed-PDF clouds, then reduced CLUBB | cloud cover and water from moisture variance, the CESM2 route | largest | not started |

**Option 3 smoke test, measured.** One call on the reference with
`shallow_convection_scheme = "uw_shallow"`. It runs, but its plume never rises
(plume top 0 m, cloud base 0 m). Richardson mixing provides no turbulence scale, so
it uses a default TKE of 0.1 m2/s2. Its tendency sits only in the two saturated
layers: +1.2 and +2.9 g/kg/day of vapour at 865 and 910 hPa, with -2.9 and -7.3
K/day. The cooling is exactly Lv/cp times the vapour gain and water is conserved,
so it is evaporating the deck's cloud water in place. That moistens and cools the
deck instead of exporting its moisture. To work, the plume needs a surface-driven
velocity scale. The EDMF line got stuck at that same step on 20 levels.

**Already ruled out** (see the list at the top): a stronger `simple` shallow scheme,
the EDMF line, autoconversion and sedimentation changes, the plume spectrum, a
lower dry-adjustment trigger, and imposed subsidence.

**Recommendation.** Keep the lab on option 1. For goal 1, option 4, which is also
first in `docs/scm_physics_suite_plan.md`. The UW boundary layer is the only change
so far that took the band off saturation, and its failures are specific missing
pieces rather than a wrong design. First step, cheap and contained: reproduce the
BOMEX failure (cloud-water path 0.227 kg/m2 against a 0.10 limit) and trace where
the extra condensate comes from. No column runs until BOMEX passes. [Later on 11 Sep: paused after two failed
fixes; see "UW on BOMEX: step 5".]

### UW on BOMEX: step 1, 11 Sep. Failure reproduced and located.

Same case as the recorded failure: six-hour BOMEX, 20 levels, 900 s host step,
`run_bomex(..., scheme='uw', shallow_scheme='uw')`. BOMEX has no rain, so all cloud
water comes from the two schemes. Cloud water made over the 6 hours, g/m2:

| case | water path kg/m2 | max cloud fraction | made by turbulence | made by shallow |
|---|---|---|---|---|
| UW turbulence with layer closure, alone | 0.0623 | 0.000 | +62.3 | 0 |
| UW turbulence with layer closure + UW shallow | **0.2270** | **0.183** | **+234.7** | -7.8 |
| UW turbulence without layer closure + UW shallow | 0.0393 | 0.032 | +34.4 | +4.8 |

The recorded numbers (0.0623 and 0.2270) reproduce exactly. With shallow
convection on, the turbulence scheme makes 228 g/m2 in a single layer at 804 m.
Shallow convection itself removes a little. So the excess is made by the
turbulence scheme with `uw_layer_closure = true`, not by shallow convection.
Without the layer closure, the same pair passes the 0.10 kg/m2 gate, matching
the 0.039 in `docs/scm_physics_suite_plan.md`. The turbulence-only case still
makes condensate with zero cloud fraction, the known inconsistency.

### UW on BOMEX: step 2, 11 Sep. The plume cannot launch on 20 levels.

At 804 m in the failing run, the turbulence scheme supplies +8.0 g/kg/day of vapour
and makes +2.2 g/kg/day of cloud water. Its entrainment velocity is 0.010 m/s,
against 0.0018 without the layer closure. The shallow plume barely acts.

UW shallow plume over the 24 steps of the six-hour run, no layer closure:

| levels | steps with a plume | highest top | water path kg/m2 |
|---|---|---|---|
| 20 | 0 of 24 | none | 0.039 |
| 40 | 14 of 24 | 1128 m | 0.041 |
| 80 | 15 of 24 | 1195 m | 0.037 |

With the layer closure: 2 of 24 steps at 20 levels (water path 0.227), and 22 of 24
at 80 levels (water path 0.088, under the 0.10 limit).

So the recorded BOMEX failure is a 20-level failure. On 20 levels the UW shallow
plume never launches, and the 0.039 "pass" happens without shallow convection. On
40 and 80 levels it launches and reaches about 1.1-1.2 km. The EDMF line failed the
same way (its 20-level plume reached 56 m). The water-path gate did not catch this,
because it does not check that the plume rises.

The column screens of the same UW pair (`uw_candidate_v1`, 20 levels) also failed:
TOA +9.6 to +12.5 W/m2, and up to 40% of the column's mass saturated.

**Verdict (corrected in step 4: the cause is a bookkeeping bug, not the grid): a coarse-grid limit, not a flaw in the layer closure.** The 20-level grid
has 7 levels below 2 km, too few for a surface-launched plume to reach cloud base.
This is the same wall the production band hits: on 20 levels nothing carries
moisture out of the boundary-layer top. Next: run the boundary-layer and shallow
physics on a finer internal grid (`scm/physics_grid.py`, already in the code and in
the plan) and repeat the 20-level BOMEX.

### UW on BOMEX: step 3, 11 Sep. The finer internal grid launches the plume but makes fog.

The same 20-level BOMEX run with the UW turbulence and shallow schemes on
`scm/physics_grid.py`. It splits every host layer below sigma 0.70 into finer
pieces, runs the schemes there, and maps the results back conservatively.

| run | internal levels | plume launched | water path g/m2 | in the lowest level | above 400 m | max RH, lowest level |
|---|---|---|---|---|---|---|
| native 20 | 20 | 0 of 24 steps | 39.3 | 0.0 | 39.3 | 0.868 |
| native 40 | 40 | 14 of 24 | 41.2 | 0.0 | 41.2 | 0.870 |
| native 80 | 80 | 15 of 24 | 37.0 | 0.0 | 26.8 | 0.869 |
| 20 + grid x2 | 29 | 0 of 24 | 0.0 | 0.0 | 0.0 | not measured |
| 20 + grid x4 | 47 | 11 of 24 | 38.0 | 38.0 | 0.0 | 1.001 |
| 20 + grid x8 | 83 | 24 of 24 | 72.6 | 72.6 | 0.0 | 1.008 |

With the finer grid the plume launches, but all the cloud water sits in the lowest
level. The turbulence scheme saturates it (RH above 1) and condenses 89-90 g/m2
there. That is surface fog. No native run does this at any resolution, so the fog
comes from the internal-grid path, not from thin layers as such. The high cloud
fraction in these runs (0.25 and 0.47) is that fog, read out by the shallow scheme
as cloud water divided by 3 g/kg. The layer closure on the finer grid breaks
outright: 5.4 kg/m2 of cloud water in six hours.

Likely cause, not yet tested: each step rebuilds the fine layers as flat copies of
the 20-level values. The fine near-surface profile of temperature and TKE is lost,
and the surface moisture lands in a thin layer that cannot mix it away.

**Verdict.** UW turbulence plus UW shallow works on native 40- and 80-level grids:
the plume launches, there is no fog, and the cloud water sits in the cumulus layer.
On 20 levels it cannot launch. The finer internal grid is the right idea for a
20-level column, but as implemented it makes fog, so it is not usable yet.

### UW on BOMEX: step 4, 11 Sep. The real cause is a bookkeeping bug.

**Correction to step 2.** Its verdict, "a coarse-grid limit", was wrong. CESM
(32 levels, about 8 below 2 km) and GFDL AM4 (33 levels) run UW-type shallow
schemes on grids about as coarse as ours (7 below 2 km), so the 20-level failure
needed a closer look.

Traced inside the plume, 20-level BOMEX, two steps (launch w2 0.18 m2/s2):

| step | first layer | plume behaviour | where it stops |
|---|---|---|---|
| 8 | 803 -> 1239 m (436 m) | buoyant at once, 0.5 g/kg of cloud water, w2 up to 2.0 | 194 m into the layer |
| 23 | 804 -> 1240 m (436 m) | buoyant at once, 0.8 g/kg of cloud water, w2 up to 4.4 | 242 m into the layer |

The plume launches, forms cloud and rises about 200-240 m. It then stops inside
the first model layer. When that happens, `_integrate_one_column` in
`scm/convection_uw.py` records no flux and no plume top, so the whole plume is
discarded. On 80 levels the first layer is 94 m thick, so plumes cross several
layers before stopping and are kept. That is the entire resolution difference.

Ruled out as the cause: launch strength. Feeding the plume the boundary-layer-mean
TKE (0.200 against 0.093 at the top level, as CAM's `tkeavg` does) changes nothing
at 20 levels. CAM's `uwshcu.F90` declares a penetration depth for a plume that
stops inside a layer (`ppen`, `kpen`); the part that computes it was not visible
in the fetched excerpt, so its exact form is unverified.

**Verdict: fixable in our code, no grid change needed.** When the plume stops
after crossing into the next layer, its transport up to that point must be kept
and its mass left in that layer. The finer internal grid is not needed for this.

### UW on BOMEX: step 5, 11 Sep. Two fixes tried, both fail. Paused.

**Keep the partial layer (tried, reverted).** When the plume stopped inside the layer
above, keep what it had carried across the boundary. Criteria fixed before: plume
kept in at least 12 of 24 steps at 20 levels, under 0.10 kg/m2 of cloud water, no
fog, all tests passing. Result: kept in 3 of 24 (was 0), 22.5 g/m2, no fog, but
`test_uw_bomex_long_timestep_is_bounded_at_development_resolutions` failed (cloud
fraction 0.20 against 0.15 at 40 levels). Reverted; `scm/convection_uw.py` matches
HEAD. Across all 24 steps the plume rises a median 242 m (range 47-291) and crosses
the first boundary (218 m up) in 15. The check on 50 m sub-steps caught only 3,
because many plumes stop in the very sub-step that crosses.

**Why the plume stops so low: it mixes too hard.** Our lateral mixing rate at launch
is 9.4 per km. GFDL AM4's shallow plume uses a constant 3 per km (Zhao et al. 2018,
as summarized in a search; not checked against the paper text). Our plumes stop
near 1 km at every resolution. BOMEX clouds reach about 1.5-2 km.

**GFDL's mixing rate in our plume (diagnostic, original code).** Criteria fixed
before: kept in at least 12 of 24 steps at 20 levels, tops of at least 1400 m,
under 0.10 kg/m2, no fog, similar tops across resolutions.

| levels | plume kept | median top | cloud water g/m2 | max cloud fraction |
|---|---|---|---|---|
| 20 | 9 of 24 | 1241 m | 354 | 0.23 |
| 40 | 12 of 24 | 1375 m | 252 | 0.32 |
| 80 | 19 of 24 | 1319 m | 282 | 0.38 |

The plume goes deeper and crosses boundaries more often, but it makes 2.5-3.5 times
the cloud-water limit. Fails.

**Verdict: UW is not a quick fix.** Each correction exposes the next defect: the
launch bookkeeping, the dilution, the cloud water it then makes, a phase partition
that disagrees with production (`partition_mse` at full saturation against
production's 0.90), and a cloud fraction read from cloud water. Making it work is a
component-by-component project with BOMEX profile targets (plume mass flux, cloud
fraction and cloud water by height, against the LES), not only the water-path gate.
That gate missed that the plume never launched. Paused, not dropped: come back to it
after the longwave work, or sooner if that work points back to clouds. The lab stays
on option 1.

Full test suite with the stop on in `atm407.toml`: 148 passed, 1 expected-fail, 0 failures.

### Link map and fix classification, 10 Sep

The open problems fall into three clusters. Problems inside a cluster share a
source and should be fixed together. The clusters are independent of each other.

**Cluster A: cloud deck and surface. Macro.**
- The boundary layer carries about 3 g/kg/day of water to its top (865-910 hPa).
  Deep convection removes about half. The rest condenses, so those layers
  saturate and fill with cloud: 89% cover, about 245 g/m2 of cloud water.
- Clouds are radiatively off, so the deck is invisible. Switched on, it would
  reflect about 160 W/m2, far more than observed.
- With the deck invisible, the surface absorbs clear-sky sunlight (212 W/m2) and
  must shed it. Evaporation is limited, so sensible heat carries the rest. That
  gives the 4.15 K surface-air gap. This link is a hypothesis, not yet tested.
- A second lead (10 Sep, untested): deep convection cools the levels below
  865 hPa by 20.4 W/m2. That includes 8.4 W/m2 (-14 K/day) in the 5 hPa lowest
  layer. The surface has to resupply that heat as sensible heat.
- Shallow convection cannot help. Above 85% RH its formula moves water down.
- Fix: redesign how moisture leaves the boundary-layer top, then turn clouds on
  and re-derive the albedo. A design decision and a new equilibrium. Options
  laid out 11 Sep; see "Cloud deck: design options".

**Cluster B: stratosphere and model top. Small fixes, large effect.**
- The plume does not stop at neutral buoyancy and cools the stratosphere. Verified
  above. **Stop tested alone 10 Sep: not promoted at first, promoted later that day.** It fixes 10 hPa (34 K too
  cold becomes 5 K) but cools 305-380 hPa by 4-5 K. **Overshoot tested 10 Sep:
  not promoted.** It leaves a 205 K layer at 235 hPa. In both tests most of the
  upper-troposphere cooling follows colder surface air, not the plume top. See
  "Plume-top fix alone" and "Plume overshoot".
- Ozone: **not a problem in production** (it already uses the profile path). The
  count-weighting bug only affects configs using plain `multiband` or `semi_gray`.
- `p_top = 0`, so the top layer spans 0-20 hPa with unbounded depth.
- Result: stratospheric temperature is not physical. The top layer is 194 K, and
  it moved 13 K when only the boundary-layer mixing changed. [With the stop, now in
  production: 222.7 K, within 5 K of the US Standard Atmosphere.]
- Fix: open. The neutral stop fixes the stratosphere. Decided 10 Sep: promote
  it. The colder surface was traced to removing the plume's 5.4 W/m2 heat pump,
  not to a flaw in the stop. Promoted after a settling run. The overshoot is worse. A plume spectrum
  (`mf_plume_count`) was tested 11 Sep and is not the lever. The cool, dry source
  air limits the plume top; see "Low plume top".
  The ozone profile is already on. The model top is a separate, larger change.

**Cluster C: code hygiene. Micro, independent.**
- Phase-partition recycling: **not a production defect.** The old test checked a
  pair production does not use. New contract test added; production pair stable.
- Silent defaults: **done.** 26 written into `atm407.toml`, bit-identical.
- Heights 21 m low: **tried and reverted.** The fix barely moves the column but
  breaks four tests (two energy tests, two EDMF benchmarks). Deferred.
- Cloud optical depths not saved in checkpoints: **fixed** in the reference
  generator.
- Legacy mass-flux path: **removed 10 Sep** (see Cleanup). Before that: diagnosed and fenced off. Its raw tendencies lose
  122.6 W/m2 of column moist static energy, and the MSE correction spreads that
  back as a uniform +1 K/day at every level. It does not run in production, but
  it was the code default, so any caller without a config got it. The code
  default is now 'flux', matching every shipped config. The leak itself is not
  fixed. Switching the default exposed a real production problem; see the
  paragraph on closure resolution dependence.
- Checked and fine: `[mass_flux] detrainment_rate` maps correctly to
  `mf_detrainment_rate` (both 5e-6 in the current config).
- None of the changes kept moves the lab's equilibrium. The 26 settings are
  bit-identical, the default switch does not reach the lab (its config sets
  'flux'), and the checkpoint change only matters with cloud radiation on.

**No longer problems:** upper-troposphere dryness (RH 0.46 at 305 hPa, C-shaped
profile), the upside-down boundary layer (fixed by `k_diff = 40`), and TOA 1.06
(explained; the fix added only 0.13 W/m2).

**Suggested order:** Cluster C is done. The resolution fix (fixed-depth parcel and
plume source) was withdrawn; see Plan item 6. Cluster B: the neutral stop and the overshoot
were both tested and not promoted. The neutral stop is promoted.
Cluster A last. It is the real science problem and needs design decisions.

## Plan: two goals, one column

**Primary goal: a realistic high-resolution column.** Fix what is wrong, then add
complexity. **Secondary goal: a reduced-complexity column ready for GCM
integration.** That is a *subset* of the first -- you strip the full column back,
you do not build a second one.

Guiding rule, from the project owner: if it works, do not change it without
strong evidence of improvement. The tiers below are ordered by that rule.
Restructuring proposals are deliberately last, and most are marked "not now".

### Tier 1: defects -- wrong at any complexity, fix regardless of goal

These are bugs. None is a design question and none becomes acceptable at higher
resolution.

1. **[RESOLVED 10 Sep: `k_diff = 40`, theta_v drop 3.16 -> 0.39 K]** **Static instability in the lowest kilometre.** Five unstable interfaces in
   the accepted checkpoint; theta_v falls 282.95 -> 279.81 K from 998 to 865 hPa.
   The 3 K/km dry-adjustment tolerance permits it and the BL diagnosis pins at
   its ceiling. This is the surface/mixing/adjustment coupling documented
   throughout this file, now visible in the shipped state.
2. **Count-weighted absorbers.** `/nlevels` instead of mass weighting survives in
   `semi_gray.py` (lines 45, 64, 123) and, on the active path,
   `multiband.py:198` for shortwave ozone. The bottom 5 hPa layer gets ten times
   its mass share. Ozone is the worse case: it peaks in the stratosphere, so
   count-weighting puts stratospheric absorber in the boundary layer.
3. **Phase-partition recycling.** `partition_mse` (full saturation) and
   `partial_condensation` (RH 0.95) disagree about the same layer, so condensate
   cycles 0 <-> 0.0245 g/kg per call. `scm/phase_partition.py` unified the
   implementation but not the contract.
4. **149 parameters on silent code defaults**, eleven on the active path
   (`ri_crit`, `k_diff_cap_factor`, `unstable_diffusion_boost`,
   `surface_flux_coupling`, ...). This is the failure mode that hid
   `bl_diagnose_depth = False`. Write the active-path defaults into
   `atm407.toml` so they are visible and diffable. Cheap, and it prevents a
   recurrence.
5. **Mass-flux vertical structure.** Warming spans 865-380 hPa while nearly all
   vapour drying sits below 865 hPa; compensating subsidence should couple them
   through the plume depth. Water conserves, so this is structure, not a leak.
   Related: the environmental-descent expression uses temperature from below
   without the pressure-work relation.
   [10 Sep: that expression is in the legacy path, which production does not
   run. See 'Which mass-flux code actually runs'. The legacy path
   was removed later on 10 Sep.]
   **Re-measured 10 Sep on the flux path** (reference state, one call): heating
   spans 380-810 hPa, and all the drying sits at 865 hPa and below. 460-750 hPa
   moistens slightly. So the structure point still stands on the production path.
6. **Flux-path closure is resolution-dependent.** CAPE response per unit mass
   flux differs about 61% across 10/20/40 levels (test limit 15%). This blocks
   the high-resolution goal. See the 10 Sep audit.
   Separate from the plume-top problem in Cluster B (measured). Fix before any
   change of vertical resolution. Its fix changes the 20-level answer, so batch
   it with Cluster B's re-equilibration.
   **Cause identified 10 Sep:** the CAPE parcel and the plume source are both tied
   to the single lowest level. Fix direction: a fixed-depth source layer for both
   (verified to converge in a scratch test). Needs a correct implementation.
   **Downgraded later on 10 Sep:** on a smooth native sounding the closure converges
   from 40 levels; 20 levels is under-resolved (about 1.8x). The source-layer fix is
   not adopted. See the correction under the resolution trace.

### Tier 2: missing physics -- adds realism, the path to goal 1

Add in roughly this order. Each is independent and each can be validated against
the RRTMG harness or a published benchmark before the next is started.

6. **[In progress 11 Sep.] Fifth longwave band** for the water-vapour rotation region, fitted against
   the existing RRTMG comparison. This is the identified cause of the
   upper-tropospheric cooling deficit (0.9-1.4 K/day short at 305-540 hPa): a
   grey one-kappa-per-band model cannot represent line saturation, so opacity
   falls linearly with `q` when it should fall far more slowly.
7. **Water-vapour continuum** (`lw_band_wv_continuum`, implemented, default
   zero). Real MT_CKD physics the model lacks. Tested against the wrong problem
   earlier and found inert; it has never been evaluated against something it
   should actually change.

   **Longwave, step 1, 11 Sep: baseline against RRTMG on the current reference.**
   , clear sky, ozone off in both, same T, q
   and grid. Ours traps too much: OLR 245.5 against 263.3 W/m2
   (-17.8), surface downward longwave 306.3 against
   277.7 (+28.6). Heating RMS 1.02 K/day.

   | hPa | ours K/day | RRTMG K/day | difference |
   |---|---|---|---|
   | 10 | -0.32 | -1.69 | +1.37 |
   | 75 | -0.55 | -1.12 | +0.57 |
   | 235 | -0.01 | -0.97 | +0.96 |
   | 305 | +0.11 | -1.11 | +1.22 |
   | 380 | -0.16 | -1.28 | +1.11 |
   | 460 | -0.74 | -1.11 | +0.38 |
   | 540 | -1.32 | -1.21 | -0.11 |
   | 685 | -2.60 | -1.39 | -1.21 |
   | 810 | -3.21 | -1.62 | -1.59 |
   | 865 | -3.85 | -2.08 | -1.76 |
   | 998 | +0.59 | +4.21 | -3.62 |

   Too little cooling from 10 to 460 hPa, too much from 615 to 910 hPa, and far
   too little heating in the lowest layer, where the surface is 4.8 K warmer than
   the air. That last one is the same line-saturation limit seen from the other
   side: a thin layer next to a warm surface is opaque in strong water-vapour lines,
   and a grey band cannot be. It bears on the surface-air gap.

   Lesson carried from 6 Sep: the rest of the column is tuned around this too-opaque
   longwave, and the too-opaque clear sky partly stands in for the missing cloud
   longwave effect. So the new longwave is built and checked offline against RRTMG
   first. Adopting it in the column is a separate, planned step that re-derives the
   albedo; it is not judged against the old equilibrium.

   **Found 11 Sep: more grey bands do not help.** Three fits against RRTMG from 7 Sep,
   never written up (`outputs/column/diagnostics/radiation_rrtmg_fit_*.json`, made
   by `scripts/fit_multiband_rrtmg.py` over 15 perturbed profiles):

   | fit | worst heating RMS K/day | worst OLR error W/m2 | worst surface-down error W/m2 |
   |---|---|---|---|
   | 8 streams | 0.77 | 5.7 | 7.3 |
   | 12 streams | 0.83 | 5.3 | 4.5 |
   | 16 bands on RRTMG's band edges | 0.84 | 7.2 | 8.2 |

   All stall near 0.6-0.8 K/day. The OLR error flips sign between the halved-humidity
   case (+4 to +7) and the 1.5x case (-5 to -7), so the opacity responds too
   strongly to humidity. That is the grey-band limit, and adding bands does not
   remove it.

   **Where the best grey fits still fail, on today's reference** (heating error
   against RRTMG, K/day):

   | hPa | production | 8 streams | 16 bands |
   |---|---|---|---|
   | 10 | +1.37 | +1.30 | +1.48 |
   | 75 | +0.57 | +0.58 | +0.83 |
   | 235 | +0.96 | +0.74 | +0.76 |
   | 305 | +1.22 | +0.89 | +0.71 |
   | 380 | +1.11 | +0.50 | +0.27 |
   | 810 | -1.59 | -0.12 | -0.63 |
   | 998 | -3.62 | +3.59 | +2.91 |

   The stratosphere keeps about 1 K/day too little cooling in every version, so
   fitting does not reach it. With ozone off that cooling is CO2's, and CO2 is grey
   too. The upper troposphere improves but stays 0.3-0.9 K/day short. The lowest
   layer flips from far too little heating to far too much.

   **Design, following RRTMG (CESM2's longwave): a k-distribution.** Split each
   spectral band into a few g-points. Each g-point gets its own water-vapour and CO2
   absorption strength and a share of the band's Planck emission. One g-point per
   band reproduces today's model exactly, so it is off by default. Fit it with an
   extended `scripts/fit_multiband_rrtmg.py` over the same 15 profiles.

   **Offline acceptance criteria, fixed before fitting.** On all 15 profiles: heating
   RMS at most 0.3 K/day, OLR within 2 W/m2 and surface downward longwave within
   5 W/m2 of RRTMG. On the reference: 235-460 hPa and 10-75 hPa within 0.3 K/day and
   the lowest layer within 1 K/day. CO2-doubling forcing within 10% of RRTMG's on the
   same state (3.63 W/m2, so 3.27-3.99). Radiation cost at most three times today's
   (1.14 ms per longwave call, so at most 3.4 ms).

   **Criterion revised before any fitting.** It first said 3.7 +/- 0.4 W/m2. That is an
   adjusted forcing; the check computes an instantaneous one. On the same clear-sky,
   ozone-off basis production gives 4.72 and RRTMG 3.63, so the fair test is
   against RRTMG on the same state.

   **Found while setting that criterion: the lab's CO2 forcing is about 30% too strong.**
   On the current reference, clear sky, ozone off, instantaneous: production 4.72 W/m2
   for a CO2 doubling, RRTMG 3.63. The earlier "calibrated to 3.708" used a different
   basis. This matters for the notebook's CO2 experiments. The k-distribution fit
   includes a doubled-CO2 case, so a candidate that passes fixes this too.

   **Longwave, step 2, 11 Sep: 4 bands x 3 g-points, shared g-point shares. Fails.**
   `scripts/fit_longwave_kdistribution.py --edges four --gpoints 3`, 4000 steps.

   | criterion | result | |
   |---|---|---|
   | heating RMS, all 15 profiles | 0.68 K/day worst | fail (0.3) |
   | OLR error, all profiles | 2.11 W/m2 worst | fail, just (2) |
   | surface-down error | 4.26 W/m2 worst | pass |
   | reference 230-470 hPa | 1.30 K/day worst | fail (0.3) |
   | reference 10-80 hPa | 0.33 K/day worst | fail, just (0.3) |
   | reference lowest layer | +1.45 K/day | fail (1) |
   | CO2-doubling forcing | 3.63 W/m2 (RRTMG 3.63) | pass |
   | cost | 2.89 ms per call | pass |

   What it fixed: the humidity response (OLR error +2.1 and -1.7 W/m2 for the dry and
   moist cases, against +/-5-7 for every grey fit), the stratosphere (from +1.4 to
   within 0.33 K/day), the CO2 forcing, and most of the lowest-layer error (from -3.6).
   What it did not: the upper troposphere is still +0.8 to +1.3 K/day short at
   235-380 hPa, and RMS stays at 0.6-0.7.

   Why, from the fitted numbers: the three g-point shares are shared by all bands
   (0.17 / 0.62 / 0.21). So every band gets a 21% share of strong absorption,
   including the window. The rotation band's strongest g-point settled at only
   3.9 m2/kg, too weak to make the dry upper troposphere emit, probably because a
   stronger one would make 21% of that band opaque at every level. Next: let each
   band have its own shares. Same criteria.

   **Longwave, step 3, 11 Sep: per-band shares. Fails the same way.** Same criteria.
   Heating RMS 0.67 K/day worst, OLR 2.80 W/m2, upper troposphere 1.17 K/day worst
   (305 hPa), stratosphere 0.42, lowest layer +1.60. The CO2 forcing (3.63) and the
   cost (2.93 ms) pass. The rotation band's strongest g-point settled at 5.8 m2/kg
   with a 29% share, even though it was now free to go stronger. So the shared shares
   were not what held it back. That idea is refuted.

   Every fit so far (grey with 8, 12 and 16 bands; 4x3 with shared shares; 4x3 per
   band) stalls near 0.6 K/day with the same pattern: too little cooling at 235-380
   hPa and too much at 685-865 hPa. When every set of coefficients lands in the same
   place, the cause is probably something no coefficient can change.

   **Longwave, step 4, 11 Sep: not the layer discretization.** Hypothesis: our solver
   treats each layer as one temperature, while RRTMG lets the Planck source vary
   across the layer, and that errs most in thick layers. Test: the same profile
   interpolated to 40 and 80 levels, against RRTMG on the same grids (ours minus
   RRTMG, K/day):

   | version | levels | 230-470 hPa mean | 600-900 hPa mean | RMS |
   |---|---|---|---|---|
   | production | 20 / 40 / 80 | +0.92 / +1.01 / +0.95 | -1.36 / -1.42 / -1.41 | 1.02 / 1.11 / 1.23 |
   | k-distribution fit | 20 / 40 / 80 | +0.75 / +0.72 / +0.66 | -0.62 / -0.67 / -0.67 | 0.60 / 0.58 / 0.59 |

   The error does not shrink with resolution, and RRTMG itself barely changes (305 hPa:
   -1.11 at 20 levels, -1.01 at 80). Refuted.

   What RRTMG has that we do not: each g-point's absorption depends on pressure (and
   temperature). Line centres get stronger at low pressure, line wings weaker. Ours
   is the same at every pressure. Next test: a per-g-point pressure exponent on the
   water-vapour absorption (Tier 2 item 8, pressure broadening). Off by default. Same
   criteria.

   **Longwave, step 5, 11 Sep: per-g-point pressure exponents. Fails.** Same criteria:
   heating RMS 0.69 K/day worst, OLR 2.06, upper troposphere 0.97 K/day worst, 10-80 hPa
   0.43, lowest layer +2.33. CO2 forcing (3.63) and cost (2.81 ms) pass. The fitted
   exponents are modest (strongest rotation g-point -0.31, weak ones positive). It
   helps the upper troposphere a little (1.17 to 0.97) and nothing else.

   **Where this leaves the longwave.** Six structures (grey 8, 12, 16; 4x3 shared; 4x3
   per band; 4x3 with pressure exponents) stall near 0.6 K/day RMS with the same shape.
   None meets the criteria. The best candidate against production, on the reference:

   | | production | 4x3 k-distribution with pressure exponents |
   |---|---|---|
   | heating RMS | 1.02 K/day | 0.58 |
   | OLR error | -17.8 W/m2 | -0.2 |
   | 230-470 hPa mean error | +0.92 | about +0.6 |
   | 10-80 hPa mean error | +0.95 | about -0.1 |
   | CO2-doubling forcing | 4.72 (RRTMG 3.63) | 3.63 |

   A large improvement, but short of the bar set before fitting, so it is not adopted.
   The options (`lw_gpoint_fractions`, `lw_band_wv_pressure_exponent`) stay in the
   code, off by default, with tests. Fit results are kept in
   `outputs/column/diagnostics/longwave_kdistribution_*.json` (local, not in git);
   `scripts/fit_longwave_kdistribution.py` reproduces them.

   **Reassessment against CESM2.** CESM2 does not approximate RRTMG; it runs it.
   RRTMG is already installed here (climlab). For goal 1, the realistic column, calling
   RRTMG directly is simpler and more faithful than fitting a 12-stream copy, and it
   brings cloud radiation with it, which the cloud deck (problem 1) needs. The
   multiband scheme stays for the lab, where it could later take the fitted
   coefficients. Measured: an RRTMG longwave call costs about 1 ms here, under 1% of a model
   step, since radiation runs every 8 steps. climlab's `RRTMG_LW` and `RRTMG_SW` accept
   cloud optical depth (`tauc`), which the column already computes.

   **Decision, 11 Sep: not yet.** Accurate radiation would expose the cloud deck (too
   thick, about 160 W/m2 of reflection once it radiates) and the missing upward moisture
   route at the same time. Order: fix the cloud deck first (the boundary-layer and
   shallow-convection rebuild), then move the realistic column to RRTMG. The lab keeps
   today's radiation. Small fix to do first: the CO2 forcing is 30% too strong. The CO2
   term only acts away from the control CO2, so correcting it changes the notebook's CO2
   experiments but not the lab's equilibrium.
8. **Pressure broadening.** Line absorption currently has no p-dependence beyond
   layer mass.
9. **Condensate sedimentation** (`cloud_sedimentation_speed`, implemented,
   default zero, conservation-tested). Made humidity worse on its own because the
   air below is already near-saturated -- worth revisiting once the band is
   fixed, not before.
10. **Model top and stratosphere.** `p_top = 0` means the top layer has unbounded
    geometric depth and there is no stratosphere. This is a genuine requirement
    for goal 1 and it is where ozone belongs. It invalidates the radiation tuning
    and every checkpoint, so schedule it as a deliberate epoch with a full
    retune, not as a side change.

### Tier 3: GCM-readiness -- only for goal 2, and only when goal 1 is close

Do not do these now. They are recorded so the decision is deliberate later.

- **Conserved-variable interface** (`qt` plus dry or moist static energy). This
  is what CAM actually couples on, and it would close item 3 as a side effect.
  The strongest Tier 3 candidate, and the only one that is pure refactoring
  against existing tests.
- **Surface fluxes as a boundary condition** rather than a body source spread by
  `surface_heat_sigma_depth`. Must change together with the turbulence closure:
  the physical form made dry adjustment worse under Richardson (14.4 -> 33.9
  K/day) and zero under UW.
- **Hybrid sigma-pressure or dry-mass coordinate.** CAM-SE moved to dry-mass
  coordinates for condensate and energy treatment
  ([Lauritzen et al. 2018](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2017MS001257)).
  Only relevant when a host is actually in view.
- **Documented tendency contract**: fields required per step, fields returned,
  units, host assumptions. Write it as a test, not prose.

### Not changing, and why

- **Temperature stays the prognostic thermodynamic variable.** Hydrostatic
  models store `T`; the models that store theta (WRF, MPAS) are nonhydrostatic
  and solve a compressible set. `T` is also what radiation, the surface bulk
  formulae and the parcel ascent each want directly. The existing pattern --
  store `T`, convert to theta locally where a scheme needs it, as
  `scm/dry_adjustment.py` does with Exner-weighted mass -- is correct.
- **Scheme signature** `(state, grid, params)` -> tendencies. Already the right
  shape; it is what made the whole per-scheme audit in this document possible.
- **Leading batch dimension.** Maps directly to host columns.
- **Sequential operator splitting.** Ordering-sensitive in principle, but no
  measured problem, and changing it would be a rewrite.

### One caution carried forward

`cloud_ls_precip_fraction = 0.95` looks like a bug -- it removes 95% of
condensate before microphysics sees it, leaving `qc` an order of magnitude below
its own autoconversion threshold. It is load-bearing: routing water through the
reservoir instead made humidity worse. Anything in Tier 1 or 2 that touches
microphysics should re-check this rather than assuming it is wrong.

## September 9: origin of the near-surface instability, measured

The professor's static-instability finding was traced to its source with a new
per-scheme stability budget (`scripts/trace_static_stability.py`, 2 days, no
physics changed). The budget brackets every scheme with a virtual-potential-
temperature gradient reading, so each scheme's contribution to static stability
is measured rather than inferred from parameter perturbation -- the method that
sent earlier sessions in circles.

**Static stability budget, K/km per interface, positive = stabilising:**

| p hPa | radiation | surface | boundary layer | dry adj | shallow | deep | mean state |
|---|---|---|---|---|---|---|---|
| 887.5 | -0.008 | 0.000 | **-0.002** | -0.012 | 0.044 | 0.032 | -2.460 |
| 927.5 | -0.026 | 0.000 | **-0.006** | -0.036 | 0.015 | 0.004 | -2.340 |
| 957.5 | -0.037 | 0.000 | **-0.012** | -0.050 | -0.001 | 0.021 | -2.912 |
| 978.8 | -0.063 | -1.249 | **+0.038** | +1.300 | -0.001 | -0.022 | -2.665 |
| 992.5 | -0.116 | -1.049 | **+0.477** | -0.313 | -0.001 | +1.023 | -2.885 |

**The boundary-layer scheme is doing essentially nothing.** At 978.8 hPa it
contributes +0.038 K/km against dry adjustment's +1.300 -- about 3%. Through the
persistently unstable layer at 887-957 hPa its contribution is negative and
within noise of zero. Surface fluxes destabilise (-1.0 to -1.2) and *dry
adjustment*, a numerical backstop, is the only thing restoring stability, and
only at one interface. Everywhere else the instability simply persists at -2.3
to -2.9 K/km.

**Direct measurement of the cause.** Richardson diffusivity through the unstable
layer, production config, on the accepted checkpoint:

| interface hPa | 887.5 | 927.5 | 957.5 | 978.8 | 992.5 |
|---|---|---|---|---|---|
| K m2/s | 0.0000 | 0.0005 | 0.0016 | 0.0037 | 0.0073 |

A convective marine boundary layer has K of order 10-100 m2/s. **This is roughly
four orders of magnitude too small.** The scheme is not mixing, which is why the
profile is upside down, why the diagnosed depth pins at its 900 m ceiling, and
why dry adjustment is load-bearing rather than a backstop.

This unifies findings that were previously recorded as separate problems: the
dry-adjustment activity, the surface/mixing/adjustment coupling, the moisture
supply into the mid-troposphere, and the 5.73 K surface-to-air temperature jump
are all consequences of a boundary layer that does not mix. It also explains why
UW turbulence eliminated dry adjustment entirely -- it actually mixes.

**Why K is this small: three limiters, measured.** Printing every term in
`richardson_diffusivity` per interface on the accepted checkpoint:

| interface hPa | height m | Ri | stability factor | depth factor | K m2/s | limiter |
|---|---|---|---|---|---|---|
| 837.5 | 1405 | +0.069 | 0.785 | 0.00000 | 0.0000 | depth factor |
| 887.5 | 948 | **-0.634** | 3.536 | **0.00000** | **0.0000** | depth factor |
| 927.5 | 595 | -0.382 | 2.527 | 0.11477 | 0.1450 | stability x depth |
| 957.5 | 337 | -0.118 | 1.472 | 0.39192 | 0.2884 | stability x depth |
| 992.5 | 42 | -0.022 | 1.087 | 0.90968 | 0.4945 | stability x depth |

1. **`k_diff = 0.5` m2/s sets the scale, and it is 20-200x too small.** Even at
   stability factor 1 and depth factor 1 the scheme cannot exceed 0.5, and the
   cap allows only 2.0. A convective marine boundary layer is 10-100 m2/s. This
   is the primary limiter and no amount of stability-function tuning can escape
   it.
2. **The 900 m depth ceiling cuts mixing off inside the unstable layer.**
   `bl_max_depth_m = 900` in both `atm407.toml` and `default.toml`; the diagnosed
   depth sits exactly on it. The interface at 887.5 hPa is at 948 m -- above the
   ceiling -- so its depth factor is exactly zero and K is exactly zero, *despite*
   Ri = -0.634, the most unstable interface in the column. The scheme correctly
   detects the instability (stability factor 3.5, its largest anywhere) and is
   then structurally forbidden from acting on it.
3. **The depth factor is quadratic and decays fast.** `((depth - z)/depth)^2`
   gives 0.115 at 595 m within a 900 m layer, damping mixing to a ninth
   two-thirds of the way up a layer that should be well mixed throughout.

**A previous hypothesis is refuted.** Earlier work suspected the
`k_diff_cap_factor` / `unstable_diffusion_boost` pair (both silently defaulting
to 4, multiplicative). They are not the limiter here: the stability factor peaks
at 3.5 and K never approaches the 2.0 cap. Raising them helped previously only
because it lifted the whole product, not because either was binding.

**Suggested order of work**, cheapest and most defensible first: raise
`bl_max_depth_m` so the ceiling is above the unstable layer (it is pinned, which
is itself a warning); then set `k_diff` to a physically defensible convective
value and check against the dry-transport benchmark already in this document
(Richardson currently leaves a 47 K/km superadiabatic gradient under 100 W/m2
where UW leaves 1.0-1.5); then revisit the quadratic depth profile. Measure the
static-stability budget after each with `scripts/trace_static_stability.py`
rather than moving more than one at a time.

**Caution on the RH95 metric.** The accepted checkpoint reports
`rh95_mass_fraction = 0.0`, but RH at 865/910/945 hPa is 0.928/0.937/0.937. The
band did not disappear; it dropped just below the 0.95 threshold. Do not read
that statistic as evidence the humidity problem is solved.

## September 9-10: the mixing strength was the cause. Fixed and promoted.

Short answer: **the boundary layer mixes about 20 to 100 times too weakly**, and
raising it fixes the upside-down air. Whether the column can live with the change
is still being tested.

### What was ruled out first

Raising the 900 m depth ceiling does **not** fix it. Two-day tests:

| ceiling m | theta_v drop K | dry adjustment K/day |
|---|---|---|
| 900 | 3.16 | 13.88 |
| 1200 | 2.69 | 13.80 |
| 1500 | 2.86 | 13.74 |
| 3000 | 2.82 | 13.73 |

The drop stops improving after about 2.8 K. The ceiling was ranked first in the
previous entry as the cheapest fix; that ranking was wrong.

### What does work

Varying `k_diff` with the ceiling held at 1500 m, two days:

| k_diff | theta_v drop K | dry adjustment K/day | TOA | CAPE |
|---|---|---|---|---|
| 0.5 (shipped) | 2.86 | 13.74 | +0.95 | 766 |
| 5.0 | 2.78 | 6.57 | +1.70 | 642 |
| 15.0 | 1.53 | **0.00** | +3.23 | 496 |
| 40.0 | **0.60** | **0.00** | +4.48 | 441 |

At 40 the column is nearly upright and dry adjustment stops entirely. The
boundary layer does its own job for the first time in this project.

### Equilibrium test, 400 days, 5 m slab

Criteria were fixed before the runs: theta_v drop < 1.0 K, |TOA| <= 1.0,
|surface| <= 1.0, CAPE >= 400, deep rain > large-scale rain.

| | k_diff 15 | k_diff 40 |
|---|---|---|
| theta_v drop K | 1.41 | **0.41** |
| TOA | +1.50 | +2.32 |
| surface | +0.46 | +1.04 |
| CAPE | 537 | 584 |
| deep / LS rain | 2.20 / 0.64 | 2.11 / 1.08 |
| ts K | 288.48 | 289.77 |

**The stability problem is solved at k_diff = 40**: the drop falls from 3.16 K to
0.41 K, which is the professor's finding removed. **The energy balance is not
yet acceptable**: TOA +2.32 against a limit of 1.0.

Both runs were still warming at 400 days, so `k_diff = 40` was continued to
1000 days. It settled (slope 0.0009 K/day, window drift 0.044 K):

| criterion | limit | k_diff 40, 1000 days | |
|---|---|---|---|
| theta_v drop | < 1.0 K | **0.39** | pass |
| TOA | <= 1.0 | **1.062** | **fail by 0.06** |
| surface | <= 1.0 | 0.214 | pass |
| CAPE | >= 400 | 653 | pass |
| deep > large-scale rain | | 2.23 > 1.13 | pass |
| dry adjustment | < 2 K/day | 0.00 | pass |

Five of six pass. The only failure is TOA, 6% over its limit, in a column that has
genuinely stopped moving -- so it is a real offset, not transit. The model's own
`equilibrium_passed` gate fails on the same single metric.

Two side effects to weigh:

- **Better:** surface minus lowest-air temperature falls from 5.73 K to 4.15 K.
  Still large against the observed 1-2 K, but moving the right way.
- **Worse:** the humidity band gets wetter. RH at 865/910 hPa rises from
  0.93/0.94 to **0.99/0.98**. Stronger mixing carries more surface moisture up.
  This fix trades the stability problem for a worse humidity problem.

**Promoted on 10 September**, by the project owner's decision, accepting the
0.06 W/m2 TOA overshoot. `atm407.toml` now sets `k_diff = 40`,
`k_diff_cap_factor = 40` and `bl_max_depth_m = 1500`. The canonical reference
`atm407_equilibrium_20level` is the 1000-day checkpoint; one physics step from it
gives TOA +1.04 against the recorded +1.06. The previous `atm407_flux_v1`
reference is in git history.

`scm/configs/default.toml` was **not** changed. It still has
`bl_max_depth_m = 900` and no `k_diff`, so any config that does not inherit
`atm407.toml` still gets the old weak mixing (code default 0.5). That is the
silent-default hazard recorded elsewhere in this file; decide deliberately.

### Why mixing changes the TOA balance (it barely does)

Mixing does not touch radiation directly. It changes the profile, and the
profile sets the radiation. Swapping temperature and humidity between the old
and new equilibria and recomputing clear-sky radiation:

| column | OLR | ASR | TOA |
|---|---|---|---|
| old | 242.56 | 243.23 | +0.66 |
| new | 246.47 | 247.12 | +0.65 |
| old T, new q (humidity only) | 229.78 | 247.12 | +16.68 change |
| new T, old q (temperature only) | 260.19 | 243.23 | -17.63 change |

Stronger mixing lifts more moisture off the ocean: column water vapour rises
from 9.52 to 15.53 kg/m2. That traps about 17 W/m2 more longwave. The column
warms 3.6 K (287.43 -> 291.02 K) until it radiates that away. The two cancel to
within about 1 W/m2. At equilibrium TOA moved only from +0.93 to +1.06. The
+4.48 seen after two days was the column part-way through that adjustment, not a
lasting cost. An earlier entry described the TOA change as the price of the fix;
the price is 0.13 W/m2, and most of the overshoot was already in the baseline.

### Where the band's water comes from now

With dry adjustment off, the boundary layer is the supplier. Two-day vapour
budget from the promoted reference, g/kg/day:

| p hPa | RH | boundary layer | shallow | deep | condensation | net |
|---|---|---|---|---|---|---|
| 810 | 0.89 | +0.170 | -0.005 | -0.164 | 0.000 | 0.000 |
| 865 | 0.99 | **+3.112** | -0.030 | -1.482 | **-1.599** | 0.001 |
| 910 | 0.98 | **+2.629** | 0.001 | -1.718 | **-0.911** | 0.001 |
| 945 | 0.90 | +0.921 | 0.013 | -0.933 | 0.000 | 0.001 |

The band sits at the top of the boundary layer. Mixing brings about 3 g/kg/day
up to 865-910 hPa. Deep convection removes about half. The rest can only leave
by condensing, and condensation only happens at saturation, so those layers pin
near RH 1. At 810 hPa, above the mixed layer, supply and deep convection balance
without condensation and RH stays at 0.89.

Shallow convection is almost silent in this band: -0.030 g/kg/day at 865 hPa
against +3.1 from the boundary layer. In the real trade-wind atmosphere, shallow
cumulus is what carries moisture out of the top of the boundary layer. It is not
suppressed. It is throttled by its own limiter.

**Why shallow convection is silent: its temperature cap binds.** The scheme's
strength is 0.094 (RH factor 0.51 x humidity factor 0.52 x MSE factor 1.00 x
CAPE factor 0.36), so it is switched on. But it relaxes the upper layer's moist
static energy toward the subcloud value, and the two differ by 21.8 kJ/kg. That
implies strong heating, which hits `shallow_max_dt_day = 1.5` K/day exactly. The
limiter then scales the whole transport down, moisture included. Measured peak
moisture tendency is 0.028 g/kg/day, against 3.1 from the boundary layer.

This is why turning CAPE suppression off (600 -> 1e6) changed nothing to three
decimals in a two-day run: peak |dT| is 1.500 K/day at both settings. Raising the
strength just makes the cap bite harder.

The scheme's moisture target is `min(q_low, 0.85 qs)`. At 865 hPa (RH 0.99) that
target is 0.737 g/kg below the current humidity, so the scheme is trying to dry
the band, not moisten it. **Lifting the cap does not remove the band. It moves it down.** Two days from the
promoted reference, only the shallow caps changed:

| dT cap K/day | RH 750 | 810 | 865 | 910 | 945 | RH95 | dry adj | TOA | CAPE |
|---|---|---|---|---|---|---|---|---|---|
| 1.5 (shipped) | 0.70 | 0.89 | 0.99 | 0.98 | 0.90 | 0.09 | 0.00 | +1.04 | 654 |
| 5 | 0.61 | 0.75 | 0.99 | 0.99 | 0.95 | 0.12 | 0.00 | +0.78 | 553 |
| 15 | 0.59 | 0.74 | 0.81 | 0.99 | 0.99 | 0.09 | 4.12 | +5.31 | 150 |

The free troposphere above the band dries (810 hPa 0.89 -> 0.74), but the
saturation shifts down to 910-945 hPa instead of disappearing. At cap 15 deep
convection collapses (CAPE 150), dry adjustment returns, and TOA goes to +5.3.

This follows from the scheme's formula. When the upper layer is wetter than its
85% target, the scheme removes water from it and, to conserve water, adds that
water to the subcloud layer below. So a stronger shallow scheme pumps the band's
moisture downward. That is the reverse of what trade cumulus does in nature,
which is carry moisture up and out of the boundary layer. Ruled out as a fix;
the scheme's design, not its cap, is what is wrong for this purpose.

### The band is a cloud deck, and its clouds are radiatively switched off

What the model diagnoses in the band, on the promoted reference:

| p hPa | height m | RH | cloud fraction | cloud water g/kg |
|---|---|---|---|---|
| 810 | 1702 | 0.89 | 0.000 | 0.0000 |
| 865 | 1177 | 0.99 | **0.707** | **0.2439** |
| 910 | 764 | 0.98 | **0.611** | **0.2333** |
| 945 | 452 | 0.90 | 0.000 | 0.0000 |

This is a stratocumulus-like deck at the top of the boundary layer: saturated air,
60-70% cloud cover, about 0.24 g/kg of cloud water, capped by drier air above.
That is what real subtropical marine boundary layers look like at this surface
temperature (291 K). Cloud water and cover are both in the observed range for
stratocumulus.

**So the band may not be a humidity error.** Saturation at the top of a mixed
layer is where marine cloud forms. The inconsistency is elsewhere:
`atm407.toml` has `[radiation.clouds] enabled = false`, so the model builds a
cloud deck that does not reflect sunlight and does not emit as cloud. A real deck
like this would reflect strongly and cool the surface, which would change the
whole equilibrium.

This reframes the band from "why is this layer saturated" to "should this cloud
deck be allowed to radiate". **Radiative size of the deck, measured.** Enabling clouds through the config and
stepping the promoted reference, from the second step on:

| | clouds off | clouds on |
|---|---|---|
| absorbed sunlight (ASR) | 247.1 | 85.8 |
| OLR | 246.1 | 239.9 |
| shortwave cloud effect | 0 | **-161.3** |
| longwave cloud effect | 0 | +6.1 |

The deck reflects about two thirds of the absorbed sunlight, a net effect of about
-155 W/m2. For scale, the observed global-mean shortwave cloud effect is about
-47 W/m2, and subtropical stratocumulus regions reach -100 to -150 locally. The
deck holds roughly 245 g/m2 of cloud water at 89% total cover, which is thick and
near-overcast even for stratocumulus.

So turning clouds on is not a toggle. With this deck radiating, the column would
lose over 150 W/m2 and cool far below 291 K. The deck being this thick is the
humidity band seen from the radiation side: the boundary-layer top saturates and
fills with cloud water. Note the promoted `k_diff = 40` made the band wetter
(RH 0.93 -> 0.99 at 865-910 hPa), so it also thickened the deck.

**Restart note.** Cloud optical depths are not saved in checkpoints (arrays: t, q,
qc, cloud_fraction, ts, ps, sigma_full). Microphysics computes them, and it runs
after radiation. So the first radiation call after any restart sees zero cloud
optical depth. With clouds off this has no effect. With clouds on, a one-step
diagnostic on a restarted state reads a cloud effect of exactly zero. That is how
this measurement was first misread; step at least twice.
**[Fixed 10 Sep.]** `scripts/make_atm407_reference.py` now saves
`cloud_sw_tau_layer` and `cloud_lw_tau_layer` in new checkpoints, and restores
them when seeding from a checkpoint that has them. Older checkpoints lack them
and fall back to the old behaviour. The notebooks and diagnostic scripts still
load only t, q, qc and cloud fraction; with cloud radiation off, as in the lab,
that makes no difference.
Turning clouds on also needs the surface albedo re-derived. The albedo 0.28
pairing noted in `atm407.toml` was measured while the cloud flag was broken, so
it cannot be trusted. That would be a new equilibrium, and a decision for the
project owner.

Caution: this is a plausibility argument from the cloud diagnostics, not a
measured cause. It is consistent with every "relocates instead of removes"
result in this document, since moving a cloud deck's moisture does not remove
the reason the boundary-layer top saturates.



### Note for whoever continues this

`scripts/make_atm407_reference.py` has no `--k-diff` flag. It uses argparse and
**errors loudly** on an unknown flag, so a mistyped run stops rather than
producing a wrong answer. Set `k_diff` in a config file instead, and confirm it
loaded with `extract_param_overrides` before trusting the result. An earlier note
here claimed the script silently ignored the flag; that was wrong.

## September 8: teaching audit exposed unresolved physical problems

Passing the bulk equilibrium gates is not enough to accept the vertical profile. The canonical checkpoint has five statically unstable interfaces in its lowest kilometre: virtual potential temperature falls from 282.95 K at 998 hPa to 279.81 K at 865 hPa. The dry-adjustment tolerance of 3 K/km permits this, and the boundary-layer diagnosis therefore reaches its 900 m ceiling on the unmodified sounding at every tested wind speed. The surface is also 5.73 K warmer than the lowest air layer; sensible and latent fluxes are 41.4 and 73.1 W/m2, giving an evaporative fraction of only 0.638. These are related structural warnings, not notebook-plotting errors. The checkpoint remains useful for reproducing the current model but should not be described as a physically accepted marine profile until this is corrected and a new equilibrium is generated.

The flux-form mass-flux scheme has a second profile problem. At the six-hour closure timescale it warms broadly from roughly 865 to 380 hPa, but nearly all net vapour drying lies below 865 hPa. An established deep-convection scheme should couple compensating-subsidence warming and drying through more of the plume depth. The column-integrated vapour loss still equals convective rain, so this is a vertical-structure problem rather than a water leak.

The four-band longwave scheme also lacks a separate super-strong water-vapour rotation band. Its upper-tropospheric cooling is therefore not spectrally realistic even though its bulk TOA response is calibrated. The implementation already accepts arbitrary band counts, but adding a fifth band requires fitting its spectral weight and absorption coefficients against the existing RRTMG comparison harness before retuning and regenerating equilibrium.

Notebook 01 now labels these limitations, uses controlled stable soundings for the boundary-layer and dry-adjustment demonstrations, uses unique control-variable names so out-of-order execution cannot change another experiment, and no longer claims that longwave cools every layer. These presentation fixes do not resolve the underlying SCM problems above.

Written as a handoff. The model is a single-column atmospheric model (`scm/`)
used for an undergraduate atmospheric dynamics lab
(`notebooks/02_experiments.ipynb`). Physics is mass-flux deep convection
(Zhang-McFarlane inspired), a simple shallow scheme, Richardson boundary-layer
diffusion, large-scale condensation, prognostic cloud condensate, and a
four-band longwave / three-band shortwave radiation scheme. Configuration is
`scm/configs/atm407.toml`, layered over `scm/configs/default.toml`.

The reference state is a saved equilibrium in
`notebooks/data/atm407_equilibrium_20level.npz`, regenerated by
`scripts/make_atm407_reference.py`. The lab loads it so students start from a
balanced column rather than waiting for a spin-up.

The lab is delivered as two notebooks. `01_meet_the_column.ipynb` introduces
each parameterization in isolation -- one section per scheme, each calling
`scheme(state, grid, params)` against the saved reference and plotting what it
does on its own. `02_experiments.ipynb` puts them together and runs the
column forward. The split exists because the schemes are separable in the code
and the physics is easier to teach that way; it also means the remaining
humidity artefacts are shown to students in section 10 of the first notebook
rather than hidden. Notebook 2 is titled "the column at work"; the earlier
forecast-office framing ("can convection keep up", the tournament, the medal
ratings) was removed at the user's request, though the predict-before-you-run
device was kept.

The following historical introduction described this as a current picture, not an append-only log. Two
earlier top-priority problems (subgrid condensation, moist subcloud layer) have
been traced to code defects and are recorded under "Fixed and verified" rather
than in the open list.


## September 7 (later): the notebook 02 phase portrait is aliased, not wrong code

The "Storm phase portrait: does rain chase CAPE?" figure in
`notebooks/02_experiments.ipynb` plots exactly what its code says it
plots, but the picture is dominated by an artifact and reads as "rain and CAPE
are unrelated", which is the opposite of the model's actual behaviour.

**The column carries an intrinsic relaxation oscillation.** With no large-scale
forcing at all, the saved 20-level equilibrium run at dt = 900 s does not sit
still. Deep-convective rain and dilute CAPE execute a repeating cycle of period
roughly 6-8 steps (1.5-2 hours), spanning CAPE 725-842 J/kg and deep rain
1.38-1.74 mm/day. Sampled every step, control CAPE and control deep rain
correlate at **-0.92**: rain is lowest exactly when CAPE spikes. This is the
trial-plume closure in `scm/convection_mf.py`, where
`mb = target_reduction / cape_response` -- when a step destabilises the column,
`cape_response` (CAPE consumed per unit mass flux) rises faster than
`cape_excess`, so the diagnosed mass flux and rain fall while CAPE rises.

**It is not a radiation-cadence artifact.** Rerunning the control with
`rad_interval = 1` instead of 8 leaves the oscillation unchanged (CAPE standard
deviation 34.63 vs 34.57 J/kg, same spike spacing). The 2-hour radiation refresh
is exonerated.

**Three-hourly diagnostics alias it.** `experiment['diagnostic_hours'] = 3`
samples every 12 steps, close to twice the oscillation period, so successive
plotted points land on unrelated phases of the cycle. The phase portrait's
entire deep-rain spread is 0.39 mm/day while the unforced oscillation alone
spans 0.35 mm/day: about ninety per cent of the vertical scatter students see is
the limit cycle, not the imposed ascent. Point-to-point CAPE-rain correlation in
the plotted series is +0.12, i.e. noise.

**The physics underneath is fine.** Daily means over the same forced run are
clean and monotonic: control CAPE 762 J/kg and rain 1.611 mm/day; forced day 0
913 and 1.676, day 1 1040 and 1.768, day 2 980 and 1.751. Rain does chase CAPE.
Only the sampling and the instantaneous plotting hide it.

**Fixed in the notebook on September 8, without touching the model.** Mission 5
now samples every step instead of every three hours, so the cycle is resolved
rather than aliased, and plots a centred six-hour running mean -- exactly one
cycle period -- as the trajectory, with the raw every-step trace kept underneath
so students can see the oscillation they are averaging over. `runningmean` in the
helper cell pads the ends with the first and last full-window means rather than
with the first and last samples; padding with a single sample put one phase of
the cycle straight back into the two endpoints, which moved the final plotted
rain by 0.074 mm/day, about forty per cent of the whole forced signal. The
unforced star is now the mean of a six-hour control run rather than one
instantaneous `physics_step`, which is why the trajectory now starts on the star
instead of 0.06 mm/day above it. The design-target metrics are computed from the
smoothed series.

Measured effect at the default sliders: CAPE-rain correlation over the plotted
series goes from +0.12 to **+0.91**, and the rating goes from "2 of 3 met" to
"all 3 met" -- the old figure was failing the rain-increase target (0.139 against
a 0.15 floor) purely because `max()` over an aliased series landed on the wrong
phase. Added run cost is about eight per cent, from the extra six-hour baseline
run.

The oscillation itself is untouched and remains open: damping it is a model
change and should not be done to fix a figure. Missions 6 and 7 and the
nine-column sweep sample at 3, 6 and unchecked intervals respectively and have
the same exposure; they were left alone because smoothing would change the lag
and recovery-time quantities they measure. Nothing here invalidates the accepted
`atm407_flux_v1` configuration -- its gates are evaluated on long means, which
average the cycle out.

---

# Earlier handoff: superseded where contradicted by the linked audit

## The one thing to know

**Dry adjustment is not the cause of the saturated band. It is a label on a
transport that happens regardless of which scheme performs it.**

Two invisible parameters in `scm/boundary_layer.py` (`unstable_diffusion_boost`,
default 4, and `k_diff_cap_factor`, default 4) throttle how much the Richardson
scheme may mix in unstable conditions. They are multiplicative -- the boost is
clipped by the cap -- so raising either alone does nothing and raising both takes
dry-adjustment activity from 14.38 to 0.72 K/day, a twenty-fold cut, with no
change of turbulence scheme.

The band does not improve. RH >= 95% mass goes 0.100 -> 0.155. The vapour budget
at 865 hPa shows why:

| process | control | boost 100 + cap 40 |
|---|---|---|
| dry adjustment | **+0.508** | -0.495 |
| boundary layer | +0.083 | **+0.619** |
| condensation | -0.472 | -0.044 |

The supply is unchanged; only its attribution moves. Any scheme that mixes the
boundary layer carries surface moisture upward, and the mid-troposphere receives
it either way. **Earlier sections of this document that treat dry adjustment as
the causal agent are wrong, including ones written on September 7.** It is a
symptom.

**The column is in water balance and the band is a steady state, not an
accumulation.** Over ten days evaporation is 2.2833 mm/day and total
precipitation 2.2835 mm/day, an imbalance of -0.0001. Locally the same holds: net
vapour tendency at 810 hPa is +0.001 g/kg/day against individual terms ten times
larger. Nothing is piling up anywhere.

So the band is not a leak, not a clamp, and not an efficiency shortfall. It is a
balance that happens to sit at RH ~ 1: upward moisture transport and condensation
are equal and opposite, and the equilibrium point of that balance is at
saturation. Moving it requires changing the *ratio* of transport to removal at
those levels, not the magnitude of either -- which is why every intervention that
scaled one term simply relocated the band or swapped which scheme carried it.

An earlier version of this section claimed a 2.37 vs 1.87 mm/day precipitation
shortfall. That was a single-step snapshot read as a mean and it was wrong; the
ten-day figures above supersede it.

**The band is grid-converged.** Same config, 10 days, 20 vs 40 levels: max RH in
780-940 hPa is 0.996 vs 0.997, RH95 mass 0.100 vs 0.090, TOA -0.25 vs -0.24,
CAPE 1312 vs 1452. Doubling vertical resolution changes nothing, so this is real
model behaviour and not a discretization artifact. Do not spend time on the grid.

## What was measured and eliminated

All by experiment, not argument. Do not re-litigate these.

| hypothesis | test | outcome |
|---|---|---|
| autoconversion threshold too high | 0.2 -> 0 g/kg | inert; band unmoved |
| condensate re-evaporating | `dry_factor` = 0 above RH 0.75 | cannot fire in the band |
| BL depth ceiling | 1500 -> 6000 m | depth free at 3894 m, dry adjustment unchanged |
| surface moisture stencil | 0.005 -> 0.05 | 14.4 -> 13.5, band unmoved |
| microphysics reservoir starved | `cloud_ls_precip_fraction` 0.95 -> 0.05 | reservoir activates, humidity **worsens** 0.10 -> 0.15 |
| condensate sedimentation | new code, w = 0.02-0.10 m/s | monotonically worse; clamp propagates downward |
| near-surface instability is numerical | dt 900 -> 100 s | dry adjustment unchanged (14.38 -> 14.30): physical |
| condensation clamps at saturation | direct partition test | false; it settles below saturation as designed |
| dry adjustment is the supply | boost/cap raised | false; boundary layer takes over the same flux |

Two configurations were run to equilibrium, scored against criteria fixed in
advance, and reverted: the RRTMG band calibration (ts 285.45 -> 279.36, CAPE
-39%) and `surface_heat_sigma_depth = 0.05` (six of seven criteria, first config
ever to reach `equilibrium_passed: true`, but the band relocated rather than
dissolving). Both checkpoints are kept as evidence.

## Ways forward, ranked

**1. Mid-tropospheric precipitation efficiency.** The reframing above points
here and nothing has tested it. Total precipitation is 1.87 mm/day against 2.37
mm/day of evaporation. Look at why deep convection removes only ~0.10 g/kg/day
from 865 hPa when dry adjustment and turbulence deliver five times that. Suspect
the mass-flux detrainment profile and `precip_efficiency`; note the environmental
descent expression is separately known to be wrong (it uses temperature from
below without the pressure-work relation).

**2. Make UW usable.** UW drives dry adjustment to exactly 0.00 and is the only
change that broke the band at all three levels together (810/865/910 ->
0.87/0.75/0.60). It costs TOA +9.9 W/m2 and CAPE 1312 -> 335, i.e. convection
nearly shuts off. This is the Bretherton-Park closure as a partial
implementation; it fails BOMEX at 0.227 kg/m2 water path against a 0.10 limit.
Real work, open-ended.

**3. Sub-band radiation structure.** The upper troposphere is 0.9-1.4 K/day short
of RRTMG cooling at 305-540 hPa, and four candidate causes were eliminated. The
cause is the grey band model: one `kappa` per band cannot represent line
saturation, so opacity falls linearly with `q` when it should fall far more
slowly. Needs more bands or a strong/weak-line split per band. The RRTMG
comparison harness now exists (climlab is installed).

**4. Leave it.** Both notebooks pass, the reference is promoted and converged,
and the band is documented for students as a known artefact. Defensible if ATM
407 is the goal.

## Defects found in code review, not yet fixed

**Count-weighted absorbers remain in four places.** The `/nlevels` bug fixed in
the multiband longwave still exists in `scm/radiation_schemes/semi_gray.py` lines
45, 64 and 123, and -- on the active path -- `multiband.py:198` for shortwave
ozone. On the 20-level grid the bottom 5 hPa layer receives **ten times** its
mass share. Ozone is worse than CO2 was: it is not well mixed and peaks in the
stratosphere, so count-weighting places stratospheric absorber in the boundary
layer. Present shortwave heating is +0.19 K/day at 10 hPa against +1.49 at 998
hPa. A `multiband_ozone_profile` scheme exists and fixes this; `atm407.toml` does
not select it.

**149 parameters are read from silent code defaults** and set by no config. Most
belong to inactive schemes, but these are on the active path and load-bearing:
`ri_crit` (0.25), `k_diff_min` (0.05), `k_diff_cap_factor` (4.0),
`unstable_diffusion_boost` (4.0), `bl_shear_floor` (1.0), `surface_flux_coupling`
('distributed'), `cloud_evaporation_scheme`, `cloud_autoconversion_scheme`,
`cloud_fraction_from_condensation`, `lw_band_wv_continuum`, `condensation_scheme`.
This is the same failure mode as `bl_diagnose_depth` defaulting to `False`, which
silently disabled a verified fix in every config that did not inherit
`atm407.toml`. Recommend writing the active-path defaults explicitly into
`atm407.toml` so they are visible and diffable.

**Surface fluxes are a body source, not a boundary condition.** Sensible and
latent heat are spread over the lowest layers by `surface_heat_sigma_depth`
rather than applied as a flux boundary condition on the turbulence solver. These
are not equivalent. The physical form exists (`surface_flux_coupling =
'boundary_layer'`) and makes dry adjustment worse under Richardson (14.4 -> 33.9)
but gives 0.00 under UW -- so the coupling and the closure must change together.

## Fixed this session

- **UW discarded its cloud fraction.** `partition_water` returns a fraction from
  the same subgrid distribution it condensed from; `boundary_layer_uw.py`
  assigned and dropped it, so UW-mixed layers carried condensate while reporting
  zero cloud. Now returned as `condensation_cloud_fraction` and merged into the
  microphysics diagnosis. (A first attempt returned it as `cloud_fraction`, which
  tripped the EDMF plume-handoff branch in `column_model.py:311` and broke three
  runs with `KeyError: 'plume_condensate'`; the separate key avoids that.)
- **Condensate sedimentation** added to `scm/cloud_microphysics.py` as a
  conservative upwind flux, `cloud_sedimentation_speed`, default 0.0. Water
  residual 3.1e-11 with it active against 3.6e-11 without. Inert at the default;
  kept because it is real physics the model lacked, but it is not a fix.

## State

`scm/configs/atm407.toml` is unmodified from the last commit. The promoted
reference is untouched. It reports `equilibrium_passed: false`, but only on the
0.05 K window-drift gate (0.105); TOA 0.241, surface 0.488 and column residual
0.0097 all pass comfortably, and the slope is 0.0021 K/day. Describe it as
nearly converged, not converged. Test suite
124 passing. Both notebooks execute end to end. All experimental checkpoints are
kept under their own labels.


## Where the column stands right now

### September 6: longwave validated against RRTMG

climlab (`conda install -c conda-forge climlab climlab-rrtmg`) is installed in
the `gcm` environment, giving an independent longwave reference. The promoted
reference profile was passed to both `compute_longwave_multiband` and
`RRTMG_LW`, clear sky, same T and q.

**OLR: multiband 242.13, RRTMG 256.20 W/m2 -- the scheme traps 14 W/m2 too
much.** Heating rates localize it, and the error changes sign with height:

| p hPa | q g/kg | multiband K/day | RRTMG K/day | diff |
|---|---|---|---|---|
| 305 | 0.009 | -0.200 | -1.121 | +0.922 |
| 380 | 0.037 | -0.198 | -1.611 | +1.413 |
| 460 | 0.096 | -0.321 | -1.643 | +1.322 |
| 540 | 0.171 | -0.500 | -1.401 | +0.901 |
| 810 | 1.746 | -3.707 | -2.712 | -0.995 |
| 865 | 2.517 | -3.754 | -2.547 | -1.206 |
| 997 | 5.115 | -1.546 | -0.596 | -0.950 |

Too little cooling aloft, too much below. Longwave absorption here is linear in
`q` (`kappa_wv * q * dp/g`), so it vanishes as the air dries, while RRTMG keeps
cooling the upper troposphere because CO2 carries that band where vapour is
absent. That is self-reinforcing with the dry upper troposphere in problem 1:
nothing cools the layer, so convection is never required to moisten it.

This implicates the CO2 mass-weighting recorded under "Fixed and verified".
Mass-weighting is correct for a well-mixed gas in an optically thin band, but it
concentrates CO2 optical depth in the thick lower layers, where water vapour has
already saturated the absorption. The fix removed a grid-dependence bug and
should not be reverted; the vertical distribution still needs separate work.

Caveat: climlab's 20-level grid is not identical to `make_grid(20)`, so the
per-level differences are indicative. The OLR gap is the robust number.
Reproduce with the comparison in this session's scratch, or rebuild it from
`compute_longwave_multiband` plus `climlab.radiation.RRTMG_LW`.

### September 6: longwave recalibrated against RRTMG, tested, and REVERTED

The calibration below was applied, run to equilibrium, judged against criteria
fixed in advance, and **reverted**. `scm/configs/atm407.toml` is back to the
committed coefficients. The promoted reference was never overwritten; the test
run is kept as `atm407_equilibrium_20level_rrtmg_cal_5m` for the record.

Cause of the 14 W/m2 OLR gap was isolated by elimination: scaling CO2 up made
OLR worse (2x -> 221 W/m2), scaling water vapour down fixed OLR but degraded
heating-rate RMS from 0.805 to 1.326, and only the **atmospheric window
fraction** improved both together. `lw_band_weights[0]` was 0.10, letting just
10% of the longwave through the window; 0.26 closed OLR to within 0.08 W/m2 of
RRTMG (242.13 -> 256.28 against 256.20) and cut heating-rate RMS to 0.632.
`lw_band_co2_log_factor` was re-solved by bisection to restore 3.700 W/m2 per
doubling, which the band change had knocked to 3.214.

**In isolation the radiation was better. In the coupled column it was worse.**
400 days, 5 m slab, from the promoted reference:

| criterion (set before the run) | baseline | calibrated | |
|---|---|---|---|
| \|TOA\| <= 1.0 W/m2 | 0.24 | **1.287** | fail |
| \|surface\| <= 1.0 W/m2 | 0.49 | **1.062** | fail |
| ts drift <= 0.005 K/day | 0.0021 | 0.0040 | pass |
| RH95 mass <= 0.15 | 0.11 | 0.09 | pass |
| CAPE 800-2500 | 1275 | **781** | fail |
| deep rain > large-scale | 1.72 / 0.67 | 1.05 / 0.19 | pass |

Surface temperature fell 285.45 -> 279.36 K. The direction was expected -- more
outgoing longwave forces cooling -- but the magnitude broke convection: CAPE
dropped 39% and deep rain 39%, and the column had not re-equilibrated after 400
days.

**The lesson is worth keeping.** The old coefficients were not merely mistuned;
the column's convection was calibrated *against* the too-opaque radiation, so
correcting one alone breaks the balance. A future radiation improvement has to
be accompanied by re-tuning convection, or evaluated on a column allowed to find
a genuinely new equilibrium rather than judged at 400 days. Note also that
humidity slightly improved (RH95 0.11 -> 0.09), so the radiative direction is
probably right even though this configuration is not promotable.

### September 7: the near-surface instability is physical, not a timestep artifact

Step 1 of the chain -- why the 988/998 hPa interface is unstable at essentially
every step -- had never been tested directly. If it were a discretization
artifact (the bottom layer is 5 hPa, about 42 m, taking +17.6 K/day of surface
heat on a 900 s step) the cure would be cheap substepping. It is not.

| dt s | steps/day | 988/998 unstable | 945/970 | max dry-adjustment K/day |
|---|---|---|---|---|
| 900 | 96 | 100.0% | 43.2% | 14.38 |
| 300 | 288 | 85.2% | 37.2% | 14.23 |
| 100 | 864 | 83.2% | 33.2% | 14.30 |

A ninefold reduction in timestep leaves dry-adjustment magnitude **unchanged**
(14.38 -> 14.30) and the interface unstable at 83% of steps. There is mild
timestep sensitivity in the firing frequency but none in the effect. The surface
genuinely destabilizes that interface faster than local diffusion can mix it.

This closes off the cheap fix and confirms the expensive one is the right target.
It is consistent with the dry-transport benchmark already recorded: under 100
W/m2 of surface heating, Richardson diffusion leaves a 47 K/km superadiabatic
gradient while UW leaves 1.0-1.5. Local K-diffusion is structurally inadequate
for a convective boundary layer, which is why every serious model uses nonlocal
or mass-flux transport there.

Note also that surface fluxes are currently injected as a *body source* spread
over the lowest layers (`surface_heat_sigma_depth`), rather than applied as a
flux boundary condition on the turbulence solver. The two are not equivalent,
and `surface_flux_coupling = 'boundary_layer'` (the physical form) made dry
adjustment *worse* under Richardson (14.4 -> 33.9) while giving 0.00 under UW.
The coupling and the turbulence closure have to be fixed together; neither alone
is sufficient.

### September 7 (later): causal chain closed, and a correction

**Correction to the entry below.** It states the cause as "condensation is a
one-way sink that can never push a layer below saturation". That is true of the
`rh_crit = 1.0` branch, but production runs `rh_crit = 0.95`, which dispatches to
`partial_condensation` / `phase_partition.partition_water`. Testing that partition
directly at 810 hPa conditions (T 260.3 K, qs 1.756 g/kg, rh_crit 0.95).
[10 Sep: production changed to 0.90 in commit 22fb0fe, late on 7 Sep. At 0.90
the mean stays below saturation until total water is about 15% above `qs`
(RH 0.971 / 0.989 / 0.998 / 1.000 at qt/qs 1.00 / 1.05 / 1.10 / 1.15). The
conclusion below still holds.]

| qt/qs | grid-mean RH after | qc g/kg | cloud fraction |
|---|---|---|---|
| 1.00 | 0.9855 | 0.019 | 0.462 |
| 1.05 | 0.9990 | 0.066 | 0.861 |
| 1.10 | 1.0000 | 0.129 | 1.000 |

It settles the mean **below** saturation with partial cloud, exactly as a
Sundqvist closure should. The formulation is not clamped and is not the bug. What
the table does show is the real limit: the mean only stays under 1 while total
water is within about 5% of `qs`. Beyond that the subgrid distribution is fully
saturated and the mean pins. **The band is a supply problem, not a sink problem.**

#### The chain, every link measured

1. The near-surface column is unstable at nearly every step (988/998 hPa exceeds
   the dry-adjustment trigger at 100% of sampled steps).
2. Dry adjustment therefore fires continuously, at 14.4 K/day.
3. It is the dominant moisture source into 810-910 hPa: +0.404 g/kg/day against
   condensation -0.301 and deep convection -0.095, netting +0.001.
4. That supply pushes total water past the ~5% subgrid width, so the grid mean
   pins at RH 1.

#### UW turbulence confirms link 2-3 by removing them

Ten days, production config otherwise, from the promoted reference:

| case | dry adj K/day | 810 | 865 | 910 | RH95 | TOA | CAPE | deep rain |
|---|---|---|---|---|---|---|---|---|
| richardson (control) | 14.42 | 1.00 | 0.95 | 0.97 | 0.100 | -0.25 | 1312 | 1.75 |
| uw_moist | **0.00** | 1.00 | 0.99 | 0.81 | 0.390 | +9.03 | 585 | 0.90 |
| uw_moist + layer closure | **0.00** | **0.87** | **0.77** | **0.62** | 0.360 | +10.20 | 320 | 0.47 |
| uw + BL surface coupling | **0.00** | **0.83** | **0.76** | **0.61** | 0.300 | +9.39 | 323 | 0.46 |

Dry adjustment goes to exactly zero in every UW variant, and with the layer
closure the band comes off saturation at all three levels together for the first
time. The diagnosis is therefore confirmed rather than inferred.

The cost is that UW is not yet usable: TOA +9.4 W/m2, CAPE collapses 1312 -> 323,
deep rain 1.75 -> 0.46, and RH95 mass *triples* to 0.30 because saturation
reappears elsewhere. Convection is close to shut off. This reproduces the
September 6 screening and is why UW is not default.

**The remaining problem is UW's moist response, specifically its condensate
handling** -- the BOMEX water-path failure (0.227 against a 0.10 limit) and the
turbulence-only case leaving cloud fraction at zero despite carrying condensate.
That inconsistency is the next thing to fix, and it now sits on the critical path
to problem 1 rather than beside it.

#### Condensate sedimentation: implemented, tested, not enabled

`cloud_sedimentation_speed` (default 0.0) adds a conservative upwind flux of
condensate between layers in `scm/cloud_microphysics.py`. Water residual is
3.1e-11 kg/m2/s with it active against 3.6e-11 without, so the transport itself
is clean. **It makes humidity monotonically worse:**

| cloud_ls_precip_fraction | w m/s | RH95 | 810 | 865 | 910 | 945 |
|---|---|---|---|---|---|---|
| 0.95 | 0.00 | 0.110 | 0.99 | 0.97 | 0.94 | 0.86 |
| 0.20 | 0.02 | 0.180 | 0.99 | 0.99 | 0.99 | 0.97 |
| 0.20 | 0.05 | 0.205 | 0.97 | 0.97 | 0.97 | 0.99 |

945 hPa goes 0.86 -> 0.99: condensate falls into the layer below and evaporates
there, but that air is already near-saturated, so the clamp propagates downward
instead of draining. The code is kept because it is real physics the model
lacked and it is inert at the default, but it is not a fix and should not be
enabled on its own.

### September 7: the saturated band is a clamp, not a leak -- cause established

The band at 810-910 hPa is not maintained by any removable process. It is held
at saturation by a balance, and the mechanism is now measured rather than
argued.

**The decisive measurement.** Ten-day mean vapour budget at 810 hPa, production
config, from the promoted reference:

| process | g/kg/day |
|---|---|
| dry adjustment | **+0.404** |
| condensation | -0.301 |
| deep convection | -0.095 |
| shallow | -0.011 |
| boundary layer | +0.003 |
| **net** | **+0.001** |

Final state: RH 0.9961, `qc` **0.045 g/kg**. Two facts follow, and together they
close the question.

**The layer is saturated with essentially no condensate in it.** No microphysics
change can dry a layer that holds no cloud water. That is why autoconversion,
evaporation and the precipitation split were all inert -- they act on
condensate, and there is none here to act on.

**Condensation is a one-way sink.** It removes vapour above `qs` and can never
push a layer below saturation. So any layer with a persistent moisture supply is
clamped at exactly RH = 1 and held there indefinitely; the supply rate sets how
much condenses, not how wet the layer is. Dry adjustment is that supply, and it
keeps firing because the near-surface column is unstable at nearly every step
(988/998 hPa exceeds the trigger at 100% of steps).

This explains why every intervention relocated the band instead of removing it.
Widening the sensible-heat stencil changed *where* dry adjustment fires, so the
saturated layer moved with it -- 910 hPa improved 0.95 -> 0.65 while 750 hPa
became saturated and RH95 mass rose 0.11 -> 0.12. Nothing tested changed the
fact that a moisture-fed layer gets pinned.

#### Eliminated this session, all measured

| hypothesis | test | result |
|---|---|---|
| autoconversion threshold too high | scan 0.2 -> 0 g/kg | inert; RH95 0.10 -> 0.11, band unmoved |
| condensate re-evaporating | `dry_factor` = 0 above RH 0.75 | cannot fire in the band |
| BL depth ceiling | raise 1500 -> 6000 m | depth diagnoses to 3894 m, dry adjustment unchanged at 14.4 K/day |
| surface moisture stencil | 0.005 -> 0.05 | dry adjustment 14.4 -> 13.5, 810 hPa still 1.00 |
| microphysics reservoir bypassed | `cloud_ls_precip_fraction` 0.95 -> 0.05 | reservoir activates (autoconversion 0.000 -> 0.47, LWP 0.004 -> 0.070) but humidity **worsens**: RH95 0.10 -> 0.15 |

The last one is worth keeping. `cloud_ls_precip_fraction = 0.95` looked like a
bug -- it removes 95% of condensate before microphysics sees it, leaving 0.045
g/kg against a 0.2 g/kg autoconversion threshold, so the reservoir never
activates. It is instead load-bearing: dumping water out of the column fast is
partially compensating for the band. Routing water through the reservoir makes
it linger and re-evaporate, and the layer gets wetter.

#### What would actually fix it

Two candidates, both structural changes rather than tuning, neither attempted:

1. **Stop dry adjustment being a moisture supply.** It is a numerical backstop
   and it is currently the dominant moisture source in the mid-troposphere. This
   is the connected-layer/UW turbulence work already identified as the next
   target -- the September 6 audit found UW eliminates dry-adjustment activity
   entirely, though its moist response is not yet acceptable.
2. **Give condensation a subsaturated exit.** Condensate sedimentation between
   layers (problem 5) would let water leave a layer physically instead of being
   removed in place, so the layer can fall below saturation.

Reverted after testing and not promoted: the RRTMG band calibration, and
`surface_heat_sigma_depth = 0.05` (six of seven criteria passed, `equilibrium_passed:
true`, but RH95 mass rose because the band relocated rather than dissolved --
kept as `atm407_equilibrium_20level_heatstencil_5m`).

### September 6: the upper-troposphere cooling deficit is a band-model limit

Four candidate causes were tested against RRTMG for the 0.9-1.4 K/day shortfall
at 305-540 hPa. **All four failed**, and the negative results are the finding.

| candidate | result |
|---|---|
| water-vapour continuum, 10-200 | no help aloft (it is quadratic in `q`, so it acts where vapour already is); RMS degrades past 25 |
| CO2 concentrated aloft, `massfrac * (p/p0)^n`, n = -0.5 to -1.5 | worse: upper levels flip to net *warming* |
| CO2 scaled 2-8x | OLR collapses to 221-179 W/m2 |
| sublinear absorption, `tau ~ (q dp/g)^e`, e = 0.4-0.8 | optimizer returns to e = 1.0; every sublinear fit trades the lower troposphere away |

**RRTMG says the cooling is water vapour, not a trace gas.** Zeroing CO2 *and*
ozone in RRTMG changes 380 hPa cooling only from -1.611 to -1.677 K/day, while
OLR moves 256 -> 290. So the missing opacity is vapour opacity in a layer
holding 0.037 g/kg, and no redistribution of a well-mixed absorber can supply
it.

The structural reason: `tau_wv = kappa * q * dp/g` is a **grey band model**.
Within one band it has a single absorption coefficient, so a layer's opacity is
proportional to its vapour. Real bands contain thousands of lines of hugely
varying strength; in a dry layer the weak lines go transparent while the strong
line cores stay saturated, so opacity falls far more slowly than `q`. That is
what a correlated-k scheme like RRTMG represents with its k-distribution, and it
cannot be recovered by tuning a single coefficient per band -- which is exactly
what these four experiments demonstrate.

Fixing it properly means sub-band structure: either more bands with a spread of
`kappa`, or a two-coefficient (strong-line / weak-line) split per band. That is
a real change to `multiband.py`, not a recalibration, and it should be done
against the RRTMG comparison already built.

Consequence for problem 1: **the dry upper troposphere has a radiative cause
that is now identified but not fixed.** Too little cooling aloft means
convection is never obliged to moisten that layer. Do not attribute it solely
to convection until the band model is improved.

### Phase-partition contract: module landed, contract did not

`scm/phase_partition.partition_water` is now imported by both
`scm/boundary_layer_uw.py` and `scm/condensation.py`, which was the prescribed
next step. **The recycling it was meant to remove still reproduces.** [Superseded 10 Sep:
that test alternates a pair production no longer uses. With the production pair
the partition is stable; see `scripts/diagnose_partition_contract.py`.] Rerunning
`scripts/diagnose_partition_handoff.py` (removed 10 Sep) gives condensate alternating between
0 and 0.0245097 g/kg across all five cycles, unchanged. The script alternates
`partition_mse` (grid-mean full saturation) with `partial_condensation` (RH
0.95): the two call sites now share an implementation but are still asked for
different answers. The shared *contract* is outstanding.

### Latest structural development: experimental connected-layer closure

`scm/uw_layers.py` implements cloud-fraction-weighted moist buoyancy from
liquid static energy and total water (including condensate loading), separate
connected convective/shear layers, a length scale based on each layer's own
thickness, layer-mean TKE relaxation and energy-limited entrainment at stable
edges. The surface no longer supplies an elevated layer's length scale or TKE
across a strong stable barrier. This follows the organization in Bretherton
and Park (2009), DOI 10.1175/2008JCLI2556.1, but remains a reduced development
implementation: surface forcing, layer extension and entrainment quadrature
are simplified, and cloud-top radiative entrainment is not implemented.
It must not be described as a complete CAM UWMT port.

Select it explicitly with `uw_layer_closure = true` or the budget script's
`--uw-layer-closure`. It is not enabled by default because it fails the moist
acceptance test. Long host steps use internal turbulence steps of at most
60 s; this alone did not remove the BOMEX failure. Surface fluxes are applied
once per substep and aggregate returned tendencies conserve the full-step
surface heat and water input. UW residual diagnostics now check the actual
returned thermodynamic state rather than only the pre-partition solver state.

New tests cover neutrality at uniform moist-conserved variables, independence
of a detached turbulent layer from surface buoyancy, and full-step surface
budgets under subcycling. The broad suite passed 124 tests with the accepted
default path; the three structural tests also pass after the final stable-
surface-layer diagnostic correction. These passes do not override the known
experimental BOMEX failure:

| six-hour BOMEX, 20 levels, 900 s host step | water path kg/m2 | diagnosed depth m |
|---|---|---|
| new turbulence alone | 0.0623 | 640 |
| new turbulence plus UW shallow convection | 0.2270 | 1023 |

The combined case exceeds the existing water-path limit 0.10 kg/m2 and cloud
fraction limit 0.15 (it reaches 0.183). At 40 levels water path is 0.2064 and
cloud fraction 0.255. This demonstrates a significant turbulence/shallow
interaction; it does not prove shallow convection alone is defective.
An additional benchmark deficiency is that the turbulence-only case leaves
cloud fraction zero despite condensate, so its moist stability input is not
self-consistent. A proper cloud-diagnosis handoff is required before using
that case to validate a moist closure.

The dry heated test improves: maximum unstable theta gradient is about
1.00 K/km, versus 1.47 for the earlier UW candidate and 47 for Richardson.
The surface-connected layer diagnoses approximately 2.10 km rather than
sticking at an imposed 1.50 km ceiling. Energy remains conserved.
In the two-day ATM407 screening, however, deep/LS/cloud rain is
0.309/0.005/3.245 mm/day, surface flux -57.24 W/m2, TOA +6.00 W/m2, CAPE
203 J/kg and saturated mass 28%. Thus the structural candidate is implemented
and testable but is not a successful replacement for production physics.
The next work must resolve cloud diagnosis and the shallow/turbulence
interface on BOMEX, including condensate and source-layer budgets, before
attempting more equilibrium tuning.

Reproduce dry tests with `scripts/diagnose_dry_surface_transport.py
--uw-layer-closure` and ATM407 with the previous coupled UW budget command
plus `--uw-layer-closure`. Results are
`outputs/column/diagnostics/dry_surface_connected_layers_6hour.json` and
`atm407_uw_connected_layers_2day.json` in that directory. Earlier dry-candidate
output was regenerated during development; use the connected-layer filename
for the explicitly selected new closure. Defaults and the promoted reference
have not changed.

### September 6 audit: current production budget

#### Controlled boundary-layer comparison

#### UW moist-column screening result

Actual-transport trace identifies a major contributor to UW's wet layer.
The diagnostic BL depth is 1500 m, but `uw_diffusivity` retains locally
generated TKE/diffusion above that depth up to a separate 5000 m limit.
Over the corrected two-day run, mean diffusivities at 1.94/2.57/3.30 km are
47.84/37.88/14.91 m2/s. Corresponding upward total-water fluxes are
2.800/1.768/0.565 kg/m2/day. These are fluxes inferred directly from the
implicit total-water solve, not reconstructed from an unrelated TKE scheme.
Thus "1500 m boundary layer" does not describe the actual transport extent.

A diagnostic ablation limiting UW turbulence to 1500 m reduces cloud rain
from 1.853 to 0.931 mm/day, saturated mass from 31% to 24%, and surface cooling
flux from -48.09 to -37.61 W/m2. Deep rain rises from 0.624 to 0.960 and LS
rain from 0.143 to 0.669 mm/day; CAPE rises from 373 to 570 J/kg. Primary
energy residual remains 0.00023 W/m2 and water residual 3.09e-11 kg/m2/s.
The elevated transport therefore materially contributes, but is not the sole
cause: saturation remains and dry-adjustment activity returns (maximum
absolute mean tendency 2.61 K/day). A hard cutoff is not a validated fix.

The structural gap is the candidate's treatment of elevated turbulence and
its connection to the surface/cloud layer. It uses a dry virtual-potential-
temperature gradient (without condensate loading) for local TKE, retains that
local mixing above the diagnosed top, and lacks the published UW scheme's
full moist convective-layer diagnosis and layer TKE transport. Its good dry
test does not validate those missing moist processes. Bretherton and Park
(2009), DOI 10.1175/2008JCLI2556.1, describe unified turbulent-layer treatment,
explicit entrainment and layer-mean TKE transport. The name UW in this repo
denotes a partial implementation, not equivalence to the established model.
The next physical correction should address those layer boundaries and moist
buoyancy energetics, rather than selecting a new arbitrary cutoff.

Reproduce the trace with `scripts/trace_uw_transport.py`; results are
`outputs/column/diagnostics/uw_transport_trace.json`. The ablation is
`atm407_uw_above_bl_ablation_2day.json` in that directory, produced by adding
`--uw-max-height 1500` to the UW coupled diagnostic command. All modifications
in this investigation are diagnostics; production physics was not changed.

Shared-partition correction implemented in `scm/phase_partition.py`.
UW and partial condensation now call the same nonprecipitating total-water /
moist-enthalpy solver at the configured condensation RH threshold. Rain is
still removed explicitly by condensation/microphysics, never by the shared
solver. Three new unforced alternating-call tests cover dry, partially cloudy
and supersaturated initial states; condensate, total water and enthalpy remain
stationary. All 21 focused partition/condensation/UW tests pass.
The first broad suite run gave 120 passes and one BOMEX grid-comparison
failure after the full-saturation solver also changed numerical precision.
The correction is now scoped to RH thresholds below 1; the established
full-saturation UW solver remains unchanged. All 17 affected UW convection,
turbulence and shared-partition tests pass on rerun, including that BOMEX test.
The entire suite was not repeated after this scoped correction.

The corrected two-day UW run does **not** remove the moist-column bias:
RH95 mass remains 31%; deep/large-scale/cloud rain is
0.624/0.143/1.853 mm/day, versus 0.580/0.400/1.792 previously.
TOA is +5.156 W/m2, surface -48.092 W/m2, surface drift -0.394 K and CAPE
373 J/kg. Primary residual is +0.0055 W/m2, water residual 4.26e-11
kg/m2/s, and potential-energy reconciliation -0.0012 W/m2.
The eliminated partition recycling was a real defect but is not sufficient
to explain the wet layer. Do not represent this correction as a validated
climate improvement. The next target is the vertical distribution of UW
total-water flux and diagnosed mixing/entrainment extent, using the stored
layer budgets and an independently specified moist case before more tuning.
Result: `outputs/column/diagnostics/atm407_uw_shared_partition_2day.json`.
The earlier standalone handoff script deliberately retains the old
full-saturation/partial-cloud pair as a reproducer of the original failure;
`scm/test_phase_partition.py` verifies the repaired contract.

The timestep/handoff audit is complete. At 300 s versus 900 s, two-day
UW surface flux changes from -49.38 to -48.70 W/m2, CAPE from 351 to 363,
and final RH95 mass remains 31%. Thus the broad moist response persists at
the shorter timestep. Rain partition is more sensitive: deep/large-scale/cloud
rain changes from 0.580/0.400/1.792 to 0.603/0.576/1.594 mm/day. A factor-three
step reduction is not a convergence proof, particularly for precipitation.

UW's boundary-layer stage supplies net condensate at 2.208 kg/m2/day at
900 s and 2.014 at 300 s, concentrated at 615-810 hPa. Existing cloud source
diagnostics do not label this as a plume source. At 900 s microphysics removes
1.811 kg/m2/day from its condensate reservoir, mostly as 1.792 mm/day cloud
rain; the remainder includes the signed condensation handoff. These stage
budgets identify turbulent transport/partitioning as the upstream source,
but do not distinguish all transported condensate from newly condensed vapor.

A specific thermodynamic inconsistency is independently reproduced:
UW calls `partition_mse`, which uses grid-mean full saturation, while ATM407
later calls partial condensation with RH threshold 0.95. At 280 K, 850 hPa,
initial RH 0.98, zero forcing and precipitation disabled, alternating these
solvers repeatedly switches condensate between zero and 0.0245097 g/kg.
Water and moist enthalpy errors are exactly zero in this float64 test.
Conservation therefore does not certify a consistent phase partition. This
recycling is established; its contribution to the full-column cloud-rain
bias is not yet quantified.

Next use one shared nonprecipitating phase-partition contract for UW and
condensation, leaving precipitation to an explicit reservoir sink. Require
the unforced alternating-call test to preserve the partition before running
the same two-day comparison. Do not tune autoconversion to mask the conflict.
Reproduce the isolated failure with `scripts/diagnose_partition_handoff.py` (removed 10 Sep).
Results: `outputs/column/diagnostics/uw_partition_handoff.json` and
`atm407_uw_handoff_dt900_2day.json` / `atm407_uw_handoff_dt300_2day.json` in
the same directory. The budget script now records timestep and the BL
condensate tendency explicitly. Physics defaults remain unchanged.

Two-day ATM407 runs now separate surface coupling from the turbulence scheme.
Both start from the promoted reference at 20 levels, 900 s and 5 m slab; all
radiation, convection and cloud settings remain ATM407's. The new CLI options
are `--surface-coupling boundary_layer --bl-scheme richardson` or
`--surface-coupling boundary_layer --bl-scheme uw_moist` in
`scripts/diagnose_atm407_budget.py`. The surface routine returns zero direct
atmospheric tendencies in this mode, and turbulence receives the fluxes once.

| configuration | TOA W/m2 | surface W/m2 | CAPE J/kg | deep / LS / cloud rain mm/day | final RH95 mass | max absolute mean dry-adjustment heating K/day |
|---|---|---|---|---|---|---|
| original | -0.146 | -0.513 | 1276 | 1.719 / 0.516 / 0 | 0.15 | 14.38 |
| Richardson, coupled surface | +0.004 | +0.026 | 1287 | 1.751 / 0.596 / 0 | 0.10 | 33.60 |
| UW, coupled surface | +5.069 | -49.380 | 351 | 0.580 / 0.400 / 1.792 | 0.31 | 0 |

UW eliminates dry-adjustment activity in this run, confirming that stronger,
conserved-variable turbulent transport can replace that numerical backstop.
However, the moist response is not yet acceptable for promotion: cloud rain
dominates, saturated mass increases to 31%, and surface cooling reaches
0.404 K over two days. The lowest-level RH falls to 52.5%. The original
surface/atmosphere state is far from balance under this scheme, so these
transient fluxes do not establish UW's equilibrium temperature or runaway.
Primary energy residual is +0.00095 W/m2, water residual -6.22e-11 kg/m2/s,
and MSE/primary reconciliation including potential energy is -0.0021 W/m2.
Thus the large surface cooling is resolved energy exchange, not a measured
conservation leak. Diagnosed depth still sits at 1500 m.

Next isolate UW total-water/condensate redistribution and saturation
partitioning at each stage, including sensitivity to the 900 s host timestep,
before any longer run. Do not tune radiation or promote either coupling
variant from these short results. UW's profile diagnostics currently group
surface input into the boundary-layer tendency (no separate surface/mixing
split), unlike Richardson; compare their combined surface+BL tendencies.
The budget script's reconstructed diffusivity uses the separate TKE-v2
formula and is not a valid UW diffusivity measurement.

Evidence: `outputs/column/diagnostics/atm407_richardson_coupled_2day.json`
and `outputs/column/diagnostics/atm407_uw_moist_coupled_2day.json`.

Existing-candidate follow-up: the same six-hour dry test now compares
`boundary_layer_tke_v2.tke_boundary_layer` and
`boundary_layer_uw.uw_moist_turbulence` with the Richardson controls.
Momentum is initialized to zero in every case; prognostic candidate TKE and
returned scalar/momentum tendencies are advanced at every step. The 100 W/m2
heated cases give:

| scheme | largest unstable theta gradient K/km, 60 s | at 30 s | maximum K m2/s |
|---|---|---|---|
| Richardson, temperature | 47.182 | 47.183 | 1.50 |
| Richardson, MSE | 52.357 | 52.358 | 1.68 |
| prognostic TKE v2 | 2.708 | 2.813 | 100 |
| UW diagnostic TKE | 1.47077 | 1.47085 | 200 |

Both candidates preserve the unheated lower layer much better than raw
temperature diffusion. All energy residuals are below 1e-9 W/m2. UW is the
preferred candidate for the next controlled moist-column comparison on this
evidence, not a newly validated production default. Six existing UW tests
also pass, including dry-layer grid comparisons and BOMEX checks; these do
not amount to independent validation of the full scheme against LES.

The candidates hit diffusivity caps. Doubling UW's cap from 200 to 400 m2/s
changes the maximum unstable gradient only from 1.47077 to 1.47296 K/km.
Raising its boundary-depth ceiling alone from 1500 to 3000 m gives diagnosed
depth 1837 m and gradient 1.37625 K/km, so UW is not forced to the new ceiling
in this test. These checks support proceeding with UW without tuning its
diffusivity cap merely to improve the result. Other profile/entrainment
sensitivities remain unvalidated.

Reproduce with `python scripts/diagnose_dry_surface_transport.py`; all 14
cases are saved in `outputs/column/diagnostics/dry_surface_candidates_6hour.json`.
Any moist follow-up must override the current ATM407 parameters directly:
the old `uw_turbulence_only_v1.toml` also changes radiation and is not a
controlled comparison. Test conservative surface-flux handoff together with
the scheme, since the dry experiment injected surface heat directly into it.

Follow-up isolated dry test completed with
`scripts/diagnose_dry_surface_transport.py`. It calls only boundary-layer
mixing: no radiation, convection, condensation or dry adjustment. Initial
potential temperature is 300 K below 850 hPa with a stable atmosphere above;
vapor and condensate are zero. Tests use the native 20-level grid, six hours,
and either zero or 100 W/m2 prescribed surface sensible heat flux.

| transport | flux W/m2 | maximum temperature change K | largest unstable potential-temperature gradient K/km |
|---|---|---|---|
| temperature | 0 | 0.71235 | 0.43750 |
| moist static energy | 0 | 0.0000011 | 0.0000028 |
| temperature | 100 | 10.3641 | 47.1822 |
| moist static energy | 100 | 11.3836 | 52.3568 |

This establishes two distinct shortcomings: raw temperature diffusion does
not preserve a dry-neutral lower layer, while the existing MSE option almost
does; neither closure transports the imposed surface heating without severe
superadiabatic gradients. Halving the timestep from 60 to 30 seconds changes
the forced gradients by less than 0.002 K/km, excluding a large timestep
artifact in this test. Energy residuals are below 1e-9 W/m2. The forced runs
hit the 1500 m depth ceiling, with maximum diagnosed diffusivities only
1.50 and 1.68 m2/s. This is a controlled consistency/stress test, not a claim
of validation against LES or observations.

The next development target is turbulence strength and nonlocal heat
transport driven by surface buoyancy, using conserved thermal variables.
An established implementation should first pass this dry test and a published
convective-boundary-layer benchmark before entering the moist equilibrium.
ECMWF documents this separation: conserved-variable transport plus EDMF in
unstable boundary layers, rather than local diffusion alone
([ECMWF atmospheric physics](https://www.ecmwf.int/en/research/modelling-and-prediction/atmospheric-physics)).
Review the repository's existing turbulence candidates against these tests
before introducing another scheme. No physics default was changed.
Results: `outputs/column/diagnostics/dry_surface_transport_6hour.json`.

The requested two-by-two test is complete: the same promoted checkpoint,
ATM407 config, 20 levels, 900 s, 5 m slab and two days in every case. Only
`bl_max_depth_m` (1500 or 3000 m) and the existing
`bl_mix_moist_static_energy` option (false or true) differ. The unchanged
control is the production-budget run below.

| heat transport / ceiling | TOA W/m2 | surface W/m2 | deep rain mm/day | large-scale rain mm/day | final RH95 mass |
|---|---|---|---|---|---|
| temperature / 1500 m | -0.146 | -0.513 | 1.719 | 0.516 | 0.15 |
| temperature / 3000 m | +0.022 | -0.394 | 1.728 | 0.479 | 0.15 |
| moist static energy / 1500 m | -0.079 | -2.999 | 1.659 | 0.722 | 0.15 |
| moist static energy / 3000 m | -0.035 | -3.191 | 1.654 | 0.716 | 0.15 |

Neither intervention removes the saturated band. Increasing the ceiling
simply makes the diagnosed depth reach the new 3000 m limit. Dry adjustment
still has a maximum absolute mean heating/cooling tendency near 14 K/day in
all four cases. Its vapor input at 910 hPa rises from 0.868 g/kg/day in the
control to 1.065, 0.971 and 1.164 respectively. The MSE option increases
large-scale rain and initially cools the surface faster. These short responses
reject promotion of these switches as a demonstrated fix; they do not establish
their eventual equilibria or discredit MSE transport as a physical approach.

All cases retain small primary energy residuals (absolute values below
0.019 W/m2), water residuals below 1.1e-11 kg/m2/s, and MSE/primary residual
reconciliation including potential energy within 0.002 W/m2. The independent
two-layer diffusion-rate and water/enthalpy conservation tests both pass.

The next investigation should isolate surface-forced turbulent transport in
a dry convective boundary-layer case, with dry adjustment disabled for that
component test. Measure heat flux, static stability and mixing depth against
a specified benchmark. The current diffusivity magnitude and depth diagnosis
must be validated together; neither changing the transported variable alone
nor lifting the depth ceiling resolves the surface instability. This is a
bounded component test, not authorization to remove adjustment in production.

Results: `outputs/column/diagnostics/atm407_bl_depth_2day.json`,
`atm407_bl_mse_2day.json`, and `atm407_bl_both_2day.json` in the same directory.
Reproduce with `scripts/diagnose_atm407_budget.py --reference
notebooks/data/atm407_equilibrium_20level.npz --config scm/configs/atm407.toml
--ocean-depth 5 --days 2`, adding `--bl-max-depth 3000` and/or
`--bl-heat-transport moist-static-energy`, and a distinct `--output` path.
No production physics, default configuration or checkpoint was changed.

This audit supersedes the causal claims below that diagnosed boundary-layer
depth eliminated dry adjustment, and that the Betts-Miller warming mechanism
has been established. The latter configuration failed to converge, but its
cause has not been isolated. Its current config also has condensation RH 1.0,
whereas ATM407 now uses 0.95; it is no longer a closure-only comparison.

A two-day continuation of the promoted reference using `atm407.toml`, 20
levels, 900 s and a 5 m slab gives mean TOA -0.146 W/m2, surface -0.513 W/m2,
surface drift -0.0040 K, deep rain 1.719 and large-scale rain 0.516 mm/day.
Primary energy residual is -0.0082 W/m2 and water residual -1.06e-11
kg/m2/s. MSE residual is +0.392 W/m2 and must be interpreted with potential
energy storage, not reported as zero. These are short-run diagnostics, not a
new equilibrium certification. End-of-run RH >=95% mass is 15%, showing that
the saved 11% threshold statistic alone does not describe persistence.

Mean vapor tendencies in g/kg/day:

| pressure hPa | boundary layer | dry adjustment | shallow | deep | condensation |
|---|---|---|---|---|---|
| 810 | +0.003 | +0.300 | -0.011 | -0.095 | -0.241 |
| 865 | +0.082 | +0.471 | -0.014 | -0.103 | -0.442 |
| 910 | +0.053 | +0.868 | +0.001 | -0.453 | -0.365 |

**[Superseded 7 Sep -- see HANDOFF. Raising the boundary layer's diffusion
limits moves this same supply from dry adjustment to turbulent mixing with no
humidity benefit, so dry adjustment is a label on the transport, not its cause.]**
Dry adjustment supplies most of the local vapor input in this band; deep
convection locally dries it. Diagnosed boundary-layer depth averages exactly
its 1500 m ceiling. A separate stage trace using the same config shows the
970/987.5 hPa interface exceeds the dry-adjustment trigger at every sampled
step. Surface forcing raises its mean lapse excess from 2.52 to 3.79 K/km;
boundary-layer mixing only reduces this to 3.73 before adjustment. Thus the
surface/mixing/adjustment coupling remains load-bearing even with diagnosed
depth. The next controlled test should examine that coupling, including the
1500 m ceiling and the choice of transported thermal variable. Do not infer
that raising the ceiling or increasing diffusivity is already a validated fix.

Upper-tropospheric net moisture tendencies are tiny over these two days, so
this budget does not isolate why that region became dry during spin-up.
The mass-flux heat/moisture descent mismatch remains a separate structural
concern and needs component tests before any replacement is promoted.

Evidence: `outputs/column/diagnostics/atm407_current_audit_2day.json` and
`outputs/column/diagnostics/atm407_current_stage_trace_2day.json`.
`trace_dry_adjustment.py` now accepts explicit config/reference arguments and
defaults to ATM407 and the promoted reference, removing the old silent
default-config confound.

Every checkpoint below evaluated under the code as it currently stands, one
physics step from rest, `atm407.toml`, 5 m slab. "As generated" is what the
metadata recorded when the file was written.

| checkpoint | ts (K) | TOA now | surface now | TOA as generated | subcloud RH (910-997 hPa) |
|---|---|---|---|---|---|
| `atm407_equilibrium_20level` (superseded) | 285.23 | +5.77 | +2.60 | +1.36 | 1.00 / 1.00 / 0.97 / 0.96 / 1.00 |
| `..._partial_cloud_rh100_5m_radswitch` | 289.70 | +1.02 | -0.09 | +0.97 | 1.00 / 1.00 / 0.96 / 0.95 / 1.00 |
| `..._partial_cloud_rh095_5m_radswitch` | 287.12 | -0.03 | -0.27 | -0.23 | 0.99 / 0.95 / 0.90 / 0.91 / 0.99 |
| `..._partial_cloud_rh095_diffusionfix_5m` | 290.48 | +0.58 | +5.56 | +0.47 | 0.84 / 0.74 / 0.67 / 0.65 / 0.75 |

That table is the record of how stale the pre-fix checkpoints became; it is kept
because it is the evidence that metadata cannot be trusted across a physics fix.

**The reference has been replaced.** `notebooks/data/atm407_equilibrium_20level`
is now the `atm407_current_rh095_5m` run: `atm407.toml` under the current code,
400 days on a 5 m slab. `condensation_rh_crit` in `atm407.toml` moved from 1.0
to 0.95 at the same time, so the config the notebook runs and the reference it
loads are the same column -- verified at -0.26 W/m2 TOA one step from rest
against the -0.24 recorded. The notebook's instructor banner was rewritten to
describe the current physics and the two remaining humidity artefacts.

**The subcloud layer is fixed and the fix is the diffusion correction.** The
`diffusionfix` checkpoint was the first to show it, and the regenerated
mass-flux references confirm it under the production config: 945-998 hPa now
runs 0.75-0.86 against 0.90-1.00 before, with 0.75-0.85 observed. That is
Problem 3 from the previous version of this document, closed.

**Three references regenerated under the current code**, all 400 days on a 5 m
slab, all `atm407.toml` except for the convection closure. `bm_conservative_v2`
is a copy of `atm407.toml` with the closure swapped (at the time, a four-key diff:
`convection_scheme`, `bm_conserve_enthalpy`, `rhbm`, `tau_bm`), built because
`bm_conservative_v1` layered over `default.toml` and so ran with
`bl_diagnose_depth` off, the uncalibrated CO2 and the untuned cloud shortwave.
**That diff is no longer closure-only.** `condensation_rh_crit` in
`atm407.toml` later moved to 0.95 while `bm_conservative_v2.toml` still sets
1.0, so the 1000-day runaway below confounds the closure with a condensation
change. The flat-70%-RH water-vapour-feedback mechanism given for it is a
plausible reading, not an established one.

| | MF, rh_crit 1.0 | MF, rh_crit 0.95 | BM conservative v2 |
|---|---|---|---|
| ts | 286.27 K | 285.45 K | 293.60 K |
| TOA | -1.19 | **-0.24** | +5.79 |
| surface | -1.52 | **-0.49** | +3.07 |
| ts slope, 50-day window | 0.0046 K/day | **0.0021** | 0.0114 |
| mass at RH >= 95% | 0.15 | 0.11 | 0.03 |
| CAPE | 1353 | 1275 | 1635 |
| deep / large-scale precip | 1.87 / 0.69 | 1.72 / 0.67 | 2.86 / 0.002 |
| column water residual | -2.5e-10 | 3.9e-11 | 4.3e-10 |

**The fair fight does clear Betts-Miller of the runaway.** `v1` reached 321 K
with CAPE 57 000 and zero deep precipitation; `v2` convects, precipitates almost
entirely convectively, and conserves. The 321 K blowup was the missing
`bl_diagnose_depth`.

**But its humidity is prescribed, not solved, and the RH >= 95% mass fraction is
a misleading way to see that.** Read the profiles instead:

| p (hPa) | 305 | 380 | 460 | 540 | 615 | 685 | 750 | 810 | 865 | 910 | 945 | 970 | 988 | 998 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MF 1.0 | .05 | .20 | .37 | .47 | .50 | .50 | .57 | 1.00 | 1.00 | 1.00 | .89 | .80 | .76 | .85 |
| MF 0.95 | .04 | .19 | .37 | .47 | .50 | .50 | .57 | 1.00 | .97 | .94 | .86 | .78 | .75 | .84 |
| BM v2 | .70 | .70 | .70 | .70 | .70 | .70 | .70 | .70 | .70 | .73 | .96 | .93 | .87 | .97 |

Betts-Miller relaxes humidity toward `rhbm * qs`, and with the whole column
convecting the column simply *is* the reference profile: a flat 0.70 from 305 to
910 hPa. It has no saturated slab because it cannot have one, not because
anything was fixed. A real tropical sounding has a mid-tropospheric minimum
around 30-40% and a dry upper troposphere; a constant 70% is not that.

That also explains the warmth, and the continuation settles it. At 305 hPa
Betts-Miller holds 70% RH where mass flux holds 5%, and upper-tropospheric water
vapour is what sets OLR. Continued to 1000 days the column does not converge:

| days | ts | TOA | ts slope |
|---|---|---|---|
| 0 | 289.70 K | +1.02 | -- |
| 400 | 293.60 K | +5.79 | 0.0114 K/day |
| 1000 | 299.98 K | +6.23 | 0.0108 K/day |

The imbalance grows and the drift rate does not decay. This is a slow runaway,
driven by the water-vapour feedback of a prescribed moist upper troposphere, and
600 further days bought no convergence. `bm_conservative_v1`'s 321 K was the
same trajectory accelerated by the missing `bl_diagnose_depth`, not a separate
failure. Betts-Miller at `rhbm = 0.7` is not a candidate for this column.

**Mass flux is in much better shape than this document previously recorded**,
because the older numbers came from checkpoints generated before the cloud-flag
and diffusion fixes. At `rh_crit = 0.95` the column is nearly converged -- TOA
-0.24 W/m2, drifting 0.002 K/day -- and RH >= 95% mass is 0.11 against the 0.24-0.30
recorded earlier. `equilibrium_passed` is still false only on the 0.05 K
window-drift gate (`scm/diagnostics.py`), which is strict; TOA, surface and
column-residual gates all pass.

Two real problems remain visible in the mass-flux profiles: a saturated band at
810-910 hPa (narrower than the 685-865 previously recorded, and at `rh_crit
0.95` no longer pinned at 1.00), and an upper troposphere that is too dry, 4-5%
RH at 305 hPa where observations are 20-40%. **[Superseded 10 Sep: in the current reference RH is 0.46 at
305 hPa and 0.71 at 380 hPa, with a mid-tropospheric minimum of 0.28 at 540 hPa.
That is close to the observed tropical C-shape. The 4-5% figure came from an
older reference. It was already 0.34 at 305 hPa before the k_diff change.]**

**The saturated slab is narrower but not gone.** Every checkpoint still has RH at or near 1
between about 685 and 865 hPa, 24-30% of column mass at RH >= 95%.

---

## Open problem 1: near-saturated band at 810-910 hPa

**Status: much improved, not closed.** Under the current code the band is
810-910 hPa rather than 685-865, and RH >= 95% column mass is 0.11 at
`rh_crit = 0.95` against the 0.24-0.30 recorded below. Betts-Miller shows 0.03
but only because it prescribes RH, so it is not evidence about this problem
either way. The record below of what was tried against the slab still stands;
the numbers in it predate the cloud-flag and diffusion fixes and should be
re-measured before any of it is treated as a closed door.

The layers between roughly 685 and 865 hPa sit at or very near 100% RH.
Observed tropical mean RH at those levels is 60-80%. A deep saturated layer in a
*mean* profile is not physical: it is what a cloudy sub-volume looks like, which
is what cloud fraction is supposed to represent.

Partial condensation now works (see "Fixed and verified"), and it does lower the
slab, but only from 1.00 to about 0.95-0.98 at `rh_crit = 0.95` -- it moves the
pin rather than removing it. So the layer is not merely being clipped by
saturation adjustment; something is delivering moisture to it faster than
anything removes it.

The best evidence about what that something is comes from a stage trace of the
`diffusionfix` checkpoint: at 810 hPa, dry adjustment adds 0.770 g/kg/day, deep
convection removes 0.121, and condensation removes 0.658. Radiation cools the
level 3.231 K/day, balanced by dry adjustment (+0.411), deep convection
(+1.173), and condensation (+1.638). On that reading the layer is maintained by
repeated dry-adjustment transport against radiative cooling, while deep
convection locally dries it.

**Caveat that must be resolved before acting on that trace.**
`scripts/trace_dry_adjustment.py` calls `load_run_config(None)`, so it runs on
`default.toml` alone. `bl_diagnose_depth` defaults to `False` in code
(`scm/boundary_layer.py:30`), so the trace ran with the *fixed* `bl_top_sigma`
cutoff -- precisely the configuration in which the dry adjustment is already
known to become load-bearing, and which `bl_diagnose_depth = true` in
`atm407.toml` reduces but does not remove (see the retraction below). The trace should be repeated against
`scm/configs/atm407.toml` before its attribution is trusted. It may well hold;
it has not yet been shown to hold for the configuration the lab runs.

Approaches already tried and rejected, with the reason:

- **Lowering the dry-adjustment trigger to zero.** Saturation spreads *upward*
  to 540-750 hPa, large-scale rain rises 0.933 -> 2.592 mm/day, the surface
  cools 0.678 K at a mean surface flux of -16.55 W/m2. Energy residual stays at
  0.006 W/m2, so this is not a conservation leak -- lowering the trigger
  intensifies the moisture-transport/condensation cascade. Results:
  `outputs/column/diagnostics/rh095_dry_neutral_10day.json`.
- **Imposed subsidence.** Dries the column only by destroying convection: at
  0.5 hPa/hr CAPE falls 2026 -> 481 and precipitation 2.69 -> 1.50 mm/day. The
  drying and the convective suppression are the same process, so no setting is
  useful.
- **Autoconversion thresholds.** Condensate drains (cloud water path 0.67 ->
  0.07) but RH *rises*, because the condensate returns to vapour rather than
  precipitating.
- **Plume spectrum** (`mf_plume_count`, `mf_plume_entrainment_spread`). No
  measurable effect: `buoyancy_detrainment_weight = 0.0` means detrainment does
  not respond to buoyancy, so every plume detrains identically. Enabling
  buoyancy detrainment made the profile worse.
- **Common-interface convective heat/vapour fluxes.** Over ten days this
  severely depleted upper-tropospheric vapour, raised large-scale rain to
  1.714 mm/day, and developed a water residual of -2.01e-7 kg/m2/s. Removed.
  Not a clean single-factor test (it also used a 2500 Pa CAPE step and immediate
  condensate fallout). Record: `rh095_convective_flux_10day.json`.
- **Widening the surface-flux stencil.** Regressed the column; reverted.

The mass-flux environmental-descent expression is a live suspect: it uses
temperature from below without the appropriate pressure-work relation. Trials
correcting its direction, throughflow and vertical gradient did not produce a
validated replacement -- the two-day drying persisted at 685 hPa but 750-865 hPa
remained saturated after ten days, and the closure-response resolution test
failed (19-23% spread against a 15% limit). Those trials were removed. The
expression is still wrong; nobody has yet found the right form.

---

## Open problem 2: no working alternative to the mass-flux closure

**Status: Betts-Miller is ruled out; nothing has replaced it.** The runaway is
understood rather than open (see the 1000-day table above): the closure holds a
flat 70% RH through the deep troposphere, and the resulting water-vapour
feedback carries the column past 300 K without converging. What remains open is
that the mass-flux closure has no validated competitor, so the two humidity
artefacts in problem 1 have no second scheme to be checked against.


`scm/configs/bm_conservative_v1.toml` is a Frierson-style Betts-Miller with a
repaired energy contract: reference temperature and humidity solved together to
match column moist enthalpy, one common relaxation factor so limiting preserves
that constraint, contiguous convective layer including the source air, no
negative-rain solutions, all removed vapour returned as precipitation with an
explicit zero retained-condensate return so the legacy cloud fallback cannot
manufacture cloud water from the same rain. The approach follows
[Frierson (2007)](https://doi.org/10.1175/JAS3935.1). It is a simplified
adjustment scheme at prescribed reference RH 0.7, not a CESM/GFDL port.

It does what it was built to do for humidity: in a 50-day continuation, RH at
685-865 hPa settles at 72-81% and dry-adjustment transport there is zero.

It does not close energetically. The saved 600-day checkpoint
(`notebooks/data/atm407_equilibrium_20level_bm_conservative_v1_5m.npz`) is a
runaway:

| ts | TOA | surface | CAPE | deep precip | large-scale precip |
|---|---|---|---|---|---|
| 321.36 K | +42.4 W/m2 | +8.9 W/m2 | 57 375 J/kg | 0.00 mm/day | 0.94 mm/day |

Deep precipitation is exactly zero and CAPE is 57 000 J/kg, so convection has
switched off entirely and the column is warming with nothing to stabilise it.
Note this supersedes the 200-day description that appeared in earlier versions
of this document (293.6 K, TOA +11.1, still warming at +0.024 K/day) -- that run
was continued to 600 days and the file was overwritten. The 200-day state was
not an equilibrium, it was a waypoint on the way here.

**Before this is debugged as a convection problem, fix the comparison.**
`bm_conservative_v1.toml` layers over `default.toml`, not over `atm407.toml`, so
the candidate ran without three settings the baseline depends on:

| setting | atm407 | bm_conservative_v1 |
|---|---|---|
| `bl_diagnose_depth` | `true` | absent -> `False` (code default) |
| `lw_band_co2_log_factor` | calibrated to 3.7 W/m2 per doubling | default, 2.1 W/m2 |
| `cloud_sw_scattering_efficiency` | 0.60 | 0.05 |
| `dry_adjustment_max_lapse_excess` | 3.0 | absent -> code default |

The first of those is the significant one: the candidate re-introduced the fixed
boundary-layer cutoff that `bl_diagnose_depth` was added to remove. A candidate
config that is meant to change one thing should inherit `atm407.toml`, and this
one does not. Re-run it that way before concluding anything about the closure.

---

## Open problem 3: no model top / no stratosphere

`make_grid` uses `p_top = 0`, so the top layer extends to zero pressure and has
unbounded geometric depth. CAM6 caps at 2.25 hPa and GFDL AM4 near 1 hPa, both
on hybrid sigma-pressure grids. There is no stratosphere, and ozone is lumped
into the well-mixed trace bucket rather than given a vertical profile (a
`multiband_ozone_profile` radiation scheme exists and some candidate configs use
it, but the ATM407 config does not).

Changing this alters the whole vertical grid, so it invalidates the radiation
tuning and any saved reference. Deliberately deferred. Note the vertical
*distribution* is fine: layer thickness peaks in the mid-troposphere in pressure
terms, but pressure is mass; in height the grid coarsens monotonically upward
(42 m at the surface to ~6 km at the top), matching CAM6 and AM4.

---

## Open problem 4: cloud radiative effects are too weak

Disabled in the baseline (`[radiation.clouds] enabled = false`), with
`albedo = 0.32` carrying the whole planetary albedo. The machinery works and is
tuned: `enabled = true` with `albedo = 0.28` gives a net cloud radiative effect
near -8 W/m2, correctly negative for a low-cloud regime.

It is off because the cloud field is not good enough to rest an equilibrium on.
Cloud fraction is patchy and small (0.02-0.19 against 0.3-0.6 observed), and
because every cloud in this column is low and warm the longwave effect is only
about +2.5 W/m2 against +26 observed -- there are no high cold anvils. Toggling
clouds is a good lab experiment; building the reference on them is not.

Until very recently this flag did not actually work; see "Fixed and verified".

---

## Open problem 5: missing radiation physics

Checked against RRTMG as used by CESM2 and GFDL:

- **No water-vapour continuum.** Longwave absorption is `kappa * q * dp/g`,
  strictly linear in humidity. The MT_CKD continuum's self-broadened part scales
  with vapour amount times vapour pressure -- roughly quadratic -- and dominates
  the 8-12 micron window in moist air. A parameterisation exists
  (`lw_band_wv_continuum`, default zero, quadratic in `q` and pressure
  dependent). It was tested against the cold drift that was then blamed on
  condensation and does not affect it at any strength up to 150; that drift has
  since been traced elsewhere, so the continuum has not been evaluated against
  anything it might genuinely change. It remains a real physics gap.
- **No pressure broadening.** Line absorption has no p-dependence beyond layer
  mass, overweighting upper-level absorption.
- **No condensate sedimentation.** Condensate is removed in place and never
  falls between layers.
- **Random cloud overlap only** (`cover = 1 - prod(1 - cf)`); both reference
  models use maximum-random.

---

## Fixed and verified

Do not re-litigate these.

### Cloud radiative effects were on when the config said they were off

`scm/cloud_optics.py`. `clouds_enabled()` returned true whenever cloud
microphysics was active, and `cloud_optical_properties()` selected an optics
scheme from `cloud_optics_scheme = "auto"` without consulting the radiation
flag. `atm407.toml` sets `[radiation.clouds] enabled = false`, which resolves to
`cloud_radiative_effects_enabled = False`, and that was ignored. The explicit
flag now takes precedence in both places.

The consequence is larger than a flag: **every ATM407 run described as clear-sky
was not.** Clouds were reflecting and absorbing while `albedo = 0.32`, the value
chosen to carry the whole planetary albedo *in their absence*, was applied on
top. That double-counting is why the baseline sat at 285 K. The same config with
the flag honoured settles at 289.7 K with TOA +1.0.

It is also the whole of the former "Problem 1". Subgrid condensation was blamed
for driving the column to 223-265 K, non-monotonically in `rh_crit`, and was
disabled on that evidence. Those runs had unintended cloud shortwave absorption.
With the flag honoured:

| rh_crit | ts | TOA | CAPE | before the fix |
|---|---|---|---|---|
| 1.00 | 289.70 | +0.97 | 2171 | 285.23 |
| 0.95 | 287.12 | -0.23 | 1410 | 223.69 |
| 0.90 | 285.41 | +4.38 | 1587 | 223.48 |

There is no collapse and no non-monotonicity. Partial condensation is a working
scheme, not a broken one, and `condensation_rh_crit` is a usable knob again. The
default is still 1.0 only because no reference has been promoted at 0.95 yet.

### Boundary-layer diffusion was too strong by a factor of g

`scm/boundary_layer.py`. The Richardson and constant-K solvers carried an extra
factor of gravity. Interface conductance is already `rho K / dz =
K g rho^2 / dp_interface`, so the implicit exchange coefficient must be
`dt * conductance / (dp/g)` and not another `g` larger. Mixing was roughly 9.8x
too strong.

This closes the former "Problem 3", the moist subcloud layer. A ten-day
continuation with only this correction changes RH at 685/750 hPa from 99.7/97.1%
to 70.8/68.2% and the lowest level from 98.4% to 83.5%. In the settled
`diffusionfix` checkpoint the subcloud layer runs 65-84%, against 90-93% before
and 75-85% observed.

Convective downdrafts had been implemented for this problem
(`mf_downdraft_fraction`) and do work -- the draft arrives about 5 K colder and
2 g/kg drier and CAPE falls ~15% -- but they were never the fix: a downdraft
brings down air that is colder *and* drier, so `q` and `qs(T)` fall together and
RH is roughly preserved. Downdrafts act on moist static energy, not humidity.

Covered by an independent two-layer backward-Euler exchange test against the
physical mixing rate, and a moist-transport test for column enthalpy and
total-water conservation.

### Parcel ascent mixed a moist lapse rate with a latent increment

The ascent used by both mass-flux convection and dilute CAPE applied a moist
lapse rate *and* an additional latent-temperature increment from condensation,
double-counting the phase change. It now applies exact dry Poisson pressure work
followed by an implicit saturation solve conserving `cp*T + Lv*q`. This is a
finite-step pressure-work/saturation split, not a new closure. Analytical
dry-ascent, small-step moist-lapse, saturation and enthalpy tests pass.

The CAPE default internal pressure interval drops from 2500 to 1000 Pa: the
10/20/40-level spread falls from 7.16% to 2.60% after the correction. That is a
grid comparison, not a proof of pressure-step convergence.

### Partial condensation solves at fixed moist enthalpy

`scm/condensation.py`. Total water is partitioned at fixed moist enthalpy and a
signed condensate increment is passed to microphysics; the path owns
evaporation, so the separate grid-mean evaporation rule is bypassed for it.
Isolated-layer tests cover repeated calls, pre-existing condensate, evaporation
and precipitation.

### Dry convective adjustment

`scm/dry_adjustment.py`, new. The reference previously carried a layer at
910/945 hPa with a lapse rate of 38 K/km, about four times dry adiabatic, theta_v
decreasing 9 K upward. Boundary-layer mixing stopped at a fixed `bl_top_sigma`
and the convection schemes drew from their own prescribed source layers, so that
interface had no flux coupling and nothing could remove it. The adjustment
conserves column enthalpy exactly and water to float32 roundoff, and its trigger
is a lapse-rate excess in K/km so it carries across vertical grids.

### Well-mixed gas optical depth

`scm/radiation_schemes/multiband.py`. CO2 and trace gases were divided by
`nlevels` rather than weighted by layer mass, so the 5 hPa bottom layer received
ten times its share and the radiative answer depended on how the levels were
cut. Column totals are preserved, so the tuned band coefficients keep their
meaning. Grid-shape sensitivity fell from 1.73 to 0.79 W/m2. Covered by
`scm/test_wellmixed_gas_weighting.py`, which fails on the old code.

### Compensating-subsidence drying

`scm/convection_mf.py`. The scheme applied subsidence *warming* with no matching
moisture term. In Zhang-McFarlane both exist and are driven by the same mass
flux. Adding it dried the mid-troposphere from a flat 70% to 36-50%.

### Diagnosed boundary-layer depth

`bl_diagnose_depth = true` replaces the fixed `bl_top_sigma` cutoff with a bulk
Richardson diagnosis, as KPP, Mellor-Yamada and EDMF all do. Dry-adjustment
activity fell substantially, but **not to zero, and the earlier claim in this
document that it did was wrong**. Measured under `atm407.toml` with
`bl_diagnose_depth = true`, mean dry-adjustment heating peaks at -14.38 K/day at
987.5 hPa (`outputs/column/diagnostics/atm407_current_audit_2day.json`). The
September 6 audit above is correct that the surface/mixing/adjustment coupling
remains load-bearing. The adjustment is smaller than it was, not a backstop. It also incidentally fixed shallow convection, which
had been entirely suppressed by an RH trigger the subcloud layer never reached.
**This is a code default of `False`**, so any config that does not inherit
`atm407.toml` silently reverts to the broken behaviour.

### CO2 forcing calibration

A doubling forced 2.12 W/m2; now 3.708. The term multiplies `log(co2/co2_ref)`
and the control runs at `co2 = co2_ref`, so this provably cannot shift the
equilibrium, only the forcing.

### Config loader silently dropped keys

Any `[mass_flux]` key not explicitly listed in `scm/configuration.py` was
discarded without warning, so settings that worked when passed directly had no
effect when written into a config. Downdraft, rain-evaporation,
subsidence-drying and plume parameters are now mapped.

### Energy budget accounting

The budget reporter was omitting dry convective adjustment; it is now included.
A frequent-sampling budget gives TOA +0.482 W/m2, surface -0.052, primary
residual -0.00833, MSE residual +0.00767. The directly calculated
potential-energy tendency is -0.01569 W/m2, and adding it to the MSE-minus-primary
difference leaves 0.00031 W/m2. The earlier 1.45 W/m2 sampled MSE figure was a
sampling artefact, not a sustained energy leak. Results:
`rh095_diffusionfix_settled_budget_10day.json`. Zero-filled optional clear-sky
and TKE entries in those reports are unavailable diagnostics, not measurements.

---

## Practical notes

- Test suite: `python -m pytest scm/ benchmarks/ -q`, 145 passing plus 1 expected-fail (~4 min).
  The `gcm` conda
  environment has torch and pytest; the base environment has neither. Use
  `~/miniconda3/envs/gcm/bin/python`.
- Reference regeneration on a 5 m slab: mass flux runs 400 model days in about
  75 minutes, Betts-Miller the same 400 days in 18. Mass flux is the expensive
  one because the implicit saturation solve in the parcel ascent is called twice
  per step by its closure. Budget accordingly -- the old "600 days in 9 minutes"
  note in earlier versions of this document was a Betts-Miller timing and does
  not apply to the production configuration. A 50 m slab needs of order 1000+
  days to converge, a 5 m slab roughly a tenth of that, which is the fast way to
  compare configurations. Promote to 50 m only at the end.
- Beware judging a configuration from a short run. Several wrong conclusions in
  this project came from comparing columns still in transit. Check that TOA
  imbalance is actually shrinking before drawing conclusions -- and check the
  surface flux separately, since a column can look balanced at the top while the
  slab is still gaining.
- **Check a checkpoint against the current code before trusting its metadata.**
  Metadata records the state at generation. After the cloud-radiation and
  diffusion fixes, several saved "equilibria" are no longer equilibria. One
  physics step from rest, comparing TOA and surface flux against the recorded
  values, takes seconds and is worth doing every time.
- New candidate configs should inherit `atm407.toml`, not `default.toml`. The
  loader merges a user config over `default.toml` only
  (`scm/configuration.py:26`), so anything layered directly on the default
  silently drops `bl_diagnose_depth`, the CO2 calibration and the cloud
  shortwave tuning.
- `matplotlib-base` and `ipython` are installed in the `gcm` environment, so
  both notebooks can be executed and verified locally rather than shipped
  unrun. There is no `pip` in that environment; use
  `~/miniconda3/bin/conda install -n gcm -c conda-forge <package>`.
- Verify a notebook by exec'ing its code cells in order under the `Agg`
  backend. Both notebooks pass that way: 36 cells in `01_meet_the_column`,
  20 code cells in `02_experiments`. Notebook 2 needs `IPython` for its
  animation, and without it thirteen cells fail on one cascading import.
