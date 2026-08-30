# SCM moisture-budget diagnosis

The accepted 20-level SCM state developed nearly saturated conditions from the
surface through the lower and middle troposphere. A process-resolved budget was
constructed with `scripts/diagnose_atm407_budget.py` to separate radiation,
surface exchange, boundary-layer mixing, shallow convection, deep convection,
large-scale condensation, and cloud microphysics.

The original budget showed that boundary-layer diffusion transported surface
moisture as high as sigma 0.72. Shallow convection covered the same lower-
tropospheric region and could only moisten its upper layer. Large-scale
condensation removed almost exactly the combined moisture supplied by those two
schemes. Deep convection dried the lowest levels but had little influence in
the saturated layer because buoyant-plume detrainment was nearly disabled.

The corrected development configuration is `scm/configs/mf_response_v3.toml`.
It makes four structural changes relative to the accepted configuration:

- the closure measures the CAPE response to a trial plume;
- the boundary-layer top is sigma 0.94 rather than sigma 0.72;
- shallow convection relaxes toward an 85% detrainment RH from either side;
- plume detrainment remains active while the plume is buoyant.

These changes now define the accepted standalone default and checkpoint. The
response-only `mf_response_v2.toml` is retained to reproduce the intermediate
800-day checkpoint used to diagnose the remaining saturation. Coupled-model
results produced with the previous default require rerunning before comparison.

## Twenty-level slab equilibrium

After 200 adjustment days from the response-only checkpoint, the corrected
configuration produced:

- surface temperature: 284.44 K;
- TOA net flux: +0.56 W m-2;
- surface total flux: +0.31 W m-2;
- final 50-day surface-temperature drift: 0.006 K;
- CAPE: 1434 J kg-1;
- mass at RH greater than or equal to 95%: 3.5%;
- deep-convective rain: 1.67 mm day-1;
- large-scale rain: 0.23 mm day-1;
- cloud rain: 0.01 mm day-1.

The mass-flux, temperature-tendency, and moisture-tendency caps were inactive.
The maximum column-energy residual was 0.084 W m-2, the maximum column-MSE
residual was 0.013 W m-2, and the column-water residual remained negligible.

## Native-grid comparison

Separate 100-day fixed-SST integrations were initialized natively at 10, 20,
and 40 levels. CAPE was 1479, 1452, and 1593 J kg-1. Total precipitation was
1.92, 1.93, and 1.86 mm day-1, while deep-convective precipitation was 1.66,
1.71, and 1.70 mm day-1. The RH95 mass fractions were 7.0%, 3.5%, and 5.5%.
No mass-flux or tendency caps activated, and the absolute atmospheric-energy
residual remained below 0.04 W m-2 at every resolution.

These results support the structural correction and remove the earlier extreme
resolution sensitivity. The configuration has therefore been promoted to the
standalone default; coupled-model validation remains a separate required step.

## Reproduction

Run a process budget with:

```bash
python scripts/diagnose_atm407_budget.py \
  --reference notebooks/data/atm407_equilibrium_20level_mf_response_v3.npz \
  --config scm/configs/mf_response_v3.toml \
  --days 10
```

Run the native-grid comparison with:

```bash
python scripts/compare_atm407_resolutions.py \
  --config scm/configs/mf_response_v3.toml \
  --levels 10 20 40 \
  --days 100 \
  --surface-temperature 284.44
```

## Fractional-optics v20 promotion gate

The `mf_edmf_fractional_optics_v20.toml` configuration substantially improves the 20-level column but does not yet pass the native-grid promotion gate. Independent 50-day continuations at 10, 20, and 40 levels produced CAPE values of 1675, 1043, and 695 J kg-1 and total precipitation rates of 3.67, 3.36, and 3.48 mm day-1. The similar total rain hides a large change in its source: large-scale precipitation was 2.37, 0.66, and 1.25 mm day-1, respectively. Boundary-layer depth also changed from the imposed 4000 m upper limit at 10 levels to 3101 m at 20 levels and 3641 m at 40 levels in one-day continuation diagnostics.

The 20- and 40-level radiative states were reasonably close to balance, but the 10-level state retained a TOA imbalance near -5 W m-2 and a surface imbalance near +5 W m-2. The 10-level clear-sky TOA balance also had the opposite sign from the 20- and 40-level results. All three cases conserved column water and avoided the mass-flux and tendency caps, so the disagreement is not explained by numerical clipping or a water leak.

These results reject v20 as a resolution-converged default. The dominant unresolved behavior is the grid dependence of diagnosed boundary-layer depth, CAPE, condensation, and the partition between deep, large-scale, and cloud precipitation. The current production default and student checkpoint should remain unchanged while that coupling is corrected. The v20 20-level checkpoint remains useful as an experimental diagnostic case, not as a production initial condition.

The reference generator now accepts `--levels`, conservatively remaps total water when a checkpoint is used to seed a different grid, and writes the native level count into the output name and metadata. The budget diagnostic infers its grid from the checkpoint and restores checkpoint TKE, which prevents a hidden turbulence reinitialization during continuation tests.

Code inspection after this gate showed that the reported TKE boundary-layer depth does not control either diffusivity or the shallow plume in v20. Diffusivity is computed locally from TKE and mixing length, while the plume uses its own fixed 2500 m maximum height. Switching the diagnostic to a bulk-Richardson depth therefore left all physical tendencies unchanged and was rejected as a new configuration. The TKE depth diagnostic itself was corrected to interpolate the threshold crossing at the top of the surface-connected turbulent layer and to ignore disconnected TKE aloft, removing its avoidable model-level quantization without claiming an effect on the column physics.

The precipitation budgets locate the remaining resolution dependence downstream of the depth diagnostic. At 10 levels, cloud precipitation is essentially zero and large-scale condensation supplies most of the rain; at 20 and 40 levels, prognostic cloud condensate and cloud precipitation are substantial. The next controlled development target is therefore the grid dependence of condensate production, detrainment, and autoconversion. Any correction must first reproduce a prescribed continuous cloudy profile across 10, 20, and 40 levels while conserving total water and moist energy, before returning to equilibrium tuning.

A controlled one-hour BOMEX calculation showed that the 10-level sigma grid has only three model levels below 2 km and produces a cloud-water path of 0.18 kg m-2. The 20-, 40-, and 80-level grids have 7, 14, and 27 levels below 2 km and produce 0.41, 0.49, and 0.43 kg m-2. The 10-level column therefore does not resolve the boundary-layer and shallow-cloud structure required by this EDMF formulation. It should remain useful for dry dynamics and simplified-physics tests, but it is not an appropriate convergence target for the moist production suite. Moist-physics convergence should be assessed over 20, 40, and 80 levels, while the coupled GCM must use at least the validated 20-level vertical grid unless a distinct coarse-grid closure is developed.

The subsequent TKE audit found that v20 included local shear and buoyancy production and local dissipation but omitted vertical transport of TKE. It also showed that the 100 m diagnosed depth arose because near-wall dissipation kept the lowest cell below the original 0.01 m2 s-2 threshold even while surface buoyancy production was active and TKE increased immediately above it. The experimental `mf_edmf_tke_transport_v21.toml` adds conservative implicit vertical TKE diffusion, records production, dissipation, and transport profiles, and uses a lower 0.001 m2 s-2 diagnostic threshold appropriate to the near-wall cells. The transport switch is opt-in so v20 remains reproducible.

In like-for-like five-day fixed-SST screens, v21 reduced the 20-to-80-level CAPE spread from 230 to 173 J kg-1, the deep-rain spread from 0.42 to 0.27 mm day-1, the large-scale-rain spread from 1.21 to 0.76 mm day-1, and the total-rain spread from 0.96 to 0.65 mm day-1. It also reduced the surface-flux spread from 14.6 to 5.6 W m-2. These are meaningful improvements but not sufficient for promotion: the precipitation partition remains resolution dependent, and a longer screen is required after the TKE budget and boundary-layer depth are verified under v21.

The subsequent 50-day fixed-SST screen did not confirm a broad convergence improvement. From 20 to 80 levels, v21 produced CAPE of 891, 587, and 370 J kg-1; deep rain of 1.12, 0.82, and 0.60 mm day-1; large-scale rain of 0.66, 1.25, and 2.34 mm day-1; cloud rain of 1.42, 1.33, and 0.73 mm day-1; and total rain of 3.21, 3.40, and 3.67 mm day-1. Relative to v20, the CAPE spread decreased by about 7%, the deep- and cloud-rain spreads by about 6-7%, and the surface-flux spread by about 19%, but the large-scale-rain spread increased slightly and the total-rain spread increased from 0.32 to 0.46 mm day-1. Diagnosed turbulent depth was 3546, 2752, and 503 m, demonstrating that the absolute TKE threshold still does not define a resolution-convergent layer top. Vertical TKE transport is retained as an experimental, conservative structural option, but v21 fails the promotion gate and should not replace the default or initialize long slab-ocean integrations.
