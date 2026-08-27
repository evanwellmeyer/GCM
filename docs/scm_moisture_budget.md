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
