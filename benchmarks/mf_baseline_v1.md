# MF Baseline v1

This is the first stabilized single-member mass-flux baseline after the energy-closure,
cloud-radiation, shallow-convection, and slab-accumulator fixes.

## Configuration

- Config: `scm/configs/mf_baseline_v1.toml`
- Command:

```bash
python -m scm.run_scm --config scm/configs/mf_baseline_v1.toml --device cpu --no-plot
```

## Reference Result

From `scm_full_mf_fixed_slabocean_spin2000d_pert8000d_results.pt`:

### 1xCO2

- Ts: `290.58 K`
- ASR: `246.2 W/m2`
- OLR: `245.2 W/m2`
- TOA net: `+1.02 W/m2`
- Surface total: `+0.13 W/m2`
- Column residual: `+0.88 W/m2`
- Precipitation: `2.51 mm/day`
- Late Ts drift: `0.027 K/window`
- Equilibrium check: `NOT CONVERGED`

### 2xCO2

- Ts: `293.83 K`
- ASR: `248.8 W/m2`
- OLR: `247.8 W/m2`
- TOA net: `+0.98 W/m2`
- Surface total: `+0.08 W/m2`
- Column residual: `+0.89 W/m2`
- Precipitation: `2.75 mm/day`
- Late Ts drift: `0.016 K/window`
- Equilibrium check: `PASS`

### Sensitivity

- ECS: `3.26 K`
- Delta precipitation: `0.244 mm/day`
- Hydrological sensitivity: `2.99 %/K`

## Notes

- The `1xCO2` control is very close to equilibrium but still misses the strict TOA criterion by a small margin.
- The `2xCO2` branch is the current reference response for the fixed-parameter mass-flux baseline.
- Future physics changes should be compared against this baseline before changing the reference config.
