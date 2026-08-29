# SCM boundary-layer case benchmarks

These cases separate boundary-layer and shallow-convection behavior from the radiative-convective equilibrium column.

The dry mixed-layer case starts with a neutral 300 K layer below 1 km, a 2 K inversion, and a 100 W m-2 prescribed surface sensible heat flux. It measures boundary-layer growth, mixed-layer uniformity, and column-energy conservation after six hours.

The BOMEX case follows the standard nonprecipitating trade-cumulus setup used by the CLUBB single-column model. It uses the published piecewise sounding, a surface potential-temperature flux of 0.008 K m s-1, a surface total-water flux of 5.2e-5 kg kg-1 m s-1, subsidence peaking at -0.0065 m s-1 near 1.5 km, lower-tropospheric radiative cooling of 2 K day-1, and prescribed low-level drying. The implementation was transcribed from the official CLUBB `bomex_sounding.in`, `bomex_sfc.in`, and `src/Benchmark_cases/bomex.F90` definitions.

Run the cases with:

```bash
python scripts/run_scm_case_benchmarks.py
```

The output compares 20, 40, and 80 full-atmosphere levels. For BOMEX it runs both boundary-layer-only and boundary-layer-plus-shallow-convection configurations.

The first baseline result shows that the current Richardson diffusion does not converge in the dry mixed-layer case: diagnosed depth grows from about 1.55 km at 20 levels to 2.33 km at 80 levels, while the potential-temperature spread also increases. In BOMEX, shallow convection slightly reduces diagnosed boundary-layer depth but raises maximum cloud-layer relative humidity toward saturation. These results identify boundary-layer transport as the primary development target and the shallow closure as a secondary target.

References:

- Siebesma et al. (2003), *A Large Eddy Simulation Intercomparison Study of Shallow Cumulus Convection*.
- The official [CLUBB repository](https://github.com/larson-group/clubb_release), BOMEX benchmark definitions.
