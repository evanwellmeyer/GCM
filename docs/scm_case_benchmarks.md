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

The isolated one-equation TKE experiment uses flux-form pressure-coordinate transport, local buoyancy and shear production, dissipation, and a height-limited mixing length. It is not yet registered in the production physics suite. In the six-hour dry case, its diagnosed depth is 1.20, 1.25, and 1.22 km at 20, 40, and 80 levels. The mixed-layer potential-temperature spread is 0.64, 0.74, and 0.83 K. In BOMEX, its boundary-layer-only depth is 0.91, 0.96, and 0.97 km. These are substantial convergence improvements over the Richardson baseline, although the moist case still produces supersaturation at the finer resolutions and therefore needs a cloud-aware shallow-plume treatment before coupled integration.

The cloud-aware plume transports total water and moist static energy conservatively while using an entraining liquid-water-potential-temperature plume to diagnose buoyancy and condensate. Combined with TKE, its six-hour BOMEX boundary-layer depths are 0.99, 1.02, and 1.00 km at 20, 40, and 80 levels. Maximum cloud-layer relative humidity is 0.93, 1.00, and 1.00, replacing the 1.12-1.14 supersaturation produced by TKE with the legacy shallow adjustment.

Fractional cloud is diagnosed from plume area and environmental condensate rather than treating grid-mean saturation as complete cloud cover. Plume area is mass flux divided by plume density and vertical velocity. Environmental cloud fraction is grid-mean condensate divided by a representative in-cloud condensate of 3 g kg-1. After six hours, maximum cloud fraction is approximately 10%, 10%, and 9% at 20, 40, and 80 levels, consistent with the low cloud cover expected in BOMEX and substantially more convergent than binary grid-mean cloud.

The combined TKE, plume, and fractional-cloud package is registered only as the experimental `mf_tke_plume_cloud_v10` configuration. Its first three-day continuation from the v9 RCE checkpoint failed the coupled-column acceptance screen: boundary-layer depth reached its 1.5 km ceiling, RH above 95% increased to 26% of atmospheric mass, surface flux reached -48 W m-2, and precipitation shifted strongly toward large-scale condensation and cloud autoconversion. This demonstrates that controlled-case convergence is necessary but not sufficient.

The next experiment replaced the fixed 0.03 kg m-2 s-1 plume mass flux with the established EDMF structure used by Han and Bretherton: cloud-base velocity is tied to boundary-layer TKE, plume area multiplies density and velocity to give mass flux, entrainment varies with distance from the surface and plume top, and vertical velocity responds to entrainment, drag, and buoyancy. Each model layer is internally substepped at 50 m to keep the plume integration from changing merely because the host grid is coarse. Surface heat and moisture fluxes set the plume-source anomalies, so the closure responds to the surface state rather than prescribing the same plume in every column.

In the one-hour BOMEX test, diagnosed cloud-base mass flux is 0.015-0.025 kg m-2 s-1 across 20, 40, and 80 levels and remains bounded without supersaturation. Boundary-layer depth and cloud condensate do not yet converge, however. A three-day RCE transfer is substantially worse than the fixed-flux experiment: surface flux reaches -85 W m-2, cloud precipitation reaches 6.5 mm day-1, and column water and moist-static-energy residuals become large. The likely cause is that local TKE diffusion and nonlocal plume transport are currently applied as separate complete transports rather than as a single partitioned EDMF flux. The responsive closure is therefore retained as research code only. Production remains on `mf_profile_v4`; neither the default configuration nor the production checkpoint should be changed.

References:

- Siebesma et al. (2003), *A Large Eddy Simulation Intercomparison Study of Shallow Cumulus Convection*.
- Han and Bretherton (2019), *TKE-Based Moist Eddy-Diffusivity Mass-Flux (EDMF) Parameterization for Vertical Turbulent Mixing*.
- Olson et al. (2019), *A Description of the MYNN-EDMF Scheme and the Coupling to Other Components in WRF-ARW*.
- The official [CLUBB repository](https://github.com/larson-group/clubb_release), BOMEX benchmark definitions.
- The official [MYNN-EDMF repository](https://github.com/NCAR/MYNN-EDMF), reference implementation and cases.
