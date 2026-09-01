# PyTorch physics-suite plan

The accepted `mf_response_v3` configuration remains the production control while more complete parameterizations are developed. New schemes will be independent PyTorch implementations of published formulations rather than direct translations of Fortran source. Original model code may be used to identify expected inputs, outputs, constants, limiters, and reference behavior, subject to its license, but the equations and scientific documentation are the implementation specification.

The first shared component is `scm/physics_grid.py`. It refines host pressure layers below a configurable vertical-coordinate threshold, maps layer means to the internal grid, and maps tendencies back to the host by exact pressure overlap. The map conserves the pressure integral of temperature-like, water-like, momentum-like, and tendency fields. Refinement is opt-in through `[physics_grid]`; the default configuration does not enable it and therefore retains its existing numerical path.

The first UW moist-turbulence increment is registered as `uw_moist`. It implements the surface-connected convective layer and local stable-turbulence paths using total water, liquid moist static energy, diagnostic TKE, Galperin stability functions, the UW asymptotic mixing length, and dry convective-top entrainment. It also returns heat, water, condensate, and momentum tendencies. Elevated convective layers, moist enhancement of cloud-top entrainment, and radiatively driven cloud-top layers are not yet included, so `uw_moist` remains experimental.

In a six-hour 100 W m-2 dry mixed-layer test, `uw_moist` produced boundary-layer depths of 1.286, 1.304, and 1.319 km on 20, 40, and 80 host levels. Mixed-layer potential-temperature spreads were 0.33, 0.46, and 0.53 K, and energy errors were -0.006 W m-2. In six-hour boundary-layer-only BOMEX screens, diagnosed depths were 0.890, 0.934, and 0.969 km and cloud-water paths were 0.052, 0.049, and 0.076 kg m-2. These results pass the initial boundedness, conservation, and host-resolution gates but do not validate the complete UW suite because shallow convection is not yet active.

The UW shallow-convection implementation now includes independently tested PyTorch versions of its height-dependent lateral mixing rate, explicit CIN mass-flux closure, implicit long-timestep CIN correction, vertically substepped entraining plume, buoyancy-based mixing acceptance and detrainment, conservative heat, total-water, and momentum flux divergence, condensate detrainment, a 1 g kg-1 plume-condensate ceiling, precipitation evaporation, and cloud-area diagnosis. It is registered as `uw_shallow` but remains experimental.

At the production 900 s physics timestep, six-hour combined UW BOMEX screens produced cloud-water paths of 0.039 and 0.041 kg m-2, maximum cloud fractions of 3.2% and 6.0%, cloud-base mass fluxes of 0.0475 and 0.0457 kg m-2 s-1, and boundary-layer depths of 0.856 and 0.751 km at 20 and 40 levels. One-step water residuals are below 2e-8 kg m-2 s-1 and energy residuals below 0.1 W m-2. These gates permit a two-day, 20-level fixed-SST ATM407 screen using `scm/configs/uw_candidate_v1.toml`; they do not permit an equilibrium run or replacement of the production default.

The first canonical replacement suite will pair the University of Washington moist-turbulence and shallow-convection schemes. The published formulation transports total water and liquid-water potential temperature, treats stable and convective turbulent layers in one framework, diagnoses TKE for diffusivity, includes explicit entrainment at convective-layer boundaries, and uses a CIN-based shallow-plume closure designed for long climate-model timesteps. Implementation will proceed component by component, with the UW paper equations and controlled reference behavior recorded beside each test.

The next canonical suite will follow the GFDL AM4 double-plume structure, keeping separately closed shallow and deep plumes rather than blending them into the UW implementation. A general TKE-EDMF suite will remain a third alternative. An assumed-PDF cloud macrophysics module will then provide a shared lower-order PDF option, followed by a reduced CLUBB suite with prognostic moments. A full CLUBB-like hierarchy will only be expanded after the reduced suite passes the same cases.

The schemes will use the existing `run_physics_scheme` registry and return host-grid tendencies through a common conservation boundary. A scheme may own additional state such as TKE, scalar variance, covariance, or vertical-velocity moments, but optional state must be explicit in restart metadata. Canonical suites will be validated before arbitrary cross-scheme combinations are admitted to PPEs.

Promotion gates are fixed before ATM407 tuning. Turbulence must pass a dry convective boundary layer and GABLS. Shallow-cloud coupling must pass BOMEX, RICO, DYCOMS, and ARM cases using profile, turbulent-flux, cloud-fraction, liquid-water-path, and precipitation targets. Deep convection must pass a separately forced deep case. Every component must preserve water and moist-energy budgets and show acceptable timestep and host-grid convergence. Only a suite that passes its component cases will receive one 20-level slab-ocean ATM407 equilibrium run.

The implementation order is:

1. Conservative physics grid and shared remapping.
2. UW moist turbulence.
3. UW shallow convection and the complete UW suite.
4. GFDL-style shallow and deep double plumes.
5. General TKE-EDMF.
6. Assumed-PDF cloud macrophysics.
7. Reduced CLUBB and additional higher-order moments.
8. Canonical-suite comparison, PPEs, and coupled-column promotion.

Primary scientific references are Bretherton and Park (2009), Park and Bretherton (2009), Bretherton, McCaa, and Grenier (2004), the GFDL AM4 model description, and the CLUBB scientific documentation. Their documented benchmark cases and the official public implementations provide behavioral comparisons; they do not replace repository-owned conservation and regression tests.
