# Column model: active paths and experiments

Checked 15 September 2026. This separates teaching use from ongoing development; it is not a claim that either column has no remaining physical deficiencies.

## Teaching path

`default.toml` now extends `atm407.toml`: the generic default and teaching configuration are identical. `legacy_base.toml` only preserves inheritance for older explicit experiments. No alternate physics was promoted during this unification.

Both `notebooks/01_meet_the_column.ipynb` and `notebooks/02_experiments.ipynb` explicitly load `scm/configs/atm407.toml` and `notebooks/data/atm407_equilibrium_20level.npz`. The paired JSON records the September 12 reference: 20 levels, 900 s, a 5 m equilibration slab, 289.05 K, and passing recorded equilibrium gates. Some notebook experiments override the slab depth and other parameters; the TOML's 50 m depth is not the depth used to generate that checkpoint.

| Component | Main implementation |
|---|---|
| Orchestration and process order | `scm/column_model.py` |
| Scheme selection | `scm/physics_suites.py` |
| Config merging and parameter mapping | `scm/configuration.py` |
| Richardson boundary layer | `scm/boundary_layer.py` |
| Mass-flux deep convection | `scm/convection_mf.py` |
| Simple shallow convection | `scm/convection_shallow.py` |
| Condensation and water partition | `scm/condensation.py`, `scm/phase_partition.py` |
| Radiation dispatch and implementations | `scm/radiation.py`, `scm/radiation_schemes/` |
| Shared surface, thermodynamics and cloud code | `scm/surface.py`, `scm/thermo.py`, `scm/cloud_microphysics.py`, `scm/cloud_optics.py` |

This path uses clear-sky radiative effects in the baseline, Richardson mixing, and separate mass-flux deep convection. It does not use the experimental UW shallow plume. Its bulk equilibrium does not validate its remaining cloud-layer structure.

## Current experimental path

`scm/configs/uw_candidate_v1.toml` selects `scm/boundary_layer_uw.py` and `scm/convection_uw.py`. The latter now has an optional internally substepped wrapper and the original `shallow_step` operator. The launch-layer drying under investigation is in this path.

The working diagnostic chain is:

1. `scripts/audit_bomex_launch_budget.py`: six-hour instrumented BOMEX state/flux archive; includes explicit parameter overrides.
2. `scripts/check_bomex_launch_flux.py`: frozen-state interface/source-cap audit.
3. `scripts/test_bomex_launch_closure.py`: diagnostic Gaussian launch substitution; not production physics.
4. `scripts/check_bomex_inversion_substeps.py`: short evolving-state comparison and internal-substep verification.
5. `scripts/check_uw_conserved_environment.py`: saved-state environment/sorting-closure tests.
6. `scripts/screen_uw_conserved_environment.py`: coupled 20-level screening; use `--diagnose-sorting` to include the newer cloud-depth coupling. A one-hour result is not six-hour validation.

Results are under `outputs/column/diagnostics/`. Current interpretation is in `docs/column_open_problems.md`; detailed evidence is in `docs/column_bomex_transport_audit_2026-09-14.md`. These JSON archives, rather than the notebook checkpoint, supply current BOMEX replay inputs.

## Older does not mean unused

`boundary_layer_tke_v2.py`, `boundary_layer_edmf_v3.py`, `shallow_plume_v2.py`, and `convection_bm.py` remain imported by the physics registry. In addition, the UW plume imports thermodynamic partition helpers from `shallow_plume_v2.py`. Deleting that apparently older file would break the current UW implementation.

The other trace, fitting, grid-comparison and screening scripts are retained research utilities, not notebook entry points. Their historical outputs are not automatically evidence for current code. Alternative checkpoint pairs in `notebooks/data/` are experiments; only the unsuffixed pair is selected by the notebooks.

No scientific files have been removed in this organization pass. The checkout contains substantial uncommitted work from earlier development; preserving it is safer than guessing which experiments can be discarded. [Configuration inventory](../scm/configs/README.md) lists every current TOML preset and its intended role.
