# Which SCM configuration?

Status checked 15 September 2026. A configuration's presence is not evidence that it passed validation.

| Use | Configuration | Status |
|---|---|---|
| Current teaching notebooks and their reference | `atm407.toml` | Teaching baseline; known cloud-layer limitations |
| Generic CLI with no configuration supplied | `default.toml` | Alias of `atm407.toml`; identical resolved configuration |
| Compatibility base | `legacy_base.toml` | Preserves old experiment inheritance; not a recommended run configuration |
| Current UW turbulence/shallow development | `uw_candidate_v1.toml` | Experimental; BOMEX moisture profile still fails screening |
| UW turbulence without shallow plume | `uw_turbulence_only_v1.toml` | Component experiment, not a replacement teaching baseline |
| Previous mass-flux experiments | `atm407_flux_v1.toml`, `mf_baseline_v1.toml`, `mf_response_v2.toml`, `mf_response_v3.toml`, `mf_flowdev_v1.toml` | Historical alternatives; do not pair with the canonical notebook checkpoint |
| Benchmark runners | `benchmark_suite.toml`, `benchmark_flowdev_v1.toml` | Test setups, not equilibrium initial conditions |
| Examples and calibration | `clouds_example.toml`, `large_scale_forcing_example.toml`, `trace_gases_example.toml`, `radiative_adjustment_example.toml`, `radiation_calibration.toml`, `simplified_physics.toml` | Purpose-specific examples/calibration |

The generic default and teaching setup now resolve identically. `default.toml` extends `atm407.toml`, which is the single maintained baseline. ATM407 is the course name, not a separate physics model. The reference generator also selects it when `--config` is omitted. This does not promote the experimental UW schemes.

`load_run_config()` resolves `default.toml`. Explicit files retain the pre-unification compatibility base (`legacy_base.toml`) plus their overrides and optional `extends` parents, so existing experiments do not silently acquire promoted teaching parameters. New configurations intended to modify the supported baseline should explicitly use `extends = "atm407.toml"`. Code-level defaults and runner overrides can add further settings. In particular, the archived BOMEX audit uses `uw_candidate_v1.toml` **plus overrides in `scripts/audit_bomex_launch_budget.py`**; saved `plume_params` in its JSON are the reproducible replay inputs, not the TOML alone.

The optional `params.uw_shallow_maximum_timestep_s = 225.0` enables refined shallow time integration. It is not enabled in the existing configs. The Gaussian launch closure is still a diagnostic replacement, not a selectable production configuration.

Historical files retain their names because scripts, tests and archived commands refer to them. Do not delete or rename them merely because a newer numbered experiment exists. Old names such as `mf_profile_v4` in the development log are historical references, not available current presets.

These alternatives are not immutable snapshots: for example, `atm407_flux_v1.toml` extends the current `atm407.toml`. Reproducing an old run requires its recorded code/config revision and effective parameters, not just the old filename.

See [the code map](../../docs/column_code_map.md) and [current scientific problems](../../docs/column_open_problems.md).
