# ATM407 reference data

`atm407_equilibrium_20level.npz` is the equilibrium checkpoint the notebooks
start from. It was made by `scripts/make_atm407_reference.py`. The matching JSON
file records the configuration, the spin-up, the diagnostics, and the runtime.

The current reference uses `scm/configs/atm407.toml` (label `atm407_plume_stop`):
boundary-layer mixing `k_diff = 40`, and a deep-convection plume that stops at its
level of neutral buoyancy. It ran 800 days on a 5 m slab after the stop was
switched on. The slab speeds up the surface response without changing the
equilibrium. 50-day means: surface temperature 289.0 K, TOA +0.46 W m-2,
surface -0.23 W m-2, CAPE 566 J kg-1, and a surface-temperature drift of
0.035 K over the window. It passes the model's equilibrium check. Known
limitations: relative humidity near 0.99 at 865-910 hPa, a cloud deck at the top
of the boundary layer, and a low convective top. See `docs/column_open_problems.md`.

The reference is on the native 20-level grid. The notebooks can interpolate it
to another grid. That gives a balanced first guess, not an exact equilibrium, so
a short adjustment run follows.

Older development checkpoints were removed on 10 Sep 2026. Git history keeps
them. Once that removal is committed, `git log --diff-filter=D --name-only --
notebooks/data` lists them.
