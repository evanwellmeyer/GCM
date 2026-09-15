# ATM407 reference data

`atm407_equilibrium_20level.npz` is the equilibrium checkpoint the notebooks
start from. It was made by `scripts/make_atm407_reference.py`. The matching JSON
file records the configuration, the spin-up, the diagnostics, and the runtime.

The current reference uses `scm/configs/atm407.toml`: boundary-layer mixing
`k_diff = 40`, a deep-convection plume that stops at its level of neutral buoyancy,
no instant condensation rain-out (`cloud_ls_precip_fraction = 0`), and a closure
CAPE parcel that mixes ten times faster than the plume
(`cape_entrainment_rate = 5e-5`). The last two were promoted on 12 Sep 2026 after the
BOMEX observed-balance test; it ran 400 days on a 5 m slab from the previous
reference, which is kept in `outputs/atm407_tests/`. The slab speeds up the surface
response without changing the equilibrium. 50-day means: surface temperature 289.05 K,
TOA +0.75 W m-2, surface +0.00 W m-2, CAPE 183 J kg-1, cloud water path
0.338 kg m-2, and a surface-temperature drift of 0.0001 K over the window. It passes
the model's equilibrium check. Known limitations: relative humidity near 0.99 at
865-910 hPa, a cloud deck at the top of the boundary layer, a low convective top, and
a cloud layer that nothing ventilates. See `docs/column_open_problems.md`.

The reference is on the native 20-level grid. The notebooks can interpolate it
to another grid. That gives a balanced first guess, not an exact equilibrium, so
a short adjustment run follows.

Older development checkpoints were removed on 10 Sep 2026. Git history keeps
them. Once that removal is committed, `git log --diff-filter=D --name-only --
notebooks/data` lists them.

Suffixed sensitivity checkpoints are local generated results and are ignored by
Git. Keep them for comparison; only the unsuffixed NPZ/JSON pair is distributed
with the notebooks. Source configurations and regression fixtures remain tracked.
