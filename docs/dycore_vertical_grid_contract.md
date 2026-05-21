# Dycore Vertical Grid Contract

The coupled model should treat the dycore as the owner of vertical geometry.
The SCM owns column physics, but it should not invent pressure levels when it is
being called from a host model.

## Ownership

- The dycore owns `p_interface`, `p_full`, `dp`, `ps`, model-top pressure,
  vertical indexing, and any hybrid-coordinate coefficients.
- The SCM owns radiation, clouds, convection, condensation, boundary-layer
  mixing, surface flux tendencies, and diagnostics on the supplied column grid.
- The adapter/coupler owns unit conversions, sign conventions, and any temporary
  remapping needed while interfaces are still changing.

## SCM Modes

- Standalone SCM runs may call `make_grid(...)`. This preserves the historical
  sigma-like grid and remains useful for calibration and component tests.
- Coupled runs should pass a dycore-defined grid using either
  `grid_from_hybrid_coefficients(A, B)` or
  `grid_from_pressure_interfaces(p_interface)`.

## Required Conventions

- Vertical index 0 is model top; the last interface is the surface.
- Pressure interfaces and full levels are ordered top to bottom.
- `dp` is positive layer pressure thickness in Pa.
- Full-level atmospheric fields use shape `(batch, nlevels)`.
- Surface fields use shape `(batch,)` or `(batch, 1)` at the adapter boundary.

## Near-Term Refactor Path

1. Keep standalone behavior bitwise close to the old sigma-grid path.
2. Route all pressure and layer-mass calculations through the grid helpers in
   `scm.thermo`.
3. Replace direct `grid["sigma_full"]` indexing inside physics with helper
   calls that can return per-column coordinates.
4. Let the VFS adapter pass the dycore grid directly once the tendency and
   diagnostic conventions are fully tested.
