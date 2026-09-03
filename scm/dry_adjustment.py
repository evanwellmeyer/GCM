# dry convective adjustment.
#
# the boundary layer scheme mixes only inside a prescribed sigma depth
# (bl_top_sigma), so a statically unstable interface sitting just above that
# depth cannot be removed by any other process: turbulence is not allowed to
# reach it, and the convection schemes draw from their own prescribed source
# layers. that leaves the two sides of the interface with no flux coupling at
# all, so their energy budgets equilibrate independently and the gap between
# them is unconstrained.
#
# this relaxes contiguous statically unstable layers toward a dry-neutral
# (uniform potential temperature) profile. it acts only where theta_v
# decreases upward, so a stably stratified column is left untouched.
#
# the mixing is flux-form and implicit, so it is unconditionally stable and
# conserves the column integrals exactly. potential temperature is mixed with
# exner-weighted layer mass, which is what makes uniform-theta mixing conserve
# cp*T rather than T; water species are mixed with plain layer mass.

import torch
from scm.boundary_layer import tridiag_solve
from scm.thermo import g, kappa, p0


def _mix_implicit(field, layer_mass, exchange, timestep):
    """implicit flux-form mixing that conserves sum(field * layer_mass)."""

    nlevels = field.shape[1]
    a = torch.zeros_like(field)
    b = torch.ones_like(field)
    c = torch.zeros_like(field)

    coeff_upper = timestep * exchange / layer_mass[:, :-1]
    coeff_lower = timestep * exchange / layer_mass[:, 1:]

    c[:, :-1] = -coeff_upper
    b[:, :-1] = b[:, :-1] + coeff_upper
    a[:, 1:] = -coeff_lower
    b[:, 1:] = b[:, 1:] + coeff_lower

    return tridiag_solve(a, b, c, field, 0, nlevels)


def dry_adjustment(state, grid, params):
    """relax statically unstable layers toward a dry-neutral profile."""

    del grid

    t = state['t']
    q = state['q']
    qc = state.get('qc', torch.zeros_like(q))
    p = state['p']
    dp = state['dp']

    timestep = params.get('dt', 900.0)
    tau = float(params.get('dry_adjustment_tau', 1800.0))
    # this is a safety net for pathological layers, not a convection scheme.
    # a moist-convecting column carries theta_v wiggles of a few tenths of a
    # kelvin between levels, and neutralizing those sets off a mixing cascade:
    # each neutralized interface warms the layer above it and destabilizes the
    # next one up, which ends with the whole lower troposphere mixed to a
    # dry-neutral slab. the tolerance keeps the scheme off that structure and
    # aimed only at genuinely unphysical layers. it is a per-interface value,
    # so it is tied to the level spacing of the grid it was tuned on.
    tolerance = float(params.get('dry_adjustment_tolerance', 1.0))

    zero = torch.zeros_like(t)
    if tau <= 0.0:
        return {'dt': zero, 'dq': zero, 'dqc': torch.zeros_like(qc)}

    exner = (p / p0).clamp(min=1.0e-6) ** kappa
    theta = t / exner
    theta_v = theta * (1.0 + 0.608 * q.clamp(min=0.0) - qc.clamp(min=0.0))

    # interface i couples level i (above) with level i+1 (below). the column is
    # statically unstable there when theta_v decreases upward.
    deficit = theta_v[:, 1:] - theta_v[:, :-1]
    unstable = (deficit > tolerance).to(t.dtype)

    mass = dp / g
    # exner-weighted mass: mixing theta against these weights conserves cp*T.
    heat_mass = mass * exner

    heat_exchange = unstable * torch.minimum(
        heat_mass[:, :-1], heat_mass[:, 1:]
    ) / tau
    water_exchange = unstable * torch.minimum(mass[:, :-1], mass[:, 1:]) / tau

    active = unstable.any(dim=1, keepdim=True)
    if not bool(active.any()):
        return {
            'dt': zero,
            'dq': zero,
            'dqc': torch.zeros_like(qc),
            'dry_adjustment_active': active.squeeze(1).to(t.dtype),
        }

    theta_mixed = _mix_implicit(theta, heat_mass, heat_exchange, timestep)
    q_mixed = _mix_implicit(q, mass, water_exchange, timestep)
    qc_mixed = _mix_implicit(qc, mass, water_exchange, timestep)

    t_new = theta_mixed * exner

    # a column with no unstable interface is left bit-for-bit alone: without
    # this mask the theta round trip through exner leaves a small float
    # residue every step even when nothing mixes.
    return {
        'dt': torch.where(active, (t_new - t) / timestep, zero),
        'dq': torch.where(active, (q_mixed - q) / timestep, zero),
        'dqc': torch.where(active, (qc_mixed - qc) / timestep, torch.zeros_like(qc)),
        'dry_adjustment_active': active.squeeze(1).to(t.dtype),
    }
