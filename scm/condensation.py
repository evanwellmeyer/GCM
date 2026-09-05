# large-scale condensation with finite precipitation efficiency.
# when humidity exceeds saturation, a fraction of the excess is removed
# as precipitation. the rest persists as "cloud water" (really just
# supersaturation that we allow to exist). this crudely represents the
# fact that clouds take time to precipitate and provide a greenhouse
# effect while they exist.

import torch
from scm.thermo import Lv, cp, g, saturation_specific_humidity


def condensation(state, grid, params):
    """saturation adjustment with tunable precipitation efficiency.
    precip_fraction controls how much of the excess is removed per
    timestep (1.0 = instant removal, 0.1 = only 10% removed)."""

    t = state['t']
    q = state['q']
    p = state['p']
    dp = state['dp']

    # how much of the supersaturation to actually remove as precip.
    # allowing some to persist crudely represents cloud water.
    precip_frac = params.get('ls_precip_fraction', 0.1)
    cloud_microphysics = bool(params.get('cloud_microphysics_enabled', False))

    # Condensation begins before the grid mean reaches saturation. A model
    # layer is not uniform: part of it is cloudy and saturated while the rest
    # is clear, so cloud forms once the mean crosses a critical humidity below
    # 1. Sundqvist-type schemes and the assumed-PDF closures that CESM2 and
    # GFDL use both work this way. With a hard grid-mean adjustment instead,
    # any layer that keeps receiving moisture is driven to exactly 100% and
    # stays there, which is what this column was doing between 685 and 865 hPa.
    rh_crit = float(params.get('condensation_rh_crit', 1.0))
    rh_crit = min(max(rh_crit, 0.5), 1.0)

    qs = saturation_specific_humidity(t, p)

    # find the excess above saturation
    t_new = t.clone()
    q_new = q.clone()

    # Subgrid distribution. Total water in a layer is spread uniformly about
    # its mean over a half width of (1 - rh_crit) * qs, so the layer starts to
    # condense once the mean passes rh_crit and is fully cloudy only once it
    # passes saturation by that same margin. Condensing the part of the
    # distribution that lies above qs gives a condensate that grows
    # quadratically from the onset rather than in a straight line, which is
    # what lets the grid mean settle somewhere between rh_crit and 1 instead
    # of being pinned to either. The cloud fraction below comes out of the
    # same distribution, so the two can no longer disagree.
    # NOTE: this is a partial-condensation scheme and it is NOT correct yet.
    # The intent is a diagnostic split of total water into vapour and
    # condensate over a uniform subgrid distribution, which is what lets a
    # layer hold partial cloud while its grid mean stays below saturation.
    # As written it is applied as an incremental sink on vapour instead, and
    # any setting below 1.0 drives the column into a runaway cold-and-dry
    # state (about -13 K over 200 days at 0.90, and non-monotonic in rh_crit,
    # which is the sign that the formulation rather than the tuning is wrong).
    # Rewriting it to work from total water was tried and made matters worse,
    # so the cause is not yet understood. rh_crit is therefore left at 1.0,
    # where this reduces exactly to a saturation adjustment and the column is
    # stable. Fixing it properly would remove the saturated slab between 685
    # and 865 hPa that the saturation adjustment produces.
    cloud_fraction_diag = torch.zeros_like(q)
    for _ in range(3):
        qs_current = saturation_specific_humidity(t_new, p)
        half_width = ((1.0 - rh_crit) * qs_current).clamp(min=1.0e-12)
        above = q_new + half_width - qs_current
        cloud_fraction_diag = (above / (2.0 * half_width)).clamp(min=0.0, max=1.0)
        partial = (above.clamp(min=0.0) ** 2) / (4.0 * half_width)
        saturated_excess = (q_new - qs_current).clamp(min=0.0)
        excess = torch.where(cloud_fraction_diag >= 1.0, saturated_excess, partial)
        excess = torch.clamp(excess, min=0.0)

        dqsdt = Lv * qs_current / (461.5 * t_new * t_new)
        correction = 1.0 + Lv / cp * dqsdt
        dq = -excess / correction
        dt_heating = -Lv / cp * dq

        t_new = t_new + dt_heating
        q_new = q_new + dq

    # full saturation adjustment (negative where condensation)
    full_dq = q_new - q
    full_dt = t_new - t

    # handle scalar or batched precip_frac
    if isinstance(precip_frac, torch.Tensor) and precip_frac.dim() == 1:
        pf = precip_frac.unsqueeze(1)
    else:
        pf = precip_frac

    if cloud_microphysics:
        cloud_precip_frac = params.get('cloud_ls_precip_fraction', 0.8)
        if isinstance(cloud_precip_frac, torch.Tensor) and cloud_precip_frac.dim() == 1:
            cpf = cloud_precip_frac.unsqueeze(1)
        else:
            cpf = cloud_precip_frac

        # In the microphysics-coupled path, vapor is adjusted fully to
        # saturation and the condensed water is split between precipitation
        # and an explicit cloud condensate reservoir.
        dt_tend = full_dt
        dq_tend = full_dq
        condensate_total = (-full_dq).clamp(min=0.0)
        precip_removed = cpf * condensate_total
        cloud_source = (1.0 - cpf) * condensate_total
        precip = torch.sum(precip_removed * dp / g, dim=1)
    else:
        # Legacy simplified path: only a fraction of the supersaturation
        # is removed from vapor, and the rest remains implicitly in q.
        dt_tend = pf * full_dt
        dq_tend = pf * full_dq
        condensate = (-dq_tend).clamp(min=0.0)
        cloud_source = torch.zeros_like(q)
        precip = torch.sum(condensate * dp / g, dim=1)

    result_cloud_fraction = cloud_fraction_diag

    return {
        'condensation_cloud_fraction': result_cloud_fraction,
        'dt': dt_tend,
        'dq': dq_tend,
        'precip': precip,
        'cloud_source': cloud_source,
    }
