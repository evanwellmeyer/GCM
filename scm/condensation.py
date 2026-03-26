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

    qs = saturation_specific_humidity(t, p)

    # find the excess above saturation
    t_new = t.clone()
    q_new = q.clone()

    for _ in range(3):
        qs_current = saturation_specific_humidity(t_new, p)
        excess = torch.clamp(q_new - qs_current, min=0.0)

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

    return {
        'dt': dt_tend,
        'dq': dq_tend,
        'precip': precip,
        'cloud_source': cloud_source,
    }
