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

    # only remove a fraction of the condensate as precipitation.
    # the rest stays in the column as "cloud water" (supersaturation).
    full_dq = q_new - q  # full adjustment (negative where condensation)
    full_dt = t_new - t  # full heating

    # handle scalar or batched precip_frac
    if isinstance(precip_frac, torch.Tensor) and precip_frac.dim() == 1:
        pf = precip_frac.unsqueeze(1)
    else:
        pf = precip_frac

    dt_tend = pf * full_dt
    dq_tend = pf * full_dq

    # precipitation from the removed fraction
    condensate = (-dq_tend).clamp(min=0.0)  # kg/kg removed
    precip = torch.sum(condensate * dp / g, dim=1)

    return {
        'dt': dt_tend,
        'dq': dq_tend,
        'precip': precip,
    }
