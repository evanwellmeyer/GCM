# simplified mass-flux convection scheme.
# a single entraining plume rises from the boundary layer. entrainment
# dilutes the plume with environmental air. where the plume is buoyant,
# it heats and dries the environment. cloud-base mass flux is set by
# a CAPE closure.
#
# the approach is:
#   1. march a plume upward, computing the normalized (per unit Mb)
#      tendencies at each level from detrainment of plume air.
#   2. compute column-integrated MSE change per unit Mb.
#   3. set Mb = CAPE / (tau * cp * column_mse_change) so the column
#      stabilizes over timescale tau.
#
# this produces a different vertical heating profile than betts-miller
# and responds differently to warming because the entrainment rate
# controls how sensitive the scheme is to free-tropospheric humidity.

import torch
from scm.thermo import (
    cp, Lv, g, Rd, Rv, cape,
    saturation_specific_humidity, virtual_temperature
)


def mass_flux_convection(state, grid, params):
    """simplified mass-flux scheme. returns tendency dict."""

    t = state['t']
    q = state['q']
    p = state['p']
    dp = state['dp']

    entrainment = params.get('entrainment_rate', 1.5e-4)  # per Pa
    tau_cape = params.get('tau_cape', 3600.0)
    precip_eff = params.get('precip_efficiency', 0.9)
    cape_threshold = params.get('cape_threshold', 100.0)
    timestep = params.get('dt', 900.0)

    batch = t.shape[0]
    nlevels = t.shape[1]

    cape_val = cape(t, q, p, grid)
    activation = torch.sigmoid((cape_val - cape_threshold) / 50.0)

    # initialize plume from the lowest level
    t_plume = t[:, -1].clone()
    q_plume = q[:, -1].clone()

    # per-unit-mass-flux tendencies: how much each layer changes per
    # kg/m2/s of cloud-base mass flux. units: K/s and kg/kg/s respectively.
    dt_norm = torch.zeros_like(t)
    dq_norm = torch.zeros_like(q)
    condensate_total = torch.zeros(batch, device=t.device)

    plume_active = torch.ones(batch, device=t.device)

    for k in range(nlevels - 2, -1, -1):
        p_here = p[:, k]
        dp_layer = dp[:, k]
        dp_step = (p[:, k + 1] - p[:, k]).abs()

        # entrainment: mix plume with environment
        mix = (entrainment * dp_step).clamp(max=0.5)
        t_plume = (1.0 - mix) * t_plume + mix * t[:, k]
        q_plume = (1.0 - mix) * q_plume + mix * q[:, k]

        # adiabatic cooling as plume rises
        qs_p = saturation_specific_humidity(t_plume, p_here)
        saturated = (q_plume >= qs_p).float()

        gamma_dry = Rd * t_plume / (cp * p_here)
        num = (Rd * t_plume / (cp * p_here)) * (1.0 + Lv * qs_p / (Rd * t_plume))
        den = 1.0 + Lv * Lv * qs_p / (cp * Rv * t_plume * t_plume)
        gamma_moist = num / den
        gamma = (1.0 - saturated) * gamma_dry + saturated * gamma_moist

        dp_rise = p_here - p[:, k + 1]  # negative (going up)
        t_plume = t_plume + gamma * dp_rise

        # condense excess moisture in the plume
        qs_p = saturation_specific_humidity(t_plume, p_here)
        condensate = torch.clamp(q_plume - qs_p, min=0.0)
        q_plume = q_plume - condensate
        t_plume = t_plume + Lv / cp * condensate

        # buoyancy check (soft)
        tv_plume = virtual_temperature(t_plume, q_plume)
        tv_env = virtual_temperature(t[:, k], q[:, k])
        buoyant = torch.sigmoid((tv_plume - tv_env) * 5.0)
        plume_active = plume_active * (0.3 + 0.7 * buoyant)

        # detrainment tendencies for this layer.
        # the environment is heated and dried by mixing with detrained plume air.
        # per unit mass flux (Mb in kg/m2/s), the tendency is:
        #   dT/dt = epsilon * Mb * (T_plume - T_env) / (dp/g)   ... but we factor
        # out Mb, so dt_norm = epsilon * (T_plume - T_env) / (dp/g)
        # this has units: [1/Pa] * [K] / [Pa / (m/s2 * kg/m3)]... let me just
        # think in terms of the mass budget.
        #
        # mass detrained into layer k = Mb * epsilon * dp_step / (dp_layer/g)... no.
        # simpler: the fractional replacement rate of layer k is
        #   Mb * epsilon * g / dp_layer   [1/s per unit Mb]
        # and the T tendency per unit Mb is that rate * (T_plume - T_env).
        detrainment_rate = entrainment * g  # [1/(Pa * s) * m/s^2]... actually
        # let me get the units right once and for all.
        # Mb has units kg/(m2 s).
        # epsilon has units 1/Pa (fractional entrainment per Pa of ascent).
        # the mass detrained per unit area per unit time into a layer of
        # thickness dp is: Mb * epsilon * dp  [kg/(m2 s) * 1/Pa * Pa = kg/(m2 s)]
        # the mass of the layer is dp/g [kg/m2].
        # so the mixing rate is: Mb * epsilon * dp_step * g / dp_layer  [1/s]
        # and the tendency per unit Mb is: epsilon * dp_step * g / dp_layer * (T_c - T)

        rate = entrainment * dp_step * g / dp_layer  # 1/s per unit Mb
        dt_norm[:, k] = plume_active * rate * (t_plume - t[:, k])
        dq_norm[:, k] = plume_active * rate * (q_plume - q[:, k])

        # accumulate condensate (per unit Mb, in kg/m2/s)
        condensate_total = condensate_total + plume_active * condensate * dp_layer / g

    # CAPE closure.
    # the column-integrated heating per unit Mb is:
    #   H = sum(dt_norm * dp/g)  [K * kg/m2 / s per unit Mb]
    # the energy input is cp * H [J/m2/s per unit Mb] = [W/m2 per unit Mb]
    # CAPE has units J/kg. to relate: CAPE ~ integral of buoyancy.
    # we want the heating to stabilize the column over tau_cape.
    # rough relation: the heating over time tau stabilizes if
    #   cp * H * Mb * tau_cape ~ CAPE * dp_total/g
    # so Mb ~ CAPE * (dp_total/g) / (cp * H * tau_cape)
    col_heating = torch.sum(dt_norm.clamp(min=0.0) * dp / g, dim=1)  # K*kg/m2/s per unit Mb
    dp_total = dp.sum(dim=1)
    col_mass = dp_total / g  # kg/m2

    # avoid division by zero
    col_heating_safe = col_heating.clamp(min=1e-10)
    mb = cape_val * col_mass / (cp * col_heating_safe * tau_cape)
    mb = mb * activation
    mb = mb.clamp(min=0.0, max=0.5)  # kg/m2/s, conservative cap

    dt_tend = dt_norm * mb.unsqueeze(1)
    dq_tend = dq_norm * mb.unsqueeze(1)

    # limit tendencies for numerical stability
    max_dt = 10.0 / 86400.0  # K/s
    max_dq = 5.0e-3 / 86400.0  # kg/kg/s
    dt_tend = dt_tend.clamp(-max_dt, max_dt)
    dq_tend = dq_tend.clamp(-max_dq, max_dq)

    # convective precipitation: condensate produced in the plume
    precip = precip_eff * condensate_total * mb  # kg/m2/s
    # sanity: cap at ~50 mm/day
    precip = precip.clamp(max=50.0 / 86400.0)

    return {
        'dt': dt_tend,
        'dq': dq_tend,
        'precip': precip,
    }
