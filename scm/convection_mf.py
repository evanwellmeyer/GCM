# simplified mass-flux convection scheme inspired by Zhang-McFarlane.
#
# key improvements over the previous version:
#   - detrainment moistens the free troposphere at the level where
#     the plume loses buoyancy, depositing saturated air. this is the
#     main mechanism for the water vapor feedback under warming.
#   - uses dilute CAPE (entraining parcel) for the closure, which
#     gives a more realistic sensitivity to warming.
#   - the heating profile comes from compensating subsidence warming
#     plus latent heat release, not just local mixing.

import torch
from scm.thermo import (
    cp, Lv, g, Rd, Rv,
    saturation_specific_humidity, virtual_temperature
)


def dilute_cape(t, q, p, entrainment):
    """CAPE computed with an entraining parcel. more realistic than
    undilute CAPE because it accounts for how environmental humidity
    affects buoyancy. returns (batch,) in J/kg."""

    batch = t.shape[0]
    nlevels = t.shape[1]

    t_parcel = t[:, -1].clone()
    q_parcel = q[:, -1].clone()
    p_parcel = p[:, -1].clone()

    dcape = torch.zeros(batch, device=t.device)

    for k in range(nlevels - 2, -1, -1):
        p_target = p[:, k]
        dp_step = (p_parcel - p_target).abs()

        # entrain environmental air
        mix = (entrainment * dp_step).clamp(max=0.3)
        t_parcel = (1.0 - mix) * t_parcel + mix * t[:, k]
        q_parcel = (1.0 - mix) * q_parcel + mix * q[:, k]

        # adiabatic ascent
        qs_p = saturation_specific_humidity(t_parcel, p_target)
        saturated = (q_parcel >= qs_p).float()

        gamma_dry = Rd * t_parcel / (cp * p_target)
        num = (Rd * t_parcel / (cp * p_target)) * (1.0 + Lv * qs_p / (Rd * t_parcel))
        den = 1.0 + Lv * Lv * qs_p / (cp * Rv * t_parcel * t_parcel)
        gamma_moist = num / den
        gamma = (1.0 - saturated) * gamma_dry + saturated * gamma_moist

        dp_rise = p_target - p_parcel
        t_parcel = t_parcel + gamma * dp_rise
        p_parcel = p_target

        # condensation
        qs_new = saturation_specific_humidity(t_parcel, p_target)
        excess = torch.clamp(q_parcel - qs_new, min=0.0)
        q_parcel = q_parcel - excess
        t_parcel = t_parcel + Lv / cp * excess

        # buoyancy contribution
        tv_parcel = virtual_temperature(t_parcel, q_parcel)
        tv_env = virtual_temperature(t[:, k], q[:, k])
        buoyancy = torch.clamp((tv_parcel - tv_env) / tv_env, min=0.0)
        dlnp = torch.log(p[:, k + 1].clamp(min=1.0) / p[:, k].clamp(min=1.0))
        dcape = dcape + Rd * tv_env * buoyancy * dlnp

    return dcape


def mass_flux_convection(state, grid, params):
    """simplified mass-flux scheme with detrainment moistening."""

    t = state['t']
    q = state['q']
    p = state['p']
    dp = state['dp']

    entrainment = params.get('entrainment_rate', 2.0e-4)  # per Pa
    tau_cape = params.get('tau_cape', 3600.0)
    precip_eff = params.get('precip_efficiency', 0.5)
    cape_threshold = params.get('cape_threshold', 100.0)
    timestep = params.get('dt', 900.0)

    batch = t.shape[0]
    nlevels = t.shape[1]

    # use dilute CAPE for the closure
    cape_val = dilute_cape(t, q, p, entrainment)
    activation = torch.sigmoid((cape_val - cape_threshold) / 50.0)

    # march the plume upward
    t_plume = t[:, -1].clone()
    q_plume = q[:, -1].clone()

    dt_norm = torch.zeros_like(t)
    dq_norm = torch.zeros_like(q)
    condensate_col = torch.zeros(batch, device=t.device)

    # track the plume mass flux (normalized, starts at 1)
    mf_profile = torch.ones(batch, device=t.device)

    for k in range(nlevels - 2, -1, -1):
        p_here = p[:, k]
        dp_layer = dp[:, k]
        dp_step = (p[:, k + 1] - p[:, k]).abs()

        # entrainment
        mix = (entrainment * dp_step).clamp(max=0.3)
        t_plume = (1.0 - mix) * t_plume + mix * t[:, k]
        q_plume = (1.0 - mix) * q_plume + mix * q[:, k]

        # mass flux increases from entrainment
        mf_profile = mf_profile * (1.0 + mix)

        # adiabatic cooling
        qs_p = saturation_specific_humidity(t_plume, p_here)
        saturated = (q_plume >= qs_p).float()
        gamma_dry = Rd * t_plume / (cp * p_here)
        num = (Rd * t_plume / (cp * p_here)) * (1.0 + Lv * qs_p / (Rd * t_plume))
        den = 1.0 + Lv * Lv * qs_p / (cp * Rv * t_plume * t_plume)
        gamma_moist = num / den
        gamma = (1.0 - saturated) * gamma_dry + saturated * gamma_moist

        dp_rise = p_here - p[:, k + 1]
        t_plume = t_plume + gamma * dp_rise

        # condense
        qs_p = saturation_specific_humidity(t_plume, p_here)
        condensate = torch.clamp(q_plume - qs_p, min=0.0)
        q_plume = q_plume - condensate
        t_plume = t_plume + Lv / cp * condensate

        # buoyancy
        tv_plume = virtual_temperature(t_plume, q_plume)
        tv_env = virtual_temperature(t[:, k], q[:, k])
        buoyant = torch.sigmoid((tv_plume - tv_env) * 5.0)

        # detrainment: where plume loses buoyancy, detrain mass.
        # detrainment rate increases where buoyancy decreases.
        detrain_frac = (1.0 - buoyant) * 0.3  # up to 30% per level
        mf_detrained = mf_profile * detrain_frac
        mf_profile = mf_profile * (1.0 - detrain_frac)

        # the detrained air is a mix of plume and environment.
        # it warms the environment (subsidence warming from mass removal)
        # and moistens it (plume air is close to saturation).
        mass_layer = dp_layer / g  # kg/m2
        detrain_rate = mf_detrained * g / dp_layer  # 1/s per unit Mb

        # temperature tendency: warming from plume air mixing in
        dt_norm[:, k] = detrain_rate * (t_plume - t[:, k])

        # moisture tendency: the detrained air moistens the environment,
        # but only up to a target RH. this represents the fact that
        # convective detrainment creates anvil clouds at ~80% RH, not
        # saturated air. the key: under warming, qs increases, so the
        # target q increases, and the column retains more vapor —
        # that's the water vapor feedback.
        qs_env = saturation_specific_humidity(t[:, k], p_here)
        q_target = 0.9 * qs_env  # detrained air reaches ~90% RH
        # only moisten (positive tendency), don't dry
        dq_detrain = torch.clamp(q_target - q[:, k], min=0.0)
        dq_norm[:, k] = detrain_rate * dq_detrain

        # also add compensating subsidence warming below cloud top.
        # the mass flux profile decreasing with height means air must
        # subside to compensate, warming adiabatically.
        if k < nlevels - 2:
            subsidence_warming = mf_profile * entrainment * g / dp_layer
            dt_norm[:, k] = dt_norm[:, k] + subsidence_warming * (t[:, k + 1] - t[:, k])

        # accumulate condensate for precipitation
        condensate_col = condensate_col + mf_profile * condensate * dp_layer / g

        # kill plume where it's clearly not buoyant
        mf_profile = mf_profile * (0.1 + 0.9 * buoyant)

    # boundary layer drying: the updraft removes moist air from the
    # lowest level. the drying rate is proportional to Mb * q_bl / mass_bl.
    bl_drying_rate = g / dp[:, -1]  # 1/s per unit Mb
    dq_norm[:, -1] = dq_norm[:, -1] - bl_drying_rate * q[:, -1] * 0.05  # 5% removal rate

    # CAPE closure: Mb so that dilute CAPE is removed over tau_cape
    col_heating = torch.sum(dt_norm.clamp(min=0.0) * dp / g, dim=1)
    col_mass = dp.sum(dim=1) / g
    col_heating_safe = col_heating.clamp(min=1e-8)
    mb = cape_val * col_mass / (cp * col_heating_safe * tau_cape)
    mb = mb * activation
    mb = mb.clamp(min=0.0, max=0.3)

    dt_tend = dt_norm * mb.unsqueeze(1)
    dq_tend = dq_norm * mb.unsqueeze(1)

    # limit tendencies
    max_dt = 10.0 / 86400.0
    max_dq = 5.0e-3 / 86400.0
    dt_tend = dt_tend.clamp(-max_dt, max_dt)
    dq_tend = dq_tend.clamp(-max_dq, max_dq)

    # precipitation
    precip = precip_eff * condensate_col * mb
    precip = precip.clamp(max=50.0 / 86400.0)

    return {
        'dt': dt_tend,
        'dq': dq_tend,
        'precip': precip,
    }
