import torch

from scm.boundary_layer import boundary_layer_mixing
from scm.thermo import Lv, cp, g, geopotential, saturation_specific_humidity


def edmf_boundary_layer(state, grid, params):
    """Combine local diffusion with a conservative boundary-layer updraft."""

    dt = float(params.get('dt', 900.0))
    localparams = dict(params)
    localparams['boundary_layer_scheme'] = 'richardson'
    localparams['bl_mix_moist_static_energy'] = True
    local = boundary_layer_mixing(state, grid, localparams)

    mixed = dict(state)
    mixed['t'] = state['t'] + dt * local['dt']
    mixed['q'] = state['q'] + dt * local['dq']
    mixed['qc'] = state.get('qc', torch.zeros_like(state['q'])) + dt * local['dqc']

    updraft = edmf_updraft(mixed, grid, params, local['boundary_layer_depth_m'])
    return {
        'dt': local['dt'] + updraft['dt'],
        'dq': local['dq'] + updraft['dq'],
        'dqc': local['dqc'] + updraft['dqc'],
        'boundary_layer_depth_m': local['boundary_layer_depth_m'],
        'edmf_activity': updraft['activity'],
        'edmf_mass_flux_kgm2s': updraft['mass_flux'],
        'edmf_condensate_kgm2s': updraft['condensate'],
    }


def edmf_updraft(state, grid, params, boundary_depth):
    """Move conservative surface-layer properties through the PBL top."""

    t = state['t']
    q = state['q']
    qc = state.get('qc', torch.zeros_like(q))
    p = state['p']
    dp = state['dp']
    batch = t.shape[0]
    dtype = t.dtype
    device = t.device
    dt = float(params.get('dt', 900.0))

    height = geopotential(t, q, p, grid)
    mass = dp / g
    mse = cp * t + Lv * q + g * height

    source_depth = float(params.get('edmf_source_depth_m', 250.0))
    overshoot_depth = max(float(params.get('edmf_overshoot_depth_m', 300.0)), 1.0)
    source_weight = (height <= source_depth).to(dtype)
    plume_edge = torch.exp(
        -torch.clamp(height - boundary_depth.unsqueeze(1), min=0.0) / overshoot_depth
    )
    destination_weight = (height > source_depth).to(dtype) * plume_edge
    destination_weight = torch.where(
        destination_weight >= 1.0e-3,
        destination_weight,
        torch.zeros_like(destination_weight),
    )

    source_mass = torch.sum(source_weight * mass, dim=1).clamp(min=1.0e-8)
    destination_mass = torch.sum(destination_weight * mass, dim=1).clamp(min=1.0e-8)
    source_mse = torch.sum(source_weight * mass * mse, dim=1) / source_mass
    destination_mse = torch.sum(destination_weight * mass * mse, dim=1) / destination_mass

    mse_scale = max(float(params.get('edmf_mse_scale_jkg', 8000.0)), 1.0)
    activity = torch.clamp((source_mse - destination_mse) / mse_scale, min=0.0, max=1.0)
    valid = (source_weight.sum(dim=1) > 0) & (destination_weight.sum(dim=1) > 0)
    activity = activity * valid.to(dtype)

    tau = torch.as_tensor(
        params.get('edmf_updraft_tau_s', 21600.0), device=device, dtype=dtype
    )
    if tau.dim() == 0:
        tau = tau.expand(batch)
    tau = tau.clamp(min=1.0)
    relax = (dt / tau).clamp(max=1.0)
    dmse = conservative_exchange(mse, mass, source_weight, destination_weight, activity, relax)
    source_q = weighted_mean(q, mass, source_weight)
    detrain_rh = float(params.get('edmf_detrain_rh', 0.85))
    humidity_target = torch.minimum(
        source_q.unsqueeze(1),
        detrain_rh * saturation_specific_humidity(t, p),
    )
    dq = conservative_target_exchange(
        q, humidity_target, mass, source_weight, destination_weight, activity, relax
    )
    dqc = conservative_exchange(qc, mass, source_weight, destination_weight, activity, relax)
    water_fraction = float(params.get('edmf_water_transport_fraction', 0.35))
    water_fraction = min(max(water_fraction, 0.0), 1.0)
    dq = water_fraction * dq
    dqc = water_fraction * dqc

    limiter = tendency_limiter(t, q, qc, dmse, dq, dqc, params, dt)
    dmse = dmse * limiter.unsqueeze(1)
    dq = dq * limiter.unsqueeze(1)
    dqc = dqc * limiter.unsqueeze(1)
    dq, dqc, condensed = condense_updraft_water(t, q, qc, p, dmse, dq, dqc)
    dtemperature = (dmse - Lv * dq) / cp
    mass_flux = activity * source_mass / tau
    condensate = torch.sum(condensed * mass, dim=1) / dt

    return {
        'dt': dtemperature / dt,
        'dq': dq / dt,
        'dqc': dqc / dt,
        'activity': activity,
        'mass_flux': mass_flux,
        'condensate': condensate,
    }


def conservative_exchange(tracer, mass, source, destination, activity, relax):
    source_mass = torch.sum(source * mass, dim=1).clamp(min=1.0e-8)
    source_mean = weighted_mean(tracer, mass, source)
    destination_step = (
        relax.unsqueeze(1) * activity.unsqueeze(1)
        * destination * (source_mean.unsqueeze(1) - tracer)
    )
    transported = torch.sum(destination_step * mass, dim=1)
    source_step = -source * (transported / source_mass).unsqueeze(1)
    return source_step + destination_step


def conservative_target_exchange(tracer, target, mass, source, destination, activity, relax):
    source_mass = torch.sum(source * mass, dim=1).clamp(min=1.0e-8)
    destination_step = (
        relax.unsqueeze(1) * activity.unsqueeze(1) * destination * (target - tracer)
    )
    transported = torch.sum(destination_step * mass, dim=1)
    source_step = -source * (transported / source_mass).unsqueeze(1)
    return source_step + destination_step


def weighted_mean(tracer, mass, weight):
    selected_mass = torch.sum(weight * mass, dim=1).clamp(min=1.0e-8)
    return torch.sum(weight * mass * tracer, dim=1) / selected_mass


def tendency_limiter(t, q, qc, dmse, dq, dqc, params, dt):
    dtemperature = (dmse - Lv * dq) / cp
    max_temperature = float(params.get('edmf_max_dt_day', 10.0)) * dt / 86400.0
    max_moisture = float(params.get('edmf_max_dq_day', 10.0)) * 1.0e-3 * dt / 86400.0

    temperature_peak = torch.amax(dtemperature.abs(), dim=1).clamp(min=1.0e-12)
    moisture_peak = torch.amax(dq.abs(), dim=1).clamp(min=1.0e-12)
    scale = torch.minimum(
        torch.clamp(max_temperature / temperature_peak, max=1.0),
        torch.clamp(max_moisture / moisture_peak, max=1.0),
    )

    qscale = positive_tracer_scale(q, dq, 1.0e-7)
    qcscale = positive_tracer_scale(qc, dqc, 0.0)
    return torch.minimum(scale, torch.minimum(qscale, qcscale))


def positive_tracer_scale(tracer, step, floor):
    available = (tracer - floor).clamp(min=0.0)
    ratio = torch.where(step < 0.0, available / (-step).clamp(min=1.0e-12), torch.ones_like(step))
    return torch.amin(ratio, dim=1).clamp(min=0.0, max=1.0)


def condense_updraft_water(t, q, qc, p, dmse, dq, dqc):
    """Convert EDMF supersaturation to cloud water without changing MSE."""

    qnew = q + dq
    qcnew = qc + dqc
    tnew = t + (dmse - Lv * dq) / cp
    condensed = torch.zeros_like(q)

    for _ in range(3):
        saturation = saturation_specific_humidity(tnew, p)
        excess = torch.clamp(qnew - saturation, min=0.0)
        slope = Lv * saturation / (461.5 * tnew * tnew)
        vapor_change = -excess / (1.0 + Lv / cp * slope)
        qnew = qnew + vapor_change
        qcnew = qcnew - vapor_change
        tnew = tnew - Lv / cp * vapor_change
        condensed = condensed - vapor_change

    return dq - condensed, dqc + condensed, condensed
