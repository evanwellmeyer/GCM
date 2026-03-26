import torch

from scm.thermo import g, saturation_specific_humidity


def initialize_cloud_state(batch, grid, device='cpu'):
    nlevels = grid['nlevels']
    zeros = torch.zeros(batch, nlevels, device=device)
    return {
        'qc': zeros.clone(),
        'cloud_fraction': zeros.clone(),
        'cloud_sw_tau_layer': zeros.clone(),
        'cloud_lw_tau_layer': zeros.clone(),
    }


def _to_col(value, batch, device, dtype):
    t = torch.as_tensor(value, dtype=dtype, device=device)
    if t.dim() == 0:
        return t.view(1, 1)
    if t.dim() == 1:
        if t.shape[0] == 1:
            return t.view(1, 1)
        if t.shape[0] == batch:
            return t.unsqueeze(1)
    raise ValueError(f"cannot broadcast cloud parameter with shape {tuple(t.shape)} to batch={batch}")


def _quadratic_autoconversion(qc, dt, params, batch, device, dtype):
    """Return updated condensate and precipitated condensate from autoconversion.

    The sink only acts on condensate above a threshold and accelerates
    quadratically with excess condensate. This prevents thin clouds from
    raining out immediately while making optically thick clouds precipitate
    efficiently enough to avoid unrealistic long-lived anvils.
    """

    tau = _to_col(params.get('cloud_autoconv_tau', 7200.0), batch, device, dtype).clamp(min=1.0)
    qc_thresh = _to_col(
        params.get('cloud_autoconv_qc_thresh', params.get('cloud_qc_ref', 1.0e-4)),
        batch, device, dtype
    ).clamp(min=1.0e-8)
    qc_scale = _to_col(
        params.get('cloud_autoconv_qc_scale', params.get('cloud_qc_ref', 1.0e-4)),
        batch, device, dtype
    ).clamp(min=1.0e-8)
    power = max(float(params.get('cloud_autoconv_power', 2.0)), 1.0)

    excess = torch.clamp(qc - qc_thresh, min=0.0)
    sink = (dt / tau) * excess * torch.pow(excess / qc_scale, power - 1.0)
    sink = torch.minimum(sink, excess)
    return qc - sink, sink


def cloud_microphysics_step(state, grid, params, cond_out, conv_out):
    """Very simple prognostic cloud condensate and cloud optics.

    This is intentionally lightweight: one total condensate reservoir `qc`
    is carried in the state, with diagnostic liquid/ice partitioning for
    radiative properties.
    """

    q = state['q']
    t = state['t']
    p = state['p']
    dp = state['dp']
    batch = q.shape[0]
    dtype = q.dtype
    device = q.device

    zeros = torch.zeros_like(q)
    qc_prev = state.get('qc', zeros)

    if not params.get('cloud_microphysics_enabled', False):
        return {
            'qc': zeros,
            'cloud_fraction': zeros,
            'cloud_sw_tau_layer': zeros,
            'cloud_lw_tau_layer': zeros,
            'lwp': zeros,
            'iwp': zeros,
            'precip': torch.zeros(batch, device=device, dtype=dtype),
        }

    dt = float(params.get('dt', 900.0))
    qs = saturation_specific_humidity(t, p)
    rh = q / qs.clamp(min=1.0e-8)

    ls_source = cond_out.get('cloud_source', zeros)

    sigma = grid['sigma_full'].to(device=device, dtype=dtype)
    anvil_center = float(params.get('conv_cloud_anvil_center_sigma', 0.45))
    anvil_width = float(params.get('conv_cloud_anvil_width_sigma', 0.18))
    anvil_profile = torch.exp(-((sigma - anvil_center) / max(anvil_width, 1.0e-3)) ** 2)
    anvil_profile = anvil_profile / anvil_profile.sum().clamp(min=1.0e-8)

    conv_cloud_eff = _to_col(params.get('conv_cloud_efficiency', 0.03), batch, device, dtype)
    conv_precip = conv_out.get('precip', torch.zeros(batch, device=device, dtype=dtype)).unsqueeze(1)
    conv_source = conv_cloud_eff * conv_precip * dt * g / dp.clamp(min=1.0) * anvil_profile.unsqueeze(0)

    source = ls_source + conv_source

    evap_tau = _to_col(params.get('cloud_evap_tau', 3600.0), batch, device, dtype)
    rh_evap = _to_col(params.get('cloud_rh_evap', 0.75), batch, device, dtype).clamp(min=1.0e-3)

    qc = torch.clamp(qc_prev + source, min=0.0, max=float(params.get('cloud_qc_max', 0.01)))
    qc, autoconv_sink = _quadratic_autoconversion(qc, dt, params, batch, device, dtype)
    dry_factor = torch.clamp((rh_evap - rh) / rh_evap, min=0.0, max=1.0)
    qc = qc * torch.exp(-(dt / evap_tau.clamp(min=1.0)) * dry_factor)
    qc = torch.clamp(qc, min=0.0, max=float(params.get('cloud_qc_max', 0.01)))

    t_liq = float(params.get('cloud_liquid_temp', 273.15))
    t_ice = float(params.get('cloud_ice_temp', 258.15))
    liquid_fraction = torch.clamp((t - t_ice) / max(t_liq - t_ice, 1.0e-3), min=0.0, max=1.0)

    lwp = qc * liquid_fraction * dp / g
    iwp = qc * (1.0 - liquid_fraction) * dp / g

    rh_min = float(params.get('cloud_rh_min', 0.75))
    qc_ref = float(params.get('cloud_qc_ref', 1.0e-4))
    qc_min = float(params.get('cloud_cf_qc_min', 0.0))
    rh_power = max(float(params.get('cloud_cf_rh_power', 1.0)), 1.0)
    qc_power = max(float(params.get('cloud_cf_qc_power', 0.5)), 0.5)
    cf_max = float(params.get('cloud_cf_max', 1.0))
    rh_cloud = torch.clamp((rh - rh_min) / max(1.0 - rh_min, 1.0e-3), min=0.0, max=1.0)
    qc_cloud = torch.clamp(
        (qc - qc_min) / max(qc_ref - qc_min, 1.0e-8),
        min=0.0, max=1.0
    )

    # Require both near-saturation and nontrivial condensate before
    # diagnosing large cloud fraction. The previous max-based diagnostic
    # saturated to cloud_fraction ~= 1 almost anywhere RH was high.
    cloud_fraction = (torch.pow(rh_cloud, rh_power) * torch.pow(qc_cloud, qc_power))
    cloud_fraction = cloud_fraction.clamp(min=0.0, max=cf_max)
    cloud_fraction = cloud_fraction * (qc > 1.0e-8).to(dtype)

    k_liq_sw = float(params.get('cloud_k_liq_sw', 80.0))
    k_ice_sw = float(params.get('cloud_k_ice_sw', 40.0))
    k_liq_lw = float(params.get('cloud_k_liq_lw', 12.0))
    k_ice_lw = float(params.get('cloud_k_ice_lw', 6.0))

    cloud_sw_tau_layer = cloud_fraction * (k_liq_sw * lwp + k_ice_sw * iwp)
    cloud_lw_tau_layer = cloud_fraction * (k_liq_lw * lwp + k_ice_lw * iwp)
    precip = torch.sum(autoconv_sink * dp / g, dim=1)

    return {
        'qc': qc,
        'cloud_fraction': cloud_fraction,
        'cloud_sw_tau_layer': cloud_sw_tau_layer,
        'cloud_lw_tau_layer': cloud_lw_tau_layer,
        'lwp': lwp,
        'iwp': iwp,
        'precip': precip,
    }
