# column model: time integration, physics dispatch, and state management.
# this is the core loop that steps the model forward. physics is applied
# sequentially: radiation -> surface fluxes -> boundary layer -> convection
# -> large-scale condensation. state is updated after each component.

import torch
from scm.thermo import (
    make_grid, pressure_at_full, pressure_at_half, dp_from_ps,
    saturation_specific_humidity, geopotential, g, cp, Lv, Rd, c_water
)
from scm.radiation import radiation
from scm.surface import surface_fluxes, slab_ocean_tendency, slab_heat_capacity
from scm.boundary_layer import boundary_layer_mixing
from scm.condensation import condensation
from scm.cloud_microphysics import initialize_cloud_state, cloud_microphysics_step
from scm.convection_shallow import shallow_convection
from scm.convection_bm import betts_miller
from scm.convection_mf import mass_flux_convection


def initial_state(batch, grid, params, device='cpu'):
    """create a reasonable initial atmospheric profile for the tropics.
    temperature follows a lapse rate of ~6.5 K/km with a tropopause,
    moisture is a fraction of saturation that decreases with height."""

    nlevels = grid['nlevels']
    ps = torch.full((batch,), params.get('ps0', 1e5), device=device)
    p = pressure_at_full(grid, ps)

    # temperature: start from surface and decrease with a standard lapse rate
    ts = torch.full((batch,), params.get('ts_init', 290.0), device=device, dtype=torch.float64)

    # use a simple analytic profile: T decreases with log-pressure height
    # this gives a roughly realistic tropospheric profile
    sigma = grid['sigma_full'].unsqueeze(0).expand(batch, -1)
    t = ts.to(dtype=p.dtype).unsqueeze(1) * sigma ** (Rd * 6.5e-3 / g)
    # impose a tropopause: don't let temperature fall below 200K
    t = torch.clamp(t, min=200.0)

    # moisture: a fraction of saturation, decreasing with height
    rh_profile = 0.8 * sigma ** 2  # humid near surface, dry aloft
    qs = saturation_specific_humidity(t, p)
    q = rh_profile * qs
    q = torch.clamp(q, min=1e-7)

    state = {
        't': t,
        'q': q,
        'ts': ts,
        'slab_ts_ref': ts.clone(),
        'slab_energy': torch.zeros_like(ts, dtype=torch.float64),
        'ps': ps,
    }
    state.update(initialize_cloud_state(batch, grid, device=device))
    return state


def update_derived(state, grid):
    """recompute pressure fields from surface pressure. call this after
    any state update."""
    state['p'] = pressure_at_full(grid, state['ps'])
    state['dp'] = dp_from_ps(grid, state['ps'])
    return state


def atmospheric_energy_content(state, grid):
    """Primary atmospheric storage term for energy closure in J/m2.

    For TOA/surface flux closure we use column moist enthalpy, cp*T + Lv*q.
    In this fixed-mass hydrostatic column, adding geopotential to the storage
    term can make conservative vertical redistributions look like spurious
    energy sources/sinks in the closure diagnostic.
    """

    del grid
    mhe = cp * state['t'] + Lv * state['q']
    return torch.sum(mhe * state['dp'] / g, dim=1)


def atmospheric_mse_content(state, grid):
    """Column-integrated atmospheric moist static energy in J/m2."""

    z = geopotential(state['t'], state['q'], state['p'], grid)
    mse = cp * state['t'] + Lv * state['q'] + g * z
    return torch.sum(mse * state['dp'] / g, dim=1)


def step(state, grid, params, rad_cache=None):
    """advance the column model by one timestep. returns the updated state
    and a diagnostics dict.

    if rad_cache is provided and fresh, reuse it instead of recomputing
    radiation. this lets us call radiation less frequently."""

    dt = params.get('dt', 900.0)
    use_slab = params.get('use_slab_ocean', True)
    debug_nan = params.get('debug_nan', False)

    def check_nan(label, tensor):
        if debug_nan and torch.isnan(tensor).any():
            n = torch.isnan(tensor).sum().item()
            print(f"  NaN detected in {label}: {n} values")
            return True
        return False

    # make sure derived fields are current
    if 'slab_ts_ref' not in state:
        state['slab_ts_ref'] = state['ts'].clone().to(torch.float64)
    if 'slab_energy' not in state:
        state['slab_energy'] = (
            slab_heat_capacity(params) * (state['ts'].to(torch.float64) - state['slab_ts_ref'])
        )
    else:
        state['ts'] = state['slab_ts_ref'] + state['slab_energy'] / slab_heat_capacity(params)
    state = update_derived(state, grid)
    ts_prev = state['ts'].clone()
    slab_energy_prev = state['slab_energy'].clone()
    atm_energy_prev = atmospheric_energy_content(state, grid)
    atm_energy_start = atm_energy_prev.clone()
    atm_mse_prev = atmospheric_mse_content(state, grid)

    # --- radiation ---
    if rad_cache is None:
        rad_out = radiation(state, grid, params)
    else:
        rad_out = rad_cache

    check_nan('rad dt', rad_out['dt'])
    # guard radiation outputs
    rad_dt = torch.nan_to_num(rad_out['dt'], nan=0.0).to(state['t'].dtype)
    rad_dq = torch.nan_to_num(rad_out['dq'], nan=0.0).to(state['q'].dtype)
    state['t'] = state['t'] + rad_dt * dt
    state['q'] = state['q'] + rad_dq * dt
    atm_energy_after_rad = atmospheric_energy_content(state, grid)

    # --- surface fluxes ---
    state = update_derived(state, grid)
    sfc_out = surface_fluxes(state, grid, params)
    check_nan('sfc dt', sfc_out['dt'])
    check_nan('sfc dq', sfc_out['dq'])
    sfc_dt = torch.nan_to_num(sfc_out['dt'], nan=0.0).to(state['t'].dtype)
    sfc_dq = torch.nan_to_num(sfc_out['dq'], nan=0.0).to(state['q'].dtype)
    state['t'] = state['t'] + sfc_dt * dt
    state['q'] = state['q'] + sfc_dq * dt
    atm_energy_after_surface = atmospheric_energy_content(state, grid)

    # --- boundary layer mixing ---
    state = update_derived(state, grid)
    bl_out = boundary_layer_mixing(state, grid, params)
    check_nan('bl dt', bl_out['dt'])
    bl_dt = torch.nan_to_num(bl_out['dt'], nan=0.0).to(state['t'].dtype)
    bl_dq = torch.nan_to_num(bl_out['dq'], nan=0.0).to(state['q'].dtype)
    state['t'] = state['t'] + bl_dt * dt
    state['q'] = state['q'] + bl_dq * dt
    atm_energy_after_bl = atmospheric_energy_content(state, grid)

    # --- shallow convection ---
    state = update_derived(state, grid)
    shallow_out = shallow_convection(state, grid, params)
    check_nan('shallow dt', shallow_out['dt'])
    check_nan('shallow dq', shallow_out['dq'])
    shallow_dt = torch.nan_to_num(shallow_out['dt'], nan=0.0).to(state['t'].dtype)
    shallow_dq = torch.nan_to_num(shallow_out['dq'], nan=0.0).to(state['q'].dtype)
    state['t'] = state['t'] + shallow_dt * dt
    state['q'] = state['q'] + shallow_dq * dt
    atm_energy_after_shallow = atmospheric_energy_content(state, grid)

    # --- convection ---
    state = update_derived(state, grid)
    conv_out = dispatch_convection(state, grid, params)
    check_nan('conv dt', conv_out['dt'])
    check_nan('conv dq', conv_out['dq'])
    conv_dt = torch.nan_to_num(conv_out['dt'], nan=0.0).to(state['t'].dtype)
    conv_dq = torch.nan_to_num(conv_out['dq'], nan=0.0).to(state['q'].dtype)
    state['t'] = state['t'] + conv_dt * dt
    state['q'] = state['q'] + conv_dq * dt
    atm_energy_after_conv = atmospheric_energy_content(state, grid)

    # --- large-scale condensation (instantaneous adjustment) ---
    state = update_derived(state, grid)
    cond_out = condensation(state, grid, params)
    check_nan('cond dt', cond_out['dt'])
    cond_dt = torch.nan_to_num(cond_out['dt'], nan=0.0).to(state['t'].dtype)
    cond_dq = torch.nan_to_num(cond_out['dq'], nan=0.0).to(state['q'].dtype)
    state['t'] = state['t'] + cond_dt
    state['q'] = state['q'] + cond_dq
    atm_energy_after_cond = atmospheric_energy_content(state, grid)

    # --- cloud microphysics / cloud-radiative state ---
    cloud_out = cloud_microphysics_step(state, grid, params, cond_out, conv_out)
    state['qc'] = cloud_out['qc']
    state['cloud_fraction'] = cloud_out['cloud_fraction']
    state['cloud_sw_tau_layer'] = cloud_out['cloud_sw_tau_layer']
    state['cloud_lw_tau_layer'] = cloud_out['cloud_lw_tau_layer']

    # --- clamp to physical range ---
    state['q'] = torch.clamp(state['q'], min=1e-7, max=0.1)
    state['t'] = torch.clamp(state['t'], min=150.0, max=350.0)

    precip_shallow = shallow_out.get('precip', torch.zeros_like(state['ts']))
    precip_conv = conv_out.get('precip', torch.zeros_like(state['ts']))
    precip_ls = cond_out.get('precip', torch.zeros_like(state['ts'])) / dt
    precip_cloud = cloud_out.get('precip', torch.zeros_like(state['ts'])) / dt
    precip_total = (precip_shallow + precip_conv + precip_ls + precip_cloud).clamp(
        max=100.0 / 86400.0
    )

    surface_net_flux = (
        rad_out['sw_absorbed_sfc']
        + rad_out['lw_down_sfc']
        - rad_out['lw_up_sfc']
        - sfc_out['shf']
        - sfc_out['lhf']
    )

    if params.get('include_precip_enthalpy_flux', True):
        precip_temperature = state['t'][:, -1]
        precip_heat_flux = c_water * (precip_temperature - ts_prev) * precip_total
    else:
        precip_heat_flux = torch.zeros_like(state['ts'])

    surface_total_flux = surface_net_flux + precip_heat_flux

    # --- slab ocean ---
    slab_energy_tendency = torch.zeros_like(state['ts'])
    if use_slab:
        dts_dt = slab_ocean_tendency(
            state, rad_out, sfc_out, params, precip_heat_flux=precip_heat_flux
        )
        check_nan('slab dts', dts_dt)
        dts_dt = torch.nan_to_num(dts_dt, nan=0.0)
        heat_capacity = float(slab_heat_capacity(params))
        state['slab_energy'] = state['slab_energy'] + heat_capacity * dts_dt.to(torch.float64) * dt
        slab_energy_min = heat_capacity * (200.0 - state['slab_ts_ref'])
        slab_energy_max = heat_capacity * (350.0 - state['slab_ts_ref'])
        state['slab_energy'] = torch.clamp(state['slab_energy'], min=slab_energy_min, max=slab_energy_max)
        state['ts'] = state['slab_ts_ref'] + state['slab_energy'] / heat_capacity
        slab_energy_tendency = (state['slab_energy'] - slab_energy_prev) / dt

    atm_energy_now = atmospheric_energy_content(state, grid)
    atm_mse_now = atmospheric_mse_content(state, grid)
    rad_energy_tendency = (atm_energy_after_rad - atm_energy_start) / dt
    surface_energy_tendency = (atm_energy_after_surface - atm_energy_after_rad) / dt
    bl_energy_tendency = (atm_energy_after_bl - atm_energy_after_surface) / dt
    shallow_energy_tendency = (atm_energy_after_shallow - atm_energy_after_bl) / dt
    conv_energy_tendency = (atm_energy_after_conv - atm_energy_after_shallow) / dt
    condensation_energy_tendency = (atm_energy_after_cond - atm_energy_after_conv) / dt
    atm_energy_tendency = (atm_energy_now - atm_energy_prev) / dt
    atmos_flux_convergence = rad_out['toa_net'] - surface_total_flux
    atmos_energy_residual = atmos_flux_convergence - atm_energy_tendency
    column_energy_tendency = atm_energy_tendency + slab_energy_tendency
    column_energy_residual = rad_out['toa_net'] - column_energy_tendency
    atm_mse_tendency = (atm_mse_now - atm_mse_prev) / dt
    atmos_mse_residual = atmos_flux_convergence - atm_mse_tendency
    column_mse_tendency = atm_mse_tendency + slab_energy_tendency
    column_mse_residual = rad_out['toa_net'] - column_mse_tendency

    # diagnostics
    diag = {
        'olr': rad_out['olr'],
        'asr': rad_out['asr'],
        'toa_net': rad_out['toa_net'],
        'precip_shallow': precip_shallow,
        'precip_conv': precip_conv,
        'precip_ls': precip_ls,
        'precip_cloud': precip_cloud,
        'precip_total': precip_total,
        'shf': sfc_out['shf'],
        'lhf': sfc_out['lhf'],
        'sw_absorbed_sfc': rad_out['sw_absorbed_sfc'],
        'sw_reflected_toa': rad_out['sw_reflected_toa'],
        'lw_down_sfc': rad_out['lw_down_sfc'],
        'lw_up_sfc': rad_out['lw_up_sfc'],
        'surface_net_flux': surface_net_flux,
        'surface_total_flux': surface_total_flux,
        'precip_heat_flux': precip_heat_flux,
        'rad_energy_tendency': rad_energy_tendency,
        'surface_energy_tendency': surface_energy_tendency,
        'bl_energy_tendency': bl_energy_tendency,
        'shallow_energy_tendency': shallow_energy_tendency,
        'conv_energy_tendency': conv_energy_tendency,
        'condensation_energy_tendency': condensation_energy_tendency,
        'atmos_flux_convergence': atmos_flux_convergence,
        'atmos_energy_tendency': atm_energy_tendency,
        'atmos_energy_residual': atmos_energy_residual,
        'atmos_mse_tendency': atm_mse_tendency,
        'atmos_mse_residual': atmos_mse_residual,
        'slab_energy_tendency': slab_energy_tendency,
        'column_energy_tendency': column_energy_tendency,
        'column_energy_residual': column_energy_residual,
        'column_mse_tendency': column_mse_tendency,
        'column_mse_residual': column_mse_residual,
        'cape': conv_out.get('cape', torch.zeros_like(state['ts'])),
        'shallow_mse_residual': shallow_out.get('mse_residual', torch.zeros_like(state['ts'])),
        'cloud_cover': 1.0 - torch.prod(
            1.0 - cloud_out['cloud_fraction'].clamp(min=0.0, max=1.0), dim=1
        ),
        'lwp': cloud_out['lwp'].sum(dim=1),
        'iwp': cloud_out['iwp'].sum(dim=1),
        'conv_mse_residual': conv_out.get('mse_residual', torch.zeros_like(state['ts'])),
    }

    return state, diag, rad_out


def dispatch_convection(state, grid, params):
    """call the appropriate convection scheme(s). if scheme_mask is present
    in params, compute both schemes and blend for the mixed structural
    ensemble. otherwise use whichever single scheme is specified."""

    scheme = params.get('convection_scheme', 'betts_miller')
    scheme_mask = params.get('scheme_mask', None)

    if scheme_mask is not None:
        if torch.all(scheme_mask < 0.5):
            out = betts_miller(state, grid, params)
            for k in ['dt', 'dq', 'precip']:
                out[k] = torch.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0)
            return out

        if torch.all(scheme_mask >= 0.5):
            out = mass_flux_convection(state, grid, params)
            for k in ['dt', 'dq', 'precip']:
                out[k] = torch.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0)
            return out

        # mixed structural ensemble: compute both and blend
        bm_out = betts_miller(state, grid, params)
        mf_out = mass_flux_convection(state, grid, params)

        # guard against NaN from extreme parameter combinations
        for out in [bm_out, mf_out]:
            for k in ['dt', 'dq', 'precip']:
                out[k] = torch.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0)

        # scheme_mask is (batch,): 0 = betts-miller, 1 = mass-flux
        mask = scheme_mask.unsqueeze(1)  # (batch, 1) for broadcasting
        out = {
            'dt': (1.0 - mask) * bm_out['dt'] + mask * mf_out['dt'],
            'dq': (1.0 - mask) * bm_out['dq'] + mask * mf_out['dq'],
            'precip': (1.0 - scheme_mask) * bm_out['precip'] + scheme_mask * mf_out['precip'],
        }
        return out

    elif scheme == 'betts_miller':
        out = betts_miller(state, grid, params)
    elif scheme == 'mass_flux':
        out = mass_flux_convection(state, grid, params)
    elif scheme == 'none':
        return {
            'dt': torch.zeros_like(state['t']),
            'dq': torch.zeros_like(state['q']),
            'precip': torch.zeros(state['t'].shape[0], device=state['t'].device),
        }
    else:
        raise ValueError(f"unknown convection scheme: {scheme}")

    # guard against NaN
    for k in ['dt', 'dq', 'precip']:
        out[k] = torch.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0)
    return out


def run(state, grid, params, nsteps, rad_interval=8, diag_interval=100,
        callback=None):
    """run the model for nsteps. radiation is computed every rad_interval
    steps. diagnostics are collected every diag_interval steps.

    callback is an optional function called every diag_interval steps
    with (step_number, state, diag) for monitoring progress."""

    diag_history = []
    rad_cache = None
    effective_rad_interval = max(1, int(rad_interval))

    # When cloud condensate is prognostic, let radiation see the updated
    # cloud state more frequently than in the clear-sky/semi-gray path.
    if params.get('cloud_microphysics_enabled', False):
        effective_rad_interval = min(
            effective_rad_interval,
            max(1, int(params.get('rad_interval_microphysics_steps', 1))),
        )

    for n in range(nsteps):
        # recompute radiation on schedule
        if n % effective_rad_interval == 0:
            state = update_derived(state, grid)
            rad_cache = radiation(state, grid, params)

        state, diag, rad_cache_out = step(state, grid, params, rad_cache)

        if n % diag_interval == 0:
            # snapshot diagnostics
            snapshot = {k: v.detach().clone() for k, v in diag.items()}
            snapshot['step'] = n
            snapshot['ts'] = state['ts'].detach().clone()
            snapshot['t'] = state['t'].detach().clone()
            snapshot['q'] = state['q'].detach().clone()
            snapshot['qc'] = state['qc'].detach().clone()
            diag_history.append(snapshot)

            if callback is not None:
                callback(n, state, diag)

    return state, diag_history
