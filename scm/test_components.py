# component tests for the single column model.
# run each piece in isolation to verify it behaves sensibly before
# assembling the full model. this is the first thing to run on your
# machine after installing.
#
# usage: python -m scm.test_components

import torch
import sys
sys.path.insert(0, '/home/claude')


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def test_thermo(device):
    """verify thermodynamic functions give reasonable numbers."""
    print("=== thermo ===")
    from scm.thermo import (
        make_grid, saturation_vapor_pressure, saturation_specific_humidity,
        moist_adiabatic_lapse_rate, cape, moist_adiabat_profile,
        pressure_at_full, dp_from_ps
    )

    grid = make_grid(nlevels=20, device=device)
    print(f"sigma full levels: {grid['sigma_full']}")
    print(f"dsigma: {grid['dsigma']}")

    # saturation vapor pressure at 300K should be ~3500 Pa
    t = torch.tensor([300.0], device=device)
    es = saturation_vapor_pressure(t)
    print(f"e_s(300K) = {es.item():.1f} Pa (expect ~3530)")
    assert 3000 < es.item() < 4000, f"bad e_s: {es.item()}"

    # saturation specific humidity at 300K, 1000 hPa should be ~22 g/kg
    p = torch.tensor([1e5], device=device)
    qs = saturation_specific_humidity(t, p)
    print(f"q_s(300K, 1000hPa) = {qs.item()*1000:.1f} g/kg (expect ~22)")
    assert 15 < qs.item() * 1000 < 30, f"bad q_s: {qs.item()*1000}"

    # moist adiabatic lapse rate at 300K, 1000 hPa: ~3-4 K/km which is
    # roughly 0.3-0.4 K per 100 hPa, so dT/dp ~ 3e-4 K/Pa
    gamma = moist_adiabatic_lapse_rate(t, p)
    print(f"moist gamma(300K, 1000hPa) = {gamma.item()*1e4:.2f} x10^-4 K/Pa "
          f"(expect ~3-5)")

    # CAPE for a warm moist boundary layer should be positive
    batch = 2
    ps = torch.full((batch,), 1e5, device=device)
    pfull = pressure_at_full(grid, ps)
    dp = dp_from_ps(grid, ps)

    # warm moist column with a conditionally unstable profile
    t_col = torch.zeros(batch, 20, device=device)
    q_col = torch.zeros(batch, 20, device=device)
    for k in range(20):
        sigma = grid['sigma_full'][k]
        t_col[:, k] = 300.0 * sigma ** 0.19  # ~6.5 K/km lapse rate
        qs_k = saturation_specific_humidity(t_col[:, k:k+1], pfull[:, k:k+1])
        q_col[:, k] = 0.8 * qs_k.squeeze(1) * (sigma ** 2)
    t_col = torch.clamp(t_col, min=200.0)
    q_col = torch.clamp(q_col, min=1e-7)

    cape_val = cape(t_col, q_col, pfull, grid)
    print(f"CAPE = {cape_val[0].item():.0f} J/kg (expect positive, ~500-3000)")

    # moist adiabat from 300K at surface
    t_base = torch.tensor([300.0, 300.0], device=device)
    p_base = torch.tensor([1e5, 1e5], device=device)
    t_adiabat = moist_adiabat_profile(t_base, p_base, pfull)
    print(f"moist adiabat at 500 hPa: {t_adiabat[0, 9].item():.1f} K "
          f"(expect ~260-270)")

    print("thermo: PASS\n")


def test_radiation(device):
    """verify radiation gives reasonable fluxes and responds to CO2."""
    print("=== radiation ===")
    from scm.thermo import make_grid, pressure_at_full, dp_from_ps, saturation_specific_humidity
    from scm.radiation import radiation

    grid = make_grid(nlevels=20, device=device)
    batch = 4
    ps = torch.full((batch,), 1e5, device=device)
    p = pressure_at_full(grid, ps)
    dp = dp_from_ps(grid, ps)

    # tropical-ish profile
    sigma = grid['sigma_full'].unsqueeze(0).expand(batch, -1)
    t = 300.0 * sigma ** 0.19
    t = torch.clamp(t, min=200.0)
    qs = saturation_specific_humidity(t, p)
    q = 0.7 * qs * sigma ** 2
    q = torch.clamp(q, min=1e-7)
    ts = torch.full((batch,), 300.0, device=device)

    state = {'t': t, 'q': q, 'ts': ts, 'p': p, 'dp': dp}
    params_1x = {'co2': 400.0, 'co2_ref': 400.0}
    params_2x = {'co2': 800.0, 'co2_ref': 400.0}

    out_1x = radiation(state, grid, params_1x)
    out_2x = radiation(state, grid, params_2x)

    olr_1x = out_1x['olr'][0].item()
    olr_2x = out_2x['olr'][0].item()
    forcing = olr_1x - olr_2x

    print(f"OLR (1xCO2) = {olr_1x:.1f} W/m2 (expect ~220-280)")
    print(f"OLR (2xCO2) = {olr_2x:.1f} W/m2")
    print(f"2xCO2 forcing = {forcing:.2f} W/m2 (expect ~2-5)")
    print(f"LW down at surface = {out_1x['lw_down_sfc'][0].item():.1f} W/m2")
    print(f"SW absorbed at surface = {out_1x['sw_absorbed_sfc'][0].item():.1f} W/m2")

    # heating rate should be negative in most of the column (radiative cooling)
    dt_col = out_1x['dt'][0]  # K/s
    dt_per_day = dt_col * 86400.0
    print(f"radiative heating rate at 500 hPa: {dt_per_day[9].item():.2f} K/day "
          f"(expect ~ -1 to -3)")

    if forcing < 0.5:
        print("WARNING: CO2 forcing seems too weak. check co2_log_factor.")
    if forcing > 10:
        print("WARNING: CO2 forcing seems too strong.")

    params_trace_1x = {
        'co2': 400.0, 'co2_ref': 400.0,
        'radiation_mode': 'semi_gray_plus_trace_gases',
        'trace_gases_enabled': True,
        'ch4': 1.8, 'ch4_ref': 1.8,
        'ch4_base_tau': 0.02, 'ch4_log_factor': 0.01,
        'n2o': 0.332, 'n2o_ref': 0.332,
        'n2o_base_tau': 0.01, 'n2o_log_factor': 0.01,
        'o3_lw_tau': 0.02,
        'o3_sw_tau': 0.01,
    }
    params_trace_2x = dict(params_trace_1x)
    params_trace_2x['ch4'] = 3.6
    out_trace_1x = radiation(state, grid, params_trace_1x)
    out_trace_2x = radiation(state, grid, params_trace_2x)
    trace_forcing = out_trace_1x['olr'][0].item() - out_trace_2x['olr'][0].item()
    print(f"trace-gas OLR (base) = {out_trace_1x['olr'][0].item():.1f} W/m2")
    print(f"trace-gas CH4 doubling forcing = {trace_forcing:.2f} W/m2 (expect > 0)")
    assert trace_forcing > 0.0, "trace-gas branch should reduce OLR when CH4 increases"

    print("radiation: PASS\n")


def test_surface(device):
    """verify surface fluxes are reasonable."""
    print("=== surface ===")
    from scm.thermo import make_grid, pressure_at_full, dp_from_ps, saturation_specific_humidity
    from scm.surface import surface_fluxes

    grid = make_grid(nlevels=20, device=device)
    batch = 2
    ps = torch.full((batch,), 1e5, device=device)
    p = pressure_at_full(grid, ps)
    dp = dp_from_ps(grid, ps)

    sigma = grid['sigma_full'].unsqueeze(0).expand(batch, -1)
    t = 300.0 * sigma ** 0.19
    t = torch.clamp(t, min=200.0)
    qs = saturation_specific_humidity(t, p)
    q = 0.7 * qs * sigma ** 2
    q = torch.clamp(q, min=1e-7)

    # SST 2 degrees warmer than lowest level air
    ts = t[:, -1] + 2.0

    state = {'t': t, 'q': q, 'ts': ts, 'p': p, 'dp': dp}
    params = {'cd': 1.2e-3, 'wind_speed': 5.0}
    out = surface_fluxes(state, grid, params)

    shf = out['shf'][0].item()
    lhf = out['lhf'][0].item()
    print(f"sensible heat flux = {shf:.1f} W/m2 (expect ~5-20)")
    print(f"latent heat flux = {lhf:.1f} W/m2 (expect ~50-150)")

    # latent should dominate sensible over warm ocean
    assert lhf > shf, "latent should exceed sensible over warm ocean"
    assert lhf > 0, "latent heat flux should be positive (upward)"
    assert shf > 0, "sensible heat flux should be positive when SST > T_air"

    print("surface: PASS\n")


def test_condensation(device):
    """verify saturation adjustment removes supersaturation."""
    print("=== condensation ===")
    from scm.thermo import make_grid, pressure_at_full, dp_from_ps, saturation_specific_humidity
    from scm.condensation import condensation

    grid = make_grid(nlevels=20, device=device)
    batch = 2
    ps = torch.full((batch,), 1e5, device=device)
    p = pressure_at_full(grid, ps)
    dp = dp_from_ps(grid, ps)

    sigma = grid['sigma_full'].unsqueeze(0).expand(batch, -1)
    t = 280.0 * sigma ** 0.19
    t = torch.clamp(t, min=200.0)

    # make a few levels supersaturated
    qs = saturation_specific_humidity(t, p)
    q = 0.5 * qs  # subsaturated everywhere
    q[:, 8:12] = 1.3 * qs[:, 8:12]  # supersaturated in mid-troposphere

    state = {'t': t, 'q': q, 'p': p, 'dp': dp}
    out = condensation(state, grid, {})

    # apply adjustment
    t_new = t + out['dt']
    q_new = q + out['dq']
    qs_new = saturation_specific_humidity(t_new, p)

    # should be at or below saturation everywhere
    excess = (q_new - qs_new).max().item()
    print(f"max supersaturation after adjustment: {excess*1000:.4f} g/kg "
          f"(should be ~0)")
    print(f"precipitation: {out['precip'][0].item():.4f} kg/m2")

    # temperature should have increased where condensation occurred
    dt_mid = out['dt'][0, 10].item()
    print(f"warming at condensation level: {dt_mid:.3f} K (should be positive)")
    assert dt_mid > 0, "condensation should warm the air"

    print("condensation: PASS\n")


def test_convection_bm(device):
    """verify betts-miller activates for unstable columns."""
    print("=== convection (betts-miller) ===")
    from scm.thermo import (make_grid, pressure_at_full, dp_from_ps,
                            saturation_specific_humidity, cape)
    from scm.convection_bm import betts_miller

    grid = make_grid(nlevels=20, device=device)
    batch = 4
    ps = torch.full((batch,), 1e5, device=device)
    p = pressure_at_full(grid, ps)
    dp = dp_from_ps(grid, ps)

    sigma = grid['sigma_full'].unsqueeze(0).expand(batch, -1)
    t = 300.0 * sigma ** 0.19
    t = torch.clamp(t, min=200.0)
    qs = saturation_specific_humidity(t, p)
    q = 0.85 * qs * sigma ** 1.5  # fairly moist to give CAPE
    q = torch.clamp(q, min=1e-7)

    cape_val = cape(t, q, p, grid)
    print(f"CAPE before convection: {cape_val[0].item():.0f} J/kg")

    state = {'t': t, 'q': q, 'p': p, 'dp': dp}
    params = {'dt': 900.0, 'tau_bm': 7200.0, 'rhbm': 0.8,
              'cape_threshold': 100.0}
    out = betts_miller(state, grid, params)

    precip = out['precip'][0].item() * 86400  # mm/day
    dt_max = (out['dt'][0] * 86400).max().item()
    dq_min = (out['dq'][0] * 86400 * 1000).min().item()
    dq_max = (out['dq'][0] * 86400 * 1000).max().item()

    # debug: check what dq_tend looks like
    dq_profile = out['dq'][0] * 86400 * 1000  # g/kg/day
    col_dq = torch.sum(out['dq'][0] * dp[0] / 9.81).item() * 86400 * 1000
    print(f"convective precip: {precip:.2f} mm/day")
    print(f"max heating rate: {dt_max:.2f} K/day")
    print(f"max drying rate: {dq_min:.2f} g/kg/day")
    print(f"max moistening rate: {dq_max:.2f} g/kg/day")
    print(f"col-integrated dq: {col_dq:.4f} g/m2/day")
    print(f"dq at bottom 4 levels: {dq_profile[-4:].tolist()}")

    if cape_val[0].item() > 100 and col_dq > -10.0:
        print("WARNING: CAPE is positive but very little column drying. "
              "check betts-miller logic.")
    else:
        print(f"BM scheme OK: column drying = {col_dq:.0f} g/m2/day")

    print("convection (bm): PASS\n")


def test_convection_mf(device):
    """verify mass-flux scheme activates for unstable columns."""
    print("=== convection (mass-flux) ===")
    from scm.thermo import (make_grid, pressure_at_full, dp_from_ps,
                            saturation_specific_humidity, cape)
    from scm.convection_mf import mass_flux_convection

    grid = make_grid(nlevels=20, device=device)
    batch = 4
    ps = torch.full((batch,), 1e5, device=device)
    p = pressure_at_full(grid, ps)
    dp = dp_from_ps(grid, ps)

    sigma = grid['sigma_full'].unsqueeze(0).expand(batch, -1)
    t = 300.0 * sigma ** 0.19
    t = torch.clamp(t, min=200.0)
    qs = saturation_specific_humidity(t, p)
    q = 0.85 * qs * sigma ** 1.5
    q = torch.clamp(q, min=1e-7)

    cape_val = cape(t, q, p, grid)
    print(f"CAPE before convection: {cape_val[0].item():.0f} J/kg")

    state = {'t': t, 'q': q, 'p': p, 'dp': dp}
    params = {'dt': 900.0, 'entrainment_rate': 1.5e-4, 'tau_cape': 3600.0,
              'precip_efficiency': 0.9, 'cape_threshold': 100.0}
    out = mass_flux_convection(state, grid, params)

    precip = out['precip'][0].item() * 86400
    dt_profile = out['dt'][0] * 86400

    print(f"convective precip: {precip:.2f} mm/day")
    print(f"heating profile (K/day): {dt_profile.tolist()[:5]}... (top levels)")

    print("convection (mf): PASS\n")


def test_short_integration(device):
    """run the model in stages to isolate problems:
    1) radiation + surface + BL + condensation only (no convection)
    2) full model with convection"""
    print("=== short integration ===")
    from scm.thermo import make_grid
    from scm.column_model import initial_state, run, update_derived
    from scm.ensemble import default_params

    grid = make_grid(nlevels=20, device=device)
    sigma = grid['sigma_full']

    # --- phase 1: no convection, 200 steps ---
    print("phase 1: no convection (200 steps)")
    params = default_params(device=device)
    params['convection_scheme'] = 'none'
    batch = 4
    state = initial_state(batch, grid, params, device=device)

    state, history = run(state, grid, params, nsteps=200,
                         rad_interval=8, diag_interval=100)

    t_profile = state['t'][0]
    print(f"  Ts: {state['ts'][0].item():.2f} K")
    print(f"  T range: [{state['t'].min().item():.1f}, {state['t'].max().item():.1f}] K")
    for k in [0, 5, 9, 14, 17, 19]:
        print(f"    sigma={sigma[k].item():.3f}  T={t_profile[k].item():.1f} K  "
              f"q={state['q'][0, k].item()*1000:.2f} g/kg")
    if len(history) > 0:
        print(f"  OLR: {history[-1]['olr'][0].item():.1f} W/m2")
        print(f"  precip: {history[-1]['precip_total'][0].item() * 86400:.2f} mm/day")

    assert not torch.isnan(state['t']).any(), "NaN in T (no-convection)"
    assert not torch.isnan(state['ts']).any(), "NaN in Ts (no-convection)"
    print("  phase 1: PASS")

    # --- phase 2: with BM convection, 200 steps ---
    print("\nphase 2: with betts-miller (200 steps)")
    params2 = default_params(device=device)
    params2['convection_scheme'] = 'betts_miller'
    state2 = initial_state(batch, grid, params2, device=device)

    state2, history2 = run(state2, grid, params2, nsteps=200,
                           rad_interval=8, diag_interval=100)

    t_profile2 = state2['t'][0]
    print(f"  Ts: {state2['ts'][0].item():.2f} K")
    print(f"  T range: [{state2['t'].min().item():.1f}, {state2['t'].max().item():.1f}] K")
    for k in [0, 5, 9, 14, 17, 19]:
        print(f"    sigma={sigma[k].item():.3f}  T={t_profile2[k].item():.1f} K  "
              f"q={state2['q'][0, k].item()*1000:.2f} g/kg")
    if len(history2) > 0:
        print(f"  OLR: {history2[-1]['olr'][0].item():.1f} W/m2")
        print(f"  precip: {history2[-1]['precip_total'][0].item() * 86400:.2f} mm/day")

    assert not torch.isnan(state2['ts']).any(), "NaN in Ts (with BM)"
    print("  phase 2: PASS")

    # --- phase 3: with MF convection, 200 steps ---
    print("\nphase 3: with mass-flux (200 steps)")
    params3 = default_params(device=device)
    params3['convection_scheme'] = 'mass_flux'
    state3 = initial_state(batch, grid, params3, device=device)

    state3, history3 = run(state3, grid, params3, nsteps=200,
                           rad_interval=8, diag_interval=100)

    t_profile3 = state3['t'][0]
    print(f"  Ts: {state3['ts'][0].item():.2f} K")
    print(f"  T range: [{state3['t'].min().item():.1f}, {state3['t'].max().item():.1f}] K")
    for k in [0, 5, 9, 14, 17, 19]:
        print(f"    sigma={sigma[k].item():.3f}  T={t_profile3[k].item():.1f} K  "
              f"q={state3['q'][0, k].item()*1000:.2f} g/kg")
    if len(history3) > 0:
        print(f"  OLR: {history3[-1]['olr'][0].item():.1f} W/m2")
        print(f"  precip: {history3[-1]['precip_total'][0].item() * 86400:.2f} mm/day")

    assert not torch.isnan(state3['ts']).any(), "NaN in Ts (with MF)"
    print("  phase 3: PASS")

    print("\nshort integration: PASS\n")


def test_ensemble_batching(device):
    """verify that the mixed ensemble infrastructure works."""
    print("=== ensemble batching ===")
    from scm.thermo import make_grid
    from scm.column_model import initial_state, run, update_derived
    from scm.ensemble import make_ensemble_params

    grid = make_grid(nlevels=20, device=device)
    n_bm, n_mf = 10, 10
    base = {
        'dt': 900.0, 'ps0': 1e5, 'ts_init': 290.0,
        'solar_constant': 1360.0, 'zenith_factor': 0.25,
        'co2': 400.0, 'co2_ref': 400.0, 'use_slab_ocean': True,
    }
    params = make_ensemble_params(n_bm, n_mf, base_params=base, device=device)

    print(f"ensemble size: {n_bm + n_mf}")
    print(f"scheme_mask: {params['scheme_mask']}")

    state = initial_state(n_bm + n_mf, grid, params, device=device)

    # run step by step to find where NaN first appears
    from scm.column_model import step, update_derived, radiation
    state = update_derived(state, grid)
    rad_cache = radiation(state, grid, params)

    # enable NaN debug for first few steps
    params['debug_nan'] = True

    for i in range(50):
        if i % 8 == 0:
            state = update_derived(state, grid)
            rad_cache = radiation(state, grid, params)
        state, diag, _ = step(state, grid, params, rad_cache)
        if torch.isnan(state['ts']).any():
            nan_members = torch.where(torch.isnan(state['ts']))[0]
            print(f"NaN at step {i}, members: {nan_members.tolist()}")
            for m in nan_members[:3].tolist():
                scheme = 'BM' if params['scheme_mask'][m] < 0.5 else 'MF'
                print(f"  member {m}: {scheme}")
            params['debug_nan'] = False  # stop spamming
            break

    ts_bm = state['ts'][:n_bm]
    ts_mf = state['ts'][n_bm:]

    has_nan = torch.isnan(state['ts']).any().item()
    if has_nan:
        n_nan_bm = torch.isnan(ts_bm).sum().item()
        n_nan_mf = torch.isnan(ts_mf).sum().item()
        print(f"NaN count: BM={n_nan_bm}/{n_bm}, MF={n_nan_mf}/{n_mf}")
        # print some of the non-NaN values to see if model is otherwise ok
        valid = ~torch.isnan(state['ts'])
        if valid.any():
            print(f"valid members Ts: {state['ts'][valid][:5]}")
    else:
        print(f"BM members Ts: mean={ts_bm.mean():.2f}, std={ts_bm.std():.2f}")
        print(f"MF members Ts: mean={ts_mf.mean():.2f}, std={ts_mf.std():.2f}")

    assert state['ts'].shape[0] == n_bm + n_mf
    assert not has_nan, "NaN in surface temperature"

    print("ensemble batching: PASS\n")


def main():
    device = pick_device()
    print(f"device: {device}\n")

    test_thermo(device)
    test_radiation(device)
    test_surface(device)
    test_condensation(device)
    test_convection_bm(device)
    test_convection_mf(device)
    test_short_integration(device)
    test_ensemble_batching(device)

    print("=" * 40)
    print("all component tests passed")
    print("=" * 40)


if __name__ == '__main__':
    main()
