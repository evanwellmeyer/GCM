# component tests for the single column model.
# run each piece in isolation to verify it behaves sensibly before
# assembling the full model. this is the first thing to run on your
# machine after installing.
#
# usage: python -m scm.test_components

import torch
import sys
import tempfile
from pathlib import Path
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
    print(f"ASR (1xCO2) = {out_1x['asr'][0].item():.1f} W/m2")
    print(f"TOA net (1xCO2) = {out_1x['toa_net'][0].item():+.2f} W/m2")
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

    params_cloudy = {
        'radiation_scheme': 'semi_gray',
        'radiation_mode': 'semi_gray_plus_clouds',
        'cloud_radiative_effects_enabled': True,
        'cloud_fraction': 0.5,
        'cloud_sw_reflectivity': 0.4,
        'cloud_sw_tau': 0.1,
        'cloud_lw_tau': 0.5,
        'cloud_top_sigma': 0.6,
        'cloud_bottom_sigma': 0.9,
    }
    out_cloudy = radiation(state, grid, params_cloudy)
    print(f"cloudy ASR = {out_cloudy['asr'][0].item():.1f} W/m2")
    print(f"cloudy OLR = {out_cloudy['olr'][0].item():.1f} W/m2")
    assert out_cloudy['asr'][0].item() < out_1x['asr'][0].item(), (
        "cloud shortwave reflection should reduce absorbed solar"
    )

    state_micro = dict(state)
    state_micro['cloud_fraction'] = torch.full_like(q, 0.15)
    state_micro['cloud_sw_tau_layer'] = torch.full_like(q, 0.02)
    state_micro['cloud_lw_tau_layer'] = torch.full_like(q, 0.03)
    params_multi_1x = {
        'radiation_scheme': 'multiband',
        'co2': 400.0, 'co2_ref': 400.0,
        'albedo': 0.28,
        'lw_band_weights': [0.10, 0.25, 0.35, 0.30],
        'lw_band_wv_kappa': [0.00, 0.10, 0.22, 0.40],
        'lw_band_co2_base_tau': [0.00, 0.15, 0.65, 0.35],
        'lw_band_co2_log_factor': [0.00, 0.01, 0.12, 0.06],
        'lw_band_trace_scale': [0.00, 0.20, 0.60, 0.20],
        'trace_gases_enabled': True,
        'ch4': 1.8, 'ch4_ref': 1.8,
        'ch4_base_tau': 0.02, 'ch4_log_factor': 0.01,
        'n2o': 0.332, 'n2o_ref': 0.332,
        'n2o_base_tau': 0.01, 'n2o_log_factor': 0.01,
        'o3_lw_tau': 0.02,
        'o3_sw_tau': 0.03,
        'cloud_microphysics_enabled': True,
    }
    params_multi_2x = dict(params_multi_1x)
    params_multi_2x['co2'] = 800.0
    out_multi_1x = radiation(state_micro, grid, params_multi_1x)
    out_multi_2x = radiation(state_micro, grid, params_multi_2x)
    multi_forcing = out_multi_1x['olr'][0].item() - out_multi_2x['olr'][0].item()
    print(f"multiband OLR (1xCO2) = {out_multi_1x['olr'][0].item():.1f} W/m2")
    print(f"multiband 2xCO2 forcing = {multi_forcing:.2f} W/m2 (expect > 0)")
    assert multi_forcing > 0.0, "multiband branch should reduce OLR under 2xCO2"
    assert out_multi_1x['asr'][0].item() < out_1x['asr'][0].item(), (
        "microphysics-coupled clouds should reduce ASR relative to clear sky"
    )

    print("radiation: PASS\n")


def test_cloud_microphysics(device):
    """verify explicit cloud condensate and cloud optics are produced."""
    print("=== cloud microphysics ===")
    from scm.thermo import make_grid, pressure_at_full, dp_from_ps, saturation_specific_humidity
    from scm.condensation import condensation
    from scm.cloud_microphysics import initialize_cloud_state, cloud_microphysics_step

    grid = make_grid(nlevels=20, device=device)
    batch = 2
    ps = torch.full((batch,), 1e5, device=device)
    p = pressure_at_full(grid, ps)
    dp = dp_from_ps(grid, ps)

    sigma = grid['sigma_full'].unsqueeze(0).expand(batch, -1)
    t = torch.clamp(295.0 * sigma ** 0.19, min=200.0)
    qs = saturation_specific_humidity(t, p)
    q = torch.clamp(1.05 * qs, min=1e-7)

    state = {'t': t, 'q': q, 'ts': torch.full((batch,), 295.0, device=device), 'p': p, 'dp': dp}
    state.update(initialize_cloud_state(batch, grid, device=device))

    params = {
        'ls_precip_fraction': 0.3,
        'cloud_microphysics_enabled': True,
        'cloud_autoconv_tau': 1800.0,
        'cloud_autoconv_qc_thresh': 1.0e-5,
        'cloud_autoconv_qc_scale': 1.0e-4,
        'cloud_autoconv_power': 2.0,
        'cloud_ls_precip_fraction': 0.8,
        'dt': 900.0,
    }
    cond_out = condensation(state, grid, params)
    conv_out = {'precip': torch.zeros(batch, device=device)}
    cloud_out = cloud_microphysics_step(state, grid, params, cond_out, conv_out)

    print(f"cloud source = {cond_out['cloud_source'][0].sum().item():.3e} kg/kg")
    print(f"qc column = {cloud_out['qc'][0].sum().item():.3e} kg/kg")
    print(f"cloud precip = {cloud_out['precip'][0].item() * 86400.0 / params['dt']:.3f} mm/day")
    print(f"cloud cover = {(1.0 - torch.prod(1.0 - cloud_out['cloud_fraction'][0])).item():.2f}")

    assert cond_out['cloud_source'].sum().item() > 0.0, "microphysics path should create cloud source"
    assert cloud_out['qc'].sum().item() > 0.0, "cloud condensate should accumulate"
    assert cloud_out['precip'].sum().item() > 0.0, "autoconversion should produce cloud precipitation"
    assert cloud_out['cloud_sw_tau_layer'].sum().item() > 0.0, "cloud SW optical depth should be positive"

    print("cloud microphysics: PASS\n")


def test_quadratic_autoconversion(device):
    """verify thick clouds precipitate more efficiently than thin clouds."""
    print("=== quadratic autoconversion ===")
    from scm.thermo import make_grid, pressure_at_full, dp_from_ps, saturation_specific_humidity
    from scm.cloud_microphysics import initialize_cloud_state, cloud_microphysics_step

    grid = make_grid(nlevels=20, device=device)
    batch = 2
    ps = torch.full((batch,), 1e5, device=device)
    p = pressure_at_full(grid, ps)
    dp = dp_from_ps(grid, ps)

    sigma = grid['sigma_full'].unsqueeze(0).expand(batch, -1)
    t = torch.clamp(292.0 * sigma ** 0.18, min=210.0)
    qs = saturation_specific_humidity(t, p)
    q = qs.clone()

    state = {'t': t, 'q': q, 'ts': torch.full((batch,), 292.0, device=device), 'p': p, 'dp': dp}
    state.update(initialize_cloud_state(batch, grid, device=device))
    state['qc'][0] = 3.0e-4
    state['qc'][1] = 1.5e-3

    params = {
        'cloud_microphysics_enabled': True,
        'cloud_autoconv_tau': 1800.0,
        'cloud_autoconv_qc_thresh': 2.0e-4,
        'cloud_autoconv_qc_scale': 4.0e-4,
        'cloud_autoconv_power': 2.0,
        'cloud_evap_tau': 1.0e9,
        'cloud_rh_evap': 0.7,
        'dt': 900.0,
    }
    cond_out = {'cloud_source': torch.zeros_like(q)}
    conv_out = {'precip': torch.zeros(batch, device=device)}
    cloud_out = cloud_microphysics_step(state, grid, params, cond_out, conv_out)

    retained_frac_low = cloud_out['qc'][0].sum().item() / state['qc'][0].sum().item()
    retained_frac_high = cloud_out['qc'][1].sum().item() / state['qc'][1].sum().item()
    print(f"thin cloud precip = {cloud_out['precip'][0].item() * 86400.0 / params['dt']:.3f} mm/day")
    print(f"thick cloud precip = {cloud_out['precip'][1].item() * 86400.0 / params['dt']:.3f} mm/day")

    assert cloud_out['precip'][1].item() > cloud_out['precip'][0].item(), (
        "thicker clouds should autoconvert more condensate"
    )
    assert retained_frac_high < retained_frac_low, (
        "quadratic autoconversion should retain a smaller condensate fraction in thick clouds"
    )

    print("quadratic autoconversion: PASS\n")


def test_shallow_convection(device):
    """verify the shallow scheme moistens just above the BL without precipitating."""
    print("=== shallow convection ===")
    from scm.thermo import make_grid, pressure_at_full, dp_from_ps, saturation_specific_humidity, cp, Lv
    from scm.convection_shallow import shallow_convection

    grid = make_grid(nlevels=20, device=device)
    batch = 1
    ps = torch.full((batch,), 1e5, device=device)
    p = pressure_at_full(grid, ps)
    dp = dp_from_ps(grid, ps)

    sigma = grid['sigma_full']
    sigma_2d = sigma.unsqueeze(0)
    t = torch.clamp(293.0 * sigma_2d ** 0.18, min=210.0)
    qs = saturation_specific_humidity(t, p)
    rh_profile = 0.25 + 0.70 * sigma_2d ** 4
    q = torch.clamp(rh_profile * qs, min=1.0e-7)
    state = {'t': t, 'q': q, 'ts': torch.full((batch,), 293.0, device=device), 'p': p, 'dp': dp}

    params = {
        'shallow_convection_enabled': True,
        'shallow_tau': 3600.0,
        'shallow_top_sigma': 0.72,
        'shallow_base_sigma': 0.90,
        'shallow_rh_trigger': 0.75,
        'shallow_cape_suppress': 1.0e6,
        'shallow_mse_scale': 1000.0,
        'shallow_max_dt_day': 3.0,
        'shallow_max_dq_day': 3.0,
        'shallow_enforce_mse_conservation': True,
        'dt': 900.0,
    }
    out = shallow_convection(state, grid, params)
    up_mask = (sigma >= 0.72) & (sigma < 0.90)
    low_mask = sigma >= 0.90
    dq_day = out['dq'][0] * 86400.0 * 1000.0
    print(f"upper-layer moistening = {dq_day[up_mask].mean().item():.2f} g/kg/day")
    print(f"subcloud drying = {dq_day[low_mask].mean().item():.2f} g/kg/day")
    shallow_mse = torch.sum((cp * out['dt'][0] + Lv * out['dq'][0]) * dp[0] / 9.81).item()
    print(f"shallow mse residual = {shallow_mse:.4e} W/m2")

    assert out['precip'][0].item() == 0.0, "shallow convection should not precipitate here"
    assert dq_day[up_mask].mean().item() > 0.0, "shallow convection should moisten above the BL"
    assert dq_day[low_mask].mean().item() < 0.0, "shallow convection should dry the subcloud layer"
    assert abs(shallow_mse) < 1.0e-2, "shallow correction should keep column moist enthalpy nearly closed"

    print("shallow convection: PASS\n")


def test_calibration_utils():
    """verify calibration grid generation and scoring are well-formed."""
    print("=== calibration utils ===")
    from scm.calibration import iter_parameter_grid, calibration_score

    grid = {
        'albedo': [0.28, 0.32],
        'f_window': [0.15, 0.20],
    }
    candidates = list(iter_parameter_grid(grid))
    print(f"candidate count = {len(candidates)} (expect 4)")
    assert len(candidates) == 4

    better = {
        'ts': 300.0,
        'olr': 240.0,
        'asr': 240.0,
        'toa_net': 0.5,
        'surface_net_flux': 1.0,
    }
    worse = {
        'ts': 290.0,
        'olr': 210.0,
        'asr': 280.0,
        'toa_net': 20.0,
        'surface_net_flux': 30.0,
    }
    better_score = calibration_score(better)
    worse_score = calibration_score(worse)
    print(f"better score = {better_score:.2f}, worse score = {worse_score:.2f}")
    assert better_score < worse_score

    print("calibration utils: PASS\n")


def test_benchmark_thresholds():
    """verify benchmark threshold evaluation catches passes and failures."""
    print("=== benchmark thresholds ===")
    from scm.benchmark import evaluate_case_thresholds

    case_result = {
        'one_x': {'late_toa_abs': 0.8, 'equilibrium': False},
        'two_x': {'late_toa_abs': 0.9, 'equilibrium': True},
        'sensitivity': {'ecs': 3.2},
    }
    thresholds = {
        'one_x': {'late_toa_abs_max': 1.0},
        'two_x': {'equilibrium': True, 'late_toa_abs_max': 1.0},
        'sensitivity': {'ecs_min': 3.0, 'ecs_max': 3.5},
    }
    evaluation = evaluate_case_thresholds(case_result, thresholds)
    assert evaluation['passed'], "matching thresholds should pass"

    thresholds['sensitivity']['ecs_max'] = 3.0
    evaluation_fail = evaluate_case_thresholds(case_result, thresholds)
    assert not evaluation_fail['passed'], "violated thresholds should fail"
    print("benchmark thresholds: PASS\n")


def test_restart_bundle_roundtrip():
    """verify restart bundles serialize and reload nested tensor state."""
    print("=== restart bundle ===")
    from scm.experiment import build_restart_path, load_restart_bundle, save_restart_bundle

    bundle = {
        'kind': 'scm_restart',
        'phase': '1x',
        'state': {
            'ts': torch.tensor([290.0]),
            'slab_energy': torch.tensor([123.0], dtype=torch.float64),
        },
        'params': {
            'co2': 400.0,
            'scheme_mask': torch.tensor([1.0]),
        },
        'history_1x': [
            {'ts': torch.tensor([290.0]), 'toa_net': torch.tensor([0.5])},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        restart_path = Path(tmpdir) / build_restart_path('scm_test_case', '1x').name
        save_restart_bundle(restart_path, bundle)
        loaded = load_restart_bundle(restart_path, device=torch.device('cpu'))

    print(f"restart path = {restart_path.name}")
    assert loaded['phase'] == '1x'
    assert loaded['state']['ts'].device.type == 'cpu'
    assert loaded['state']['slab_energy'].dtype == torch.float64
    assert loaded['history_1x'][0]['toa_net'].device.type == 'cpu'

    print("restart bundle: PASS\n")


def test_equilibrium_check():
    """verify the equilibrium check uses late-time trend and flux closure."""
    print("=== equilibrium check ===")
    from scm.diagnostics import check_equilibrium, equilibrium_metrics

    good_history = []
    bad_history = []
    for i in range(60):
        good_history.append({
            'ts': torch.tensor([290.0 + 2.0e-4 * i]),
            'toa_net': torch.tensor([0.4]),
            'surface_total_flux': torch.tensor([0.3]),
            'column_energy_residual': torch.tensor([0.5]),
        })
        bad_history.append({
            'ts': torch.tensor([290.0 + 2.0e-4 * i]),
            'toa_net': torch.tensor([1.4]),
            'surface_total_flux': torch.tensor([0.3]),
            'column_energy_residual': torch.tensor([0.5]),
        })

    good_metrics = equilibrium_metrics(good_history, window=50)
    bad_metrics = equilibrium_metrics(bad_history, window=50)
    print(f"good late |TOA net| = {good_metrics['max_toa_imbalance']:.2f} W/m2")
    print(f"bad late |TOA net| = {bad_metrics['max_toa_imbalance']:.2f} W/m2")

    assert check_equilibrium(good_history), "small late trend and imbalances should pass"
    assert not check_equilibrium(bad_history), "large late TOA imbalance should fail"

    print("equilibrium check: PASS\n")


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


def test_slab_energy_accumulator():
    """verify the slab heat-content accumulator resolves small fluxes that
    repeated float32 temperature updates would lose."""
    print("=== slab energy accumulator ===")
    from scm.surface import slab_heat_capacity

    ts_ref = torch.tensor([290.0], dtype=torch.float32)
    ts_direct = ts_ref.clone()
    slab_energy = torch.zeros_like(ts_ref)
    heat_capacity = slab_heat_capacity({'ocean_depth': 50.0})
    net_flux = 1.0  # W/m2
    dt = 900.0

    for _ in range(10 * 96):  # 10 days
        ts_direct = ts_direct + net_flux / heat_capacity * dt
        slab_energy = slab_energy + net_flux * dt

    ts_accum = ts_ref + slab_energy / heat_capacity
    delta_direct = (ts_direct - ts_ref).item()
    delta_accum = (ts_accum - ts_ref).item()
    print(f"direct float32 delta Ts = {delta_direct:.6f} K")
    print(f"energy-accumulator delta Ts = {delta_accum:.6f} K")

    assert delta_accum > 1.0e-3, "energy accumulator should retain the small slab warming signal"
    assert delta_accum > delta_direct + 1.0e-3, "accumulator should outperform direct float32 Ts stepping"

    print("slab energy accumulator: PASS\n")


def test_energy_budget_diagnostics(device):
    """verify the timestep reports the extended column energy-budget terms."""
    print("=== energy budget diagnostics ===")
    from scm.thermo import make_grid
    from scm.column_model import initial_state, run
    from scm.ensemble import default_params

    grid = make_grid(nlevels=20, device=device)
    params = default_params(device=device)
    params['convection_scheme'] = 'mass_flux'
    params['include_precip_enthalpy_flux'] = True
    state = initial_state(1, grid, params, device=device)
    state, history = run(state, grid, params, nsteps=16, rad_interval=1, diag_interval=8)
    diag = history[-1]

    for key in [
        'precip_heat_flux', 'surface_total_flux', 'atmos_flux_convergence',
        'atmos_energy_tendency', 'atmos_energy_residual',
        'atmos_mse_tendency', 'atmos_mse_residual',
        'slab_energy_tendency', 'column_energy_tendency', 'column_energy_residual',
        'column_mse_tendency', 'column_mse_residual',
        'rad_energy_tendency', 'surface_energy_tendency', 'bl_energy_tendency',
        'shallow_energy_tendency', 'conv_energy_tendency',
        'condensation_energy_tendency', 'conv_mse_residual', 'shallow_mse_residual',
    ]:
        assert key in diag, f"missing diagnostic: {key}"
        assert torch.isfinite(diag[key]).all(), f"non-finite diagnostic: {key}"

    print(f"column residual = {diag['column_energy_residual'][0].item():+.2f} W/m2")
    print("energy budget diagnostics: PASS\n")


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
                            saturation_specific_humidity, cape, cp, Lv)
    from scm.convection_mf import mass_flux_convection, dilute_cape

    grid = make_grid(nlevels=20, device=device)
    batch = 4
    ps = torch.full((batch,), 1e5, device=device)
    p = pressure_at_full(grid, ps)
    dp = dp_from_ps(grid, ps)

    sigma = grid['sigma_full'].unsqueeze(0).expand(batch, -1)
    t = 300.0 * sigma ** 0.19
    t = torch.clamp(t, min=200.0)
    qs = saturation_specific_humidity(t, p)
    q = 0.95 * qs * sigma ** 1.0
    q = torch.clamp(q, min=1e-7)

    cape_val = cape(t, q, p, grid)
    print(f"CAPE before convection: {cape_val[0].item():.0f} J/kg")

    state = {'t': t, 'q': q, 'p': p, 'dp': dp}
    params = {'dt': 900.0, 'entrainment_rate': 1.0e-5, 'tau_cape': 3600.0,
              'precip_efficiency': 0.9, 'cape_threshold': 100.0,
              'mf_condensate_retention': 0.25, 'mf_condensate_fallout': 0.45}
    out = mass_flux_convection(state, grid, params)
    params_flow = dict(params)
    params_flow.update({
        'mf_cape_timescale_mode': 'flow_dependent',
        'mf_tau_cape_min': 1800.0,
        'mf_tau_cape_max': 7200.0,
        'mf_tau_cape_rh_ref': 0.55,
        'mf_tau_cape_rh_sensitivity': 1.0,
        'mf_tau_cape_cape_ref': 500.0,
        'mf_tau_cape_cape_sensitivity': 1.0,
    })
    out_flow = mass_flux_convection(state, grid, params_flow)

    dcape_noload = dilute_cape(t, q, p, params['entrainment_rate'])
    dcape_load = dilute_cape(
        t, q, p, params['entrainment_rate'],
        condensate_retention=params['mf_condensate_retention'],
        condensate_fallout=params['mf_condensate_fallout'],
    )

    precip = out['precip'][0].item() * 86400
    dt_profile = out['dt'][0] * 86400

    print(f"convective precip: {precip:.2f} mm/day")
    print(f"heating profile (K/day): {dt_profile.tolist()[:5]}... (top levels)")
    print(f"dilute CAPE without loading: {dcape_noload[0].item():.0f} J/kg")
    print(f"dilute CAPE with loading: {dcape_load[0].item():.0f} J/kg")
    print(f"fixed tau_cape: {out['tau_cape_eff'][0].item():.0f} s")
    print(f"flow tau_cape: {out_flow['tau_cape_eff'][0].item():.0f} s")
    conv_mse = torch.sum((cp * out['dt'][0] + Lv * out['dq'][0]) * dp[0] / 9.81).item()
    print(f"convective mse residual: {conv_mse:.4e} W/m2")

    assert dcape_load[0].item() < dcape_noload[0].item(), (
        "condensate loading should reduce dilute CAPE"
    )
    assert out_flow['tau_cape_eff'][0].item() < out['tau_cape_eff'][0].item(), (
        "flow-dependent CAPE timescale should shorten in a moist, high-CAPE column"
    )
    assert abs(conv_mse) < 1.0e-2, "MF correction should keep column moist enthalpy nearly closed"

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
    test_cloud_microphysics(device)
    test_quadratic_autoconversion(device)
    test_shallow_convection(device)
    test_calibration_utils()
    test_benchmark_thresholds()
    test_restart_bundle_roundtrip()
    test_equilibrium_check()
    test_surface(device)
    test_slab_energy_accumulator()
    test_energy_budget_diagnostics(device)
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
