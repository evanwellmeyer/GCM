# %%
# diagnostic: run a single member to near-equilibrium, then analyze
# the radiation budget in detail to figure out why OLR is too high.

import torch
import sys
sys.path.insert(0, '.')

from scm.thermo import (
    make_grid, saturation_specific_humidity, g, cp, Lv, sigma_sb,
    pressure_at_full, dp_from_ps
)
from scm.column_model import initial_state, run, update_derived
from scm.radiation import radiation, compute_longwave, compute_shortwave
from scm.surface import surface_fluxes, slab_ocean_tendency
from scm.ensemble import default_params

device = 'cpu'
grid = make_grid(nlevels=20, device=device)
params = default_params(device=device)
params['convection_scheme'] = 'betts_miller'

sigma = grid['sigma_full']

# %%
# run to near-equilibrium (1000 days, single member)
print("running 1000 days to near-equilibrium...")
state = initial_state(1, grid, params, device=device)
state, history = run(state, grid, params,
                     nsteps=int(1000 * 86400 / 900),
                     rad_interval=8, diag_interval=int(50 * 86400 / 900))
print(f"final Ts = {state['ts'][0].item():.2f} K")
print(f"final OLR = {history[-1]['olr'][0].item():.1f} W/m2")
print(f"final precip = {history[-1]['precip_total'][0].item() * 86400:.2f} mm/day")

# %%
# analyze the equilibrated column
state = update_derived(state, grid)
t = state['t']
q = state['q']
ts = state['ts']
p = state['p']
dp = state['dp']

print(f"\n--- equilibrated column ---")
print(f"Ts = {ts[0].item():.2f} K")
qs = saturation_specific_humidity(t, p)
rh = q / qs.clamp(min=1e-10)
print(f"{'sigma':>8} {'T(K)':>8} {'q(g/kg)':>10} {'qs(g/kg)':>10} {'RH':>6}")
for k in range(20):
    print(f"{sigma[k].item():8.3f} {t[0,k].item():8.1f} "
          f"{q[0,k].item()*1000:10.4f} {qs[0,k].item()*1000:10.4f} "
          f"{rh[0,k].item():6.3f}")

# %%
# radiation budget
print(f"\n--- radiation analysis ---")
rad_out = radiation(state, grid, params)
lw_heat, lw_down_sfc, olr_full = compute_longwave(t, q, ts, p, dp, params)
sw_heat, sw_sfc = compute_shortwave(t, q, ts, p, dp, params)

# what the surface emits
sfc_emission = sigma_sb * ts[0].item()**4
print(f"surface blackbody emission: {sfc_emission:.1f} W/m2")
print(f"OLR: {olr_full[0].item():.1f} W/m2")
print(f"greenhouse effect: {sfc_emission - olr_full[0].item():.1f} W/m2 (want ~150)")

# column water vapor path
wvp = torch.sum(q * dp / g, dim=1)
print(f"column water vapor: {wvp[0].item():.1f} kg/m2 (Earth tropical ~40-50)")

# total longwave optical depth
kappa_wv = params.get('kappa_wv', 0.15)
co2_base_tau = params.get('co2_base_tau', 1.5)
tau_wv_total = kappa_wv * wvp[0].item()
print(f"total WV optical depth: {tau_wv_total:.2f} (at kappa_wv={kappa_wv})")
print(f"CO2 baseline optical depth: {co2_base_tau}")
print(f"total optical depth: {tau_wv_total + co2_base_tau:.2f}")

# what kappa_wv would we need for ~150 W/m2 greenhouse effect?
# greenhouse effect ≈ surface_emission - OLR
# for a gray atmosphere, OLR ≈ surface_emission / (1 + 0.75*tau)
# so tau ≈ (surface_emission/OLR - 1) / 0.75
target_olr = 240.0
target_tau = (sfc_emission / target_olr - 1.0) / 0.75
print(f"\ntarget total tau for OLR={target_olr}: {target_tau:.2f}")
print(f"current total tau: {tau_wv_total + co2_base_tau:.2f}")
needed_tau = target_tau - co2_base_tau
needed_kappa = needed_tau / max(wvp[0].item(), 1.0)
print(f"needed kappa_wv: {needed_kappa:.3f} (current: {kappa_wv})")

# %%
# energy budget
incoming = 1360.0 * 0.25
albedo = params.get('albedo', 0.1)
absorbed_solar = incoming * (1.0 - albedo)
print(f"\n--- energy budget ---")
print(f"absorbed solar: {absorbed_solar:.1f} W/m2")
print(f"OLR: {olr_full[0].item():.1f} W/m2")
print(f"TOA imbalance: {absorbed_solar - olr_full[0].item():.1f} W/m2")

sfc_out = surface_fluxes(state, grid, params)
print(f"SHF: {sfc_out['shf'][0].item():.1f} W/m2")
print(f"LHF: {sfc_out['lhf'][0].item():.1f} W/m2")

net_sfc = (rad_out['sw_absorbed_sfc'][0].item() + rad_out['lw_down_sfc'][0].item()
           - rad_out['lw_up_sfc'][0].item() - sfc_out['shf'][0].item()
           - sfc_out['lhf'][0].item())
print(f"net surface: {net_sfc:.1f} W/m2")

# %%
print(f"\n--- suggested parameter changes ---")
print(f"kappa_wv: {kappa_wv} -> {needed_kappa:.3f}")
print(f"(this should bring OLR from {olr_full[0].item():.0f} to ~{target_olr:.0f})")

