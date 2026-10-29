"""
Every number quoted in the revised manuscript (Physica Scripta version), printed in the order it appears.
NumPy/SciPy only. Photon-model curves are cached in photon_curves.npz for make_figures_v2.py.
"""
import numpy as np
from scipy.optimize import brentq
from causal_convention import (T1, Tphi, nth, t_read, t_pulse, t_eval, g1, G1, Peq, a, b, C, P_read,
                               q_c, P0_causal, P_causal_cf, tau_crit_causal, tau_abs_causal, weights)
from residual_photons_window import branch_curves, total, TWO_PI, MHz
from window_analytic import f1

G2 = G1 / 2 + 1 / Tphi
kap_pulse = np.exp(-(G1 + G2) * t_pulse / 2)
F = np.exp(-G1 * t_pulse)
T_slot = t_eval - t_read - t_pulse
PAES = [0.01, 0.02, 0.03, 0.04, 0.05]
d = 0.01

print("== Sec. 2 ==")
print(f"Gamma1 = {G1:.4e} ns^-1, 1/Gamma1 = {1/G1/1e3:.3f} us, Peq = {Peq*100:.4f} %, 1/Gamma2 = {1/G2/1e3:.3f} us")
print(f"P_read = {P_read:.6f}; pulse map a = {a:.12f}, s = {b:.12f}; Bloch kappa = {kap_pulse:.12f}; "
      f"|s|-kappa = {abs(b)-kap_pulse:.1e}; a-(1+kappa)/2 = {a-(1+kap_pulse)/2:.2e}; F = {F:.9f}")

print("\n== Table 1 (causal, delta = 1 pp) ==")
for p in PAES:
    tc = tau_crit_causal(p)
    rule = d * T1 / (P_read * (1 - 2 * p))
    fo = d / (g1 * (1 + nth))
    pp = P_read * (1 - p) + (1 - P_read) * p
    tf = np.log(1 + d * np.exp(G1 * T_slot) / (pp * C)) / G1
    print(f"{p*100:.0f}% | q={q_c(p):.4f} | P0={P0_causal(p)*100:.3f}% | tau_c={tc:.2f} | rule={rule:.2f} ({100*(rule-tc)/tc:+.2f}%)"
          f" | first-order={fo:.2f} ({100*(fo-tc)/tc:+.2f}%) | fixed-slot={tf:.2f} | p_pulse={pp:.5f}")
print(f"prefactor tau_c / rule: {min(tau_crit_causal(p)/(d*T1/(P_read*(1-2*p))) for p in PAES):.4f}"
      f"-{max(tau_crit_causal(p)/(d*T1/(P_read*(1-2*p))) for p in PAES):.4f}")
print(f"P0 at P_AE=0: {P0_causal(0.0)*100:.4f} %; eps = 0.5: q = {q_c(0.5):.2e}")
print("budget: " + ", ".join(f"{x*100:.1f} pp -> {tau_crit_causal(0.01, budget=x):.1f} ns" for x in [0.005, 0.02]))
print("Delta: " + ", ".join(f"{D:.0f} ns -> {tau_crit_causal(0.01, Delta=D):.2f}" for D in [300, 1180]))
print(f"projection at start of window (P_e = 1): {tau_crit_causal(0.01, Pe=1.0):.2f} ns "
      f"(shift {tau_crit_causal(0.01)-tau_crit_causal(0.01, Pe=1.0):.2f} ns)")
for ee, eg in [(0.02, 0.005), (0.05, 0.005), (0.005, 0.02), (0.005, 0.005)]:
    print(f"asym eps_e={ee*100:.1f}% eps_g={eg*100:.1f}%: tau_c = {tau_crit_causal(ee, eg):.2f}")
for Pe in [0.9, 0.5, 0.2]:
    print(f"P_e={Pe}: tau_c = {tau_crit_causal(0.01, Pe=Pe):.2f} ns")
Pe_min = brentq(lambda x: q_c(0.01, Pe=x) - d, 1e-6, 1)
print(f"existence (causal): q > delta requires P_e > {Pe_min:.4f}")

print("\n== Fixed-slot remark ==")
for p in [0.01, 0.05]:
    pp = P_read * (1 - p) + (1 - P_read) * p
    print(f"P_AE={p*100:.0f}%: bound = {pp*C*(1-np.exp(-G1*T_slot))*100:.4f} pp")

print("\n== Table 2 (absolute criterion) ==")
for et in [0.01, 0.02, 0.03]:
    print(f"eps_t={et*100:.0f}%: " + " | ".join(
        (lambda t: "none" if np.isnan(t) else f"{t:.0f}")(tau_abs_causal(et, p)) for p in [0.001, 0.0025, 0.005, 0.01, 0.02]))
print("floors: " + ", ".join(f"{p*100:.2f}% -> {P0_causal(p)*100:.3f}%" for p in [0.001, 0.0025, 0.005, 0.01, 0.02]))

print("\n== Published latencies (causal excess, P_AE = 1 %) ==")
for lab, tau in [("Salathe 2018 electronics", 219.0), ("Gebauer 2020 electronics", 428.0)]:
    print(f"{lab}: tau = {tau:.0f} ns -> excess {q_c(0.01)*(1-np.exp(-G1*tau))*100:.3f} pp")

print("\n== Sec. 4: photon window (P_AE = 1 %, eps_t = 2 %) ==")
devices = [("Walter2017", 7.9, 37.5, 2.5), ("Hazra2025_100", 3.2, 11.6, 7.6),
           ("Hazra2025_240", 3.2, 11.6, 1.2), ("CampagneIbarcq2013", 0.78, 1.83, 1.4),
           ("slow_n5", 0.95, 0.55, 5.0), ("slow_n20", 0.95, 0.55, 20.0)]
taus = np.concatenate([np.linspace(0, 200, 201), np.linspace(205, 3000, 560)])
cache = {"taus": taus}
eps, et = 0.01, 0.02
for lab, chi_M, kap_M, n0 in devices:
    chi, kap = TWO_PI * chi_M * MHz, TWO_PI * kap_M * MHz
    N = int(np.ceil(n0 + 7 * np.sqrt(n0) + 8))
    # expm_multiply needs a uniform grid: run the two uniform pieces separately
    c1 = branch_curves(chi, kap, n0, taus[:201], N)
    c2 = branch_curves(chi, kap, n0, np.linspace(0, 3000, 601), N)
    P = np.concatenate([total(c1, eps), total(c2, eps)[41:]])
    cache[lab] = P
    i = np.argmin(P)
    ok = P <= et
    lo = taus[np.argmax(ok)] if ok.any() else np.nan
    hi = taus[len(ok) - 1 - np.argmax(ok[::-1])] if ok.any() else np.nan
    ff = f1(chi, kap)
    w = P_read * (1 - eps)
    ts = np.log(w * ff * n0 * kap / (q_c(eps) * G1)) / (kap - G1)
    print(f"{lab}: 1/kappa={1/kap:.1f} ns N={N} | P(0)={P[0]*100:.2f}% | tau*={taus[i]:.0f} ns P*={P[i]*100:.3f}% "
          f"| window [{lo:.0f}, {hi:.0f}] | f1={ff:.4f} | tau* analytic={ts:.1f}")
cache["noph"] = np.array([P_causal_cf(t, eps) for t in taus])
np.savez("photon_curves.npz", **cache)

print("\n== Sec. 5 ==")
room = et - P0_causal(eps)
th = lambda r: -np.log(1 - r / q_c(eps)) / G1 if r > 0 else np.nan
print(f"upper edge {th(room):.1f} ns; per 0.1 % leakage {0.001/(G1*q_c(eps)):.1f} ns; "
      + ", ".join(f"L={L*100:.2f}% -> {th(room-L):.1f}" for L in [0.0012, 0.0048, 0.0214]))
T2 = 1 / G2
print("spectator: " + ", ".join(f"{t:.0f} ns -> {50*(1-np.exp(-t/T2)):.3f} %" for t in [100, 500]))
print(f"projection incompleteness Gamma1/Gamma_m: Gm=1 -> {G1/1:.1e}; Gm=0.05 -> {G1/0.05:.1e}; Gm=0.01 -> {G1/0.01:.1e}")
