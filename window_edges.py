"""Precise window edges (root-finding on the exact qubit-resonator model), P_AE = 1 %, eps_t = 2 %."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from residual_photons_window import branch_curves, total, TWO_PI, MHz

eps, et = 0.01, 0.02
def P_exact(chi, kap, n0, tau, N):
    return total(branch_curves(chi, kap, n0, np.array([0.0, tau]), N), eps)[1] if tau > 0 else \
           total(branch_curves(chi, kap, n0, np.array([0.0, 1e-9]), N), eps)[0]

devices = [("Walter2017", 7.9, 37.5, 2.5, (0, 28), (28, 700)),
           ("Hazra2025_100", 3.2, 11.6, 7.6, (0, 91), (91, 700)),
           ("Hazra2025_240", 3.2, 11.6, 1.2, (0, 65), (65, 700)),
           ("CampagneIbarcq2013", 0.78, 1.83, 1.4, (0, 127), (127, 700)),
           ("slow_n1", 0.95, 0.55, 1.0, (0, 190), (190, 700))]
for lab, cM, kM, n0, lo_br, hi_br in devices:
    chi, kap = TWO_PI * cM * MHz, TWO_PI * kM * MHz
    N = int(np.ceil(n0 + 7 * np.sqrt(n0) + 8))
    f = lambda t: P_exact(chi, kap, n0, t, N) - et
    lo = brentq(f, *lo_br, xtol=1e-3) if f(lo_br[0]) > 0 else 0.0
    hi = brentq(f, *hi_br, xtol=1e-3)
    opt = minimize_scalar(lambda t: P_exact(chi, kap, n0, t, N), bounds=(max(lo, 1e-6), hi), method="bounded",
                          options={"xatol": 0.05})
    print(f"{lab}: tau_- = {lo:.1f} ns, tau* = {opt.x:.1f} ns, P(tau*) = {opt.fun*100:.3f} %, tau_+ = {hi:.1f} ns")

# closed forms, equations (17) and (18) of the manuscript
from causal_convention import P_read, q_c, P0_causal, G1
from window_analytic import f1
w, room = P_read * (1 - eps), et - P0_causal(eps)
for lab, cM, kM, n0, *_ in devices:
    chi, kap = TWO_PI * cM * MHz, TWO_PI * kM * MHz
    F1 = f1(chi, kap)
    tstar = np.log(w * F1 * n0 * kap / (q_c(eps) * G1)) / (kap - G1)
    tminus = max(0.0, np.log(w * F1 * n0 / room) / kap)
    print(f"{lab}: f1 = {F1:.4f}, closed-form tau* = {tstar:.1f} ns, tau_- = {tminus:.1f} ns, "
          f"nbar(tau*) = {n0*np.exp(-kap*tstar):.3f}")
