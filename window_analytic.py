"""
Major 4, closed-form version of the latency window.

For small residual photon number the extra pulse error is linear in n(tau) = n0 e^{-kappa tau}:
    P(tau) ~ P_floor + q_c (1 - e^{-Gamma1 tau}) + w_ee f1 n0 e^{-kappa tau}
where f1 = d(pulse error)/d n at n -> 0 (one number per device, from a 4-level Fock simulation).
Then
    optimum delay   tau*  = (1/(kappa - Gamma1)) ln[ w_ee f1 n0 kappa / (q_c Gamma1) ]
    lower bound     tau_lo ~ (1/kappa) ln[ w_ee f1 n0 / room ]        (room = eps_target - P_floor)
    upper bound     tau_hi = -(1/Gamma1) ln(1 - room / q_c)
and the window exists iff P(tau*) <= eps_target. Each value is compared with the exact numerics.
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.sparse.linalg import expm_multiply
from causal_convention import P_read, P0_causal, q_c, G1, t_pulse, P_causal_cf
from residual_photons_window import build, coherent_dm, branch_curves, total, TWO_PI, MHz


def pulse_error_e(chi, kappa, n, N):
    """Excited population left after the pulse for a qubit projected to |e>, field coherent with mean n, tau = 0."""
    L0, Lp, Pe_op, d = build(chi, kappa, N)
    v = np.kron(np.diag([1.0, 0.0]), coherent_dm(n, N)).reshape(-1).astype(complex)
    r = expm_multiply(Lp * t_pulse, v).reshape(d, d)
    return np.real(np.trace(Pe_op @ r))


def f1(chi, kappa):
    e0 = pulse_error_e(chi, kappa, 0.0, 4)
    n = 1e-4
    return (pulse_error_e(chi, kappa, n, 6) - e0) / n


def analytic_window(chi, kappa, n0, eps, target):
    w = P_read * (1 - eps)
    F = f1(chi, kappa)
    qc, Pf = q_c(eps), P0_causal(eps)
    room = target - Pf
    P = lambda t: Pf + qc * (1 - np.exp(-G1 * t)) + w * F * n0 * np.exp(-kappa * t)
    tstar = np.log(w * F * n0 * kappa / (qc * G1)) / (kappa - G1)
    res = dict(f1=F, tau_star=tstar, P_star=P(tstar),
               tau_hi=-np.log(1 - room / qc) / G1 if room > 0 else np.nan)
    if P(tstar) <= target and P(0) > target:
        res['tau_lo'] = brentq(lambda t: P(t) - target, 0, tstar)
    elif P(0) <= target:
        res['tau_lo'] = 0.0
    else:
        res['tau_lo'] = np.nan
        res['tau_hi'] = np.nan
    if not np.isnan(res['tau_hi']) and P(tstar) <= target:
        res['tau_hi'] = brentq(lambda t: P(t) - target, tstar, 1e6)
    return res


if __name__ == "__main__":
    eps, target = 0.01, 0.02
    devices = [
        ("Walter 2017",            7.9,  37.5,  2.5, "verified"),
        ("Hazra 2025 (100 ns)",    3.2,  11.6,  7.6, "verified"),
        ("Hazra 2025 (240 ns)",    3.2,  11.6,  1.2, "verified"),
        ("Campagne-Ibarcq 2013",   0.78,  1.83, 1.4, "verified"),
        ("Riste 2012",             0.95,  0.55, 45.0, "n0 PARTIAL (read via summariser)"),
    ]
    print("device                 | f1 (per photon) | tau* analytic | P(tau*) | window analytic (P<=2%)")
    for lab, chi_M, kap_M, n0, note in devices:
        chi, kap = TWO_PI * chi_M * MHz, TWO_PI * kap_M * MHz
        r = analytic_window(chi, kap, n0, eps, target)
        win = "closed" if np.isnan(r['tau_lo']) else f"[{r['tau_lo']:.0f}, {r['tau_hi']:.0f}] ns"
        print(f"{lab:22s} | {r['f1']:.4e}      | {r['tau_star']:7.1f} ns    | {r['P_star']*100:.3f}% | {win}   ({note})")

    # exact check for the Riste-like case on a coarse grid (largest Fock space)
    chi, kap, n0 = TWO_PI * 0.95 * MHz, TWO_PI * 0.55 * MHz, 45.0
    taus = np.linspace(0, 3000, 31)
    N = int(np.ceil(n0 + 7 * np.sqrt(n0) + 8))
    P = total(branch_curves(chi, kap, n0, taus, N), eps)
    i = np.argmin(P)
    print(f"\n[exact] Riste-like case, N = {N}: min P = {P[i]*100:.3f}% at tau = {taus[i]:.0f} ns; "
          f"P <= 2% anywhere on the grid: {bool((P <= target).any())}")
    print("        P(tau) on grid: " + ", ".join(f"{t:.0f}:{p*100:.2f}%" for t, p in zip(taus[::3], P[::3])))
