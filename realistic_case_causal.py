"""
QuTiP mesolve check of the causal (end-of-slot) closed form in causal_convention.py.

Same four-branch instrument as realistic_case_EXACT.py, but every branch is evaluated at
the END OF THE PULSE SLOT, t_read + tau + t_pulse (deterministic controller timing; no
drive in the slot when the report is "g"), instead of at the fixed clock t_eval.

Run on your own machine (QuTiP 5.3.1):
    python realistic_case_causal.py
and paste back the last four lines of the output.
"""
import numpy as np
import qutip as qt
from causal_convention import P_causal_cf, tau_crit_causal, PAES

T1, nth, T_phi = 50e3, 0.01, 100e3
gamma_1, gamma_phi = 1.0 / T1, 1.0 / (2 * T_phi)       # L3 = sqrt(gamma_phi) sz, as in the archive
t_read, t_pulse = 300.0, 20.0

sz, sx, sm = qt.sigmaz(), qt.sigmax(), qt.destroy(2)
sp = sm.dag()
c_ops = [np.sqrt(gamma_1 * (1 + nth)) * sm, np.sqrt(gamma_1 * nth) * sp, np.sqrt(gamma_phi) * sz]
zeroH = qt.Qobj(np.zeros((2, 2)))
H_pulse = (np.pi / t_pulse / 2.0) * sx
rho_e, rho_g = qt.ket2dm(qt.basis(2, 1)), qt.ket2dm(qt.basis(2, 0))


def p_e(rho):
    return np.real((rho * sp * sm).tr())


P_READ = p_e(qt.mesolve(zeroH, rho_e, np.linspace(0, t_read, 301), c_ops=c_ops).states[-1])


def branch(rho, tau, pulse):
    if tau > 0:
        rho = qt.mesolve(zeroH, rho, np.linspace(0, tau, int(tau) + 1), c_ops=c_ops).states[-1]
    H = H_pulse if pulse else zeroH
    rho = qt.mesolve(H, rho, np.linspace(0, t_pulse, 201), c_ops=c_ops).states[-1]
    return p_e(rho)


def P_mesolve(tau, P_AE):
    return (P_READ * ((1 - P_AE) * branch(rho_e, tau, True) + P_AE * branch(rho_e, tau, False))
            + (1 - P_READ) * ((1 - P_AE) * branch(rho_g, tau, False) + P_AE * branch(rho_g, tau, True)))


if __name__ == "__main__":
    print(f"qutip {qt.__version__}, numpy {np.__version__}")
    print(f"P_read (mesolve) = {P_READ:.6f}")
    taus = np.arange(0, 1050, 50.0)
    dmax = 0.0
    for tau in taus:
        for p in PAES:
            dmax = max(dmax, abs(P_mesolve(tau, p) - P_causal_cf(tau, p)))
    roots = []
    for p in PAES:
        p0 = P_mesolve(0.0, p)
        lo, hi = 0.0, 1000.0
        for _ in range(40):                       # bisection on the mesolve curve itself
            mid = 0.5 * (lo + hi)
            if P_mesolve(mid, p) - p0 < 0.01:
                lo = mid
            else:
                hi = mid
        roots.append(0.5 * (lo + hi))
    print(f"max |mesolve - closed form| over the 21x5 grid: {dmax:.2e}")
    print("tau_crit from mesolve (bisection): " + ", ".join(f"{r:.2f}" for r in roots) + " ns")
    print("tau_crit from closed form:          " + ", ".join(f"{tau_crit_causal(p):.2f}" for p in PAES) + " ns")
