"""
EXACT (non-Monte-Carlo) computation of the realistic-case sweep -- CONDITIONED version.

WHY THIS WORKS WITHOUT MONTE CARLO:
sc_ops = [sigmaz()] is a QND measurement operator: it commutes with the population operator
sp*sm, so the ensemble-averaged population at the end of the readout window is the plain
mesolve result (verified: single-trajectory smesolve 0.9940 vs mesolve 0.99396). The only
random elements are two classical Bernoulli draws:
  (i)  the projective readout outcome: |e> with probability P_read, |g> otherwise;
  (ii) the assignment: the projected outcome is reported correctly with prob 1-P_AE.
Four combinations, each followed by a deterministic Lindblad evolution (pulse / no pulse):

  P_final = P_read     * [ (1-P_AE) P_pulse(|e>) + P_AE P_nopulse(|e>) ]
          + (1-P_read) * [ (1-P_AE) P_nopulse(|g>) + P_AE P_pulse(|g>) ]          (Eq. 8)

WHAT CHANGED vs. the previous version: both branches used to be evolved from the SAME
unconditioned mixed state rho_at_read (P_e ~ 0.994). That double-counts the fraction that
relaxed during the readout window and gives 1.21 % at P_AE = 0 instead of the thermal floor.
The branches are now evolved from the projected states |e> and |g>. tau_crit is unaffected
(the two versions differ by a tau-independent constant); P(0) drops by ~1.1 %.

This script also compares every grid point against the closed form of Sec. E
(closed_form_conditioned.py) and prints the maximum absolute difference -- that number
replaces the [VERIFY ...] placeholder in Sec. E of the manuscript.

Caveat: this exactness is a direct consequence of a QND (population-preserving) measurement
operator with a thresholded outcome. Measurement inefficiency / finite bandwidth / any
back-action that does NOT commute with the population operator, or feedback on the
continuous record, would require real Monte Carlo again (see Discussion).
"""

import numpy as np
import qutip as qt

T1 = 50e3
nth = 0.01
T_phi = 100e3
gamma_1 = 1.0 / T1
gamma_phi = 1.0 / (2 * T_phi)

t_read = 300.0
t_pulse = 20.0
t_eval = 1500.0

# ---------------------------------------------------------------------------------------
# Closed form of Sec. E. Imported from closed_form_conditioned.py when that file sits in the
# same directory; otherwise defined inline below so this script runs standalone. The inline
# copy must stay identical to the module -- if you edit one, edit the other. (A transcription
# error here would show up as a large max|mesolve - closed form|, not as a silent wrong result.)
# ---------------------------------------------------------------------------------------
try:
    from closed_form_conditioned import P_conditioned, tau_crit_analytic
    print("closed form: imported from closed_form_conditioned.py")
except ImportError:
    print("closed form: closed_form_conditioned.py not found -- using inline copy")
    _Gamma1 = gamma_1 * (1 + 2 * nth)                          # Eq. (11)
    _Peq = nth / (1 + 2 * nth)                                  # Eq. (11)
    _Gamma2 = _Gamma1 / 2 + 1.0 / T_phi                         # Eq. (13)
    _kappa = np.exp(-(_Gamma1 + _Gamma2) * t_pulse / 2)         # Eq. (12)
    _T_after = t_eval - t_read - t_pulse

    def _relax(P0, t):
        return _Peq + (P0 - _Peq) * np.exp(-_Gamma1 * t)

    _P_read_cf = _relax(1.0, t_read)

    def _P_pulse(P_s, tau):
        Pd = _relax(P_s, tau)
        return _relax((1 + _kappa) / 2 - _kappa * Pd, t_eval - t_read - tau - t_pulse)

    def _P_nopulse(P_s):
        return _relax(P_s, t_eval - t_read)

    def P_conditioned(tau, P_AE):
        """Eq. (8) in closed form."""
        from_e = (1 - P_AE) * _P_pulse(1.0, tau) + P_AE * _P_nopulse(1.0)
        from_g = (1 - P_AE) * _P_nopulse(0.0) + P_AE * _P_pulse(0.0, tau)
        return _P_read_cf * from_e + (1 - _P_read_cf) * from_g

    def tau_crit_analytic(P_AE, thr=0.01):
        """Eq. (15)."""
        p_pulse = _P_read_cf * (1 - P_AE) + (1 - _P_read_cf) * P_AE
        return np.log(1 + thr * np.exp(_Gamma1 * _T_after)
                      / (p_pulse * (1 + _kappa) * (0.5 - _Peq))) / _Gamma1

sz = qt.sigmaz()
sx = qt.sigmax()
sm = qt.destroy(2)
sp = sm.dag()
c_ops = [
    np.sqrt(gamma_1 * (1 + nth)) * sm,
    np.sqrt(gamma_1 * nth) * sp,
    np.sqrt(gamma_phi) * sz,
]
zeroH = qt.Qobj(np.zeros((2, 2)))


def p_e(rho):
    return np.real((rho * sp * sm).tr())


# Population at the end of the readout window for a qubit prepared in |e> (computed once).
rho_e = qt.ket2dm(qt.basis(2, 1))
rho_g = qt.ket2dm(qt.basis(2, 0))
t_list_read = np.linspace(0, t_read, int(t_read) + 1)
P_READ = p_e(qt.mesolve(zeroH, rho_e, t_list_read, c_ops=c_ops).states[-1])


def evolve_branch(rho_start, tau, pulse):
    """Delay tau, then (pulse | no pulse), then relax to t_eval. rho_start = projected state."""
    t_list_delay = np.linspace(t_read, t_read + tau, int(tau) + 1) if tau > 0 else [t_read, t_read]
    rho = qt.mesolve(zeroH, rho_start, t_list_delay, c_ops=c_ops).states[-1]
    t0, t1 = t_read + tau, t_read + tau + t_pulse
    H = (np.pi / t_pulse / 2.0) * sx if pulse else zeroH
    rho = qt.mesolve(H, rho, np.linspace(t0, t1, int(t_pulse) + 1), c_ops=c_ops).states[-1]
    rho = qt.mesolve(zeroH, rho, np.linspace(t1, t_eval, int(t_eval - t1) + 1), c_ops=c_ops).states[-1]
    return p_e(rho)


def exact_final_population(tau, P_AE):
    Pe1 = evolve_branch(rho_e, tau, pulse=True)    # projected |e>, reported e  -> pulse
    Pe0 = evolve_branch(rho_e, tau, pulse=False)   # projected |e>, reported g  -> no pulse
    Pg0 = evolve_branch(rho_g, tau, pulse=False)   # projected |g>, reported g  -> no pulse
    Pg1 = evolve_branch(rho_g, tau, pulse=True)    # projected |g>, reported e  -> pulse
    return (P_READ * ((1 - P_AE) * Pe1 + P_AE * Pe0)
            + (1 - P_READ) * ((1 - P_AE) * Pg0 + P_AE * Pg1))


if __name__ == "__main__":
    taus = np.arange(0, 1050, 50)
    PAEs = [0.01, 0.02, 0.03, 0.04, 0.05]

    print(f"Readout population before discrimination: {P_READ:.5f}\n")
    print("tau_ns," + ",".join(f"P_AE_{p*100:.0f}pct" for p in PAEs))
    max_diff = 0.0
    for tau in taus:
        row = []
        for p in PAEs:
            v = exact_final_population(float(tau), p)
            max_diff = max(max_diff, abs(v - P_conditioned(float(tau), p)))
            row.append(v)
        print(f"{int(tau)}," + ",".join(f"{v*100:.4f}" for v in row))

    print(f"\nMax |mesolve - closed form| over the 21x5 grid: {max_diff:.2e} absolute "
          f"({max_diff/0.02*1e6:.0f} ppm of a ~2% signal)  <- put this in Sec. E")
    print(f"Sanity: P_AE = 0 -> {exact_final_population(0.0, 0.0)*100:.3f}% (thermal floor; old model gave 1.21%)")
    print("tau_crit (Eq. 15):", ", ".join(f"{tau_crit_analytic(p):.1f}" for p in PAEs), "ns")
