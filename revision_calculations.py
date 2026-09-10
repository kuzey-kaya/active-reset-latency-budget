"""
revision_calculations.py — everything the two reviewer reports asked to be recomputed.

Run with:  python revision_calculations.py
Requires numpy, scipy, matplotlib.  No QuTiP needed (the exact pulse propagator is built
directly from the Liouvillian), so a reviewer can reproduce every number in seconds.

Addresses, by reviewer item:
  B9  (R1 CRITICAL) exact affine pulse map, replacing the claim that Eq. (12) is exact
  B9  (both)        the logarithm argument X, wrongly stated as 0.021 in the reviewed version
  B9  (R1)          the threshold existence condition
  B10 (R1 CRITICAL) admissible-root masking for Figure 3
  B6  (R2)          generalisation of p_pulse to an asymmetric confusion matrix
  B11 (R1 CRITICAL) sensitivity of p_pulse to the initial excited population
  B10 (R2)          the results table
  B5  (R1)          T2 reconciliation between Eqs. (3) and (13)
  B9  (R2)          what the "2.6% of trajectories" figure includes
  B10 (R1)          the 0.043% decomposition expressed relative to P_eq
  B11 (R2)          the excess corresponding to a 352 ns loop
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- parameters (Sec. II.A)
T1      = 50e3          # ns, bare relaxation parameter 1/gamma_1
n_th    = 0.01          # bath occupation parameter (NOT the equilibrium excited population)
T_phi   = 100e3         # ns
t_read  = 300.0
t_pulse = 20.0
t_eval  = 1500.0
budget_default = 0.01   # absolute increase in excitation probability (1 percentage point)

gamma_1 = 1.0 / T1
gamma_phi = 1.0 / (2 * T_phi)          # prefactor of the sigma_z collapse operator, Eq. (2)
Gamma_1 = gamma_1 * (1 + 2 * n_th)     # population equilibration rate, Eq. (11)
Gamma_2 = Gamma_1 / 2 + 1.0 / T_phi    # Eq. (13)
P_eq    = n_th / (1 + 2 * n_th)        # equilibrium excited population
T       = t_eval - t_read - t_pulse    # admissible delay window

kappa   = np.exp(-(Gamma_1 + Gamma_2) * t_pulse / 2)   # Eq. (12) slope magnitude


def relax(P0, t):
    """Eq. (11)."""
    return P_eq + (P0 - P_eq) * np.exp(-Gamma_1 * t)


# ----------------------------------------------------------------- 1. exact pulse map (B9)
def exact_pulse_map():
    """
    Exact affine population map of the finite-duration pi-pulse, obtained from the full
    Liouvillian of Eqs. (1)-(2) with H = (Omega/2) sigma_x, Omega = pi/t_pulse.
    Returns (a, b) with P_out = a + b P_in.  The map is exactly affine for diagonal inputs
    because the Liouvillian is linear; only a and b differ from Eq. (12).
    """
    Om = np.pi / t_pulse
    I2 = np.eye(2)
    sx = np.array([[0, 1], [1, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    sm = np.array([[0, 1], [0, 0]], complex)      # |g><e|
    sp = sm.conj().T
    H = 0.5 * Om * sx

    spre = lambda A: np.kron(A, I2)
    spost = lambda A: np.kron(I2, A.T)

    L = -1j * (spre(H) - spost(H))
    for c in (np.sqrt(gamma_1 * (1 + n_th)) * sm,
              np.sqrt(gamma_1 * n_th) * sp,
              np.sqrt(gamma_phi) * sz):
        cd = c.conj().T
        L += spre(c) @ spost(cd) - 0.5 * spre(cd @ c) - 0.5 * spost(cd @ c)

    U = expm(L * t_pulse)

    def P_out(P_in):
        rho = np.diag([1 - P_in, P_in]).astype(complex).reshape(-1)
        return (U @ rho).reshape(2, 2)[1, 1].real

    a = P_out(0.0)
    b = P_out(1.0) - a
    lin_err = abs(P_out(0.5) - (a + 0.5 * b))      # affine check
    return a, b, lin_err


a_exact, b_exact, affine_residual = exact_pulse_map()
a_eq12, b_eq12 = (1 + kappa) / 2, -kappa

# C = a + (b - 1) P_eq  is the only combination of the pulse map that enters the excess
C_exact = a_exact + (b_exact - 1) * P_eq
C_eq12  = a_eq12 + (b_eq12 - 1) * P_eq          # identical to (1 + kappa)(1/2 - P_eq)

# ----------------------------------------------------------------- 2. branch model (Eq. 8)
P_read = relax(1.0, t_read)                      # population at the end of the readout window


def p_pulse(P_e=P_read, eps_e=0.01, eps_g=None):
    """
    Probability that the corrective pulse fires.

    P_e   : population of the projected-excited branch, i.e. the probability that the
            projection at the end of the readout window returns |e>.  Defaults to P_read,
            the value for a qubit prepared in |e>.
    eps_e : P(report g | projected e), the missed-excited probability.
    eps_g : P(report e | projected g), the false-excited probability.  Defaults to eps_e,
            which is the symmetric confusion channel used in the manuscript.

    Symmetric case reduces to p_pulse = P_e(1 - eps) + (1 - P_e) eps.
    """
    if eps_g is None:
        eps_g = eps_e
    return P_e * (1 - eps_e) + (1 - P_e) * eps_g


def P_branch(P_start, tau, pulse, a=None, b=None):
    """One branch: delay tau, optional pulse, relax to t_eval."""
    a = a_exact if a is None else a
    b = b_exact if b is None else b
    P = relax(P_start, tau)
    if pulse:
        P = a + b * P
        return relax(P, T - tau)
    return relax(P, t_eval - t_read - tau)


def P_residual(tau, eps_e=0.01, eps_g=None, P_e=None, a=None, b=None):
    """Eq. (8): four-branch weighted sum, exact for the terminal-projection model."""
    if eps_g is None:
        eps_g = eps_e
    Pe = P_read if P_e is None else P_e
    from_e = (1 - eps_e) * P_branch(1.0, tau, True, a, b) + eps_e * P_branch(1.0, tau, False, a, b)
    from_g = eps_g * P_branch(0.0, tau, True, a, b) + (1 - eps_g) * P_branch(0.0, tau, False, a, b)
    return Pe * from_e + (1 - Pe) * from_g


# ----------------------------------------------------------------- 3. threshold, existence
def max_excess(eps_e=0.01, eps_g=None, P_e=None, C=None):
    """Largest excess reachable before the observation window ends (tau = T)."""
    C = C_exact if C is None else C
    return p_pulse(P_read if P_e is None else P_e, eps_e, eps_g) * C * (1 - np.exp(-Gamma_1 * T))


def tau_crit(budget=budget_default, eps_e=0.01, eps_g=None, P_e=None, C=None):
    """
    Eq. (15) with a general pulse map.  Returns None when the budget exceeds the maximum
    excess attainable inside the window, i.e. when no admissible root exists.
    """
    C = C_exact if C is None else C
    pp = p_pulse(P_read if P_e is None else P_e, eps_e, eps_g)
    if budget > max_excess(eps_e, eps_g, P_e, C):
        return None
    return np.log(1 + budget * np.exp(Gamma_1 * T) / (pp * C)) / Gamma_1


def tau_crit_numeric(budget=budget_default, eps_e=0.01, eps_g=None):
    """Root of Eq. (10) found numerically from Eq. (8), as an independent check."""
    P0 = P_residual(0.0, eps_e, eps_g)
    f = lambda t: P_residual(t, eps_e, eps_g) - P0 - budget
    if f(T) < 0:
        return None
    return brentq(f, 0, T)


def X_argument(budget=budget_default, eps_e=0.01, eps_g=None, C=None):
    """The additive term inside ln(1 + X) in Eq. (15)."""
    C = C_exact if C is None else C
    return budget * np.exp(Gamma_1 * T) / (p_pulse(P_read, eps_e, eps_g) * C)


# ================================================================= report
if __name__ == "__main__":
    print("=" * 78)
    print("1.  EXACT PULSE MAP versus Eq. (12)                        [B9, R1 CRITICAL]")
    print("=" * 78)
    print(f"  exact    : P_out = {a_exact:.12f} + ({b_exact:.12f}) P_in")
    print(f"  Eq. (12) : P_out = {a_eq12:.12f} + ({b_eq12:.12f}) P_in")
    print(f"  intercept difference : {a_eq12 - a_exact:.3e}")
    print(f"  slope difference     : {b_eq12 - b_exact:.3e}")
    print(f"  affinity residual    : {affine_residual:.1e}  (map is exactly affine)")
    print(f"  C = a + (b-1)P_eq    : exact {C_exact:.12f}   Eq.(12) {C_eq12:.12f}")
    print(f"  tau_crit(1 pp): exact map {tau_crit(C=C_exact):.6f} ns, "
          f"Eq.(12) map {tau_crit(C=C_eq12):.6f} ns, difference "
          f"{tau_crit(C=C_eq12) - tau_crit(C=C_exact):.2e} ns")
    print("  -> Eq. (12) is an approximation accurate to ~2e-8 in the intercept; it is not")
    print("     exact.  All results below use the exact map.")

    print()
    print("=" * 78)
    print("2.  LOGARITHM ARGUMENT X                                   [B9, both reviewers]")
    print("=" * 78)
    X = X_argument()
    print(f"  X at eps = 1%, budget = 1 pp : {X:.5f}")
    print(f"  X*(1+kappa)                  : {X * (1 + kappa):.5f}   <- the 0.021 printed in")
    print("                                   the reviewed version, i.e. (1+kappa) omitted")
    print(f"  ln(1+X)/X                    : {np.log(1 + X) / X:.5f}")

    print()
    print("=" * 78)
    print("3.  THRESHOLD EXISTENCE CONDITION                     [B9, B10, R1 CRITICAL]")
    print("=" * 78)
    print(f"  window T = t_eval - t_read - t_pulse = {T:.0f} ns")
    print(f"  max attainable excess inside window  = {max_excess()*100:.4f} percentage points")
    print("  budget    formal root    admissible?")
    for bud in (0.005, 0.01, 0.02, 0.0229, 0.03, 0.05):
        pp = p_pulse()
        formal = np.log(1 + bud * np.exp(Gamma_1 * T) / (pp * C_exact)) / Gamma_1
        ok = tau_crit(bud) is not None
        print(f"   {bud*100:5.2f} pp   {formal:8.1f} ns    {'yes' if ok else 'NO - outside window'}")
    print("  -> the 5 pp curve of the reviewed Figure 3 has no root inside the window.")

    print()
    print("=" * 78)
    print("4.  RESULTS TABLE                                              [B10, R2 (ii)]")
    print("=" * 78)
    print("  eps    p_pulse   P(0)      tau_c Eq.(10)  tau_c Eq.(15)  1.019*bud*T1/p  bud*T1")
    for eps in (0.01, 0.02, 0.03, 0.04, 0.05):
        pp = p_pulse(eps_e=eps)
        print(f"  {eps*100:>2.0f}%   {pp:.5f}   {P_residual(0.0, eps)*100:6.4f}%   "
              f"{tau_crit_numeric(eps_e=eps):9.2f} ns  {tau_crit(eps_e=eps):9.2f} ns  "
              f"{1.0192*budget_default*T1/pp:9.2f} ns  {budget_default*T1:7.2f} ns")
    print("  relative error of the simplifications, against the Eq. (10) root:")
    for eps in (0.01, 0.03, 0.05):
        ex = tau_crit_numeric(eps_e=eps); pp = p_pulse(eps_e=eps)
        print(f"    eps={eps*100:>2.0f}%: 1.019*bud*T1/p_pulse {100*(1.0192*budget_default*T1/pp/ex-1):+6.2f}%   "
              f"bud*T1/p_pulse {100*(budget_default*T1/pp/ex-1):+6.2f}%   "
              f"bud*T1 {100*(budget_default*T1/ex-1):+6.2f}%")

    print()
    print("=" * 78)
    print("5.  ASYMMETRIC CONFUSION MATRIX                                    [B6, R2 (i)]")
    print("=" * 78)
    print("  p_pulse = P_e (1 - eps_e) + (1 - P_e) eps_g, with")
    print("    eps_e = P(report g | projected e),  eps_g = P(report e | projected g).")
    print("  Caution: a measured P(g|e) from a calibrated device already contains relaxation")
    print("  during t_read.  Inserting it as eps_e double-counts that relaxation, which is")
    print("  carried here by P_read.  eps_e is the discrimination error at the end of the")
    print("  window, not the device-level P(g|e).")
    print("  eps_e  eps_g   p_pulse    P(0)       tau_crit")
    for ee, eg in ((0.01, 0.01), (0.02, 0.005), (0.05, 0.005), (0.005, 0.02), (0.10, 0.01)):
        tc = tau_crit(eps_e=ee, eps_g=eg)
        print(f"  {ee*100:4.1f}%  {eg*100:4.1f}%  {p_pulse(eps_e=ee, eps_g=eg):.5f}  "
              f"{P_residual(0.0, ee, eg)*100:7.4f}%  "
              f"{('%8.2f ns' % tc) if tc else '   none (outside window)'}")

    print()
    print("=" * 78)
    print("6.  SENSITIVITY TO THE INITIAL EXCITED POPULATION         [B11, R1 CRITICAL]")
    print("=" * 78)
    print("  The claim of weak sensitivity to assignment error holds for a qubit prepared")
    print("  in |e>.  For a general initial population the pulse probability is")
    print("  p_pulse = P_e(1-eps) + (1-P_e)eps, which depends strongly on eps when P_e is small.")
    print("   P_e      p_pulse(eps=1%)  p_pulse(eps=5%)  ratio   tau_crit(1%)  tau_crit(5%)  shift")
    for Pe in (1.0, P_read, 0.9, 0.5, 0.2, 0.05):
        p1, p5 = p_pulse(Pe, 0.01), p_pulse(Pe, 0.05)
        t1, t5 = tau_crit(eps_e=0.01, P_e=Pe), tau_crit(eps_e=0.05, P_e=Pe)
        s = f"{100*(t5/t1-1):+6.1f}%" if (t1 and t5) else "   n/a"
        print(f"  {Pe:6.4f}   {p1:.5f}        {p5:.5f}        {p5/p1:5.3f}   "
              f"{('%8.1f' % t1) if t1 else '    none'}      "
              f"{('%8.1f' % t5) if t5 else '    none'}   {s}")
    print("  -> at P_e = 1 the shift is a few percent; by P_e = 0.05 the same change in")
    print("     assignment error moves the boundary by a large factor.  The design rule must")
    print("     be stated for a qubit that is being reset from a substantially excited state.")

    print()
    print("=" * 78)
    print("7.  SMALLER ITEMS")
    print("=" * 78)
    print(f"  [B5]  T2 from Eq. (13): 1/Gamma_2 = {1/Gamma_2/1e3:.3f} us, not the 50 us of Eq. (3);")
    print(f"        Eq. (3) is the gamma_phi-only approximation.  1/Gamma_1 = {1/Gamma_1/1e3:.3f} us,")
    print(f"        distinct from the bare T1 = {T1/1e3:.1f} us.")
    print(f"  [B5]  P_eq = n_th/(1+2n_th) = {P_eq*100:.4f}%, distinct from n_th = {n_th*100:.1f}%.")
    resid0 = P_residual(0.0, 0.0)
    pulse_part = (1 - kappa) / 2 * np.exp(-Gamma_1 * T)
    thermal_part = relax(0.0, T)
    print(f"  [B10] zero-assignment-error residual = {resid0*100:.4f}%")
    print(f"          finite-pulse part  {pulse_part*100:.4f}%  = {pulse_part/P_eq*100:.1f}% of P_eq")
    print(f"          thermal repopulation {thermal_part*100:.4f}%  = {thermal_part/P_eq*100:.1f}% of P_eq")
    print(f"          total                {resid0*100:.4f}%  = {resid0/P_eq*100:.1f}% of P_eq")
    print(f"  [B9]  no-pulse branch weights at eps = 2%: projected-e misassigned "
          f"{P_read*0.02*100:.2f}%, projected-g correctly assigned {(1-P_read)*0.98*100:.2f}%, "
          f"total {(P_read*0.02+(1-P_read)*0.98)*100:.2f}%")
    for tau_ in (219.0, 352.0):
        print(f"  [B11] excess at tau = {tau_:.0f} ns (eps = 1%): "
              f"{(P_residual(tau_, 0.01)-P_residual(0.0, 0.01))*100:.4f} pp")
    taus = np.arange(0, 1050, 50.0)
    y = np.array([P_residual(t, 0.01) for t in taus])
    lin = np.polyval(np.polyfit(taus, y, 1), taus)
    print(f"  [B10] Figure 2 linearity: max deviation of the eps = 1% curve from a straight "
          f"line over 0-1000 ns is {np.max(np.abs(y-lin))*100:.4f} pp "
          f"({np.max(np.abs(y-lin))/np.ptp(y)*100:.2f}% of the swept range)")

    # ------------------------------------------------------------- Figure 3, corrected
    p_ref = p_pulse(eps_e=0.01)
    T1_axis = np.logspace(np.log10(5e3), np.log10(3e6), 400)

    def boundary(T1v, bud):
        """Loop-latency boundary with the window scaled as T = (T1v/T1) * T_ref."""
        G1 = (1 + 2 * n_th) / T1v
        Tv = T * T1v / T1
        val = np.log(1 + bud * np.exp(G1 * Tv) / (p_ref * C_exact)) / G1
        return np.where(bud <= p_ref * C_exact * (1 - np.exp(-G1 * Tv)),
                        val + t_read + t_pulse, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    for bud, ls in ((0.005, ":"), (0.01, "-"), (0.02, "--")):
        ax.plot(T1_axis / 1e3, boundary(T1_axis, bud) / 1e3, ls, color="#1f4e79", lw=1.8,
                label=f"{bud*100:g} pp budget")
    for lab, lat, ls in (("Salathé 2018: 352 ns total loop", 352.0, (0, (4, 2))),
                         ("Gebauer 2020: 428 ns electronics only", 428.0, (0, (1, 2)))):
        ax.axhline(lat / 1e3, color="0.35", lw=1.1, ls=ls)
        ax.annotate(lab, xy=(2400, lat / 1e3), fontsize=8.5, color="0.3", ha="right", va="bottom")
    ax.plot(T1 / 1e3, (tau_crit() + t_read + t_pulse) / 1e3, "o", color="crimson", ms=8, zorder=5)
    ax.annotate("this work: 838 ns at $T_1$ = 50 µs", xy=(T1 / 1e3, 0.838), xytext=(70, 1.9),
                fontsize=9, color="crimson",
                arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(5, 3000); ax.set_ylim(0.18, 400)
    ax.set_xlabel(r"Qubit lifetime $T_1$ (µs)", fontsize=12)
    ax.set_ylabel(r"Total feedback loop latency $\tau + t_{read} + t_{pulse}$ (µs)", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", title="Boundary, Eq. (15)")
    fig.savefig("Figure3_design_rule.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\n  Figure 3 written (budgets 0.5, 1 and 2 pp; inadmissible roots masked).")
