"""
Causal (end-of-slot) evaluation convention for the latency-limited active reset.

Schedule: readout [0, t_read], classical delay tau, pulse slot of length t_pulse
(drive on only if the discriminator reports "e"). The next operation starts at the end
of the slot, t_read + tau + t_pulse, in EVERY branch (deterministic controller timing).
The reset error is the excited population at that instant (Delta = 0), or a fixed
interval Delta later.

Closed form (derived in the accompanying note; checked below against full
density-matrix propagation of all four branches):

    P(tau) - P(0) = e^{-Gamma1 Delta} q_c (1 - e^{-Gamma1 tau})

    q_c = |b| [w_ee (1-Peq) - w_ge Peq] - e^{-Gamma1 t_pulse} [w_eg (1-Peq) - w_gg Peq]
        ~ (1-Peq) P_e (1 - 2 eps_e) + Peq (1-P_e) (1 - 2 eps_g)

    tau_crit = -(1/Gamma1) ln[1 - budget e^{Gamma1 Delta} / q_c]

with w_ee = P_e(1-eps_e), w_eg = P_e eps_e, w_ge = (1-P_e) eps_g, w_gg = (1-P_e)(1-eps_g),
eps_e = P(report g | projected e), eps_g = P(report e | projected g), and b the slope of
the exact affine pulse map.  Pure NumPy/SciPy: no QuTiP required.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

# ---------------- parameters (manuscript Sec. II) ----------------
T1, Tphi, nth = 50_000.0, 100_000.0, 0.01             # ns
t_read, t_pulse, t_eval = 300.0, 20.0, 1500.0          # ns
g1, gphi = 1 / T1, 1 / Tphi
G1 = g1 * (1 + 2 * nth)
Peq = nth / (1 + 2 * nth)
T_slot = t_eval - t_read - t_pulse                      # fixed-slot window, 1180 ns
BUDGET = 0.01
PAES = [0.01, 0.02, 0.03, 0.04, 0.05]

# ---------------- Liouvillians (basis |e>, |g>; row-major vec) ----------------
sm = np.array([[0, 0], [1, 0]], complex)               # |g><e|
sp = sm.conj().T
sz = np.diag([1.0, -1.0]).astype(complex)
sx = np.array([[0, 1], [1, 0]], complex)
I2 = np.eye(2)


def _D(L):
    LdL = L.conj().T @ L
    return np.kron(L, L.conj()) - 0.5 * np.kron(LdL, I2) - 0.5 * np.kron(I2, LdL.T)


def liouvillian(H, nth_=nth):
    return (-1j * (np.kron(H, I2) - np.kron(I2, H.T))
            + _D(np.sqrt(g1 * (1 + nth_)) * sm) + _D(np.sqrt(g1 * nth_) * sp)
            + _D(np.sqrt(gphi / 2) * sz))


Omega = np.pi / t_pulse
L_free = liouvillian(np.zeros((2, 2), complex))
L_drive = liouvillian(0.5 * Omega * sx)
U_pulse = expm(L_drive * t_pulse)


def rho_of(P):
    return np.array([[P, 0], [0, 1 - P]], complex).reshape(-1)


def pop(v):
    return v.reshape(2, 2)[0, 0].real


# exact affine pulse map P_out = a + b P_in
a = pop(U_pulse @ rho_of(0.0))
b = pop(U_pulse @ rho_of(1.0)) - a
C = a + (b - 1) * Peq


def relax(P, t):
    return Peq + (P - Peq) * np.exp(-G1 * t)


P_read = relax(1.0, t_read)


def weights(eps_e, eps_g=None, Pe=None):
    eps_g = eps_e if eps_g is None else eps_g
    Pe = P_read if Pe is None else Pe
    return dict(ee=Pe * (1 - eps_e), eg=Pe * eps_e, ge=(1 - Pe) * eps_g, gg=(1 - Pe) * (1 - eps_g))


# ---------------- closed forms ----------------
def q_c(eps_e, eps_g=None, Pe=None):
    w = weights(eps_e, eps_g, Pe)
    return (-b * (w['ee'] * (1 - Peq) - w['ge'] * Peq)
            - np.exp(-G1 * t_pulse) * (w['eg'] * (1 - Peq) - w['gg'] * Peq))


def P0_causal(eps_e, eps_g=None, Pe=None, Delta=0.0):
    w = weights(eps_e, eps_g, Pe)
    pulse = (w['ee'] + w['ge']) * (Peq + C) + b * (w['ee'] * (1 - Peq) - w['ge'] * Peq)
    nopulse = (w['eg'] + w['gg']) * Peq + np.exp(-G1 * t_pulse) * (w['eg'] * (1 - Peq) - w['gg'] * Peq)
    return Peq + np.exp(-G1 * Delta) * (pulse + nopulse - Peq)


def P_causal_cf(tau, eps_e, eps_g=None, Pe=None, Delta=0.0):
    return P0_causal(eps_e, eps_g, Pe, Delta) + np.exp(-G1 * Delta) * q_c(eps_e, eps_g, Pe) * (1 - np.exp(-G1 * tau))


def tau_crit_causal(eps_e, eps_g=None, Pe=None, budget=BUDGET, Delta=0.0):
    X = budget * np.exp(G1 * Delta) / q_c(eps_e, eps_g, Pe)
    return -np.log(1 - X) / G1 if X < 1 else np.nan


def tau_abs_causal(eps_target, eps_e, eps_g=None, Pe=None):
    """Absolute criterion P(tau) = eps_target; nan if the zero-latency floor already exceeds it."""
    room = eps_target - P0_causal(eps_e, eps_g, Pe)
    if room <= 0:
        return np.nan
    return -np.log(1 - room / q_c(eps_e, eps_g, Pe)) / G1


# ---------------- independent check: full density-matrix propagation ----------------
def P_causal_numeric(tau, eps_e, eps_g=None, Pe=None, Delta=0.0):
    w = weights(eps_e, eps_g, Pe)
    U_tau, U_Delta = expm(L_free * tau), expm(L_free * Delta)
    U_idle_slot = expm(L_free * t_pulse)
    tot = 0.0
    for key, wt in w.items():
        start = rho_of(1.0 if key[0] == 'e' else 0.0)
        U_slot = U_pulse if key[1] == 'e' else U_idle_slot
        tot += wt * pop(U_Delta @ U_slot @ U_tau @ start)
    return tot


def P_fixed_numeric(tau, eps):
    w = weights(eps)
    tot = 0.0
    for key, wt in w.items():
        start = rho_of(1.0 if key[0] == 'e' else 0.0)
        U_slot = U_pulse if key[1] == 'e' else expm(L_free * t_pulse)
        tot += wt * pop(expm(L_free * (T_slot - tau)) @ U_slot @ expm(L_free * tau) @ start)
    return tot


if __name__ == "__main__":
    print(f"P_read = {P_read:.6f}  a = {a:.12f}  b = {b:.12f}  C = {C:.12f}")
    print(f"downward rate gamma1(1+nth) = Gamma1(1-Peq): {g1*(1+nth):.6e} vs {G1*(1-Peq):.6e} ns^-1\n")

    taus = np.arange(0, 1050, 50.0)
    dmax = max(abs(P_causal_numeric(t, p) - P_causal_cf(t, p)) for t in taus for p in PAES)
    print(f"[check] max |full propagation - closed form|, causal, 21x5 grid: {dmax:.2e}")
    for D in [300.0, 1180.0]:
        d = max(abs(P_causal_numeric(t, p, Delta=D) - P_causal_cf(t, p, Delta=D)) for t in taus[::4] for p in PAES)
        print(f"[check] same with Delta = {D:.0f} ns: {d:.2e}")
    fixed = [brentq(lambda t: P_fixed_numeric(t, p) - P_fixed_numeric(0, p) - BUDGET, 0, T_slot) for p in PAES]
    print("[check] fixed-slot tau_crit reproduced: " + ", ".join(f"{x:.2f}" for x in fixed) + " ns\n")

    print("P_AE | q_c      approx (1-Peq)Pe(1-2e)+Peq(1-Pe)(1-2e) | P(0)     | tau_crit exact | numeric root")
    for p in PAES:
        approx = (1 - Peq) * P_read * (1 - 2 * p) + Peq * (1 - P_read) * (1 - 2 * p)
        root = brentq(lambda t: P_causal_numeric(t, p) - P_causal_numeric(0, p) - BUDGET, 0, 20000)
        print(f"{p*100:3.0f}% | {q_c(p):.6f}  {approx:.6f}                            | {P0_causal(p)*100:.4f}% | "
              f"{tau_crit_causal(p):8.2f} ns    | {root:8.2f} ns")

    print("\nCollapse (Minor 7): excess/q_c at tau = 500 ns for each P_AE:",
          ", ".join(f"{(P_causal_cf(500, p)-P0_causal(p))/q_c(p):.10f}" for p in PAES))

    print("\nDesign-rule accuracy (budget = 1 pp):")
    print("P_AE | exact    | b*T1/(Pe(1-2e)) err | b*T1/p_pulse err | b*T1 err | first-order b/gamma_down err | prefactor")
    for p in PAES:
        ex = tau_crit_causal(p)
        r1 = BUDGET * T1 / (P_read * (1 - 2 * p))
        pp = P_read * (1 - p) + (1 - P_read) * p
        r2 = BUDGET * T1 / pp
        r3 = BUDGET * T1
        r4 = BUDGET / (g1 * (1 + nth))
        print(f"{p*100:3.0f}% | {ex:7.2f}  | {r1:7.2f} {100*(r1-ex)/ex:+6.2f}%     | {r2:7.2f} {100*(r2-ex)/ex:+6.2f}%  |"
              f" {100*(r3-ex)/ex:+6.2f}% | {r4:7.2f} {100*(r4-ex)/ex:+6.2f}%            | {ex/r1:.5f}")

    print("\nBudget dependence (P_AE = 1%): " + ", ".join(
        f"{bb*100:.1f} pp -> {tau_crit_causal(0.01, budget=bb):.1f} ns" for bb in [0.005, 0.01, 0.02, 0.05]))
    print("Existence: largest attainable excess q_c = " + ", ".join(f"{q_c(p)*100:.1f}%" for p in PAES))
    print("Delta dependence (P_AE = 1%): " + ", ".join(
        f"Delta={D:.0f} -> {tau_crit_causal(0.01, Delta=D):.2f} ns" for D in [0, 300, 1180, 5000]))

    print("\nAsymmetric confusion matrix:")
    for ee, eg in [(0.02, 0.005), (0.05, 0.005), (0.005, 0.02)]:
        print(f"  eps_e={ee*100:.1f}%, eps_g={eg*100:.1f}%: q_c={q_c(ee, eg):.5f}  tau_crit={tau_crit_causal(ee, eg):.2f} ns")

    print("\nInitial-state dependence (P_AE = 1%):")
    for Pe in [1.0, 0.9, 0.5, 0.43, 0.2, 0.05]:
        print(f"  P_e={Pe:.2f}: q_c={q_c(0.01, Pe=Pe):.5f}  tau_crit={tau_crit_causal(0.01, Pe=Pe):9.2f} ns")
    print("  uninformative readout eps=0.5, P_e=0.994: q_c =", f"{q_c(0.5):.2e}")

    print("\nUnconditional-pulse benchmark (Fig. 1 protocol: nth=0, pulse always, no projection):")
    Ps = np.exp(-t_read / T1)
    # with nth = 0 the pulse map is recomputed
    U0 = expm(liouvillian(0.5 * Omega * sx, 0.0) * t_pulse)
    a0 = pop(U0 @ rho_of(0.0)); b0 = pop(U0 @ rho_of(1.0)) - a0
    tb = -np.log(1 - BUDGET / (-b0 * Ps)) * T1
    print(f"  excess = |b| P_s (1 - e^(-tau/T1)), |b| = {-b0:.9f}: tau_crit = {tb:.2f} ns "
          f"(ideal pulse, Eq. 9: {-T1*np.log(1 - BUDGET/Ps):.2f} ns)")

    print("\nAbsolute criterion P(tau) <= eps_target (causal, Delta = 0):")
    grid = [0.001, 0.0025, 0.005, 0.01, 0.02]
    print("eps_target | " + " | ".join(f"P_AE={p*100:.2f}%" for p in grid))
    for et in [0.01, 0.02, 0.03]:
        cells = []
        for p in grid:
            t = tau_abs_causal(et, p)
            cells.append("   none   " if np.isnan(t) else f"{t:7.1f} ns")
        print(f"   {et*100:.0f}%     | " + " | ".join(cells))
    print("zero-latency floor P(0): " + ", ".join(f"P_AE={p*100:.2f}% -> {P0_causal(p)*100:.4f}%" for p in grid))
    print("P(0) at P_AE = 0:", f"{P0_causal(0.0)*100:.4f}%")
