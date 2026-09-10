"""
Closed-form solution of the realistic-case feedback loop with a PROJECTIVE (conditioned)
QND readout, plus a fully analytic expression for tau_crit under the excess criterion.

Difference from closed_form_solution.py
---------------------------------------
The previous Eq. (8) started BOTH branches from the same unconditioned mixed state
(P_e = P_read ~ 0.994). A strong QND sigma_z readout projects the qubit: conditioned on the
true outcome the post-measurement state is |e> or |g>, and the assignment error P_AE is a
classical error applied on top of that projected outcome. The unconditioned version counts
the fraction that relaxed during the readout window twice (once as an unflipped "g" outcome
that keeps P_e ~ 0.994, once as the g-component of the mixed state that the pulse flips back
to e). Symptom: at P_AE = 0 the old Eq. (8) gives 1.21 %, twice the Fig. 1 zero-error offset
of 0.60 %, instead of reducing to (essentially) the thermal floor.

Conditioned model (Eq. 8'):

    P(tau, P_AE) = P_read [ (1-P_AE) P_pulse(1, tau) + P_AE P_nopulse(1) ]
                 + (1-P_read) [ (1-P_AE) P_nopulse(0) + P_AE P_pulse(0, tau) ]

where P_pulse(P_s, tau) / P_nopulse(P_s) are the deterministic populations at t_eval for a
qubit that is in population P_s at the end of the readout window and does / does not receive
the pi-pulse after a delay tau.  P_read = population at the end of the readout window for a
qubit prepared in |e>.

Analytic tau_crit
-----------------
Chaining Eq. (11)-(12): for a pulse branch starting at P_s,

    P_pulse(P_s, tau) = P_eq + (1+kappa)(1/2 - P_eq) e^{-Gamma_1 (T - tau)} - kappa (P_s - P_eq) e^{-Gamma_1 T}

with T = t_eval - t_read - t_pulse. The tau-dependent term does NOT depend on P_s, and the
no-pulse branches are tau-independent, so

    P(tau) - P(0) = p_pulse (1+kappa)(1/2 - P_eq) e^{-Gamma_1 T} (e^{Gamma_1 tau} - 1),

with p_pulse = P_read (1-P_AE) + (1-P_read) P_AE the total probability that a pulse fires.
Setting this equal to the excess threshold gives

    tau_crit = (1/Gamma_1) ln[ 1 + thr e^{Gamma_1 T} / ( p_pulse (1+kappa)(1/2 - P_eq) ) ].     (14)

Because only p_pulse depends on P_AE, and p_pulse = P_read - P_AE (2 P_read - 1) ~ 1 - P_AE,
tau_crit is insensitive to readout quality by construction. The same formula holds for the
old unconditioned Eq. (8) (same p_pulse), which is why the two models give IDENTICAL tau_crit
values and differ only by a tau-independent offset in P(0).
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- parameters (identical to the manuscript) ---------------------------------------
T1 = 50e3          # ns
nth = 0.01
T_phi = 100e3      # ns
t_read = 300.0
t_pulse = 20.0
t_eval = 1500.0
threshold = 0.01

gamma1 = 1.0 / T1
Gamma1 = gamma1 * (1 + 2 * nth)                 # Eq. (11)
Peq = nth / (1 + 2 * nth)                        # Eq. (11)
Gamma2 = Gamma1 / 2 + 1.0 / T_phi                # Eq. (13)
kappa = np.exp(-(Gamma1 + Gamma2) * t_pulse / 2)  # Eq. (12)
T_after = t_eval - t_read - t_pulse              # T in Eq. (14)


def relax(P0, t):
    """Eq. (11)."""
    return Peq + (P0 - Peq) * np.exp(-Gamma1 * t)


def P_pulse(P_s, tau):
    """Delay tau, pi-pulse (Eq. 12), relax to t_eval."""
    Pd = relax(P_s, tau)
    Pp = (1 + kappa) / 2 - kappa * Pd
    return relax(Pp, t_eval - t_read - tau - t_pulse)


def P_nopulse(P_s):
    """No pulse: relax from end of readout to t_eval (tau-independent)."""
    return relax(P_s, t_eval - t_read)


P_read = relax(1.0, t_read)


def P_conditioned(tau, P_AE):
    """Eq. (8'): projective readout, assignment error applied to the projected outcome."""
    from_e = (1 - P_AE) * P_pulse(1.0, tau) + P_AE * P_nopulse(1.0)
    from_g = (1 - P_AE) * P_nopulse(0.0) + P_AE * P_pulse(0.0, tau)
    return P_read * from_e + (1 - P_read) * from_g


def P_unconditioned(tau, P_AE):
    """Old Eq. (8) (closed_form_solution.py), kept for comparison."""
    p_assigned = P_read * (1 - P_AE) + (1 - P_read) * P_AE
    return p_assigned * P_pulse(P_read, tau) + (1 - p_assigned) * P_nopulse(P_read)


def tau_crit_numeric(P_func, P_AE, thr=threshold):
    P0 = P_func(0.0, P_AE)
    return brentq(lambda t: P_func(t, P_AE) - P0 - thr, 0, 5000)


def tau_crit_analytic(P_AE, thr=threshold):
    """Eq. (14)."""
    p_pulse = P_read * (1 - P_AE) + (1 - P_read) * P_AE
    return np.log(1 + thr * np.exp(Gamma1 * T_after) / (p_pulse * (1 + kappa) * (0.5 - Peq))) / Gamma1


if __name__ == "__main__":
    PAEs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    print(f"P_read = {P_read:.5f}   kappa = {kappa:.7f}   Peq = {Peq:.5f}\n")
    print("P_AE   P(0) uncond.   P(0) cond.   tau_c numeric(cond)   tau_c numeric(uncond)   tau_c Eq.(14)")
    for p in PAEs:
        print(f"{p:.2f}   {P_unconditioned(0, p)*100:6.2f} %      {P_conditioned(0, p)*100:6.2f} %"
              f"      {tau_crit_numeric(P_conditioned, p):8.2f} ns"
              f"          {tau_crit_numeric(P_unconditioned, p):8.2f} ns"
              f"           {tau_crit_analytic(p):8.2f} ns")

    # Sanity: conditioned model at P_AE = 0, nth -> 0 should give essentially zero
    print(f"\nCheck: the two models differ by a tau-independent constant?")
    taus = np.arange(0, 1050, 50.0)
    for p in [0.01, 0.05]:
        d = np.array([P_unconditioned(t, p) - P_conditioned(t, p) for t in taus])
        print(f"  P_AE={p:.2f}: offset = {d.mean()*100:.4f} %, spread over tau = {np.ptp(d):.2e}")

    # ---- grid + Figure 2 (conditioned model) ----------------------------------------
    PAE_grid = [0.01, 0.02, 0.03, 0.04, 0.05]
    with open("fig2_grid_conditioned.csv", "w") as f:
        f.write("tau_ns," + ",".join(f"P_AE_{int(p*100)}pct" for p in PAE_grid) + "\n")
        for t in taus:
            f.write(f"{int(t)}," + ",".join(f"{P_conditioned(t, p)*100:.4f}" for p in PAE_grid) + "\n")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(PAE_grid)))
    for p, c in zip(PAE_grid, colors):
        y = [P_conditioned(t, p) * 100 for t in taus]
        ax.plot(taus, y, "o-", color=c, markersize=4, label=rf"$P_{{AE}}$={int(p*100)}%")
        ax.axvline(tau_crit_analytic(p), color=c, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"Feedback Latency, $\tau$ (ns)", fontsize=12)
    ax.set_ylabel(r"Residual $P(|e\rangle)$ (%)", fontsize=12)
    ax.legend(title="Assignment error", loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.savefig("Figure2_conditioned.png", dpi=300, bbox_inches="tight")
    print("\nSaved fig2_grid_conditioned.csv and Figure2_conditioned.png")

    # ---- checks quoted in the manuscript ---------------------------------------------
    print("\n--- Eq. (16): the scaling law tau_crit ~ 1.02 thr T1 / p_pulse ---")
    for p in PAEs:
        pp = P_read*(1-p) + (1-P_read)*p
        print(f"  P_AE={p:.2f}: exact {tau_crit_analytic(p):7.2f} ns   thr*T1/p_pulse "
              f"{threshold*T1/pp:7.2f} ns   ratio {tau_crit_analytic(p)/(threshold*T1/pp):.5f}")
    print(f"  analytic prefactor e^(G1 T)/[(1+k)(1/2-Peq) G1 T1] = "
          f"{np.exp(Gamma1*T_after)/((1+kappa)*(0.5-Peq)*Gamma1*T1):.5f}")

    print("\n--- threshold dependence (tau_crit is very nearly proportional to thr) ---")
    for thr in [0.005, 0.01, 0.02]:
        print(f"  thr={thr:.3f}: tau_crit(P_AE=1%) = {tau_crit_analytic(0.01, thr):8.2f} ns")

    print("\n--- t_eval dependence (through T) ---")
    for te in [1000, 1500, 5000]:
        T_e = te - t_read - t_pulse
        pp = P_read*0.99 + (1-P_read)*0.01
        tc_e = np.log(1 + threshold*np.exp(Gamma1*T_e)/(pp*(1+kappa)*(0.5-Peq)))/Gamma1
        print(f"  t_eval={te:5d} ns: tau_crit(P_AE=1%) = {tc_e:7.2f} ns")

    print("\n--- where the projection sits inside the readout window ---")
    def model(t_proj, tau, P_AE):
        Pp = relax(1.0, t_proj)
        def branch(P_s, pulse):
            P_end = relax(P_s, t_read - t_proj)
            Pd = relax(P_end, tau)
            if pulse:
                Pd = (1 + kappa)/2 - kappa*Pd
                return relax(Pd, t_eval - t_read - tau - t_pulse)
            return relax(Pd, t_eval - t_read - tau)
        return (Pp*((1-P_AE)*branch(1.0, True) + P_AE*branch(1.0, False))
                + (1-Pp)*((1-P_AE)*branch(0.0, False) + P_AE*branch(0.0, True)))
    for tp, lab in [(0.0, "start of window"), (t_read, "end (model used)")]:
        p0 = model(tp, 0.0, 0.01)
        tc_p = brentq(lambda t: model(tp, t, 0.01) - p0 - threshold, 0, 5000)
        print(f"  projection at {lab:18s}: P(0) = {p0*100:.4f} %   tau_crit = {tc_p:7.2f} ns")

    print("\n--- what the 0.043 % at P_AE = 0 actually is ---")
    print(f"  P_eq (thermal equilibrium)                  = {Peq*100:.4f} %")
    print(f"  P(0) at P_AE = 0                            = {P_conditioned(0.0, 0.0)*100:.4f} %")
    print(f"  left by the finite-duration pulse, at t_eval= {(1-kappa)/2*np.exp(-Gamma1*T_after)*100:.4f} %")
    print(f"  thermal repopulation over {T_after:.0f} ns from empty = {relax(0.0, T_after)*100:.4f} %")
    print(f"  fraction of equilibrium reached by t_eval    = {relax(0.0, T_after)/Peq*100:.1f} %")

    print("\n--- why the trajectory cross-check runs at P_AE = 0.5 ---")
    for p in [0.02, 0.5]:
        outs = np.array([P_pulse(1.0, 0.0), P_nopulse(1.0), P_nopulse(0.0), P_pulse(0.0, 0.0)])
        w = np.array([P_read*(1-p), P_read*p, (1-P_read)*(1-p), (1-P_read)*p])
        m = (w*outs).sum(); sd = np.sqrt((w*(outs-m)**2).sum())
        for N in ([120] if p == 0.02 else [400]):
            print(f"  P_AE={p}: mean {m*100:6.2f} %  SD {sd*100:5.2f} %  "
                  f"N={N} -> SE {sd/np.sqrt(N)*100:5.2f} % ({sd/np.sqrt(N)/m*100:.0f} % relative)")
