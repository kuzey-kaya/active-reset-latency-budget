"""
Figures 2 and 3.

Figure 2: realistic-case sweep, now with an external anchor. The vertical marker is the
electronic feedback latency of Salathe et al., Phys. Rev. Applied 9, 034011 (2018),
tau_EL,tot = 219 ns. That is the quantity directly comparable to tau here: both exclude the
readout window and the actuator pulse. Their *total* loop latency, tau_FB = 352 ns, is a
different variable (Eq. 1 of that work: electronics + readout + actuator pulse) and is used
in Fig. 3 instead.

Figure 3: the design rule in the variable a hardware group actually quotes. The y axis is the
total feedback loop latency tau + t_read + t_pulse, so published loop latencies can be read
against it directly; the x axis is T1. Boundary lines are the asymptotic rule
tau_crit ~ 1.019 thr T1 / p_pulse, valid when the evaluation window scales with T1
(see the condition stated in the Introduction).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from closed_form_conditioned import (P_conditioned, tau_crit_analytic, P_read,
                                     t_read, t_pulse, T1, threshold)

# ---------------------------------------------------------------- Figure 2
taus = np.arange(0, 1050, 50.0)
PAE_grid = [0.01, 0.02, 0.03, 0.04, 0.05]
SALATHE_EL = 219.0          # tau_EL,tot, electronics only

fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(PAE_grid)))
for p, c in zip(PAE_grid, colors):
    ax.plot(taus, [P_conditioned(t, p) * 100 for t in taus], "o-", color=c,
            markersize=4, label=rf"$P_{{AE}}$={int(p*100)}%")
    ax.axvline(tau_crit_analytic(p), color=c, linestyle=":", linewidth=1.2)
ax.axvline(SALATHE_EL, color="0.25", linestyle="-.", linewidth=1.6)
ax.annotate("Salathé et al. (2018)\nelectronic latency, 219 ns",
            xy=(SALATHE_EL, ax.get_ylim()[1]), xytext=(SALATHE_EL + 25, 6.35),
            fontsize=9, color="0.25")
ax.set_xlabel(r"Feedback Latency, $\tau$ (ns)", fontsize=12)
ax.set_ylabel(r"Residual $P(|e\rangle)$ (%)", fontsize=12)
ax.legend(title="Assignment error", loc="upper left")
ax.grid(True, alpha=0.3)
fig.savefig("Figure2_conditioned.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
p_pulse_1pct = P_read * 0.99 + (1 - P_read) * 0.01
T1_axis = np.logspace(np.log10(5e3), np.log10(3e6), 300)      # 5 us .. 3 ms


def loop_latency_boundary(T1v, thr):
    """Critical latency in the *total loop* variable: tau_crit + t_read + t_pulse."""
    return 1.0192 * thr * T1v / p_pulse_1pct + t_read + t_pulse


fig, ax = plt.subplots(figsize=(8, 6))
for thr_, style in [(0.001, ":"), (0.01, "-"), (0.05, "--")]:
    ax.plot(T1_axis / 1e3, loop_latency_boundary(T1_axis, thr_) / 1e3, style,
            color="#1f4e79", linewidth=1.8,
            label=f"{thr_*100:g}% excess budget")

hardware = [("Salathé 2018: 352 ns total loop", 352.0),
            ("Gebauer 2019: 428 ns platform", 428.0)]
for i, (lab, lat) in enumerate(hardware):
    ax.axhline(lat / 1e3, color="0.35", linewidth=1.1,
               linestyle=(0, (4, 2)) if i == 0 else (0, (1, 2)))
    ax.annotate(lab, xy=(2400, lat / 1e3), fontsize=8.5, color="0.3",
                ha="right", va="bottom" if i else "top")

ax.plot(T1 / 1e3, (tau_crit_analytic(0.01) + t_read + t_pulse) / 1e3, "o",
        color="crimson", markersize=8, zorder=5)
ax.annotate("this work: 838 ns at $T_1$ = 50 µs", xy=(T1 / 1e3, 0.838),
            xytext=(70, 1.9), fontsize=9, color="crimson",
            arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"Qubit lifetime $T_1$ (µs)", fontsize=12)
ax.set_ylabel(r"Total feedback loop latency $\tau + t_{read} + t_{pulse}$ (µs)", fontsize=12)
ax.set_xlim(5, 3000)
ax.set_ylim(0.18, 400)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", title="Boundary, Eq. (16)")
fig.savefig("Figure3_design_rule.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("tau_crit(1%) =", round(tau_crit_analytic(0.01), 1), "ns")
print("in the loop variable:", round(tau_crit_analytic(0.01) + t_read + t_pulse, 1), "ns =",
      round((tau_crit_analytic(0.01) + t_read + t_pulse) / T1 * 100, 3), "% of T1")
print("Salathe r = 352/40000 =", round(352 / 40e3 * 100, 3), "%")
print("Saved Figure2_conditioned.png and Figure3_design_rule.png")
