"""
Figures 1-3 of the revised manuscript. Single-column width (8.5 cm), vector PDF. Same colour scheme as make_figures.py (v1.1): viridis for the
P_AE series, #1f4e79 and crimson as accents; distinct line styles keep the figures readable in greyscale.
Figure 3 reads photon_curves.npz written by paper_numbers.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from causal_convention import (G1, C, P_read, t_read, t_pulse, t_eval, q_c, P_causal_cf, P0_causal, tau_crit_causal)
BLUE, RED = "#1f4e79", "crimson"   # accent colours of the v1.1 figures

W = 8.5 / 2.54
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "mathtext.fontset": "dejavusans",
                     "axes.linewidth": 0.6, "lines.linewidth": 1.25, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})
PAES = [0.01, 0.02, 0.03, 0.04, 0.05]
greys = [matplotlib.colors.to_hex(c) for c in plt.cm.viridis(np.linspace(0, 0.9, 5))]  # viridis, as in the v1.1 figures
styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]

# ---------------- Figure 1: schedule ----------------
fig, ax = plt.subplots(figsize=(W, 1.75))
ax.set_xlim(-30, 1560); ax.set_ylim(-0.75, 2.45); ax.axis("off")
rows = [(1.55, "(a) end of slot"), (0.15, "(b) fixed slot")]
tau = 450.0
for y, lab in rows:
    ax.text(-30, y + 0.62, lab, ha="left", va="bottom")
    ax.add_patch(Rectangle((0, y), t_read, 0.45, facecolor="#c6dbef", edgecolor="0.0", lw=0.6))
    ax.text(t_read / 2, y + 0.225, "readout", ha="center", va="center", fontsize=7)
    ax.add_patch(Rectangle((t_read, y), tau, 0.45, facecolor="white", edgecolor="0.0", lw=0.6, hatch="////"))
    ax.text(t_read + tau / 2, y + 0.225, r"delay $\tau$", ha="center", va="center", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    x0 = t_read + tau
    ax.add_patch(Rectangle((x0, y), 60, 0.45, facecolor=BLUE, edgecolor="0.0", lw=0.6))
    ax.text(x0 + 30, y - 0.08, r"$\pi$ if ‘e’", ha="center", va="top", fontsize=6.5)
    if y > 1:
        ax.annotate("", xy=(x0 + 60, y + 0.62), xytext=(x0 + 60, y + 0.47),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.0"))
        ax.text(x0 + 75, y + 0.225, r"evaluate $P(\tau)$; next operation starts", va="center", fontsize=7)
    else:
        ax.add_patch(Rectangle((x0 + 60, y), t_eval - x0 - 60, 0.45, facecolor="0.93", edgecolor="0.0", lw=0.6))
        ax.text((x0 + 60 + t_eval) / 2, y + 0.225, "idle", ha="center", va="center", fontsize=7)
        ax.annotate("", xy=(t_eval, y + 0.62), xytext=(t_eval, y + 0.47),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.0"))
        ax.text(t_eval, y + 0.66, r"$t_{\rm eval}$", ha="center", va="bottom", fontsize=7)
ax.text(1560, -0.62, "pulse slot not to scale", ha="right", va="bottom", fontsize=6, color="0.35")
fig.savefig("fig1_schedule.pdf")

# ---------------- Figure 2: P(tau) and collapse ----------------
taus = np.linspace(0, 1500, 301)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 4.4), sharex=True)
for p, g, ls in zip(PAES, greys, styles):
    ax1.plot(taus, [P_causal_cf(t, p) * 100 for t in taus], color=g, ls=ls, marker="o", markersize=2.5,
             markevery=20, label=rf"$P_{{\rm AE}}={p*100:.0f}\%$")
    ax1.axvline(tau_crit_causal(p), color=g, linestyle=":", linewidth=0.9)
ax1.set_ylabel(r"$P(\tau)$ (%)")
ax1.legend(loc="upper left", ncol=2, fontsize=7)
ax1.text(0.98, 0.04, "(a)", transform=ax1.transAxes, ha="right", va="bottom")
for p, g, ls in zip(PAES, greys, styles):
    ex = np.array([P_causal_cf(t, p) - P0_causal(p) for t in taus]) / q_c(p)
    ax2.plot(taus, ex * 100, color=g, ls=ls)
T = t_eval - t_read - t_pulse
ax2.plot(taus, (1 - np.exp(-G1 * taus)) * 100, color=BLUE, lw=3.0, alpha=0.25, label=r"$1-\mathrm{e}^{-\Gamma_1\tau}$")
ax2.plot([], [], color="0.0", label=r"$[P(\tau)-P(0)]/q$, all five $P_{\rm AE}$")
ax2.set_xlabel(r"feedback delay $\tau$ (ns)")
ax2.set_ylabel(r"normalized excess (%)")
ax2.legend(loc="upper left", fontsize=7)
ax2.text(0.98, 0.04, "(b)", transform=ax2.transAxes, ha="right", va="bottom")
for a_ in (ax1, ax2):
    a_.grid(True, alpha=0.3, lw=0.4)
fig.tight_layout(h_pad=0.4)
fig.savefig("fig2_causal.pdf")

# ---------------- Figure 3: latency window with residual photons ----------------
d = np.load("photon_curves.npz")
t3 = d["taus"]
fig, ax = plt.subplots(figsize=(W, 2.9))
sel = [("Walter2017", "Walter et al.", matplotlib.colors.to_hex(plt.cm.viridis(0.0)), "-"),
       ("Hazra2025_100", "Hazra et al., 100 ns", matplotlib.colors.to_hex(plt.cm.viridis(0.3)), "--"),
       ("CampagneIbarcq2013", "Campagne-Ibarcq et al.", matplotlib.colors.to_hex(plt.cm.viridis(0.6)), "-."),
       ("slow_n5", r"slow resonator, $\bar n_0=5$", matplotlib.colors.to_hex(plt.cm.viridis(0.78)), ":")]
m = t3 >= 1.0
x = t3[m]
for key, lab, g, ls in sel:
    ax.plot(x, d[key][m] * 100, color=g, ls=ls, label=lab)
ax.plot(x, d["noph"][m] * 100, color=BLUE, lw=2.6, alpha=0.3, label="no residual photons")
ax.axhline(2.0, color=RED, lw=0.8, ls=(0, (1, 2)), label=r"target $\epsilon_{\rm t}=2\%$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1, 3000); ax.set_ylim(0.8, 100)
ax.set_xlabel(r"feedback delay $\tau$ (ns)")
ax.set_ylabel(r"$P(\tau)$ (%)")
ax.legend(loc="upper right", fontsize=6.5)
ax.grid(True, alpha=0.3, lw=0.4, which="both")
fig.savefig("fig3_window.pdf")
print("saved fig1_schedule.pdf, fig2_causal.pdf, fig3_window.pdf")
