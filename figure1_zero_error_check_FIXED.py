"""
Zero-error validation script (Figure 1) -- matches the Fig. 1 embedded in the manuscript
(excess-definition tau_crit = 506 ns, dotted line at P0 + 1%).

Changes vs. the original, and why:

1. predicted_offset is now computed from the physics (1 - exp(-t_read/T1))
   instead of hard-coded as 0.00500. The exact value is 0.00598 -- close,
   but the manuscript should not quote a rounded number as if it were exact.

2. num_trajectories raised from 2,000 to 20,000. At these low populations
   (0.6%-2.6%), 2,000 trajectories means only ~10-50 "successful" counts per
   point -- far too few for a stable nonlinear fit. Confirmed empirically:
   the "adjacent points look identical" pattern seen at 2,000 trajectories was
   plain counting-statistics noise, NOT a discretization bug (which pairs looked
   close changed with every seed and the effect vanished at 20,000).

3. curve_fit now receives sigma=y_errors_stochastic, absolute_sigma=True.
   The original script computed y_errors_stochastic but never passed it to
   curve_fit, so the fit silently treated every point as equally reliable
   even though low-tau points are much less noisy (in absolute terms) than
   high-tau points.

4. A fixed random seed is passed to mcsolve via `seeds=` for the run that produces
   the saved figure, so it is genuinely reproducible, and a separate block repeats the whole
   measurement N_REPEATS times to report an *empirical* uncertainty on
   tau_crit and offset -- more honest than trusting one fit's covariance
   when the underlying counts are this small.

With ntraj=20,000, 12 independent repeats took about a minute total, so
this is still cheap.
"""

import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qutip import basis, sigmap, Qobj, mcsolve

# =====================================================================
# 1. SIMULATION PARAMETERS
# =====================================================================
delays_ns = np.arange(0, 1050, 50)   # 0-1000 ns, 21 points
num_trajectories = 20000             # was 2000 -- see note above
T1_ns = 50000.0
t_read_ns = 300.0
threshold = 0.01
FIGURE_SEED = 42                     # fixed so the saved figure is reproducible
N_REPEATS = 12                       # for the empirical uncertainty estimate

# predicted_offset derived from physics instead of hard-coded
predicted_offset = 1 - np.exp(-t_read_ns / T1_ns)   # = 0.00598, not 0.00500

# =====================================================================
# 2. QUTIP SETUP
# =====================================================================
e_state = basis(2, 0)
g_state = basis(2, 1)
P_e = e_state * e_state.dag()
psi0 = np.sqrt(1 - predicted_offset) * g_state + np.sqrt(predicted_offset) * e_state
H = Qobj(np.zeros((2, 2)))
gamma = 1.0 / T1_ns
c_ops = [np.sqrt(gamma) * sigmap()]


def active_reset_curve(tau, T1, offset):
    return (1 - np.exp(-tau / T1)) + offset * np.exp(-tau / T1)


def run_and_fit(seed, ntraj):
    # QuTiP 5's mcsolve has its OWN random number generator: np.random.seed() does not
    # control it, so the seed must be passed via the `seeds=` argument or the run is not
    # reproducible. (This is why an earlier version of this script produced a visibly
    # different Figure 1 on every execution.)
    result = mcsolve(H, psi0, delays_ns, c_ops=c_ops, e_ops=[P_e], ntraj=ntraj,
                      seeds=seed, options={"progress_bar": False})
    y = result.expect[0]
    yerr = np.sqrt(np.abs(y * (1 - y)) / ntraj)
    yerr[yerr == 0] = 1e-6  # guard against a zero-variance point

    popt, pcov = curve_fit(active_reset_curve, delays_ns, y, p0=[T1_ns, predicted_offset],
                            sigma=yerr, absolute_sigma=True)
    fitted_T1, fitted_offset = popt
    # EXCESS definition (advisor's Day-1 criterion): P(tau_crit) - P(0) = threshold.
    # Solving Eq. (9): tau_crit = -T1 * ln(1 - threshold / (1 - offset))  -> ~506 ns.
    # (The earlier absolute-threshold form -T1*ln((1-thr)/(1-offset)) gave ~203 ns.)
    tau_crit = -fitted_T1 * np.log(1 - threshold / (1 - fitted_offset))
    return y, yerr, fitted_T1, fitted_offset, tau_crit


# =====================================================================
# 3. THE RUN USED FOR THE ACTUAL FIGURE (fixed seed -> reproducible)
# =====================================================================
print(f"Running mcsolve with ntraj={num_trajectories}, seed={FIGURE_SEED} ...")
start_time = time.time()
y_data_stochastic, y_errors_stochastic, fitted_T1, fitted_offset, tau_crit = \
    run_and_fit(FIGURE_SEED, num_trajectories)
print(f"Simulation time: {time.time() - start_time:.2f} s")

std_err = None  # placeholder; the empirical std below is what's actually reported

# =====================================================================
# 4. EMPIRICAL UNCERTAINTY FROM INDEPENDENT REPEATS
# =====================================================================
print(f"\nRunning {N_REPEATS} independent repeats for an honest uncertainty estimate...")
tau_crit_list, offset_list = [], []
for i in range(N_REPEATS):
    _, _, _, off_i, tc_i = run_and_fit(1000 + i, num_trajectories)
    tau_crit_list.append(tc_i)
    offset_list.append(off_i)

tau_crit_mean, tau_crit_std = np.mean(tau_crit_list), np.std(tau_crit_list)
offset_mean, offset_std = np.mean(offset_list), np.std(offset_list)

tau_crit_exact = -T1_ns * np.log(1 - threshold / (1 - predicted_offset))
print("\n--- RESULTS TO REPORT IN METHODS Sec. D ---")
print(f"Predicted (exact) offset: {predicted_offset:.5f}")
print(f"tau_crit (excess definition, exact from Eq. 9): {tau_crit_exact:.1f} ns")
print(f"Measured offset (empirical, {N_REPEATS} repeats): {offset_mean:.5f} +/- {offset_std:.5f}")
print(f"tau_crit (empirical, {N_REPEATS} repeats): {tau_crit_mean:.1f} +/- {tau_crit_std:.1f} ns")
print(f"(Figure 1 itself uses the single seed={FIGURE_SEED} run: "
      f"offset={fitted_offset:.5f}, tau_crit={tau_crit:.1f} ns)")

# =====================================================================
# 5. FIGURE 1
# =====================================================================
analytic_curve = active_reset_curve(delays_ns, T1_ns, predicted_offset)

plt.figure(figsize=(8, 6))
plt.plot(delays_ns, analytic_curve, color='black', linestyle='-', linewidth=2,
         label='Analytic Solution (Predicted)')
plt.errorbar(delays_ns, y_data_stochastic, yerr=y_errors_stochastic, fmt='ro',
             label=f'Stochastic Simulation (N={num_trajectories} Trajectories)',
             markersize=5, capsize=3, alpha=0.8)
plt.axvline(tau_crit_exact, color='blue', linestyle='--', linewidth=1.5,
            label=r'$\tau_{crit}$' + f' = {tau_crit_exact:.0f} ns')
plt.axvspan(tau_crit_mean - tau_crit_std, tau_crit_mean + tau_crit_std,
            color='blue', alpha=0.1)
plt.axhline(predicted_offset + threshold, color='gray', linestyle=':', linewidth=1,
            label=r'$P_0 + 1\%$ (excess threshold)')

plt.yscale('log')
plt.xlabel(r'Feedback Latency, $\tau$ (ns)', fontsize=12)
plt.ylabel(r'Residual $P(|e\rangle)$', fontsize=12)
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, which="major", linestyle='-', alpha=0.6)
plt.grid(True, which="minor", linestyle='--', alpha=0.3)

plt.savefig('Figure1_ZeroError_Validation.png', dpi=300, bbox_inches='tight')
print("\nFigure saved as 'Figure1_ZeroError_Validation.png'.")
