"""
Trajectory (stochastic) formulation of the realistic-case feedback loop -- cross-check only.

THE BUG: sc_ops = [qt.sigmax()] measures coherence in the X basis, not population.
Verified by direct simulation: p_excited at readout comes out to exactly 0.50000 on
every single trajectory, regardless of the true state -- i.e. the "measurement" carries
zero population information, discrimination becomes an uncorrelated coin flip, and the
final ~48.8% result is nothing but 50% population relaxing naturally for ~1180 ns.

THE FIX: sc_ops = [qt.sigmaz()]. Verified: p_excited at readout now correctly reads
~1.0 for a qubit prepared in |e>, consistent with genuine dispersive QND measurement
(sigma_z commutes with the population operator, so this preserves rather than destroys
population information -- matching the "Dispersive QND" label already in the comments).

CAVEAT to disclose in Methods: this models the backaction as a strong (effectively
projective) QND measurement onto the population basis. The assignment error P_AE is
still applied as an independent probability at the classical discrimination step
(sample_readout), not derived from a finite measurement SNR/efficiency. That is a
simplification worth stating explicitly rather than implying P_AE emerges from the
quantum dynamics itself.

ROLE OF THIS FILE NOW: the manuscript's primary result is the exact evaluation of Eq. (8)
(realistic_case_EXACT.py) and its closed form (closed_form_conditioned.py). This trajectory
code is kept only as a cross-check that the stochastic formulation and the exact one describe
the same physics. It is NOT used to generate any figure or headline number.

WHY THE CROSS-CHECK RUNS AT P_AE = 0.5 (an unphysical value, on purpose):
at a realistic P_AE = 2% the answer is dominated by a branch that occurs in only ~2.6% of
trajectories, so the trajectory-to-trajectory SD (13.7%) is about seven times the mean (1.99%).
120 trajectories then give a standard error of 1.25% on a 1.99% mean -- 63% relative
uncertainty, an interval equally consistent with values far from the closed form. That is not
a test. The closed form holds for ANY P_AE, so we run the comparison where the statistics are
favourable: at P_AE = 0.5 the discriminator output is independent of the qubit state, both
branches occur about half the time, mean and SD are comparable (48.8% vs 48.8%), and 400
trajectories give ~5% relative uncertainty. Choosing the test regime for statistical rather
than physical convenience is the point, not a compromise.

NOTE ON CONDITIONING: because smesolve returns the state conditioned on the measurement
record, sample_readout() draws the assignment from the (nearly projected) single-trajectory
state, so this code is consistent with the conditioned Eq. (8). If you use it for more than a
cross-check, confirm on a few trajectories that rho_at_read is close to a population eigenstate
(p_excited near 0 or 1) rather than the ensemble value 0.994 -- if it is not, the measurement
strength in sc_ops is too weak to project within t_read.
"""

import numpy as np
import qutip as qt

T1 = 50e3
nth = 0.01
T_phi = 100e3
gamma_1 = 1.0 / T1
gamma_phi = 1.0 / (2 * T_phi)  # factor of 1/2: see the T_phi note in Methods Sec. A

t_read = 300.0
t_pulse = 20.0
t_eval = 1500.0

sz = qt.sigmaz()
sx = qt.sigmax()
sm = qt.destroy(2)
sp = sm.dag()

c_ops = [
    np.sqrt(gamma_1 * (1 + nth)) * sm,   # relaxation
    np.sqrt(gamma_1 * nth) * sp,          # thermal excitation
    np.sqrt(gamma_phi) * sz,              # pure dephasing (now correctly normalized)
]
sc_ops = [qt.sigmaz()]  # FIXED (was sigmax()) -- population-preserving QND measurement


def sample_readout(rho, P_AE):
    p_excited = np.real((rho * sp * sm).tr())
    p_assigned_excited = p_excited * (1 - P_AE) + (1 - p_excited) * P_AE
    outcome = np.random.choice([0, 1], p=[1 - p_assigned_excited, p_assigned_excited])
    return outcome


# Internal time step of QuTiP's stochastic integrator. It is set by options["dt"], NOT by the
# spacing of t_list, and its default is 0.001 -- in our nanosecond units that is 300,000 steps
# per readout window. The measurement rate is 1 ns^-1 (sc_ops = sigma_z with unit prefactor),
# so 0.02 ns is still 50 steps per measurement time. Halve it and check the mean is unchanged
# if you want a convergence test.
SME_DT = 0.02


def simulate_single_trajectory(tau, P_AE):
    psi0 = qt.basis(2, 1)
    rho = qt.ket2dm(psi0)

    t_list_read = np.linspace(0, t_read, int(t_read) + 1)
    sme_result = qt.smesolve(qt.Qobj(np.zeros((2, 2))), rho, t_list_read,
                              c_ops=c_ops, sc_ops=sc_ops, ntraj=1,
                              options={"progress_bar": False, "dt": SME_DT})
    rho_at_read = sme_result.states[-1]

    outcome = sample_readout(rho_at_read, P_AE)

    if tau > 0:
        t_list_delay = np.linspace(t_read, t_read + tau, int(tau) + 1)
    else:
        t_list_delay = [t_read, t_read]  # avoid a single-point tlist at tau=0
    delay_result = qt.mesolve(qt.Qobj(np.zeros((2, 2))), rho_at_read, t_list_delay, c_ops=c_ops)
    rho_at_pulse = delay_result.states[-1]

    t_start_pulse = t_read + tau
    t_end_pulse = t_start_pulse + t_pulse
    t_list_pulse = np.linspace(t_start_pulse, t_end_pulse, int(t_pulse) + 1)

    if outcome == 1:
        omega = np.pi / t_pulse
        H_pulse = (omega / 2.0) * sx
        pulse_result = qt.mesolve(H_pulse, rho_at_pulse, t_list_pulse, c_ops=c_ops)
    else:
        pulse_result = qt.mesolve(qt.Qobj(np.zeros((2, 2))), rho_at_pulse, t_list_pulse, c_ops=c_ops)
    rho_after_pulse = pulse_result.states[-1]

    t_list_final = np.linspace(t_end_pulse, t_eval, int(t_eval - t_end_pulse) + 1)
    final_result = qt.mesolve(qt.Qobj(np.zeros((2, 2))), rho_after_pulse, t_list_final, c_ops=c_ops)

    rho_final = final_result.states[-1]
    return np.real((rho_final * sp * sm).tr())


if __name__ == "__main__":
    import time
    np.random.seed(42)          # controls sample_readout's discrimination draw
    TAU, P_AE, N = 0.0, 0.5, 400
    try:
        from closed_form_conditioned import P_conditioned
        expected = P_conditioned(TAU, P_AE)
    except ImportError:
        expected = None

    t0 = time.time()
    vals = []
    for i in range(N):
        vals.append(simulate_single_trajectory(TAU, P_AE))
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{N} trajectories, {el:.0f}s elapsed, ~{el/(i+1)*(N-i-1):.0f}s remaining", flush=True)
    mean = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(N)
    print(f"{N} trajectories at tau={TAU:.0f} ns, P_AE={P_AE} in {time.time()-t0:.1f}s")
    print(f"  sample mean      : {mean*100:.2f} % +/- {se*100:.2f} % (standard error)")
    print(f"  trajectory SD    : {np.std(vals, ddof=1)*100:.2f} %")
    print(f"  relative uncert. : {se/mean*100:.1f} %")
    if expected is not None:
        print(f"  closed form      : {expected*100:.2f} %")
        print(f"  difference       : {(mean-expected)/se:+.2f} standard errors")
    print("\n  -> put the sample mean, its standard error and N into the placeholder in Sec. E.")
