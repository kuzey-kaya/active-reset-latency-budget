"""
stochastic_verification_stratified.py  (v2)

Strengthened stochastic cross-check requested by both reviewers:
  R1 (B9 iv): test the trajectory code at NONZERO delay and REALISTIC assignment error.
  R1 (B9 v):  distinguish integration convergence from sampling agreement and report the
              difference uncertainty.
  R2 (B9 iii): use branch-conditioned sampling so that the rare branch is reached without
              relying on chance.

WHY THE OLD CHECK WAS WEAK
At a realistic P_AE the residual is dominated by the branch "projected |e>, reported g",
which occurs in ~2% of cycles. Sampling the assignment coin means most trajectories land in
the uninformative majority branch, and the standard error is huge (1.25% on a 1.99% mean
with 120 trajectories). Running at P_AE = 0.5 sidestepped this but tested only an
unphysical point and only tau = 0.

WHAT THIS SCRIPT DOES INSTEAD
The assignment coin is classical and its probabilities are known exactly, so it is
integrated out analytically rather than sampled (stratification over the reported outcome).
What remains stochastic is exactly the thing that needs testing: the readout trajectory
itself, i.e. whether smesolve with the sigma_z monitor actually leaves the qubit in a
projected population eigenstate at the end of the window, and whether the deterministic
evolution that follows reproduces the closed form.

Per readout trajectory i, with conditioned post-readout population P_i:
    q_i(eps)  = P_i (1 - eps) + (1 - P_i) eps                 # prob. the pulse fires
    est_i(tau, eps) = q_i * B_pulse(P_i, tau) + (1 - q_i) * B_nopulse(P_i, tau)
where B_pulse and B_nopulse are integrated with mesolve from the conditioned state.

WHAT v2 ADDS (after the first run of v1 on 2026-09-14)
The v1 run gave estimates above the closed form by 2.7 to 9.7 "standard errors", growing
with tau.  Both effects are now identified and reported explicitly:

  (a) The readout outcome is itself a Bernoulli draw with P(|e>) = P_read = 0.99396.  With
      N = 300 the expected number of |g> outcomes is 1.8 and the probability of observing
      none is 0.16 -- which is what happened.  The estimator then evaluates the model at the
      REALISED projected fraction (P_e = 1) rather than at P_read, and Eq. (14) predicts the
      consequence exactly: the excess is larger by p_pulse(1)/p_pulse(P_read) = 1.0060.
      The v1 standard error (spread of est_i across trajectories) did not contain this
      binomial term at all, because every trajectory sat in the same stratum.  v2 propagates
      it: SE_total^2 = SE_within^2 + (dEst/dP_e)^2 * P_e (1 - P_e) / N.
  (b) The conditioned states are not perfectly projected: mean(P_i) = 0.99998, i.e. a mean
      incompleteness delta = 2.3e-5.  A deficit delta enters the estimator twice with the
      same sign -- once through q_i (fewer pulses, more weight on the almost fully excited
      no-pulse branch) and once through the pulse-map input -- so it raises est by about
      2 * 0.976 * delta * (1 - eps) = 4.4e-5 (1 - eps), independent of tau.  v2 computes this
      prediction from the measured mean(P_i) and subtracts it.

v2 therefore reports three comparisons per grid point: the raw difference from the closed
form at P_read (what v1 printed), the difference from the closed form at the realised P_e
(removes (a)), and the residual after also subtracting the projection-incompleteness offset
(removes (b)).  It also compares the EXCESS P(tau) - P(0), which is the quantity Eq. (14)
actually predicts and which is insensitive to (b) by construction.

Trajectories are seeded (seeds=SEED), so re-running with the same N_READ reproduces the same
conditioned states and the same numbers exactly.  Raising N_READ tightens the binomial
term in (a) (SE on P_e falls as 1/sqrt(N)) and, once |g> outcomes appear, populates the
second stratum; the stratum-wise reweighting below then uses the exact weights P_read and
1 - P_read (R2's branch-conditioned sampling for the projection outcome as well).

Run:  python stochastic_verification_stratified.py
About 90 s for N_READ = 300 on a laptop; ~15 min for N_READ = 3000.
"""

import time
import numpy as np
import qutip as qt

# ----------------------------------------------------------------- parameters (Sec. II.A/B)
T1, n_th, T_phi = 50e3, 0.01, 100e3
t_read, t_pulse, t_eval = 300.0, 20.0, 1500.0
gamma_1 = 1.0 / T1
gamma_phi = 1.0 / (2 * T_phi)
SME_DT = 0.02            # stochastic integrator step, fifty steps per inverse measurement rate
N_READ = 300             # readout trajectories (reused for every grid point); try 3000 for the
                         # final run -- see docstring
SEED = 42

TAUS = [0.0, 250.0, 500.0, 1000.0]
EPSS = [0.01, 0.02, 0.05]

# closed form for comparison
try:
    from closed_form_conditioned import P_conditioned, P_pulse, P_nopulse, P_read as P_READ_CF
except ImportError:
    P_conditioned = None

# ----------------------------------------------------------------- operators (same conventions as trajectory_sim_FIXED.py)
sz, sx = qt.sigmaz(), qt.sigmax()
sm = qt.destroy(2)          # |g><e|  (basis(2,1) = |e>)
sp = sm.dag()
proj_e = qt.basis(2, 1) * qt.basis(2, 1).dag()
c_ops = [np.sqrt(gamma_1 * (1 + n_th)) * sm,
         np.sqrt(gamma_1 * n_th) * sp,
         np.sqrt(gamma_phi) * sz]
sc_ops = [sz]               # unit prefactor: measurement rate 1 ns^-1
H0 = qt.Qobj(np.zeros((2, 2)))
H_pulse = (np.pi / t_pulse / 2.0) * sx


def p_e(rho):
    return float(qt.expect(proj_e, rho))


# ----------------------------------------------------------------- 1. readout trajectories
def readout_trajectories(N, seed):
    """N conditioned post-readout states for a qubit prepared in |e>."""
    rho0 = qt.ket2dm(qt.basis(2, 1))
    tlist = np.linspace(0, t_read, int(t_read) + 1)
    res = qt.smesolve(H0, rho0, tlist, c_ops=c_ops, sc_ops=sc_ops, ntraj=N,
                      options={"progress_bar": False, "dt": SME_DT, "keep_runs_results": True},
                      seeds=seed)
    # QuTiP 5: per-trajectory states are in res.trajectories
    states = [traj.states[-1] for traj in res.trajectories]
    return states


# ----------------------------------------------------------------- 2. deterministic branches
def branch_pulse(rho, tau):
    t0 = t_read
    if tau > 0:
        rho = qt.mesolve(H0, rho, np.linspace(t0, t0 + tau, int(tau) + 1), c_ops=c_ops).states[-1]
    t1 = t0 + tau
    rho = qt.mesolve(H_pulse, rho, np.linspace(t1, t1 + t_pulse, int(t_pulse) + 1), c_ops=c_ops).states[-1]
    t2 = t1 + t_pulse
    rho = qt.mesolve(H0, rho, np.linspace(t2, t_eval, int(t_eval - t2) + 1), c_ops=c_ops).states[-1]
    return p_e(rho)


def branch_nopulse(rho, tau):
    rho = qt.mesolve(H0, rho, np.linspace(t_read, t_eval, int(t_eval - t_read) + 1), c_ops=c_ops).states[-1]
    return p_e(rho)   # tau-independent, computed once per trajectory


# ----------------------------------------------------------------- 3. branch evaluation (QuTiP part)
def evaluate_branches(states, taus):
    """Returns P_i (N,), no (N,), pu dict[tau] -> (N,).  Everything downstream is pure numpy."""
    P_i = np.array([p_e(r) for r in states])
    no = np.array([branch_nopulse(r, 0.0) for r in states])
    pu = {tau: np.array([branch_pulse(r, tau) for r in states]) for tau in taus}
    return P_i, no, pu


# ----------------------------------------------------------------- 4. closed-form helpers (pure numpy)
def closed_form(Pe, tau, eps, delta_e=0.0, delta_g=0.0):
    """
    Closed form of Sec. II.E with the projected fraction Pe as a free parameter.
    delta_e / delta_g: mean incompleteness of the conditioned states in each stratum
    (population 1 - delta_e in the |e> stratum, delta_g in the |g> stratum), so that the
    projection-incompleteness offset can be predicted from the measured mean(P_i).
    With Pe = P_read and deltas = 0 this is P_conditioned(tau, eps).
    """
    Pi_e, Pi_g = 1.0 - delta_e, delta_g
    q_e = Pi_e * (1 - eps) + (1 - Pi_e) * eps
    q_g = Pi_g * (1 - eps) + (1 - Pi_g) * eps
    from_e = q_e * P_pulse(Pi_e, tau) + (1 - q_e) * P_nopulse(Pi_e)
    from_g = q_g * P_pulse(Pi_g, tau) + (1 - q_g) * P_nopulse(Pi_g)
    return Pe * from_e + (1 - Pe) * from_g, from_e, from_g


# ----------------------------------------------------------------- 5. stratified estimator (pure numpy)
def stratified_estimate(P_i, no, pu_tau, eps, reweight=True):
    """
    est_i = q_i pu_i + (1 - q_i) no_i with the assignment coin integrated out.
    reweight=True: split trajectories by projected outcome (P_i > 0.5) and combine the two
    strata with the EXACT weights P_read, 1 - P_read instead of their sampled frequencies
    (branch-conditioned sampling of the projection outcome).  If a stratum is empty its
    branch is taken from the ideal state (|g> stratum: P_i = 0), and this is reported.
    Returns mean, SE_within, and a dict of diagnostics.
    """
    q = P_i * (1 - eps) + (1 - P_i) * eps
    est = q * pu_tau + (1 - q) * no
    N = len(est)
    if not reweight:
        return est.mean(), est.std(ddof=1) / np.sqrt(N), {}
    e_mask = P_i > 0.5
    n_e, n_g = int(e_mask.sum()), int((~e_mask).sum())
    m_e = est[e_mask].mean()
    se_e = est[e_mask].std(ddof=1) / np.sqrt(n_e) if n_e > 1 else 0.0
    if n_g > 0:
        m_g = est[~e_mask].mean()
        se_g = est[~e_mask].std(ddof=1) / np.sqrt(n_g) if n_g > 1 else 0.0
    else:
        m_g, se_g = None, 0.0     # no sampled |g> outcome: the reweighted table is skipped
    return m_e, se_e, {"n_e": n_e, "n_g": n_g, "m_g": m_g, "se_g": se_g}


# ----------------------------------------------------------------- 6. naive estimator (for contrast)
def naive(P_i, no, pu_tau, eps, rng):
    """Sample the assignment coin per trajectory instead of integrating it out."""
    vals = []
    for Pi, no_i, pu_i in zip(P_i, no, pu_tau):
        projected_e = rng.random() < Pi
        report_e = (rng.random() > eps) if projected_e else (rng.random() < eps)
        vals.append(pu_i if report_e else no_i)
    vals = np.array(vals)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))


# ================================================================= main
if __name__ == "__main__":
    t0 = time.time()
    print(f"Generating {N_READ} readout trajectories (smesolve, dt = {SME_DT} ns, seed = {SEED}) ...")
    states = readout_trajectories(N_READ, SEED)
    print(f"  done in {time.time() - t0:.0f} s")

    print("\nEvaluating branches with mesolve for every trajectory and delay ...")
    t1 = time.time()
    P_i, no, pu = evaluate_branches(states, TAUS)
    print(f"  done in {time.time() - t1:.0f} s")

    P_read_mesolve = p_e(qt.mesolve(H0, qt.ket2dm(qt.basis(2, 1)), np.linspace(0, t_read, 301),
                                    c_ops=c_ops).states[-1])
    N = len(P_i)
    e_mask = P_i > 0.5
    n_e, n_g = int(e_mask.sum()), int((~e_mask).sum())
    f_e = n_e / N
    se_fe = np.sqrt(P_read_mesolve * (1 - P_read_mesolve) / N)
    delta_e = 1.0 - P_i[e_mask].mean() if n_e else float("nan")
    delta_g = P_i[~e_mask].mean() if n_g else float("nan")
    proj_frac = np.mean((P_i < 1e-3) | (P_i > 1 - 1e-3))

    print("\n" + "=" * 86)
    print("READOUT PROJECTION   (the only sampled quantity)")
    print("=" * 86)
    print(f"  trajectories                     : {N}")
    print(f"  projected to |e> / |g>           : {n_e} / {n_g}   (expected {N*P_read_mesolve:.1f} / "
          f"{N*(1-P_read_mesolve):.1f}; P(zero |g> outcomes) = {P_read_mesolve**N:.3f})")
    print(f"  realised fraction P_e            : {f_e:.4f}   (P_read = {P_read_mesolve:.5f}, binomial SE {se_fe:.4f})")
    print(f"  within 1e-3 of an eigenstate     : {proj_frac*100:.1f}%")
    print(f"  mean incompleteness, |e> stratum : delta_e = 1 - <P_i> = {delta_e:.2e}")
    if n_g:
        print(f"  mean incompleteness, |g> stratum : delta_g = <P_i> = {delta_g:.2e}")

    if P_conditioned is None:
        print("\nclosed_form_conditioned.py not found next to this script: the comparison below needs it.")
        raise SystemExit

    print("\n" + "=" * 86)
    print("STRATIFIED CROSS-CHECK   (assignment coin integrated out; readout sampled)")
    print("=" * 86)
    print("  Columns: trajectory estimate; SE_within (spread across trajectories); SE_total (adds the")
    print("  binomial uncertainty of the realised projected fraction); closed form at P_read [paper];")
    print("  closed form at the REALISED P_e; residual after also removing the predicted projection-")
    print("  incompleteness offset.  Differences in percentage points.")
    print("  tau   eps |   est      SE_within  SE_total |  cf(P_read)  diff/SE_tot | cf(P_e real)  diff  | pred.offset  residual")
    worst_z, worst_res = 0.0, 0.0
    rows = {}
    for tau in TAUS:
        for eps in EPSS:
            m_raw, se_w, _ = stratified_estimate(P_i, no, pu[tau], eps, reweight=False)
            cf_paper, from_e, from_g = closed_form(P_read_mesolve, tau, eps)
            cf_real, _, _ = closed_form(f_e, tau, eps)
            cf_real_delta, _, _ = closed_form(f_e, tau, eps,
                                              delta_e=delta_e if n_e else 0.0,
                                              delta_g=delta_g if n_g else 0.0)
            dEst_dPe = from_e - from_g
            se_tot = np.sqrt(se_w**2 + (dEst_dPe * se_fe)**2)
            z = (m_raw - cf_paper) / se_tot
            offset_pred = cf_real_delta - cf_real
            residual = m_raw - cf_real - offset_pred
            worst_z = max(worst_z, abs(z)); worst_res = max(worst_res, abs(residual))
            rows[(tau, eps)] = m_raw
            print(f"  {tau:5.0f} {eps*100:3.0f}% | {m_raw*100:8.4f}%  {se_w*100:.4f}%   {se_tot*100:.4f}% | "
                  f"{cf_paper*100:9.4f}%   {z:+6.2f}    | {cf_real*100:9.4f}%  {100*(m_raw-cf_real):+.4f} | "
                  f"{offset_pred*100:+.4f}     {residual*100:+.5f}")
    print(f"\n  largest |diff|/SE_total against the paper's closed form : {worst_z:.2f}")
    print(f"  largest residual after removing both identified effects : {worst_res*100:.5f} pp")
    print("  (the residual is the disagreement between the mesolve pipeline and the closed form once the")
    print("   sampled projection outcome and the finite projection completeness are accounted for)")

    print("\n" + "=" * 86)
    print("EXCESS P(tau) - P(0)   (the quantity Eq. (14) predicts; insensitive to the incompleteness offset)")
    print("=" * 86)
    print("  eps   tau |  trajectory   cf(P_read)   cf(P_e real) |  traj/cf(P_read)   p_pulse(P_e)/p_pulse(P_read)")
    for eps in EPSS:
        for tau in TAUS[1:]:
            ex_t = rows[(tau, eps)] - rows[(0.0, eps)]
            ex_p = closed_form(P_read_mesolve, tau, eps)[0] - closed_form(P_read_mesolve, 0.0, eps)[0]
            ex_r = closed_form(f_e, tau, eps)[0] - closed_form(f_e, 0.0, eps)[0]
            pp_ratio = (f_e*(1-eps)+(1-f_e)*eps) / (P_read_mesolve*(1-eps)+(1-P_read_mesolve)*eps)
            print(f"  {eps*100:3.0f}% {tau:5.0f} | {ex_t*100:9.4f}   {ex_p*100:9.4f}    {ex_r*100:9.4f}   |   "
                  f"{ex_t/ex_p:.4f}            {pp_ratio:.4f}")
    print("  (if the two ratio columns agree, the tau-dependence of Eq. (14) is reproduced and the whole")
    print("   difference from the paper's value is the realised projected fraction)")

    if n_g:
        print("\n" + "=" * 86)
        print("STRATUM-REWEIGHTED ESTIMATE   (both strata sampled; combined with exact weights P_read, 1-P_read)")
        print("=" * 86)
        print("  tau   eps |  reweighted est   SE   |  cf(P_read)   diff/SE")
        for tau in TAUS:
            for eps in EPSS:
                m_e, se_e, d = stratified_estimate(P_i, no, pu[tau], eps, reweight=True)
                m = P_read_mesolve * m_e + (1 - P_read_mesolve) * d["m_g"]
                se = np.sqrt((P_read_mesolve * se_e)**2 + ((1 - P_read_mesolve) * d["se_g"])**2)
                cf_paper = closed_form(P_read_mesolve, tau, eps)[0]
                print(f"  {tau:5.0f} {eps*100:3.0f}% | {m*100:9.4f}%  {se*100:.4f}% | {cf_paper*100:9.4f}%   "
                      f"{(m-cf_paper)/se if se > 0 else float('nan'):+6.2f}")

    print("\n" + "=" * 86)
    print("NAIVE ESTIMATOR at tau = 500 ns, eps = 2%  (assignment coin sampled, for contrast)")
    print("=" * 86)
    rng = np.random.default_rng(SEED)
    m_n, se_n = naive(P_i, no, pu[500.0], 0.02, rng)
    m_s, se_s, _ = stratified_estimate(P_i, no, pu[500.0], 0.02, reweight=False)
    cf500 = closed_form(P_read_mesolve, 500.0, 0.02)[0]
    print(f"  naive      : {m_n*100:.4f}% +/- {se_n*100:.4f}%   (relative SE {se_n/m_n*100:.0f}%; "
          f"{(m_n-cf500)/se_n:+.1f} SE from the closed form)")
    print(f"  stratified : {m_s*100:.4f}% +/- {se_s*100:.4f}%   (relative SE {se_s/m_s*100:.2f}%)")
    print(f"  variance reduction factor: {(se_n/se_s)**2:.0f}x with the same {N} trajectories")

    print(f"\nTotal runtime {time.time() - t0:.0f} s.")
    print("\n-> Paste everything from 'READOUT PROJECTION' down into the chat.")
