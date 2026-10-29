# Latency limits of active reset in superconducting qubits

Simulation and analysis code for the manuscript *Latency limits of active reset in superconducting
qubits: readout contrast, residual photons and leakage* (submitted to Physica Scripta, 2026), and for
its earlier preprint version, *A Closed-Form Excess-Population Latency Budget for Active Reset of
Superconducting Qubits* (Zenodo, doi:10.5281/zenodo.23295225).

**Author:** Kuzey Kaya, Independent researcher, Gaziantep, Türkiye
**ORCID:** [0009-0003-6608-7620](https://orcid.org/0009-0003-6608-7620)

## Version 1.2 (revised manuscript)

The revised manuscript evaluates the reset error at the end of the pulse slot (the moment the next
operation can start). The error added by a feedback delay τ is then q(1 − e^(−Γ₁τ)), with a latency
coefficient q set by the readout contrast P_e(1 − 2ε). Including the readout resonator, residual
photons bound the delay from below and relaxation from above, giving a window of admissible delays
and an optimal delay. The scripts for these results are listed in `NEW_SCRIPTS.md`; run
`paper_numbers.py`, then `make_figures_v2.py`, then `window_edges.py` to reproduce every number,
table and figure of the revised manuscript.

## Version 1.1 and earlier (preprint)

The scripts below reproduce the preprint, which used a fixed evaluation clock t_eval. In that
convention the leading-order rule is: tolerable latency = error budget × T₁ / pulse-firing
probability. The revised manuscript keeps this result as its fixed-slot appendix.

## Contents

| File | What it produces |
|---|---|
| `revision_calculations.py` | The full set of results: exact affine pulse propagator, the threshold existence condition, the results table, the asymmetric confusion matrix, the initial-population sensitivity, and Figure 3. Requires only NumPy, SciPy and Matplotlib. |
| `closed_form_conditioned.py` | The closed-form solution of Sec. II.E (Eqs. 11–16), the analytic critical latency, Figure 2 and the (τ, P_AE) grid. No QuTiP needed. |
| `realistic_case_EXACT.py` | Exact four-branch master-equation evaluation of Eq. (8) with QuTiP `mesolve`, and its comparison against the closed form. |
| `figure1_zero_error_check_FIXED.py` | Zero-error validation and Figure 1: QuTiP `mcsolve`, 20,000 trajectories, plus the 12-ensemble uncertainty estimate quoted in Sec. II.D. |
| `trajectory_sim_FIXED.py` | First stochastic-trajectory cross-check with QuTiP `smesolve`, at P_AE = 0.5 and zero delay (see the docstring for why that value is used). |
| `stochastic_verification_stratified.py` | Second, stronger cross-check: the assignment channel is integrated out analytically and only the readout trajectory is sampled, so the test runs at nonzero delay (τ = 0, 250, 500, 1000 ns) and realistic assignment error (P_AE = 1%, 2%, 5%). Prints the projection diagnostics and the decomposition of the residual quoted in Sec. II.E. QuTiP; about 90 s at the default N_READ = 300. |
| `make_figures.py` | Regenerates Figures 2 and 3 with the published hardware latencies marked. |
| `fig2_grid_conditioned.csv` | The Figure 2 data. |
| `Figure1_ZeroError_Validation.png`, `Figure2_conditioned.png`, `Figure3_design_rule.png` | The figures as they appear in the manuscript. |

## Running

```bash
pip install -r requirements.txt
python revision_calculations.py        # no QuTiP required
python closed_form_conditioned.py      # no QuTiP required
python realistic_case_EXACT.py         # QuTiP
python figure1_zero_error_check_FIXED.py       # QuTiP, a few minutes
python trajectory_sim_FIXED.py                 # QuTiP, a few minutes
python stochastic_verification_stratified.py   # QuTiP, about 90 s
```

`stochastic_verification_stratified.py` imports `closed_form_conditioned.py`, so run it from
this directory.

## Reproducibility

All trajectory seeds are fixed explicitly in the scripts, so rerunning reproduces every
figure and every quoted ensemble exactly. Note that QuTiP 5's stochastic and Monte Carlo
solvers take their seed through the `seeds=` argument; `numpy.random.seed()` does not
control them.

Solver settings: `mesolve` runs at default tolerances; `smesolve` uses a fixed integration
step of 0.02 ns, which is fifty steps per inverse measurement rate. Halving that step to
0.01 ns changes the cross-check sample mean by less than a third of its standard error, so
the comparison is converged in the integration step.

The two stochastic cross-checks differ in what they sample. `trajectory_sim_FIXED.py`
samples both the readout trajectory and the assignment outcome, which is why it is run at
P_AE = 0.5, where neither branch is rare. `stochastic_verification_stratified.py` integrates
the assignment channel out analytically — its probabilities are known exactly — and samples
only the readout trajectory, which reduces the variance by a factor of 1 × 10⁵ to 7 × 10⁵ across the grid at the same 300
trajectories and makes the test usable at realistic P_AE and nonzero delay. Its output
separates three contributions to the difference from the closed form: the within-ensemble
standard error, the binomial uncertainty of the sampled projected fraction, and the finite
projection completeness of the conditioned states. At the default settings (seed 42,
N_READ = 300) the residual after the last two are accounted for is below 3 × 10⁻⁷.

Model parameters throughout: T₁ = 50 μs, n_th = 0.01, T_φ = 100 μs, t_read = 300 ns,
t_pulse = 20 ns, t_eval = 1500 ns.

## Citation

If you use this code, please cite the manuscript and this archive. The archive DOI is
minted by Zenodo on release.
