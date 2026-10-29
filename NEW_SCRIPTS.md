# Scripts added for the Physica Scripta revision (draft v1.2, not yet released)

| File | What it produces | Needs |
|---|---|---|
| `causal_convention.py` | End-of-slot closed form: excess, latency coefficient q, critical delay, absolute criterion, asymmetric matrix, initial-state dependence; checks against full density-matrix propagation; reproduces the fixed-slot values. | NumPy, SciPy |
| `realistic_case_causal.py` | QuTiP `mesolve` check of the closed form on the 21 x 5 grid (max difference 3.49e-7; roots within 0.02 ns). | QuTiP 5.3.1 |
| `residual_photons_window.py` | Qubit + readout resonator (dispersive model, exact propagation): residual photons, optimal delay, latency window. | NumPy, SciPy |
| `photon_window_mesolve.py` | QuTiP `mesolve` check of the resonator model (max difference 6.12e-7). | QuTiP 5.3.1 |
| `window_analytic.py` | Closed-form window (photon edge, relaxation edge, optimal delay) against the exact model. | NumPy, SciPy |
| `window_edges.py` | Window edges and optimal delays of table 4 by root finding on the exact model. | NumPy, SciPy |
| `paper_numbers.py` | Prints every number quoted in the manuscript and writes `photon_curves.npz` (about 10 minutes). | NumPy, SciPy |
| `make_figures_v2.py` | Figures 1-3. Run `paper_numbers.py` first. | NumPy, SciPy, Matplotlib |

Order: `python paper_numbers.py`, then `python make_figures_v2.py`, then `python window_edges.py`.
