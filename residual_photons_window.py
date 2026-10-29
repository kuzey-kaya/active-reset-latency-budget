"""
Major 4: residual readout photons and the latency window.

Model (dispersive, half-chi convention: resonator at w_r +/- chi, qubit shifted by 2 chi per photon)
    H      = chi a^dag a sigma_z                         (delay and idle slot)
    H_pi   = chi a^dag a sigma_z + (Omega/2) sigma_x      (pulse slot, drive at the BARE qubit frequency)
    c_ops  = sqrt(kappa) a, plus the qubit operators of Eq. (2)
At the end of the readout window the qubit is projected (|e> or |g>) and the resonator holds a coherent
state with mean photon number n0 (midpoint drive: |alpha_e| = |alpha_g|). The phase of alpha is irrelevant
because H and the dissipators are invariant under a -> a e^{i phi}. Each of the four branches is propagated
exactly (sparse Liouvillian, expm_multiply); the evaluation is at the end of the pulse slot (causal convention).
With n0 = 0 the model must reproduce causal_convention.py exactly (checked below).

Resonator parameters are taken from published devices (see the printed table); qubit parameters are the
manuscript's (T1 = 50 us, T_phi = 100 us, n_th = 0.01, t_read = 300 ns, t_pulse = 20 ns).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from causal_convention import (g1, gphi, nth, t_pulse, P_read, P_causal_cf, P0_causal, q_c, G1)

TWO_PI = 2 * np.pi
MHz = 1e-3  # rad/ns per (2 pi MHz) after multiplying by 2 pi


def build(chi, kappa, N):
    """Sparse Liouvillians (row-major vec) for qubit (x) Fock(N); basis qubit (|e>, |g>)."""
    I2, IN = sp.identity(2, format='csr'), sp.identity(N, format='csr')
    a = sp.diags(np.sqrt(np.arange(1, N)), 1, format='csr')
    num = sp.diags(np.arange(N, dtype=float), 0, format='csr')
    smq = sp.csr_matrix(np.array([[0, 0], [1, 0]], float))
    szq = sp.csr_matrix(np.diag([1.0, -1.0]))
    sxq = sp.csr_matrix(np.array([[0, 1], [1, 0]], float))
    H0 = chi * sp.kron(szq, num)
    Hd = (np.pi / t_pulse / 2) * sp.kron(sxq, IN)
    cops = [np.sqrt(kappa) * sp.kron(I2, a),
            np.sqrt(g1 * (1 + nth)) * sp.kron(smq, IN),
            np.sqrt(g1 * nth) * sp.kron(smq.T, IN),
            np.sqrt(gphi / 2) * sp.kron(szq, IN)]
    d = 2 * N
    Id = sp.identity(d, format='csr')

    def liou(H):
        L = -1j * (sp.kron(H, Id) - sp.kron(Id, H.T))
        for c in cops:
            cdc = (c.conj().T @ c)
            L = L + sp.kron(c, c.conj()) - 0.5 * sp.kron(cdc, Id) - 0.5 * sp.kron(Id, cdc.T)
        return sp.csr_matrix(L)

    Pe_op = sp.kron(sp.csr_matrix(np.diag([1.0, 0.0])), IN).toarray()
    return liou(H0), liou(H0 + Hd), Pe_op, d


def coherent_dm(n0, N):
    from math import factorial
    al = np.sqrt(n0)
    v = np.array([np.exp(-n0 / 2) * al ** k / np.sqrt(float(factorial(k))) for k in range(N)])
    return np.outer(v, v)


def branch_curves(chi, kappa, n0, taus, N):
    """Population at the end of the slot for the four branches, on a uniform tau grid starting at 0."""
    L0, Lp, Pe_op, d = build(chi, kappa, N)
    fld = coherent_dm(n0, N)
    out = {}
    for proj in 'eg':
        q = np.diag([1.0, 0.0]) if proj == 'e' else np.diag([0.0, 1.0])
        v0 = np.kron(q, fld).reshape(-1).astype(complex)
        dt = taus[1] - taus[0]
        states = expm_multiply(L0, v0, start=0.0, stop=taus[-1], num=len(taus), endpoint=True)
        for rep, L in (('e', Lp), ('g', L0)):
            vals = []
            for s in states:
                r = expm_multiply(L * t_pulse, s).reshape(d, d)
                vals.append(np.real(np.trace(Pe_op @ r)))
            out[proj + rep] = np.array(vals)
    return out


def total(curves, eps):
    w = dict(ee=P_read * (1 - eps), eg=P_read * eps, ge=(1 - P_read) * eps, gg=(1 - P_read) * (1 - eps))
    return sum(w[k] * curves[k] for k in w)


def interval_below(taus, P, target):
    ok = P <= target
    if not ok.any():
        return None
    i0 = np.argmax(ok)
    i1 = len(ok) - 1 - np.argmax(ok[::-1])
    return taus[i0], taus[i1]


if __name__ == "__main__":
    # ---- check 1: n0 = 0 reproduces the causal closed form ----
    taus = np.linspace(0, 1000, 21)
    c0 = branch_curves(TWO_PI * 7.9 * MHz, TWO_PI * 37.5 * MHz, 0.0, taus, 4)
    dev = max(abs(total(c0, p)[i] - P_causal_cf(t, p)) for i, t in enumerate(taus) for p in [0.01, 0.05])
    print(f"[check] n0 = 0 vs causal closed form, max |diff| = {dev:.2e}")

    # ---- devices: resonator parameters as published (chi = half splitting, kappa = energy decay) ----
    devices = [
        # label,                       chi/2pi MHz, kappa/2pi MHz, n0
        ("Walter 2017 (Purcell, fast)",       7.9,  37.5,  2.5),
        ("Hazra 2025, 100 ns readout",        3.2,  11.6,  7.6),
        ("Hazra 2025, 240 ns readout",        3.2,  11.6,  1.2),
        ("Campagne-Ibarcq 2013 (slow)",       0.78,  1.83, 1.4),
    ]
    eps, target = 0.01, 0.02
    print(f"\nP_AE = {eps*100:.0f}%, absolute target {target*100:.0f}%\n")
    for label, chi_MHz, kap_MHz, n0 in devices:
        chi, kap = TWO_PI * chi_MHz * MHz, TWO_PI * kap_MHz * MHz
        tmax = 1500.0 if kap_MHz > 5 else 4000.0
        taus = np.linspace(0, tmax, 301 if kap_MHz > 5 else 401)
        N = int(np.ceil(n0 + 7 * np.sqrt(n0) + 8))
        cv = branch_curves(chi, kap, n0, taus, N)
        cv2 = branch_curves(chi, kap, n0, taus[::10], N + 8)          # truncation check
        trunc = np.max(np.abs(total(cv, eps)[::10] - total(cv2, eps)))
        P = total(cv, eps)
        Pref = np.array([P_causal_cf(t, eps) for t in taus])
        i_star = np.argmin(P)
        win = interval_below(taus, P, target)
        win_ref = interval_below(taus, Pref, target)
        # photon-limited lower bound: first tau where photon excess falls below 1e-3 (0.1 pp)
        exc = P - Pref
        lo_1e3 = taus[np.argmax(exc <= 1e-3)] if (exc <= 1e-3).any() else np.nan
        print(f"{label}: chi/2pi={chi_MHz} MHz, kappa/2pi={kap_MHz} MHz (1/kappa={1/kap:.1f} ns), n0={n0}, N={N}, "
              f"trunc err {trunc:.1e}")
        print(f"   P(0) = {P[0]*100:.2f}%  (no photons {Pref[0]*100:.2f}%);  photon excess < 0.1 pp from tau = {lo_1e3:.0f} ns")
        print(f"   optimum tau* = {taus[i_star]:.0f} ns, P(tau*) = {P[i_star]*100:.3f}%")
        print(f"   window P <= {target*100:.0f}%: {None if win is None else f'[{win[0]:.0f}, {win[1]:.0f}] ns'}"
              f"   (no photons: {None if win_ref is None else f'[{win_ref[0]:.0f}, {win_ref[1]:.0f}] ns'})\n")
