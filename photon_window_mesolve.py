"""
QuTiP mesolve check of the residual-photon model in residual_photons_window.py.

Qubit (x) readout resonator, dispersive Hamiltonian H = chi a^dag a sz (+ Omega/2 sx during the pulse),
resonator decay sqrt(kappa) a, qubit collapse operators of Eq. (2). Resonator parameters of Walter et al. (2017):
chi/2pi = 7.9 MHz (half splitting), kappa/2pi = 37.5 MHz, n0 = 2.5. P_AE = 1 %.
Evaluation at the end of the pulse slot (causal convention).

Run on your own machine (QuTiP 5.3.1), from the folder that holds the other scripts:
    python photon_window_mesolve.py
Paste back everything it prints (about 10 lines). Run time: well under a minute.
"""
import numpy as np
import qutip as qt
from residual_photons_window import branch_curves, total

T1, nth, T_phi = 50e3, 0.01, 100e3
g1, gphi = 1 / T1, 1 / T_phi
t_read, t_pulse = 300.0, 20.0
chi, kappa, n0, eps = 2 * np.pi * 7.9e-3, 2 * np.pi * 37.5e-3, 2.5, 0.01
N = 22

a = qt.tensor(qt.qeye(2), qt.destroy(N))
sm = qt.tensor(qt.destroy(2), qt.qeye(N))          # |e> = basis(2, 1)
Pe = sm.dag() * sm
sz = 2 * Pe - qt.tensor(qt.qeye(2), qt.qeye(N))     # +1 on |e>
sx = sm + sm.dag()
H0 = chi * a.dag() * a * sz
Hp = H0 + 0.5 * (np.pi / t_pulse) * sx
c_ops = [np.sqrt(kappa) * a, np.sqrt(g1 * (1 + nth)) * sm, np.sqrt(g1 * nth) * sm.dag(),
         np.sqrt(gphi / 2) * sz]
opts = {"atol": 1e-10, "rtol": 1e-8, "nsteps": 100000}
field = qt.coherent_dm(N, np.sqrt(n0))
rho_e = qt.tensor(qt.ket2dm(qt.basis(2, 1)), field)
rho_g = qt.tensor(qt.ket2dm(qt.basis(2, 0)), field)
P_read = qt.mesolve(0 * qt.sigmaz(), qt.ket2dm(qt.basis(2, 1)), np.linspace(0, t_read, 301),
                    c_ops=[np.sqrt(g1 * (1 + nth)) * qt.destroy(2), np.sqrt(g1 * nth) * qt.create(2),
                           np.sqrt(gphi / 2) * qt.sigmaz()]).states[-1]
P_read = np.real((P_read * qt.create(2) * qt.destroy(2)).tr())


def branch(rho, tau, pulse):
    if tau > 0:
        rho = qt.mesolve(H0, rho, np.linspace(0, tau, int(tau) * 4 + 2), c_ops=c_ops, options=opts).states[-1]
    rho = qt.mesolve(Hp if pulse else H0, rho, np.linspace(0, t_pulse, 401), c_ops=c_ops, options=opts).states[-1]
    return np.real((rho * Pe).tr())


if __name__ == "__main__":
    print(f"qutip {qt.__version__}; P_read = {P_read:.6f}")
    taus = np.array([0.0, 10.0, 30.0, 100.0, 500.0])
    ref = total(branch_curves(chi, kappa, n0, np.linspace(0, 500, 101), N), eps)
    ref_at = {t: ref[int(round(t / 5))] for t in taus}
    worst = 0.0
    for t in taus:
        P = (P_read * ((1 - eps) * branch(rho_e, t, True) + eps * branch(rho_e, t, False))
             + (1 - P_read) * ((1 - eps) * branch(rho_g, t, False) + eps * branch(rho_g, t, True)))
        worst = max(worst, abs(P - ref_at[t]))
        print(f"tau = {t:5.0f} ns: mesolve P = {P*100:.5f} %   exact propagation = {ref_at[t]*100:.5f} %")
    print(f"max |mesolve - exact propagation| = {worst:.2e}")
