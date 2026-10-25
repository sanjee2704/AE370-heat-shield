"""
heating.py -- stagnation-point convective heating from an entry trajectory.

Sutton-Graves correlation (Earth air):
    q_conv = K_SG * sqrt(rho / R_n) * V^3       [W/m^2]
    K_SG   = 1.7415e-4 kg^0.5 / m               (SI units)

Reference: K. Sutton and R. A. Graves, "A General Stagnation-Point
Convective-Heating Equation for Arbitrary Gas Mixtures," NASA TR R-376, 1971.

ASSUMPTIONS (state these in the report):
  * Cold-wall convective flux. There is no hot-wall correction, so heating is
    slightly overpredicted (conservative for sizing).
  * Shock-layer radiative heating is neglected. It matters above ~10 km/s,
    so lunar-return cases are a limitation.
  * Heating is set to zero after the trajectory ends (parachute deploy).
    This is the post-flight "soak-back" phase.

Interface for the conduction model:
    hist = HeatingHistory(trajectory, R_n)
    hist.q_conv(t)    -> q at any time t [W/m^2] (linear interpolation)
    hist.q_peak, hist.t_peak, hist.Q_total
"""

import numpy as np
from scipy.special import erf

from physics.trajectory import H_SCALE, density, simulate_entry

K_SG = 1.7415e-4     # Sutton-Graves constant for Earth [kg^0.5 / m]


def sutton_graves(rho, V, R_n):
    """Stagnation-point convective heat flux [W/m^2]."""
    return K_SG * np.sqrt(np.asarray(rho) / R_n) * np.asarray(V)**3


class HeatingHistory:
    """Heat-flux history along a trajectory, callable at arbitrary times."""

    def __init__(self, trajectory, R_n):
        self.traj = trajectory
        self.R_n = float(R_n)
        self.t = trajectory.t
        self.q = sutton_graves(trajectory.rho, trajectory.V, self.R_n)
        self.t_end = self.t[-1]

        i = np.argmax(self.q)
        self.q_peak = self.q[i]
        self.t_peak = self.t[i]
        self.Q_total = np.trapezoid(self.q, self.t)          # [J/m^2]

    def q_conv(self, t):
        """Heat flux at time t [W/m^2]; zero before t = 0 and after the trajectory ends."""
        return np.interp(t, self.t, self.q, left=0.0, right=0.0)

    def __call__(self, t):
        return self.q_conv(t)

    def summary(self):
        return (f"q_peak = {self.q_peak / 1e4:7.2f} W/cm^2 at t = {self.t_peak:6.1f} s,  "
                f"Q = {self.Q_total / 1e7:6.2f} kJ/cm^2,  duration = {self.t_end:6.0f} s")


def heating_for_entry(V_E, gamma_E_deg, beta, LD=0.0, R_n=1.0, dt=0.1, V_end=150.0, **kw):
    """Convenience: simulate the entry and return its HeatingHistory."""
    tr = simulate_entry(V_E, gamma_E_deg, beta, LD=LD, dt=dt, V_end=V_end, **kw)
    return HeatingHistory(tr, R_n)


# --- exact ballistic results (Allen-Eggers limit: L/D = 0, no gravity, flat Earth)
def ballistic_q_peak(V_E, gamma_E_deg, beta, R_n, h_E=120e3):
    """Exact peak flux. It occurs where rho = beta |sin g| / (3H).

    The factor exp(a rho_E) accounts for the small but nonzero density at h_E.
    """
    s = abs(np.sin(np.radians(gamma_E_deg)))
    a = H_SCALE / (beta * s)
    rho_star = beta * s / (3 * H_SCALE)
    V_star = V_E * np.exp(-0.5 * a * (rho_star - density(h_E)))
    return K_SG * np.sqrt(rho_star / R_n) * V_star**3


def ballistic_Q(V_E, gamma_E_deg, beta, R_n, h_E=120e3, h_end=0.0):
    """Exact total heat load between h_E and h_end:
         Q = K V_E^2 e^(a rho_E) H / (sqrt(R_n) s) * sqrt(pi/a)
             * [erf(sqrt(a rho_end)) - erf(sqrt(a rho_E))],
       with s = |sin gamma_E|, a = H / (beta s), rho_E = rho(h_E).
    """
    s = abs(np.sin(np.radians(gamma_E_deg)))
    a = H_SCALE / (beta * s)
    rho_E = density(h_E)
    span = erf(np.sqrt(a * density(h_end))) - erf(np.sqrt(a * rho_E))
    return (K_SG * V_E**2 * np.exp(a * rho_E) * H_SCALE / (np.sqrt(R_n) * s)
            * np.sqrt(np.pi / a) * span)


# ---------------------------------------------------------------------------
# Self-check:  python -m physics.heating   (run from repo root)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Spot value: rho = 1e-4, V = 7 km/s, R_n = 1 m -> 1.7415e-4 * 0.01 * 3.43e11
    q1 = sutton_graves(1e-4, 7000.0, 1.0)
    assert np.isclose(q1, 1.7415e-4 * 0.01 * 7000.0**3)
    print(f"[1] Spot check: q = {q1 / 1e4:.2f} W/cm^2 (hand calc matches)")

    # 2) Exact ballistic peak flux and total heat load
    V_E, g_E, beta, R_n = 7800.0, -10.0, 300.0, 1.0
    qp_ex = ballistic_q_peak(V_E, g_E, beta, R_n)
    Q_ex = ballistic_Q(V_E, g_E, beta, R_n, h_end=10e3)
    print("[2] Ballistic check vs exact (V_E = 7.8 km/s, gamma = -10 deg, beta = 300):")
    print(f"      exact:  q_peak = {qp_ex / 1e4:.3f} W/cm^2,  Q = {Q_ex / 1e7:.5f} kJ/cm^2")
    dts, errQ = [0.4, 0.2, 0.1, 0.05], []
    for dt in dts:
        tr = simulate_entry(V_E, g_E, beta, dt=dt, h_end=10e3,
                            gravity=False, curvature=False)
        hist = HeatingHistory(tr, R_n)
        errQ.append(abs(hist.Q_total - Q_ex) / Q_ex)
        print(f"      dt = {dt:4.2f} s: q_peak err = {abs(hist.q_peak - qp_ex) / qp_ex:.1e}, "
              f"Q rel err = {errQ[-1]:.2e}")
    orders = np.log2(np.array(errQ[:-1]) / np.array(errQ[1:]))
    print(f"      observed order of Q (trapezoid): {np.round(orders, 2)}")
    assert abs(hist.q_peak - qp_ex) / qp_ex < 1e-4
    # order checked on the coarser steps; the finest approaches the error floor
    assert errQ[-1] < 1e-6 and orders[0] > 1.8

    # 3) Interface: q_conv interpolates the samples and is zero after landing
    assert np.allclose(hist.q_conv(hist.t), hist.q)
    assert hist.q_conv(hist.t_end + 100.0) == 0.0
    print("[3] q_conv(t) interface OK (matches samples, zero after t_end)")

    # 4) Peak flux vs total heat load across entry angles (the project's core tradeoff)
    print("[4] LEO return, beta = 300, L/D = 0.3, R_n = 1 m:")
    angles = [-1.5, -3.0, -6.0, -10.0]
    hists = {}
    for g in angles:
        hists[g] = heating_for_entry(7800.0, g, 300.0, LD=0.3, R_n=1.0)
        print(f"      gamma_E = {g:5.1f} deg: {hists[g].summary()}")
    print("All heating checks passed.")

    # Figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for g, h in hists.items():
        a1.plot(h.t, h.q / 1e4, label=f"gamma_E = {g:g} deg")
    a1.set(xlabel="t [s]", ylabel="q_conv [W/cm^2]", title="Stagnation-point heating")
    a1.legend()
    for g, h in hists.items():
        a2.plot(h.q_peak / 1e4, h.Q_total / 1e7, "o", ms=8, label=f"{g:g} deg")
    a2.set(xlabel="peak heat flux [W/cm^2]", ylabel="total heat load [kJ/cm^2]",
           title="Peak flux vs heat load")
    a2.legend(title="gamma_E")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "heating_overview.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
