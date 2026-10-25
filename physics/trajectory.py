"""
trajectory.py -- planar atmospheric entry, integrated with our own RK4.

State y = [V, gamma, h, s]:
    V      speed [m/s]
    gamma  flight-path angle [rad], negative when descending
    h      altitude [m]
    s      downrange distance [m]

Equations of motion (non-rotating spherical Earth, constant beta and L/D):
    dV/dt     = -rho V^2 / (2 beta) - g sin(gamma)
    dgamma/dt = (L/D) rho V / (2 beta) - (g/V - V/(R_E + h)) cos(gamma)
    dh/dt     =  V sin(gamma)
    ds/dt     =  V cos(gamma) R_E / (R_E + h)

with beta = m / (C_D S) the ballistic coefficient [kg/m^2],
g = g0 (R_E / (R_E + h))^2, and an exponential atmosphere rho = rho0 exp(-h/H).

This is a non-stiff IVP, so classical explicit RK4 (4th order) is appropriate.
It is the trajectory half of the problem; the stiff conduction half uses TR-BDF2.

Verification (in the self-check): Allen-Eggers ballistic solution. With
L/D = 0, gravity off, and flat Earth, gamma stays constant and
    V(h) = V_E exp( -H (rho(h) - rho(h_E)) / (2 beta |sin gamma_E|) )
    peak deceleration = V_E^2 |sin gamma_E| / (2 e H)
"""

from dataclasses import dataclass

import numpy as np

R_E = 6.371e6        # Earth mean radius [m]
G0 = 9.80665         # standard gravity [m/s^2]
RHO0 = 1.225         # sea-level density [kg/m^3]
H_SCALE = 7200.0     # atmospheric scale height [m] (assumption; see report)


def density(h):
    """Exponential atmosphere [kg/m^3]."""
    return RHO0 * np.exp(-np.asarray(h, dtype=float) / H_SCALE)


def rk4_step(f, t, y, dt):
    """One classical 4th-order Runge-Kutta step for y' = f(t, y)."""
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclass
class Trajectory:
    t: np.ndarray        # [s]
    V: np.ndarray        # [m/s]
    gamma: np.ndarray    # [rad]
    h: np.ndarray        # [m]
    s: np.ndarray        # [m]
    beta: float
    LD: float
    status: str          # "landed", "slowed", "skipped", or "t_max"

    @property
    def rho(self):
        return density(self.h)

    @property
    def decel_g(self):
        """Aerodynamic deceleration in Earth g's (drag only)."""
        return self.rho * self.V**2 / (2 * self.beta) / G0


def entry_rhs(beta, LD, gravity=True, curvature=True):
    """Right-hand side f(t, y) of the entry equations."""
    def f(t, y):
        V, gam, h, _ = y
        rho = RHO0 * np.exp(-h / H_SCALE)
        r = R_E + h
        g = G0 * (R_E / r)**2 if gravity else 0.0
        drag = rho * V**2 / (2 * beta)
        dV = -drag - g * np.sin(gam)
        turn = g / V - (V / r if curvature else 0.0)
        dgam = LD * drag / V - turn * np.cos(gam)
        dh = V * np.sin(gam)
        ds = V * np.cos(gam) * R_E / r
        return np.array([dV, dgam, dh, ds])
    return f


def simulate_entry(V_E, gamma_E_deg, beta, LD=0.0, h_E=120e3, dt=0.1,
                   h_end=0.0, V_end=0.0, t_max=5000.0,
                   gravity=True, curvature=True):
    """Integrate an entry from the entry interface until a stop condition.

    Stops when h <= h_end ("landed"), V <= V_end ("slowed", e.g. parachute
    deploy), h > h_E + 1 km ("skipped" out of the atmosphere), or t > t_max.
    The last point is linearly interpolated onto the stop condition.
    """
    f = entry_rhs(beta, LD, gravity, curvature)
    y = np.array([V_E, np.radians(gamma_E_deg), h_E, 0.0])
    t = 0.0
    ts, ys = [t], [y]
    status = "t_max"
    while t < t_max:
        y_new = rk4_step(f, t, y, dt)
        t_new = t + dt
        stop = None
        if y_new[2] <= h_end:
            stop, frac = "landed", (y[2] - h_end) / (y[2] - y_new[2])
        elif y_new[0] <= V_end:
            stop, frac = "slowed", (y[0] - V_end) / (y[0] - y_new[0])
        elif y_new[2] > h_E + 1e3:
            stop, frac = "skipped", 1.0
        if stop:
            ts.append(t + frac * dt)
            ys.append(y + frac * (y_new - y))
            status = stop
            break
        t, y = t_new, y_new
        ts.append(t)
        ys.append(y)
    Y = np.array(ys)
    return Trajectory(np.array(ts), Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3],
                      beta, LD, status)


def allen_eggers_V(h, V_E, gamma_E_deg, beta, h_E=120e3):
    """Exact ballistic speed vs altitude (no gravity, flat Earth, L/D = 0)."""
    sg = abs(np.sin(np.radians(gamma_E_deg)))
    return V_E * np.exp(-H_SCALE * (density(h) - density(h_E)) / (2 * beta * sg))


# ---------------------------------------------------------------------------
# Self-check:  python -m physics.trajectory   (run from repo root)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V_E, gam_E, beta = 7800.0, -10.0, 300.0

    # 1) Allen-Eggers: speed vs altitude, and RK4 order
    dts = [2.0, 1.0, 0.5, 0.25]
    errs = []
    for dt in dts:
        tr = simulate_entry(V_E, gam_E, beta, dt=dt, h_end=10e3,
                            gravity=False, curvature=False)
        Vx = allen_eggers_V(tr.h[:-1], V_E, gam_E, beta)   # skip interpolated end point
        errs.append(np.max(np.abs(tr.V[:-1] - Vx)))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print("[1] Allen-Eggers ballistic check, max |V - V_exact| along the path:")
    for dt, e in zip(dts, errs):
        print(f"      dt = {dt:5.2f} s   err = {e:.3e} m/s")
    print(f"      observed order: {np.round(orders, 2)}")
    assert errs[-1] < 1e-3 and orders[-1] > 3.5, "RK4 not converging at 4th order"

    # gamma must stay exactly constant in this limit
    assert np.allclose(tr.gamma, np.radians(gam_E), atol=1e-12)

    # 2) Peak deceleration vs Allen-Eggers
    tr = simulate_entry(V_E, gam_E, beta, dt=0.05, h_end=0.0,
                        gravity=False, curvature=False)
    a_num = np.max(tr.decel_g) * G0
    a_ex = V_E**2 * abs(np.sin(np.radians(gam_E))) / (2 * np.e * H_SCALE)
    err_a = abs(a_num - a_ex) / a_ex
    print(f"[2] Peak decel: numerical {a_num / G0:.3f} g, exact {a_ex / G0:.3f} g, "
          f"rel err {err_a:.1e}")
    assert err_a < 1e-3

    # 3) Full equations (gravity, curvature, lift) for a few entry angles
    print("[3] Full equations, LEO return (V_E = 7.8 km/s, beta = 300, L/D = 0.3):")
    cases = {}
    for g in (-1.5, -3.0, -6.0):
        tr = simulate_entry(V_E, g, beta, LD=0.3, dt=0.1, V_end=150.0)
        cases[g] = tr
        print(f"      gamma_E = {g:5.1f} deg: {tr.status}, t = {tr.t[-1]:6.0f} s, "
              f"peak decel = {tr.decel_g.max():5.2f} g")
    print("All trajectory checks passed.")

    # Figure
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4))
    tr = simulate_entry(V_E, gam_E, beta, dt=0.5, h_end=10e3,
                        gravity=False, curvature=False)
    hh = np.linspace(10e3, 120e3, 400)
    a1.plot(allen_eggers_V(hh, V_E, gam_E, beta) / 1e3, hh / 1e3, "k-", lw=1, label="exact")
    a1.plot(tr.V[::4] / 1e3, tr.h[::4] / 1e3, "o", ms=3, label="RK4")
    a1.set(xlabel="V [km/s]", ylabel="h [km]", title="Allen-Eggers check")
    a1.legend()
    a2.loglog(dts, errs, "o-", label="max error")
    a2.loglog(dts, errs[-1] * (np.array(dts) / dts[-1])**4, "k--", label="slope 4")
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set(xlabel="dt [s]", ylabel="max |V - V_exact| [m/s]", title="RK4 convergence")
    a2.legend()
    for g, tr in cases.items():
        a3.plot(tr.t, tr.h / 1e3, label=f"gamma_E = {g:g} deg")
    a3.set(xlabel="t [s]", ylabel="h [km]", title="Full equations, L/D = 0.3")
    a3.legend()
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "trajectory_verification.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
