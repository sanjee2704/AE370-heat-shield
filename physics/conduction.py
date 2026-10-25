"""
conduction.py -- 1D transient conduction through a multilayer TPS stack,
semi-discretized in space (method of lines) for the TR-BDF2 time integrator.

    dT/dt = f(t, T)          T = node temperatures, shape (N,)

Interface for the solver:
    model.f(t, T)    -> dT/dt, shape (N,)
    model.jac(t, T)  -> df/dT in scipy.linalg.solve_banded format, shape (3, N),
                        for solve_banded((1, 1), ab, rhs)

DISCRETIZATION (vertex-centered finite volume)
----------------------------------------------
Nodes sit on the surface (x = 0), on every material interface, and on the back
face. Every cell between two nodes is a single material, so no conductivity
averaging is needed at interfaces, and node 0 is the true surface temperature.

Each node j owns half of each neighboring cell. Its heat capacity is
    C_j = sum over adjacent cells of  rho * c(T_j) * h / 2      [J/m^2-K]

The flux through cell e (from node e to node e+1, +x is inward) is
    F_e = k(T_mid) * (T_e - T_{e+1}) / h,   with T_mid = (T_e + T_{e+1}) / 2

The node balance is
    C_j dT_j/dt = (flux in) - (flux out)

Boundary conditions:
    surface (x = 0):  flux in = q_conv(t) - eps*sigma*(T_0^4 - T_env^4)
    back face:        adiabatic (flux out = 0)

The Jacobian is exact: it includes the radiation term, dk/dT, and dc/dT.
dc/dT is taken by a small central difference of the tabulated c(T).
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from physics.materials import SIGMA, RCG_EMISSIVITY, li900, aluminum_2024


@dataclass
class Layer:
    material: object      # a materials.TabulatedMaterial
    thickness: float      # [m]
    n_cells: int          # uniform cells across this layer


class MultilayerSlab:
    """Multilayer 1D conduction model. Layer 0 is the heated (outer) layer."""

    def __init__(self, layers, q_conv, emissivity=RCG_EMISSIVITY, T_env=0.0):
        self.layers = list(layers)
        self.q_conv = q_conv            # callable q_conv(t) [W/m^2]
        self.emissivity = float(emissivity)
        self.T_env = float(T_env)

        # Grid
        x = [0.0]
        self._slices = []                # (cell index slice, material, h) per layer
        self.interface_nodes = []        # node index at the end of each layer
        n0 = 0
        for L in self.layers:
            h = L.thickness / L.n_cells
            x.extend(x[-1] + h * np.arange(1, L.n_cells + 1))
            self._slices.append((slice(n0, n0 + L.n_cells), L.material, h))
            n0 += L.n_cells
            self.interface_nodes.append(n0)
        self.x = np.array(x)
        self.N = len(self.x)
        self.n_cells = n0

        # Bondline = interface between layer 0 (tile) and layer 1 (structure)
        self.bondline_node = self.interface_nodes[0] if len(self.layers) > 1 else self.N - 1

    # ------------------------------------------------------------------ pieces
    def q_net(self, t, T):
        """Net heat flux into the surface [W/m^2]."""
        T0 = T[0]
        return self.q_conv(t) - self.emissivity * SIGMA * (T0**4 - self.T_env**4)

    def capacity(self, T):
        """Node heat capacities C_j [J/m^2-K] and their derivatives dC_j/dT_j."""
        C = np.zeros(self.N)
        dC = np.zeros(self.N)
        d = 1e-3
        for sl, m, h in self._slices:
            e = np.arange(self.N - 1)[sl]
            for nodes in (e, e + 1):
                Tn = T[nodes]
                C[nodes] += m.rho * m.c(Tn) * h / 2
                dc = (m.c(Tn + d) - m.c(Tn - d)) / (2 * d)
                dC[nodes] += m.rho * dc * h / 2
        return C, dC

    def _cell_fluxes(self, T):
        """Cell fluxes F_e and their partials dF/dT_e (a), dF/dT_{e+1} (b)."""
        F = np.empty(self.n_cells)
        a = np.empty(self.n_cells)
        b = np.empty(self.n_cells)
        for sl, m, h in self._slices:
            e = np.arange(self.n_cells)[sl]
            Ta, Tb = T[e], T[e + 1]
            Tm = 0.5 * (Ta + Tb)
            G = m.k(Tm) / h
            extra = m.dk_dT(Tm) / (2 * h) * (Ta - Tb)
            F[e] = G * (Ta - Tb)
            a[e] = G + extra
            b[e] = -G + extra
        return F, a, b

    def _residual(self, t, T):
        """R_j = (flux in) - (flux out) for each node [W/m^2]."""
        F, _, _ = self._cell_fluxes(T)
        R = np.zeros(self.N)
        R[0] += self.q_net(t, T)
        R[:-1] -= F
        R[1:] += F
        return R

    # --------------------------------------------------------------- interface
    def f(self, t, T):
        T = np.asarray(T, dtype=float)
        C, _ = self.capacity(T)
        return self._residual(t, T) / C

    def jac(self, t, T):
        """Exact Jacobian df/dT in banded (3, N) form for solve_banded((1, 1), ...)."""
        T = np.asarray(T, dtype=float)
        C, dC = self.capacity(T)
        F, a, b = self._cell_fluxes(T)
        R = np.zeros(self.N)
        R[0] += self.q_net(t, T)
        R[:-1] -= F
        R[1:] += F

        D = np.zeros(self.N)                 # dR_j/dT_j
        U = np.zeros(self.N - 1)             # dR_j/dT_{j+1}
        Lo = np.zeros(self.N - 1)            # dR_{j+1}/dT_j
        D[:-1] -= a
        U -= b
        Lo += a
        D[1:] += b
        D[0] += -4.0 * self.emissivity * SIGMA * T[0]**3

        # f = R / C  ->  df/dT = (dR/dT)/C - diag(R C' / C^2)
        ab = np.zeros((3, self.N))
        ab[0, 1:] = U / C[:-1]
        ab[1, :] = D / C - R * dC / C**2
        ab[2, :-1] = Lo / C[1:]
        return ab

    def jac_dense(self, t, T):
        """Dense Jacobian (for testing only)."""
        ab = self.jac(t, T)
        return np.diag(ab[1]) + np.diag(ab[0, 1:], 1) + np.diag(ab[2, :-1], -1)

    def energy(self, T):
        """Stored energy per unit area relative to T_REF [J/m^2].

        Consistent with the discretization: dE/dt = sum_j C_j dT_j/dt = q_net.
        """
        E = 0.0
        for sl, m, h in self._slices:
            e = np.arange(self.n_cells)[sl]
            E += np.sum(m.rho * h / 2 * (m.enthalpy(T[e]) + m.enthalpy(T[e + 1])))
        return E

    def stiffness_estimate(self, t, T):
        """Gershgorin bound on the largest |eigenvalue| of the Jacobian [1/s].

        Explicit Euler would need dt < ~2 / (this value) for stability.
        """
        ab = self.jac(t, T)
        row = np.abs(ab[1]).copy()
        row[:-1] += np.abs(ab[0, 1:])
        row[1:] += np.abs(ab[2, :-1])
        return row.max()

    def uniform(self, T0):
        return np.full(self.N, float(T0))


def tps_stack(tile_thickness, q_conv, al_thickness=2.0e-3, n_tile=60, n_al=4,
              pressure_pa=1013.3, emissivity=RCG_EMISSIVITY):
    """LI-900 tile on an aluminum skin.

    The bond/strain-isolation pad is not modeled separately (assumption).
    """
    layers = [Layer(li900(pressure_pa), tile_thickness, n_tile),
              Layer(aluminum_2024(), al_thickness, n_al)]
    return MultilayerSlab(layers, q_conv, emissivity=emissivity)


# ---------------------------------------------------------------------------
# Self-check:  python -m physics.conduction   (run from repo root)
# ---------------------------------------------------------------------------
def _cn_march_linear(model, T0, t_end, dt):
    """Crank-Nicolson march for the LINEAR verification case only.

    Test helper for checking the spatial discretization. It is NOT the
    project's time integrator (that is solver/tr_bdf2.py).
    """
    T = T0.copy()
    A = model.jac_dense(0.0, T)                  # constant for the linear case
    n = int(round(t_end / dt))
    t = 0.0
    lhs = -0.5 * dt * model.jac(0.0, T)
    lhs[1] += 1.0
    for _ in range(n):
        b0 = model.f(t, T) - A @ T               # f = A T + b(t)
        b1 = model.f(t + dt, T) - A @ T
        rhs = T + 0.5 * dt * (A @ T) + 0.5 * dt * (b0 + b1)
        T = solve_banded((1, 1), lhs, rhs)
        t += dt
    return T


if __name__ == "__main__":
    from pathlib import Path
    from scipy.special import erfc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physics.materials import TabulatedMaterial

    rng = np.random.default_rng(0)

    # 1) Jacobian vs finite differences (full nonlinear stack)
    model = tps_stack(0.05, q_conv=lambda t: 2.0e5, n_tile=30, n_al=4)
    T = np.sort(rng.uniform(300.0, 1500.0, model.N))[::-1]
    J = model.jac_dense(0.0, T)
    J_fd = np.zeros_like(J)
    for j in range(model.N):
        d = 1e-4 * T[j]
        Tp, Tm = T.copy(), T.copy()
        Tp[j] += d
        Tm[j] -= d
        J_fd[:, j] = (model.f(0.0, Tp) - model.f(0.0, Tm)) / (2 * d)
    err_J = np.max(np.abs(J - J_fd)) / np.max(np.abs(J_fd))
    print(f"[1] Jacobian vs finite difference: max rel err = {err_J:.2e}")
    assert err_J < 1e-4, "Jacobian does not match finite differences"

    # 2) Semi-discrete energy identity: sum_j C_j f_j == q_net
    C, _ = model.capacity(T)
    lhs_E = np.sum(C * model.f(0.0, T))
    rhs_E = model.q_net(0.0, T)
    err_E = abs(lhs_E - rhs_E) / abs(rhs_E)
    print(f"[2] Energy identity: sum C*dT/dt = {lhs_E:.6e}, q_net = {rhs_E:.6e}, "
          f"rel err = {err_E:.1e}")
    assert err_E < 1e-10

    # 3) Exact solution: constant-property semi-infinite slab, constant flux
    #    T(x,t) = T0 + (2q/k) [ sqrt(a t/pi) exp(-x^2/(4 a t)) - (x/2) erfc(x/(2 sqrt(a t))) ]
    rho, k, c = 144.0, 0.05, 1000.0
    alpha = k / (rho * c)
    q0, T_init, t_end, L = 2.0e3, 300.0, 100.0, 0.05     # penetration ~ 0.024 m << L
    const = TabulatedMaterial("const", rho, [0, 1e4], [k, k], [0, 1e4], [c, c])

    def exact(x, t):
        s = np.sqrt(alpha * t)
        return T_init + (2 * q0 / k) * (s / np.sqrt(np.pi) * np.exp(-x**2 / (4 * s**2))
                                       - x / 2 * erfc(x / (2 * s)))

    Ns, errs = [10, 20, 40, 80, 160], []
    for n in Ns:
        lin = MultilayerSlab([Layer(const, L, n)], q_conv=lambda t: q0, emissivity=0.0)
        Tn = _cn_march_linear(lin, lin.uniform(T_init), t_end, dt=0.01)
        errs.append(np.max(np.abs(Tn - exact(lin.x, t_end))))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print("[3] Semi-infinite slab, max error vs exact solution:")
    for n, e in zip(Ns, errs):
        print(f"      N = {n:4d}   err = {e:.3e} K")
    print(f"      observed order: {np.round(orders, 2)}")
    assert orders[-1] > 1.8, "spatial convergence is not 2nd order"

    # 4) Stiffness of the real stack
    lam = model.stiffness_estimate(0.0, model.uniform(300.0))
    print(f"[4] Stiffness estimate (300 K): |lambda|max ~ {lam:.1f} 1/s "
          f"-> explicit Euler dt < ~{2 / lam:.2e} s")
    print("All conduction checks passed.")

    # Figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    lin = MultilayerSlab([Layer(const, L, 40)], q_conv=lambda t: q0, emissivity=0.0)
    xf = np.linspace(0, 0.03, 300)
    for tt in (10.0, 40.0, 100.0):
        a1.plot(1e3 * xf, exact(xf, tt), "k-", lw=1)
        Tn = _cn_march_linear(lin, lin.uniform(T_init), tt, dt=0.01)
        a1.plot(1e3 * lin.x, Tn, "o", ms=3, label=f"t = {tt:g} s")
    a1.set(xlim=(0, 30), xlabel="depth x [mm]", ylabel="T [K]",
           title="Numerical (o) vs exact (-)")
    a1.legend()
    hs = L / np.array(Ns)
    a2.loglog(1e3 * hs, errs, "o-", label="max error")
    a2.loglog(1e3 * hs, errs[-1] * (hs / hs[-1])**2, "k--", label="slope 2")
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set(xlabel="h [mm]", ylabel="max |T - T_exact| [K]", title="Spatial convergence")
    a2.legend()
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "conduction_verification.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
