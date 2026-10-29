"""
two_layer.py -- two-layer lumped thermal model of a heat shield.

Layer 1 (outer): LI-900 tile, thickness L1, average temperature T1.
Layer 2 (inner): aluminum 2024-T3 skin, thickness L2, average temperature T2.

    C1(T1) dT1/dt = q(t) - eps*sigma*(T1^4 - T_env^4) - G(T1) (T1 - T2)
    C2     dT2/dt =                                     G(T1) (T1 - T2)

    T1(0) = T2(0) = T0

with (all per unit area)
    C1(T1) = rho1 c1(T1) L1          heat capacity of the tile      [J/m^2-K]
    C2     = rho2 c2 L2              heat capacity of the skin      [J/m^2-K]
    G(T1)  = 1 / ( L1/(2 k1(T1)) + L2/(2 k2) )
                                     conductance between the two layer centers [W/m^2-K]

The back face of the skin is adiabatic. The outer layer re-radiates to an
environment at T_env (default T0, so the initial state is an equilibrium).

Interface for the solver:
    model.f(t, T)   -> dT/dt, shape (2,)
    model.jac(t, T) -> 2x2 Jacobian (exact)
"""

import numpy as np

from physics.materials import SIGMA, RCG_EMISSIVITY, li900, aluminum_2024

T0_DEFAULT = 293.15     # initial temperature [K]


# ---------------------------------------------------------------------------
# Heat-flux profiles q(t) [W/m^2]
# ---------------------------------------------------------------------------
class HalfSinePulse:
    """q(t) = q_peak sin(pi t / tau) for 0 <= t <= tau, zero otherwise.

    Total heat delivered: Q = 2 q_peak tau / pi  [J/m^2]  (exact).
    """

    def __init__(self, q_peak, tau):
        self.q_peak = float(q_peak)
        self.tau = float(tau)

    @property
    def Q(self):
        return 2.0 * self.q_peak * self.tau / np.pi

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        return np.where((t >= 0) & (t <= self.tau),
                        self.q_peak * np.sin(np.pi * t / self.tau), 0.0)

    def __repr__(self):
        return (f"HalfSinePulse(q_peak={self.q_peak / 1e4:g} W/cm^2, tau={self.tau:g} s, "
                f"Q={self.Q / 1e7:.3g} kJ/cm^2)")


class ConstantFlux:
    """q(t) = q0 for t >= 0."""

    def __init__(self, q0):
        self.q0 = float(q0)

    def __call__(self, t):
        return np.where(np.asarray(t) >= 0, self.q0, 0.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TwoLayerShield:
    def __init__(self, q, L1=0.05, L2=2.0e-3, outer=None, inner=None,
                 emissivity=RCG_EMISSIVITY, T_env=T0_DEFAULT,
                 constant_props_at=None):
        """
        q                  heat-flux profile, callable q(t) [W/m^2]
        L1, L2             outer / inner layer thicknesses [m]
        outer, inner       materials (default LI-900 at 0.01 atm, Al 2024-T3)
        emissivity         surface emissivity (0 turns radiation off)
        constant_props_at  if a temperature [K] is given, k1 and c1 are frozen
                           at that temperature (linear-limit tests, sensitivity)
        """
        self.q = q
        self.L1, self.L2 = float(L1), float(L2)
        self.m1 = outer if outer is not None else li900()
        self.m2 = inner if inner is not None else aluminum_2024()
        self.eps = float(emissivity)
        self.T_env = float(T_env)
        self.T_frozen = constant_props_at
        self.k2 = float(self.m2.k(T0_DEFAULT))
        self.C2 = self.m2.rho * float(self.m2.c(T0_DEFAULT)) * self.L2

    # --- properties of the outer layer -------------------------------------
    def _k1(self, T1):
        return self.m1.k(self.T_frozen if self.T_frozen is not None else T1)

    def _dk1(self, T1):
        return 0.0 if self.T_frozen is not None else self.m1.dk_dT(T1)

    def _c1(self, T1):
        return self.m1.c(self.T_frozen if self.T_frozen is not None else T1)

    def _dc1(self, T1, d=1e-3):
        if self.T_frozen is not None:
            return 0.0
        return (self.m1.c(T1 + d) - self.m1.c(T1 - d)) / (2 * d)

    def C1(self, T1):
        return self.m1.rho * self._c1(T1) * self.L1

    def G(self, T1):
        return 1.0 / (self.L1 / (2 * self._k1(T1)) + self.L2 / (2 * self.k2))

    def _dG(self, T1):
        G = self.G(T1)
        return G**2 * self.L1 / (2 * self._k1(T1)**2) * self._dk1(T1)

    # --- right-hand side ------------------------------------------------------
    def radiation(self, T1):
        return self.eps * SIGMA * (T1**4 - self.T_env**4)

    def f(self, t, T):
        T1, T2 = T
        flow = self.G(T1) * (T1 - T2)
        R1 = self.q(t) - self.radiation(T1) - flow
        return np.array([R1 / self.C1(T1), flow / self.C2])

    def jac(self, t, T):
        T1, T2 = T
        G, dG = self.G(T1), self._dG(T1)
        C1 = self.C1(T1)
        dC1 = self.m1.rho * self._dc1(T1) * self.L1
        flow = G * (T1 - T2)
        R1 = self.q(t) - self.radiation(T1) - flow
        dflow_dT1 = G + dG * (T1 - T2)
        J = np.empty((2, 2))
        J[0, 0] = (-4 * self.eps * SIGMA * T1**3 - dflow_dT1) / C1 - R1 * dC1 / C1**2
        J[0, 1] = G / C1
        J[1, 0] = dflow_dT1 / self.C2
        J[1, 1] = -G / self.C2
        return J

    # --- diagnostics ----------------------------------------------------------
    def energy(self, T):
        """Stored energy per unit area relative to T_REF [J/m^2]."""
        T1, T2 = np.asarray(T)[..., 0], np.asarray(T)[..., 1]
        if self.T_frozen is not None:
            E1 = self.m1.rho * self.L1 * self._c1(T1) * (T1 - T0_DEFAULT)
        else:
            E1 = self.m1.rho * self.L1 * self.m1.enthalpy(T1)
        return E1 + self.C2 * (T2 - T0_DEFAULT)

    def equilibrium_temperature(self, q0):
        """Steady state under constant flux q0 (adiabatic back face): all heat re-radiated."""
        return (q0 / (self.eps * SIGMA) + self.T_env**4) ** 0.25

    def eigenvalues(self, t, T):
        return np.linalg.eigvals(self.jac(t, T))

    def initial_state(self, T0=T0_DEFAULT):
        return np.array([T0, T0], dtype=float)


def linear_exact(model, q0, t, T0=T0_DEFAULT):
    """Exact solution for constant properties, no radiation, constant flux q0.

    With theta = T1 - T2 and lam = G (1/C1 + 1/C2):
        C1 T1 + C2 T2 = (C1 + C2) T0 + q0 t
        theta(t)      = theta_inf (1 - exp(-lam t)),  theta_inf = q0 / (C1 lam)
    """
    assert model.T_frozen is not None and model.eps == 0.0
    C1, C2, G = model.C1(T0), model.C2, model.G(T0)
    lam = G * (1 / C1 + 1 / C2)
    theta = q0 / (C1 * lam) * (1 - np.exp(-lam * np.asarray(t)))
    Tbar = T0 + q0 * np.asarray(t) / (C1 + C2)
    T1 = Tbar + C2 / (C1 + C2) * theta
    T2 = Tbar - C1 / (C1 + C2) * theta
    return np.stack([T1, T2], axis=-1)


# ---------------------------------------------------------------------------
# Self-check:  python -m physics.two_layer   (run from repo root)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    pulse = HalfSinePulse(1.5e5, 400.0)
    model = TwoLayerShield(pulse)
    print(model.m1, model.m2, pulse)
    print(f"C2 = {model.C2:.0f} J/m^2-K,  C1(300 K) = {model.C1(300.0):.0f} J/m^2-K,  "
          f"G(300 K) = {model.G(300.0):.3f} W/m^2-K")

    # Jacobian vs finite differences at random states
    worst = 0.0
    for _ in range(20):
        T = np.array([rng.uniform(300, 1700), rng.uniform(300, 600)])
        t = rng.uniform(0, 400)
        J = model.jac(t, T)
        Jfd = np.zeros((2, 2))
        for j in range(2):
            d = 1e-5 * T[j]
            Tp, Tm = T.copy(), T.copy()
            Tp[j] += d
            Tm[j] -= d
            Jfd[:, j] = (model.f(t, Tp) - model.f(t, Tm)) / (2 * d)
        worst = max(worst, np.max(np.abs(J - Jfd) / np.max(np.abs(Jfd))))
    print(f"Jacobian vs finite differences: max rel err = {worst:.1e}")
    assert worst < 1e-5

    # Energy identity: d/dt E = q - radiation
    T = np.array([1200.0, 400.0])
    dE = np.array([model.m1.rho * model.L1 * model.m1.c(T[0]), model.C2]) @ model.f(100.0, T)
    print(f"Energy identity: dE/dt = {dE:.6e},  q - rad = {pulse(100.0) - model.radiation(T[0]):.6e}")
    assert np.isclose(dE, pulse(100.0) - model.radiation(T[0]), rtol=1e-12)
    print("Two-layer model checks passed.")
