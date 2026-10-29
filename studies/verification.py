"""
verification.py -- does the code converge to the solution of the CORRECT equations?
(Rubric 3(i).)  Three checks against results known independently of the code:

  [1] Linear limit, exact solution: constant properties, no radiation, constant
      flux. The 2x2 linear system has a closed-form solution (two_layer.linear_exact).
  [2] Conservation of energy: with radiation off and an adiabatic back face, all
      heat delivered by a half-sine pulse must be stored:
          E(t_end) - E(0) = Q = 2 q_peak tau / pi   (exact),
      with the full nonlinear k(T) and c(T).
  [3] Radiative equilibrium: under constant flux q0 with radiation on, both
      layers must approach T_eq with eps*sigma*(T_eq^4 - T_env^4) = q0.

Run from the repo root:   python -m studies.verification
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.two_layer import (TwoLayerShield, HalfSinePulse, ConstantFlux,
                               linear_exact)
from solver.tr_bdf2 import tr_bdf2

FIG = Path(__file__).resolve().parents[1] / "figures"


def check_linear_exact():
    q0, t_end = 3.0e3, 3000.0
    model = TwoLayerShield(ConstantFlux(q0), emissivity=0.0, constant_props_at=600.0)
    dts = [40.0, 20.0, 10.0, 5.0, 2.5]
    errs = []
    for dt in dts:
        t, Y, _ = tr_bdf2(model.f, model.jac, (0.0, t_end), model.initial_state(), dt)
        errs.append(np.max(np.abs(Y - linear_exact(model, q0, t))))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print("[1] Linear limit vs exact solution (constant k, c; no radiation; q0 = 0.3 W/cm^2):")
    for dt, e in zip(dts, errs):
        print(f"      dt = {dt:5.1f} s   max |T - T_exact| = {e:.3e} K")
    print(f"      observed order: {np.round(orders, 2)}")
    assert errs[-1] < 1e-3 and np.all(orders > 1.9)
    t, Y, _ = tr_bdf2(model.f, model.jac, (0.0, t_end), model.initial_state(), 40.0)
    return model, q0, t, Y, dts, errs


def check_energy():
    pulse = HalfSinePulse(3.0e4, 400.0)
    model = TwoLayerShield(pulse, emissivity=0.0)
    t_end = 1000.0
    dts = [20.0, 10.0, 5.0, 2.5, 1.25]
    errs = []
    for dt in dts:
        t, Y, _ = tr_bdf2(model.f, model.jac, (0.0, t_end), model.initial_state(), dt)
        stored = model.energy(Y[-1]) - model.energy(Y[0])
        errs.append(abs(stored - pulse.Q) / pulse.Q)
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print(f"[2] Energy conservation (radiation off, nonlinear k(T), c(T)), Q = {pulse.Q:.6e} J/m^2:")
    for dt, e in zip(dts, errs):
        print(f"      dt = {dt:5.2f} s   |E_stored - Q| / Q = {e:.3e}")
    print(f"      observed order: {np.round(orders, 2)}")
    assert errs[-1] < 1e-5 and orders[-1] > 1.8
    return dts, errs


def check_equilibrium():
    q0 = 2.5e3
    model = TwoLayerShield(ConstantFlux(q0))
    T_eq = model.equilibrium_temperature(q0)
    t, Y, _ = tr_bdf2(model.f, model.jac, (0.0, 1.5e5), model.initial_state(), 50.0)
    err = np.max(np.abs(Y[-1] - T_eq))
    print(f"[3] Radiative equilibrium (q0 = {q0:g} W/m^2): exact T_eq = {T_eq:.4f} K, "
          f"computed T1 = {Y[-1, 0]:.4f} K, T2 = {Y[-1, 1]:.4f} K, max err = {err:.1e} K")
    assert err < 1e-3
    return model, q0, T_eq, t, Y


if __name__ == "__main__":
    lin_model, q0, tl, Yl, dts_l, errs_l = check_linear_exact()
    dts_e, errs_e = check_energy()
    eq_model, q_eq, T_eq, te, Ye = check_equilibrium()
    print("All verification checks passed.")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    tt = np.linspace(0, tl[-1], 400)
    ex = linear_exact(lin_model, q0, tt)
    ax[0].plot(tt, ex[:, 0], "k-", lw=1, label="exact")
    ax[0].plot(tt, ex[:, 1], "k-", lw=1)
    ax[0].plot(tl, Yl[:, 0], "o", ms=4, label=r"TR-BDF2 $T_1$ (outer), $\Delta t$ = 40 s")
    ax[0].plot(tl, Yl[:, 1], "s", ms=4, label=r"TR-BDF2 $T_2$ (inner), $\Delta t$ = 40 s")
    ax[0].set(xlabel="t [s]", ylabel="T [K]", title="[1] Linear limit vs exact solution")
    ax[0].legend(fontsize=8)

    for d, e, lab in ((dts_l, errs_l, "[1] linear exact: max |T - T$_{exact}$| [K]"),
                      (dts_e, errs_e, "[2] energy: |E$_{stored}$ - Q| / Q")):
        d = np.array(d)
        ax[1].loglog(d, e, "o-", label=lab)
        ax[1].loglog(d, e[-1] * (d / d[-1])**2, "k--", lw=0.8)
    ax[1].plot([], [], "k--", lw=0.8, label="slope 2")
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set(xlabel=r"$\Delta t$ [s]", ylabel="error", title="Second-order convergence")
    ax[1].legend(fontsize=8)

    ax[2].plot(te / 3600, Ye[:, 0], label=r"$T_1$ (outer)")
    ax[2].plot(te / 3600, Ye[:, 1], label=r"$T_2$ (inner)")
    ax[2].axhline(T_eq, color="k", ls="--", lw=1, label=r"exact $T_{eq}$")
    ax[2].set_xlim(0, 10)
    ax[2].set(xlabel="t [h]", ylabel="T [K]",
              title=f"[3] Radiative equilibrium, q0 = {q_eq / 1e3:g} kW/m$^2$")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "verification.png", dpi=150)
    print(f"Saved {FIG / 'verification.png'}")
