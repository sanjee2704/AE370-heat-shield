"""
convergence.py -- convergence rate on the full nonlinear problem and the choice
of time step for the results (rubric 3(i) rate and 3(ii)).

For each test-case heat pulse, the solution is computed with a sequence of time
steps and compared with a reference solution at a very small step. The error is
the maximum over the run of |T - T_ref| at common output times (every 20 s).
All steps divide the pulse duration tau, so the pulse's corners at t = 0 and
t = tau fall on step boundaries.

Run from the repo root:   python -m studies.convergence
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.two_layer import TwoLayerShield, HalfSinePulse
from solver.tr_bdf2 import tr_bdf2

FIG = Path(__file__).resolve().parents[1] / "figures"
T_END = 3000.0
SAMPLE = 20.0                               # common output spacing [s]
DTS = [20.0, 10.0, 5.0, 2.5, 1.25, 0.625]
DT_REF = 0.078125
TOL = {"T1": 1.0, "T2": 0.1}                # accuracy targets [K]

CASES = {
    "baseline (15 W/cm$^2$, 400 s)": HalfSinePulse(1.5e5, 400.0),
    "2x peak, same Q (30 W/cm$^2$, 200 s)": HalfSinePulse(3.0e5, 200.0),
    "same peak, 2x Q (15 W/cm$^2$, 800 s)": HalfSinePulse(1.5e5, 800.0),
}


def sampled(dt, t, Y):
    stride = int(round(SAMPLE / dt))
    return Y[::stride]


if __name__ == "__main__":
    results = {}
    print(f"Accuracy targets: |T1 err| < {TOL['T1']} K, |T2 err| < {TOL['T2']} K "
          f"(max over 0-{T_END:g} s)\n")
    for name, pulse in CASES.items():
        model = TwoLayerShield(pulse)
        y0 = model.initial_state()
        tr, Yr, _ = tr_bdf2(model.f, model.jac, (0.0, T_END), y0, DT_REF)
        Yr_s = sampled(DT_REF, tr, Yr)
        lam_max = max(np.max(np.abs(model.eigenvalues(t, y).real))
                      for t, y in zip(tr[::200], Yr[::200]))
        e1, e2, steps = [], [], []
        for dt in DTS:
            t, Y, info = tr_bdf2(model.f, model.jac, (0.0, T_END), y0, dt)
            err = np.max(np.abs(sampled(dt, t, Y) - Yr_s), axis=0)
            e1.append(err[0])
            e2.append(err[1])
            steps.append(info["steps"])
        o1 = np.log2(np.array(e1[:-1]) / np.array(e1[1:]))
        o2 = np.log2(np.array(e2[:-1]) / np.array(e2[1:]))
        ok = [dt for dt, a, b in zip(DTS, e1, e2) if a < TOL["T1"] and b < TOL["T2"]]
        dt_ok = max(ok) if ok else None
        results[name] = (e1, e2, dt_ok, lam_max)
        label = name.replace("$^2$", "^2")
        print(f"{label}:  T1max = {Yr[:, 0].max():.1f} K, T2max = {Yr[:, 1].max():.2f} K")
        print("      dt [s]   steps    T1 err [K]    T2 err [K]")
        for dt, n, a, b in zip(DTS, steps, e1, e2):
            print(f"    {dt:7.3f}  {n:6d}    {a:.3e}     {b:.3e}")
        print(f"      observed order T1: {np.round(o1, 2)}")
        print(f"      observed order T2: {np.round(o2, 2)}")
        print(f"      largest dt meeting targets: {dt_ok} s")
        print(f"      max |Re(lambda)| = {lam_max:.3g} 1/s  ->  explicit RK4 stability "
              f"limit ~ {2.785 / lam_max:.0f} s")
        # least-squares slope over all steps (the tabulated k(T), c(T) have kinks,
        # so individual ratios scatter slightly around 2)
        p1 = np.polyfit(np.log(DTS), np.log(e1), 1)[0]
        p2 = np.polyfit(np.log(DTS), np.log(e2), 1)[0]
        print(f"      fitted order: T1 {p1:.2f}, T2 {p2:.2f}\n")
        assert p1 > 1.9 and p2 > 1.9, "not converging at 2nd order"

    dt_choice = min(r[2] for r in results.values())
    print(f"Time step used for all results: dt = {dt_choice} s")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    d = np.array(DTS)
    for i, (comp, tol) in enumerate((("T1", TOL["T1"]), ("T2", TOL["T2"]))):
        for name, (e1, e2, _, _) in results.items():
            ax[i].loglog(d, e1 if comp == "T1" else e2, "o-", ms=4, label=name)
        ref = list(results.values())[0][i]
        ax[i].loglog(d, ref[-1] * (d / d[-1])**2, "k--", lw=0.8, label="slope 2")
        ax[i].axhline(tol, color="r", ls=":", lw=1, label=f"target {tol:g} K")
        ax[i].axvline(dt_choice, color="gray", ls="-", lw=0.8)
        ax[i].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        layer = "outer layer $T_1$" if comp == "T1" else "inner layer $T_2$"
        ax[i].set(xlabel=r"$\Delta t$ [s]", ylabel=r"max $|T - T_{ref}|$ [K]",
                  title=f"Convergence, {layer}")
        ax[i].legend(fontsize=7)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "convergence.png", dpi=150)
    print(f"Saved {FIG / 'convergence.png'}")
