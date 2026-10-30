"""
nonlinear_effects.py -- Research Question 3: how much do radiation and the
temperature-dependent tile properties change the heat that reaches the inner layer?

Each test case is run three ways (dt = 5 s, 0 <= t <= 3000 s):
  full model           radiation on, k1(T) and c1(T) from TPSX
  radiation off        eps = 0
  constant properties  k1 and c1 frozen at their room-temperature (293.15 K) values

Reported for each run: peak outer and inner temperatures, the fraction of the
delivered heat Q that is radiated away, and the heat stored in the inner layer
at its peak, C2 (T2max - T0), as a fraction of Q.

Run from the repo root:   python -m studies.nonlinear_effects
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.two_layer import TwoLayerShield, HalfSinePulse, T0_DEFAULT
from physics.materials import RCG_EMISSIVITY
from solver.tr_bdf2 import tr_bdf2

FIG = Path(__file__).resolve().parents[1] / "figures"
DT, T_END = 5.0, 3000.0

CASES = {
    "Baseline": HalfSinePulse(1.5e5, 400.0),
    "Same Q, 2x peak": HalfSinePulse(3.0e5, 200.0),
    "Same peak, 2x Q": HalfSinePulse(1.5e5, 800.0),
}
VARIANTS = {
    "full model": dict(),
    "radiation off": dict(emissivity=0.0),
    "constant properties": dict(constant_props_at=T0_DEFAULT),
}


def run(pulse, **kw):
    model = TwoLayerShield(pulse, **kw)
    t, Y, _ = tr_bdf2(model.f, model.jac, (0.0, T_END), model.initial_state(), DT)
    rad = np.trapezoid(model.radiation(Y[:, 0]), t)
    i2 = np.argmax(Y[:, 1])
    stored_inner = model.C2 * (Y[i2, 1] - T0_DEFAULT)
    return dict(t=t, Y=Y, T1max=Y[:, 0].max(), T2max=Y[i2, 1], t2=t[i2],
                rad_frac=rad / pulse.Q, inner_frac=stored_inner / pulse.Q)


if __name__ == "__main__":
    res = {c: {v: run(p, **kw) for v, kw in VARIANTS.items()} for c, p in CASES.items()}

    print(f"{'case':18s} {'variant':20s} {'T1max':>8s} {'T2max':>8s} {'t(T2max)':>9s} "
          f"{'radiated/Q':>11s} {'inner/Q':>8s}")
    for c in CASES:
        for v in VARIANTS:
            r = res[c][v]
            print(f"{c:18s} {v:20s} {r['T1max']:8.1f} {r['T2max']:8.1f} {r['t2']:9.0f} "
                  f"{r['rad_frac']:11.3f} {r['inner_frac']:8.3f}")
        print()

    # LaTeX table rows
    print("% LaTeX rows: case & variant & T1max & T2max & radiated/Q & inner/Q")
    for c in CASES:
        for v in VARIANTS:
            r = res[c][v]
            print(f"{c} & {v} & {r['T1max']:.0f} & {r['T2max']:.0f} & "
                  f"{100 * r['rad_frac']:.0f}\\% & {100 * r['inner_frac']:.0f}\\% \\\\")

    # Figure: peak inner-layer temperature rise, grouped by case
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(CASES))
    width = 0.38
    for i, v in enumerate(("full model", "constant properties")):
        rise = [res[c][v]["T2max"] - T0_DEFAULT for c in CASES]
        bars = ax[0].bar(x + (i - 0.5) * width, rise, width, label=v)
        ax[0].bar_label(bars, fmt="%.0f", fontsize=8)
    ax[0].set_xticks(x, list(CASES))
    ax[0].set(ylabel=r"peak inner-layer rise $T_{2,\max} - T_0$ [K]",
              title="Effect of temperature-dependent properties")
    ax[0].legend(fontsize=9)

    for i, v in enumerate(("full model", "radiation off")):
        rise = [res[c][v]["T2max"] - T0_DEFAULT for c in CASES]
        bars = ax[1].bar(x + (i - 0.5) * width, rise, width, label=v)
        ax[1].bar_label(bars, fmt="%.0f", fontsize=8)
    ax[1].set_xticks(x, list(CASES))
    ax[1].set_yscale("log")
    ax[1].set(ylabel=r"peak inner-layer rise $T_{2,\max} - T_0$ [K] (log scale)",
              title="Effect of radiation")
    ax[1].legend(fontsize=9, loc="upper left")
    ax[1].set_ylim(top=3e4)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "nonlinear_effects.png", dpi=150)
    print(f"Saved {FIG / 'nonlinear_effects.png'}")
