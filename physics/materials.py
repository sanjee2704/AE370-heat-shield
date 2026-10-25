"""
materials.py -- thermophysical properties for the TPS stack.

Every material object exposes (scalars or numpy arrays accepted):
    rho            density [kg/m^3] (constant)
    k(T)           thermal conductivity [W/m-K]
    c(T)           specific heat [J/kg-K]
    dk_dT(T)       derivative of k [W/m-K^2] (for the Jacobian)
    enthalpy(T)    h(T) = integral of c dT' from T_REF to T [J/kg]
                   (needed for the energy-conservation check when c varies with T)

Tabulated data are linearly interpolated. Outside the table range the end
value is held constant (np.interp behavior); use in_range(T) to check.

SOURCES
-------
LI-900 (NASA TPSX database, LI-900 Rigid Tile, https://tpsx.arc.nasa.gov/Material?id=1):
    density            1.44e2 kg/m^3                     (property=1)
    k, thru-thickness  table vs T at 5 pressures         (property=4)
                       TPSX ref: Space Shuttle Program Thermodynamic Design
                       Data Book, SD73-SH-0226, Rockwell, 1981. Tolerance +/-18%.
    specific heat      table vs T                        (property=9), same ref, +/-10%.
    multi-use limit    1.59e3 K                          (property=14)

Aluminum 2024-T3 (constant properties):
    rho = 2780 kg/m^3, k = 121 W/m-K   -- Gabrian 2024 alloy datasheet / MatWeb
    c   = 875 J/kg-K                   -- MatWeb 2024-T3  (VERIFY + cite before report)

Surface emissivity: RCG-coated tile. NASA Ames lists high emissivity (>0.9) as
the design goal for reusable TPS coatings. 0.85 is used as a conservative
nominal value -- an ASSUMPTION; treat it as a sensitivity parameter.
NOTE: TPSX's LI-900 emissivity table is for UNCOATED tile; do not use it here.
"""

import numpy as np

T_REF = 300.0              # reference temperature for enthalpy [K]
SIGMA = 5.670374419e-8     # Stefan-Boltzmann constant [W/m^2-K^4]
RCG_EMISSIVITY = 0.85      # assumption -- see docstring


class TabulatedMaterial:
    """Material with piecewise-linear k(T) and c(T) and constant density."""

    def __init__(self, name, rho, T_k, k_vals, T_c, c_vals, max_temp=None, source=""):
        self.name = name
        self.rho = float(rho)
        self.max_temp = max_temp
        self.source = source

        self._Tk = np.asarray(T_k, dtype=float)
        self._k = np.asarray(k_vals, dtype=float)
        self._Tc = np.asarray(T_c, dtype=float)
        self._c = np.asarray(c_vals, dtype=float)
        if np.any(np.diff(self._Tk) <= 0) or np.any(np.diff(self._Tc) <= 0):
            raise ValueError("temperature tables must be strictly increasing")

        # Enthalpy at the c-table nodes (exact for piecewise-linear c).
        seg = 0.5 * (self._c[1:] + self._c[:-1]) * np.diff(self._Tc)
        self._H_nodes = np.concatenate(([0.0], np.cumsum(seg)))
        self._H_ref = 0.0
        self._H_ref = self.enthalpy(T_REF)

    # --- properties ---------------------------------------------------------
    def k(self, T):
        return np.interp(T, self._Tk, self._k)

    def c(self, T):
        return np.interp(T, self._Tc, self._c)

    def dk_dT(self, T):
        T = np.asarray(T, dtype=float)
        if len(self._Tk) < 2:
            return np.zeros_like(T)
        slopes = np.diff(self._k) / np.diff(self._Tk)
        idx = np.clip(np.searchsorted(self._Tk, T, side="right") - 1, 0, len(slopes) - 1)
        out = slopes[idx]
        return np.where((T < self._Tk[0]) | (T > self._Tk[-1]), 0.0, out)

    def enthalpy(self, T):
        """h(T) - h(T_REF) in J/kg."""
        T = np.asarray(T, dtype=float)
        Tc, c, H = self._Tc, self._c, self._H_nodes
        idx = np.clip(np.searchsorted(Tc, T, side="right") - 1, 0, len(Tc) - 2)
        inside = H[idx] + 0.5 * (c[idx] + self.c(T)) * (T - Tc[idx])
        below = c[0] * (T - Tc[0])
        above = H[-1] + c[-1] * (T - Tc[-1])
        h = np.where(T < Tc[0], below, np.where(T > Tc[-1], above, inside))
        return h - self._H_ref

    def alpha(self, T):
        """Thermal diffusivity [m^2/s]."""
        return self.k(T) / (self.rho * self.c(T))

    def in_range(self, T):
        T = np.asarray(T, dtype=float)
        lo = max(self._Tk[0], self._Tc[0])
        hi = min(self._Tk[-1], self._Tc[-1])
        return bool(np.all((T >= lo) & (T <= hi)))

    def __repr__(self):
        return f"<{self.name}: rho={self.rho:g} kg/m^3>"


# ---------------------------------------------------------------------------
# LI-900 data (TPSX, transcribed exactly; SI units)
# ---------------------------------------------------------------------------
_LI900_T_K = np.array([116.667, 255.556, 394.444, 533.333, 672.222, 811.111, 950.0,
                       1088.89, 1227.78, 1366.67, 1533.33, 1644.44, 1811.11, 1922.22])

# thru-the-thickness conductivity [W/m-K], keyed by gas pressure [Pa]
_LI900_K_BY_P = {
    10.133:   [8.65e-3, 1.30e-2, 1.59e-2, 2.16e-2, 3.03e-2, 4.03e-2, 5.33e-2,
               7.20e-2, 9.81e-2, 1.27e-1, 1.67e-1, 2.01e-1, 2.67e-1, 3.29e-1],
    101.33:   [1.30e-2, 1.73e-2, 2.16e-2, 2.89e-2, 3.74e-2, 4.76e-2, 6.06e-2,
               7.95e-2, 1.06e-1, 1.35e-1, 1.77e-1, 2.13e-1, 2.80e-1, 3.39e-1],
    1013.3:   [2.60e-2, 3.17e-2, 3.89e-2, 4.78e-2, 5.63e-2, 6.79e-2, 8.52e-2,
               1.07e-1, 1.33e-1, 1.63e-1, 2.01e-1, 2.41e-1, 3.12e-1, 3.79e-1],
    10133.0:  [3.74e-2, 4.33e-2, 5.47e-2, 6.92e-2, 8.52e-2, 1.04e-1, 1.25e-1,
               1.51e-1, 1.83e-1, 2.20e-1, 2.68e-1, 3.10e-1, 3.84e-1, 4.54e-1],
    101330.0: [4.03e-2, 4.76e-2, 5.90e-2, 7.50e-2, 9.24e-2, 1.14e-1, 1.35e-1,
               1.63e-1, 1.96e-1, 2.35e-1, 2.89e-1, 3.36e-1, 4.19e-1, 5.02e-1],
}

_LI900_T_C = np.array([116.483, 172.039, 255.372, 394.261, 533.15, 672.039, 810.928,
                       949.817, 1088.71, 1199.82, 1227.59, 1366.48, 1533.15, 1922.04])
_LI900_C = np.array([293.0, 440.0, 628.0, 879.0, 1060.0, 1150.0, 1210.0,
                     1240.0, 1260.0, 1260.0, 1270.0, 1270.0, 1270.0, 1270.0])


def li900_k_at_pressure(p_pa):
    """k(T) table at gas pressure p, interpolated linearly in log10(p) (clamped)."""
    pressures = np.array(sorted(_LI900_K_BY_P))
    table = np.array([_LI900_K_BY_P[p] for p in pressures])   # (n_p, n_T)
    logp = np.log10(np.clip(p_pa, pressures[0], pressures[-1]))
    return np.array([np.interp(logp, np.log10(pressures), table[:, j])
                     for j in range(table.shape[1])])


def li900(pressure_pa=1013.3):
    """LI-900 tile at a fixed internal gas pressure.

    Default 1013.3 Pa (0.01 atm): an assumed representative pressure during
    the high-heating phase. State it in the report; vary it as a sensitivity.
    """
    return TabulatedMaterial(
        name=f"LI-900 (p={pressure_pa:g} Pa)",
        rho=144.0,
        T_k=_LI900_T_K, k_vals=li900_k_at_pressure(pressure_pa),
        T_c=_LI900_T_C, c_vals=_LI900_C,
        max_temp=1590.0,
        source="NASA TPSX, LI-900 Rigid Tile",
    )


def aluminum_2024():
    """Aluminum 2024-T3, constant properties (small T range at the bondline)."""
    return TabulatedMaterial(
        name="Al 2024-T3",
        rho=2780.0,
        T_k=[100.0, 800.0], k_vals=[121.0, 121.0],
        T_c=[100.0, 800.0], c_vals=[875.0, 875.0],
        source="Gabrian 2024 datasheet / MatWeb",
    )


# ---------------------------------------------------------------------------
# Self-check:  python -m physics.materials   (run from repo root)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tile, al = li900(), aluminum_2024()
    print(tile, al)

    # 1) interpolation reproduces table values exactly
    assert np.allclose(li900(1013.3).k(_LI900_T_K), _LI900_K_BY_P[1013.3])
    assert np.allclose(tile.c(_LI900_T_C), _LI900_C)

    # 2) d(enthalpy)/dT == c  (central difference)
    T = np.linspace(200.0, 1800.0, 401)
    dT = 1e-3
    dhdT = (tile.enthalpy(T + dT) - tile.enthalpy(T - dT)) / (2 * dT)
    assert np.allclose(dhdT, tile.c(T), rtol=1e-6), "enthalpy inconsistent with c(T)"
    assert abs(tile.enthalpy(T_REF)) < 1e-9

    # 3) dk/dT matches finite difference away from table nodes
    Tm = 0.5 * (_LI900_T_K[1:] + _LI900_T_K[:-1])
    fd = (tile.k(Tm + dT) - tile.k(Tm - dT)) / (2 * dT)
    assert np.allclose(tile.dk_dT(Tm), fd, rtol=1e-6)

    # 4) stiffness contrast between layers
    print(f"alpha tile @ 300 K  = {tile.alpha(300.0):.2e} m^2/s")
    print(f"alpha tile @ 1500 K = {tile.alpha(1500.0):.2e} m^2/s")
    print(f"alpha Al            = {al.alpha(300.0):.2e} m^2/s")
    print("All material checks passed.")

    # figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    Tp = np.linspace(_LI900_T_K[0], _LI900_T_K[-1], 300)
    for p in sorted(_LI900_K_BY_P):
        a1.plot(Tp, li900(p).k(Tp), label=f"{p:g} Pa")
        a1.plot(_LI900_T_K, _LI900_K_BY_P[p], "k.", ms=3)
    a1.set(xlabel="T [K]", ylabel="k [W/m-K]", title="LI-900 conductivity (TPSX)")
    a1.legend(fontsize=8)
    Tp = np.linspace(_LI900_T_C[0], _LI900_T_C[-1], 300)
    a2.plot(Tp, tile.c(Tp))
    a2.plot(_LI900_T_C, _LI900_C, "k.", ms=3)
    a2.set(xlabel="T [K]", ylabel="c [J/kg-K]", title="LI-900 specific heat (TPSX)")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "li900_properties.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
