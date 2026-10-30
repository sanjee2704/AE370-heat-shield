# AE 370 Group Project 1: Two-Layer Heat Shield Thermal Model

How do total heat load and peak heat flux affect the temperatures of a heat
shield's outer and inner layers during reentry? We model the shield as two
lumped layers (LI-900 tile over an aluminum skin) with temperature-dependent
properties and surface re-radiation. That gives a coupled, nonlinear ODE initial
value problem, which we integrate with TR-BDF2 implemented from scratch.

## Setup

```
pip install -r requirements.txt
```

Run every script from the repository root with `python -m ...`.

## Structure

| Path | Contents |
|---|---|
| `solver/tr_bdf2.py` | TR-BDF2 time integrator (from scratch; Newton solve for each implicit stage) |
| `physics/materials.py` | LI-900 properties from NASA TPSX; aluminum 2024-T3 properties |
| `physics/two_layer.py` | Two-layer model f(t, T), exact Jacobian, heat-flux profiles |
| `studies/verification.py` | Checks against an exact solution, energy conservation, and radiative equilibrium |
| `studies/convergence.py` | Convergence rate and time-step selection |
| `studies/nonlinear_effects.py` | Research Question 3: runs with radiation off and with constant properties |
| `figures/` | All figures used in the report |

## Reproducing the figures

| Command | Output |
|---|---|
| `python -m physics.materials` | `figures/li900_properties.png` |
| `python -m solver.tr_bdf2` | Solver self-test (2nd order on a stiff problem; no figure) |
| `python -m physics.two_layer` | Model self-test (Jacobian, energy identity; no figure) |
| `python -m studies.verification` | `figures/verification.png` |
| `python -m studies.convergence` | `figures/convergence.png` |
| `python -m studies.nonlinear_effects` | `figures/nonlinear_effects.png` (and the RQ3 table values) |
