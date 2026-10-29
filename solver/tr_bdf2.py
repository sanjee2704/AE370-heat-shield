"""
tr_bdf2.py -- the TR-BDF2 time integrator, implemented from scratch.

Solves  y' = f(t, y),  y(t0) = y0  with a fixed step h.

One step from t_n to t_{n+1} = t_n + h  (gamma = 2 - sqrt(2), d = gamma/2, w = sqrt(2)/4):

  Stage 1 (trapezoidal rule over [t_n, t_n + gamma h]):
      Y_g - d h f(t_n + gamma h, Y_g) = y_n + d h f(t_n, y_n)

  Stage 2 (BDF2 using y_n, Y_g, y_{n+1}):
      y_{n+1} - d h f(t_{n+1}, y_{n+1}) = y_n + w h [ f(t_n, y_n) + f(t_n + gamma h, Y_g) ]

Both stages have the form  Y - d h f(t*, Y) = rhs, solved by Newton's method:
      (I - d h J) dY = -(Y - d h f(t*, Y) - rhs),   J = df/dY.

Properties: 2nd order, A-stable and L-stable. Stability function
      R(z) = (1 + (sqrt(2) - 1) z) / (1 - d z)^2,   R(z) -> 0 as z -> -inf.
"""

import numpy as np

GAMMA = 2.0 - np.sqrt(2.0)
D = GAMMA / 2.0
W = np.sqrt(2.0) / 4.0


def stability_function(z):
    """R(z) for the test equation y' = lambda y, z = h lambda."""
    z = np.asarray(z, dtype=complex)
    return (1.0 + (np.sqrt(2.0) - 1.0) * z) / (1.0 - D * z) ** 2


def _newton(f, jac, t, rhs, Y0, h, tol, max_iter):
    """Solve  G(Y) = Y - D h f(t, Y) - rhs = 0  by Newton's method."""
    Y = Y0.copy()
    n = len(Y)
    for it in range(1, max_iter + 1):
        G = Y - D * h * f(t, Y) - rhs
        A = np.eye(n) - D * h * jac(t, Y)
        dY = np.linalg.solve(A, -G)
        Y = Y + dY
        if np.linalg.norm(dY, np.inf) <= tol * (1.0 + np.linalg.norm(Y, np.inf)):
            return Y, it
    raise RuntimeError(f"Newton did not converge at t = {t:.4g} after {max_iter} iterations")


def tr_bdf2_step(f, jac, t, y, h, tol=1e-12, max_iter=25):
    """Advance one TR-BDF2 step. Returns (y_next, newton_iterations)."""
    fn = f(t, y)

    # Stage 1: trapezoidal rule to t + gamma h (initial guess: explicit Euler)
    tg = t + GAMMA * h
    Yg, it1 = _newton(f, jac, tg, y + D * h * fn, y + GAMMA * h * fn, h, tol, max_iter)
    fg = f(tg, Yg)

    # Stage 2: BDF2 to t + h (initial guess: linear extrapolation through y, Yg)
    t1 = t + h
    guess = y + (Yg - y) / GAMMA
    y1, it2 = _newton(f, jac, t1, y + W * h * (fn + fg), guess, h, tol, max_iter)
    return y1, it1 + it2


def tr_bdf2(f, jac, t_span, y0, h, tol=1e-12, max_iter=25):
    """Integrate y' = f(t, y) from t_span[0] to t_span[1] with fixed step h.

    The last step is shortened if needed to land exactly on t_span[1].
    Returns (t, Y, info) with Y of shape (n_steps + 1, len(y0)).
    """
    t0, tf = map(float, t_span)
    n = int(np.ceil((tf - t0) / h - 1e-9))
    t = t0
    y = np.array(y0, dtype=float)
    ts, ys = [t], [y.copy()]
    newton_total = 0
    for k in range(n):
        hk = min(h, tf - t)
        y, its = tr_bdf2_step(f, jac, t, y, hk, tol, max_iter)
        t = t0 + (k + 1) * h if k < n - 1 else tf
        newton_total += its
        ts.append(t)
        ys.append(y.copy())
    info = {"steps": n, "newton_iterations": newton_total,
            "f_evals_approx": 2 * n + newton_total}
    return np.array(ts), np.array(ys), info


# ---------------------------------------------------------------------------
# Self-check:  python -m solver.tr_bdf2   (run from repo root)
# Stiff test problem with exact solution y = cos t:
#     y' = -lam (y - cos t) - sin t,   y(0) = 1
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    lam = 1.0e4
    f = lambda t, y: -lam * (y - np.cos(t)) - np.sin(t)
    jac = lambda t, y: np.array([[-lam]])
    hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
    errs = []
    for h in hs:
        t, Y, _ = tr_bdf2(f, jac, (0.0, 2.0), [1.0], h)
        errs.append(np.max(np.abs(Y[:, 0] - np.cos(t))))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print(f"Stiff scalar test (lambda = {lam:g}), h*lambda up to {hs[0] * lam:g}:")
    for h, e in zip(hs, errs):
        print(f"   h = {h:7.4f}   max error = {e:.3e}")
    print(f"   observed order: {np.round(orders, 2)}")
    assert np.all(orders > 1.8), "TR-BDF2 is not 2nd order"
    assert abs(stability_function(-1e8)) < 1e-6, "not L-stable"
    print("TR-BDF2 self-check passed.")
