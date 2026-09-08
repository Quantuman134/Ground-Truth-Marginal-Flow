"""Fixed-step ODE integration for the marginal flow.

Generic over the field: everything here takes a `(x, t) -> v` callable and knows
nothing else about it. That is what lets one integrator serve both the
single-Gaussian sanity check and the real mixture.
"""

import math


def time_grid(t0, t1, h):
    """Split [t0, t1] into uniform steps no longer than h.

    Returns (steps, actual_h). Since (t1 - t0)/h rarely divides evenly, the count
    is rounded UP and the step shrunk to fit:

        steps    = ceil((t1 - t0) / h)
        actual_h = (t1 - t0) / steps        <= h

    so every step within one trajectory is the same length and the last one lands
    exactly on t1. The alternative -- keep h exact and take a short final step --
    would make the accuracy of a trajectory depend on where its remainder fell.

    Integration here only ever runs forward, from a query time towards t = 1, so a
    reversed interval is a mistake rather than a feature.
    """
    if h <= 0.0:
        raise ValueError(f"step size must be positive, got {h}")
    if t1 < t0:
        raise ValueError(f"interval runs backwards: t0={t0}, t1={t1}")

    span = t1 - t0
    if span == 0.0:
        return 0, 0.0

    steps = math.ceil(span / h)
    return steps, span / steps


def euler(field, x, t0, t1, h):
    """Integrate dx/ds = field(x, s) from t0 to t1 with the forward Euler method.

        x <- x + dt * field(x, t)

    One field evaluation per step. Not for production -- it is here as the
    first-order yardstick RK4 is measured against, since "order 4" only means
    something next to an order-1 method on the same problem.

    The whole batch advances on one time grid: `t` is a plain Python float and the
    loop is over time, never over rows. Spec 8.5 requires the perturbed and
    unperturbed trajectories to see identical arithmetic, and that is only true if
    no per-row decision exists anywhere.
    """
    steps, dt = time_grid(t0, t1, h)
    for i in range(steps):
        # t0 + i*dt, not a running t += dt: repeated addition drifts, and the
        # field is only defined on [0, 1].
        x = x + dt * field(x, t0 + i * dt)
    return x


def rk4(field, x, t0, t1, h):
    """Integrate dx/ds = field(x, s) from t0 to t1 with classical Runge-Kutta 4.

    Four field evaluations per step: the slope at the start, twice at the
    midpoint, once at the far end, combined as

        x <- x + dt/6 * (k1 + 2 k2 + 2 k3 + k4)

    Four times the cost of an Euler step and far more than four times the
    accuracy: the error falls as dt^4 rather than dt, so on the stiff sigma=0.01
    case Euler needs thousands of steps to reach an accuracy RK4 gets in a
    handful. That measurement is what chose this method.

    Like euler, the whole batch advances on one time grid and the loop is over
    time, never over rows -- spec 8.5.
    """
    steps, dt = time_grid(t0, t1, h)
    half = dt / 2.0
    for i in range(steps):
        # Stage times come from the index, and the final one is pinned to t1.
        # A running t += dt drifts by ~1e-15, and RK4's last stage evaluates at
        # the END of the step -- so the drift pushes it just past t1, outside
        # where the field is defined. Euler never sees this: it only ever
        # evaluates at the start of a step.
        t = t0 + i * dt
        t_end = t1 if i == steps - 1 else t0 + (i + 1) * dt
        t_mid = t + 0.5 * (t_end - t)

        k1 = field(x, t)
        k2 = field(x + half * k1, t_mid)
        k3 = field(x + half * k2, t_mid)
        k4 = field(x + dt * k3, t_end)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x
