"""Interpolant schedule: the scalars that define the flow.

Pure host-side math on Python floats -- no tensors, no device. Every other module
takes ``c_t`` and ``k_t`` from here, so the single-Gaussian reference path and the
GMM path cannot drift apart.

For the linear interpolant ``x_t = (1-t) x_0 + t y`` with ``x_0 ~ N(0,I)`` and a
component ``y | i ~ N(mu_i, sigma^2 I)``, the time-t component marginal is
``N(t mu_i, c_t^2 I)`` and the component velocity is ``mu_i + k_t (x - t mu_i)``.
"""

from __future__ import annotations

import math

__all__ = ["c_t_sq", "c_t", "k_t", "phi_exact", "w_exact", "t_peak"]


def _check(t, sigma):
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must lie in [0, 1], got {t}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")


def c_t_sq(t, sigma):
    """Per-component variance at time t:  (1-t)^2 + t^2 sigma^2   (Eq. 9)."""
    _check(t, sigma)
    return (1.0 - t) ** 2 + t * t * sigma * sigma


def c_t(t, sigma):
    """Per-component scale, sqrt of :func:`c_t_sq`. Strictly positive for sigma>0."""
    return math.sqrt(c_t_sq(t, sigma))


def k_t(t, sigma):
    """Velocity coefficient (t sigma^2 - (1-t)) / c_t^2   (Eq. 16).

    Equals ``0.5 * d/dt log c_t^2``, which is what makes the single-Gaussian
    amplification integrate to a closed form. Negative before the minimum-scale
    point (components contract), positive after (they expand).
    """
    _check(t, sigma)
    return (t * sigma * sigma - (1.0 - t)) / c_t_sq(t, sigma)


def phi_exact(t, sigma):
    """Amplification sigma / c_t for a single Gaussian target N(0, sigma^2 I)."""
    return sigma / c_t(t, sigma)


def w_exact(t, sigma):
    """Closed-form w(t) = sigma^2 / c_t^2 for a single Gaussian   (spec Eq. 32)."""
    return sigma * sigma / c_t_sq(t, sigma)


def t_peak(sigma):
    """Where c_t is smallest and w(t) peaks: 1 / (1 + sigma^2)   (spec Eq. 33)."""
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return 1.0 / (1.0 + sigma * sigma)
