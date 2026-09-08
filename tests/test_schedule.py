"""Tests for gtmf/schedule.py.

The oracles here are analytic: a numerical derivative, exact endpoint values, and
the location of an extremum found by brute-force scan.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtmf import schedule as sc  # noqa: E402

SIGMAS = [0.01, 0.1, 0.3, 0.6, 1.0]


@pytest.mark.parametrize("sigma", SIGMAS)
def test_k_t_is_half_dlog_c2_dt(sigma):
    """k_t == 0.5 d/dt log c_t^2.

    This identity is *why* the single-Gaussian amplification has a closed form:
    integrating k gives log(c_1/c_t). Checked against a central difference, which
    knows nothing about the k_t formula.
    """
    h = 1e-6
    for t in [0.05, 0.2, 0.5, 0.8, 0.95]:
        numeric = 0.5 * (math.log(sc.c_t_sq(t + h, sigma))
                         - math.log(sc.c_t_sq(t - h, sigma))) / (2 * h)
        assert sc.k_t(t, sigma) == pytest.approx(numeric, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("sigma", SIGMAS)
def test_endpoint_values(sigma):
    """c_0 = 1 (pure noise) and c_1 = sigma (pure data)."""
    assert sc.c_t_sq(0.0, sigma) == pytest.approx(1.0)
    assert sc.c_t(1.0, sigma) == pytest.approx(sigma)
    assert sc.k_t(0.0, sigma) == pytest.approx(-1.0)
    assert sc.k_t(1.0, sigma) == pytest.approx(1.0)
    assert sc.w_exact(1.0, sigma) == pytest.approx(1.0)


@pytest.mark.parametrize("sigma", SIGMAS)
def test_w_peaks_where_t_peak_says(sigma):
    """Brute-force argmax of w over a fine grid must land on 1/(1+sigma^2)."""
    grid = [i / 200_000 for i in range(200_001)]
    best = max(grid, key=lambda t: sc.w_exact(t, sigma))
    assert best == pytest.approx(sc.t_peak(sigma), abs=1e-4)


@pytest.mark.parametrize("sigma", SIGMAS)
def test_k_t_changes_sign_at_the_peak(sigma):
    """Components contract before the minimum-scale point and expand after."""
    tp = sc.t_peak(sigma)
    assert sc.k_t(max(tp - 1e-3, 0.0), sigma) < 0.0
    assert sc.k_t(min(tp + 1e-3, 1.0), sigma) > 0.0
    assert abs(sc.k_t(tp, sigma)) < 1e-9


@pytest.mark.parametrize("sigma", SIGMAS)
def test_phi_squared_is_w(sigma):
    for t in [0.0, 0.3, 0.7, 0.98, 1.0]:
        assert sc.phi_exact(t, sigma) ** 2 == pytest.approx(sc.w_exact(t, sigma))


@pytest.mark.parametrize("t,sigma", [(-0.1, 0.1), (1.1, 0.1), (0.5, 0.0), (0.5, -1.0)])
def test_invalid_arguments_raise(t, sigma):
    with pytest.raises(ValueError):
        sc.k_t(t, sigma)
