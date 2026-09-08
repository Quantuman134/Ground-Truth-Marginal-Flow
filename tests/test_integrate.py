"""Tests for gtmf/integrate.py."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtmf.integrate import time_grid  # noqa: E402

# the grid the experiment actually uses: 50 points on [0, 0.98], each run to t=1
REAL_TIMES = [i * 0.98 / 49 for i in range(50)]
REAL_STEPS = [1 / 512, 1 / 64]          # the two tiers from CLAUDE.md


@pytest.mark.parametrize("t0", REAL_TIMES)
@pytest.mark.parametrize("h", REAL_STEPS)
def test_steps_land_exactly_on_t1(t0, h):
    """steps * actual_h must reproduce the interval to floating-point precision;
    a trajectory that stops short or overshoots would be integrating a different
    problem than the one asked for."""
    steps, actual_h = time_grid(t0, 1.0, h)
    assert steps * actual_h == pytest.approx(1.0 - t0, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("t0", REAL_TIMES)
@pytest.mark.parametrize("h", REAL_STEPS)
def test_actual_step_never_exceeds_the_requested_one(t0, h):
    """Rounding the count up can only shrink the step. If actual_h > h the
    accuracy would be worse than the config asked for."""
    _, actual_h = time_grid(t0, 1.0, h)
    assert actual_h <= h


@pytest.mark.parametrize("t0", REAL_TIMES)
@pytest.mark.parametrize("h", REAL_STEPS)
def test_step_count_is_the_smallest_that_fits(t0, h):
    """One fewer step would need a step longer than h."""
    steps, _ = time_grid(t0, 1.0, h)
    assert steps >= 1
    if steps > 1:
        assert (1.0 - t0) / (steps - 1) > h


@pytest.mark.parametrize("span,h,expected", [
    (1.0, 1 / 512, 512), (1.0, 1 / 64, 64), (0.5, 1 / 512, 256),
    (1.0, 0.1, 10), (0.3, 0.1, 3), (0.7, 0.1, 7), (0.98, 0.01, 98),
])
def test_evenly_dividing_intervals_do_not_gain_a_spurious_step(span, h, expected):
    """ceil() on a float quotient could overshoot when span/h is an exact integer
    represented as 7.000000000000001. Measured over the real grid it never does --
    the error always lands below -- but that is luck, not a guarantee, so it is
    pinned here. A failure means one wasted step, not a wrong answer.
    """
    steps, _ = time_grid(0.0, span, h)
    assert steps == expected


def test_a_step_larger_than_the_interval_gives_one_step():
    steps, actual_h = time_grid(0.9, 1.0, h=10.0)
    assert steps == 1
    assert actual_h == pytest.approx(0.1)


def test_zero_length_interval_needs_no_steps():
    assert time_grid(1.0, 1.0, h=0.1) == (0, 0.0)


def test_steps_are_a_whole_number():
    """Used as a loop bound, so an accidental float would be a TypeError later."""
    steps, _ = time_grid(0.0, 0.98, 1 / 64)
    assert isinstance(steps, int)


@pytest.mark.parametrize("t0,t1,h", [
    (0.0, 1.0, 0.0), (0.0, 1.0, -0.1),      # non-positive step
    (1.0, 0.0, 0.1), (0.5, 0.4, 0.1),       # backwards interval
])
def test_invalid_arguments_raise(t0, t1, h):
    with pytest.raises(ValueError):
        time_grid(t0, t1, h)


# --------------------------------------------------------------------------- #
# integrators
# --------------------------------------------------------------------------- #

import torch                                                        # noqa: E402
from gtmf import schedule                                           # noqa: E402
from gtmf.integrate import euler, rk4                               # noqa: E402

# Every integrator must pass the shared checks. Append RK4 here in step 3.
INTEGRATORS = [euler, rk4]
EXPECTED_ORDER = {euler: 1, rk4: 4}

# Accuracy reachable at h = 1/4096 on the STIFFEST case (sigma = 0.01, where
# max|k| = 50 on [0.9, 1]). These differ by orders of magnitude between methods,
# and that gap is the whole reason the project uses RK4 -- see
# test_stiffness_is_what_makes_euler_expensive below.
FINE_TOLERANCE = {euler: 2e-2, rk4: 1e-8}


def linear_field(sigma):
    """The single-Gaussian field v(x, t) = k_t * x.

    Chosen because dx/ds = k_s x is linear and k_t = 0.5 d/ds log c_s^2, so it
    integrates exactly: x(t1) = (c_t1 / c_t0) x(t0). That closed form is the
    oracle -- no other integrator is involved.
    """
    return lambda x, t: schedule.k_t(t, sigma) * x


def exact_flow(x0, sigma, t0, t1):
    return x0 * (schedule.c_t(t1, sigma) / schedule.c_t(t0, sigma))


def observed_order(integrator, sigma, t0, t1, hs):
    """Slope of log(error) against log(h) -- the method's convergence order.

    Fitted rather than assumed: a mis-weighted RK4 still converges, just at a
    lower order, and the slope is the only thing that shows it.
    """
    x0 = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    exact = exact_flow(x0, sigma, t0, t1)
    errs, logs = [], []
    for h in hs:
        got = integrator(linear_field(sigma), x0, t0, t1, h)
        errs.append(float((got - exact).abs().max() / exact.abs().max()))
        logs.append(math.log(h))
    slopes = [(math.log(errs[i]) - math.log(errs[i + 1])) / (logs[i] - logs[i + 1])
              for i in range(len(errs) - 1)]
    return sum(slopes) / len(slopes)


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("sigma", [0.01, 0.3, 1.0])
@pytest.mark.parametrize("t0", [0.0, 0.3, 0.9])
def test_converges_to_the_closed_form(integrator, sigma, t0):
    """On the linear field the answer is known exactly, so shrinking h must walk
    the result towards it rather than towards something else."""
    x0 = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    exact = exact_flow(x0, sigma, t0, 1.0)
    coarse = integrator(linear_field(sigma), x0, t0, 1.0, 1 / 32)
    fine = integrator(linear_field(sigma), x0, t0, 1.0, 1 / 4096)
    assert (fine - exact).abs().max() < (coarse - exact).abs().max()
    tol = FINE_TOLERANCE[integrator]
    assert torch.allclose(fine, exact, rtol=tol, atol=tol)


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("sigma", [0.1, 1.0])
def test_convergence_order_is_what_the_method_claims(integrator, sigma):
    order = observed_order(integrator, sigma, 0.2, 1.0, [1 / 64, 1 / 128, 1 / 256])
    assert order == pytest.approx(EXPECTED_ORDER[integrator], abs=0.15)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_batched_matches_integrating_each_row_alone(integrator):
    """Spec 8.5: the perturbed and unperturbed trajectories must see identical
    arithmetic. Any per-row step decision would break this."""
    rows = torch.tensor([[1.0, 0.0], [-3.0, 2.0], [0.25, 0.25]], dtype=torch.float64)
    field = linear_field(0.2)
    together = integrator(field, rows, 0.3, 1.0, 1 / 128)
    apart = torch.cat([integrator(field, r.unsqueeze(0), 0.3, 1.0, 1 / 128)
                       for r in rows])
    assert torch.equal(together, apart)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_composition_matches_a_single_run(integrator):
    """t0 -> t1 -> t2 must agree with t0 -> t2. Needs no closed form, so this is
    the check that also applies to the real GMM field."""
    x0 = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    field = linear_field(0.3)
    h = 1 / 2048
    two_hops = integrator(field, integrator(field, x0, 0.2, 0.6, h), 0.6, 1.0, h)
    one_hop = integrator(field, x0, 0.2, 1.0, h)
    assert torch.allclose(two_hops, one_hop, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_zero_length_interval_returns_the_input_untouched(integrator):
    x0 = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    assert torch.equal(integrator(linear_field(0.3), x0, 0.7, 0.7, 1 / 64), x0)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_input_is_not_modified(integrator):
    x0 = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    before = x0.clone()
    integrator(linear_field(0.3), x0, 0.2, 1.0, 1 / 64)
    assert torch.equal(x0, before)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_the_field_sees_the_times_the_grid_promised(integrator):
    """The stage times must walk t0 -> t1 on the uniform grid, not on h."""
    seen = []
    steps, dt = time_grid(0.3, 1.0, 1 / 64)
    integrator(lambda x, t: (seen.append(t), torch.zeros_like(x))[1],
               torch.zeros((1, 2), dtype=torch.float64), 0.3, 1.0, 1 / 64)
    assert seen[0] == pytest.approx(0.3)
    assert len(set(round(s, 12) for s in seen)) >= steps
    # Strict, not 1.0 + 1e-12: RK4 evaluates at the END of each step, so float
    # drift in the stage times pushes the last one outside the field's domain.
    # A tolerant bound here is what let that bug through the first time.
    assert min(seen) >= 0.3
    assert max(seen) <= 1.0


def test_stiffness_is_what_makes_euler_expensive():
    """The measurement behind the solver choice, pinned as a test.

    On [0.9, 1] the field's max|k| is 50 at sigma=0.01 but only 1.0 at sigma=0.3,
    because k_t peaks at ~1/(2 sigma) near t = 1 - sigma. Euler's error scales as
    h * max|k|, so the stiff case costs it two extra decades of step count for the
    same accuracy -- 6554 steps to reach 6e-4, where RK4 needs a handful.
    """
    kmax = lambda sig: max(abs(schedule.k_t(0.9 + i * 0.001, sig)) for i in range(101))
    assert kmax(0.01) == pytest.approx(50.0, rel=0.05)
    assert kmax(0.3) == pytest.approx(1.0, rel=0.05)

    x0 = torch.tensor([[1.0]], dtype=torch.float64)
    exact = exact_flow(x0, 0.01, 0.9, 1.0)
    err = lambda h: float((euler(linear_field(0.01), x0, 0.9, 1.0, h) - exact).abs()
                          / exact.abs())
    assert err(1 / 4096) > 1e-3, "euler should still be short of 1e-3 here"
    assert err(1 / 65536) < 1e-3, "and should get there only ~16x later"


def test_rk4_beats_euler_at_a_matched_evaluation_budget():
    """The comparison that chose the solver: same number of field calls, since an
    RK4 step costs four. This is the honest way to compare them -- per STEP would
    be rigged in RK4's favour."""
    x0 = torch.tensor([[1.0]], dtype=torch.float64)
    exact = exact_flow(x0, 0.01, 0.2, 1.0)
    err = lambda got: float((got - exact).abs() / exact.abs())
    for n_evals in (256, 1024, 4096):
        e = err(euler(linear_field(0.01), x0, 0.2, 1.0, 0.8 / n_evals))
        r = err(rk4(linear_field(0.01), x0, 0.2, 1.0, 0.8 / (n_evals // 4)))
        assert r < e, f"rk4 should win at {n_evals} evaluations: {r:.2e} vs {e:.2e}"


def test_euler_and_rk4_converge_to_the_same_answer():
    """Both solve the same ODE, so a systematic disagreement in the small-h limit
    would mean one of them is solving a different problem."""
    x0 = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    field = linear_field(0.3)
    e = euler(field, x0, 0.3, 1.0, 1 / 262144)
    r = rk4(field, x0, 0.3, 1.0, 1 / 1024)
    assert torch.allclose(e, r, rtol=1e-5, atol=1e-8)


def test_no_stage_is_evaluated_outside_the_interval():
    """RK4's k4 sits at the end of the step, so the final one lands on t1 exactly.
    The field raises outside [0, 1], and this is the regression test for a drift
    that made it do so.
    """
    for t0 in (0.0, 0.2, 0.78, 0.98):
        seen = []
        rk4(lambda x, t: (seen.append(t), torch.zeros_like(x))[1],
            torch.zeros((1, 2), dtype=torch.float64), t0, 1.0, 1 / 64)
        assert min(seen) >= t0, f"stage before t0: {min(seen)!r}"
        assert max(seen) <= 1.0, f"stage past t1: {max(seen)!r}"
        assert max(seen) == pytest.approx(1.0), "the last stage must reach t1"


# --------------------------------------------------------------------------- #
# integrator + marginal flow, coupled
# --------------------------------------------------------------------------- #

from gtmf.marginal_flow import MarginalFlow  # noqa: E402


def gmm_flow(n=200, d=6, sigma=0.2, seed=70):
    g = torch.Generator().manual_seed(seed)
    return MarginalFlow(torch.randn((n, d), generator=g, dtype=torch.float64) * 2.0,
                        sigma, chunk=64)


def test_rk4_runs_on_the_real_field():
    """The two halves of the ODE have never met before this test: every other
    integrator check uses the analytic linear field."""
    flow = gmm_flow()
    x = flow.sample_query_states(0.3, 8, generator=torch.Generator().manual_seed(71))
    out = rk4(flow.velocity, x, 0.3, 1.0, 1 / 64)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert not torch.allclose(out, x), "the field should have moved the state"


def test_single_component_flow_reproduces_the_closed_form():
    """With N=1 at the origin the mixture IS the linear field, so integrating it
    must give c_t1/c_t0 -- the closed form, through the full field machinery."""
    flow = MarginalFlow(torch.zeros((1, 4), dtype=torch.float64), 0.3)
    x0 = torch.tensor([[1.0, -2.0, 0.5, 0.25]], dtype=torch.float64)
    got = rk4(flow.velocity, x0, 0.2, 1.0, 1 / 256)
    assert torch.allclose(got, exact_flow(x0, 0.3, 0.2, 1.0), rtol=1e-9, atol=1e-9)


def test_composition_holds_on_the_gmm_field():
    """No closed form exists here, so composition is the check that still works:
    0.3 -> 0.6 -> 1.0 must match 0.3 -> 1.0."""
    flow = gmm_flow()
    x = flow.sample_query_states(0.3, 4, generator=torch.Generator().manual_seed(72))
    h = 1 / 512
    assert torch.allclose(rk4(flow.velocity, rk4(flow.velocity, x, 0.3, 0.6, h), 0.6, 1.0, h),
                          rk4(flow.velocity, x, 0.3, 1.0, h), rtol=1e-6, atol=1e-8)


def test_batched_trajectories_stay_independent_on_the_gmm_field():
    """Spec 8.5 again, now through the real field: the perturbed and unperturbed
    rows of one batch must get exactly what they would alone."""
    flow = gmm_flow()
    x = flow.sample_query_states(0.4, 3, generator=torch.Generator().manual_seed(73))
    together = rk4(flow.velocity, x, 0.4, 1.0, 1 / 128)
    apart = torch.cat([rk4(flow.velocity, r.unsqueeze(0), 0.4, 1.0, 1 / 128) for r in x])
    assert torch.allclose(together, apart, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("t0", [0.0, 0.5, 0.98])
def test_the_field_accepts_every_stage_time_rk4_produces(t0):
    """schedule.k_t raises outside [0, 1] and RK4's last stage sits at t1, so this
    is where a drift in the stage times would surface as a crash."""
    flow = gmm_flow(n=50, d=4)
    x = flow.sample_query_states(t0, 2, generator=torch.Generator().manual_seed(74))
    assert torch.isfinite(rk4(flow.velocity, x, t0, 1.0, 1 / 64)).all()
