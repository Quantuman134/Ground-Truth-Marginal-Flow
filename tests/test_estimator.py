"""Tests for gtmf/estimator.py."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtmf.estimator import perturbed_batch  # noqa: E402


def randn(n, d, seed, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n, d), generator=g, dtype=dtype) * scale


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("scheme,blocks_for_k", [("one_sided", lambda k: 1 + k),
                                                 ("central", lambda k: 2 * k)])
@pytest.mark.parametrize("num_probes", [1, 4, 7])
def test_batch_has_one_block_per_state_the_estimator_needs(scheme, blocks_for_k, num_probes):
    x = randn(5, 3, seed=1)
    batch, n_blocks = perturbed_batch(x, 0.1, num_probes, gen(), scheme)
    assert n_blocks == blocks_for_k(num_probes)
    assert batch.shape == (n_blocks * 5, 3)


def test_one_sided_keeps_the_unperturbed_state_as_block_zero():
    """The estimator subtracts F(x) from every probe, so it must be in there,
    unchanged, in a known place."""
    x = randn(6, 4, seed=2)
    batch, n = perturbed_batch(x, 0.05, 3, gen(), "one_sided")
    assert torch.equal(batch.reshape(n, 6, 4)[0], x)


def test_one_sided_probes_differ_from_x_by_exactly_eps_times_a_unit_normal():
    """Recovering u from the batch must give back a standard normal: eps scales
    the direction, it does not reshape the distribution."""
    x = randn(200, 64, seed=3)
    eps = 0.037
    batch, n = perturbed_batch(x, eps, 5, gen(4), "one_sided")
    blocks = batch.reshape(n, 200, 64)
    u = (blocks[1:] - blocks[0]) / eps
    assert u.mean().abs() < 0.02
    assert u.std() == pytest.approx(1.0, rel=0.02)


def test_central_blocks_are_symmetric_about_the_query_state():
    """(x + eps u) and (x - eps u) must straddle x along the SAME direction --
    the whole point of a central difference. A fresh u for the minus side would
    still look plausible and would silently estimate the wrong thing.
    """
    x = randn(7, 5, seed=5)
    eps, k = 0.02, 4
    batch, n = perturbed_batch(x, eps, k, gen(6), "central")
    blocks = batch.reshape(n, 7, 5)
    plus, minus = blocks[:k], blocks[k:]
    assert torch.allclose((plus + minus) / 2, x.expand(k, 7, 5), rtol=1e-14, atol=1e-14)
    assert torch.allclose((plus - minus) / 2, plus - x, rtol=1e-14, atol=1e-14)


def test_central_carries_no_unperturbed_row():
    """F(x) is never referenced by Eq. 29, so integrating it would be waste."""
    x = randn(4, 3, seed=7)
    batch, n = perturbed_batch(x, 0.1, 2, gen(), "central")
    assert n == 4
    assert not any(torch.allclose(row, x[0]) for row in batch)


@pytest.mark.parametrize("scheme", ["one_sided", "central"])
def test_same_generator_seed_reproduces_the_batch(scheme):
    x = randn(5, 4, seed=8)
    a, _ = perturbed_batch(x, 0.1, 3, gen(9), scheme)
    b, _ = perturbed_batch(x, 0.1, 3, gen(9), scheme)
    c, _ = perturbed_batch(x, 0.1, 3, gen(10), scheme)
    assert torch.equal(a, b) and not torch.equal(a, c)


@pytest.mark.parametrize("scheme", ["one_sided", "central"])
def test_every_probe_uses_a_different_direction(scheme):
    """K probes must explore K directions; reusing one would make the average
    over directions meaningless while still producing a number."""
    x = torch.zeros((1, 32), dtype=torch.float64)
    batch, n = perturbed_batch(x, 0.1, 5, gen(11), scheme)
    rows = batch.reshape(n, 1, 32)[:, 0, :]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            assert not torch.allclose(rows[i], rows[j])


@pytest.mark.parametrize("scheme", ["one_sided", "central"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_follows_the_input(scheme, dtype):
    x = randn(4, 6, seed=12).to(dtype)
    batch, _ = perturbed_batch(x, 0.1, 3, gen(), scheme)
    assert batch.dtype == dtype


@pytest.mark.parametrize("scheme", ["one_sided", "central"])
def test_the_query_states_are_not_modified(scheme):
    x = randn(4, 6, seed=13)
    before = x.clone()
    perturbed_batch(x, 0.1, 3, gen(), scheme)
    assert torch.equal(x, before)


@pytest.mark.parametrize("scheme", ["one_sided", "central"])
def test_batch_is_contiguous_and_reshapes_back(scheme):
    """The caller recovers blocks with a reshape, which needs contiguity."""
    x = randn(9, 5, seed=14)
    batch, n = perturbed_batch(x, 0.1, 3, gen(), scheme)
    assert batch.is_contiguous()
    assert batch.reshape(n, 9, 5).reshape(-1, 5).shape == batch.shape


@pytest.mark.parametrize("kwargs", [
    dict(scheme="both"), dict(num_probes=0), dict(num_probes=-1),
    dict(eps=0.0), dict(eps=-0.1),
])
def test_invalid_arguments_raise(kwargs):
    args = dict(x=randn(3, 4, seed=15), eps=0.1, num_probes=2, scheme="one_sided")
    args.update(kwargs)
    with pytest.raises(ValueError):
        perturbed_batch(args["x"], args["eps"], args["num_probes"], gen(), args["scheme"])


def test_one_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        perturbed_batch(torch.zeros(5, dtype=torch.float64), 0.1, 2, gen())


# --------------------------------------------------------------------------- #
# w_hat, one-sided
# --------------------------------------------------------------------------- #

from gtmf import schedule                                    # noqa: E402
from gtmf.estimator import w_hat                             # noqa: E402
from gtmf.marginal_flow import MarginalFlow                   # noqa: E402


def linear_field(sigma):
    """v(x, t) = k_t x -- the single-Gaussian field, whose flow map is exactly
    linear, so the finite difference carries NO truncation error."""
    return lambda x, t: schedule.k_t(t, sigma) * x


@pytest.mark.parametrize("sigma", [0.01, 0.3, 1.0])
@pytest.mark.parametrize("t", [0.0, 0.3, 0.7, 0.9])
def test_w_hat_reproduces_the_closed_form(sigma, t):
    """On the single-Gaussian field, Phi = (sigma / c_t) I exactly, so
    w = sigma^2 / c_t^2 (spec Eq. 32) -- a number derived from none of the code
    under test. This exercises the whole chain: batch, integrator, estimator.

    Averaged over queries because w_hat is a Monte-Carlo estimate: ||u||^2/d has
    relative spread sqrt(2/d), so B*K probes bring it to ~0.3% here.
    """
    x = randn(64, 256, seed=20)
    w = w_hat(linear_field(sigma), x, t, eps=1e-3, num_probes=16,
              h=1 / 512, generator=gen(21))
    assert w.shape == (64,)
    assert float(w.mean()) == pytest.approx(schedule.w_exact(t, sigma), rel=0.02)


@pytest.mark.parametrize("sigma", [0.01, 0.3])
def test_w_hat_is_independent_of_eps_on_a_linear_field(sigma):
    """THE GATE for this phase. A linear flow map makes the difference quotient
    exact for any eps, so the whole alpha sweep must give one answer. Any
    eps-dependence here is the estimator being wrong, not physics -- on the real
    mixture that same dependence is the signal the sweep is looking for.
    """
    x = randn(32, 64, seed=22)
    ws = [w_hat(linear_field(sigma), x, 0.4, eps=e, num_probes=8,
                h=1 / 256, generator=gen(23))
          for e in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)]
    for w in ws[1:]:
        assert torch.allclose(w, ws[0], rtol=1e-9, atol=1e-12)


def test_w_hat_shrinks_its_spread_as_probes_are_added():
    """w_hat averages over K directions, so its scatter must fall like 1/sqrt(K).
    A K that silently did nothing would still return a plausible mean."""
    x = randn(256, 64, seed=24)
    spreads = [float(w_hat(linear_field(0.3), x, 0.5, eps=1e-3, num_probes=k,
                           h=1 / 256, generator=gen(25)).std())
               for k in (1, 16, 256)]
    assert spreads[0] > spreads[1] > spreads[2]
    assert spreads[0] / spreads[1] == pytest.approx(4.0, rel=0.35)   # sqrt(16)


def test_w_hat_is_positive_and_finite_on_the_real_field():
    g = torch.Generator().manual_seed(26)
    flow = MarginalFlow(torch.randn((200, 8), generator=g, dtype=torch.float64) * 2.0,
                        0.2, chunk=64)
    x = flow.sample_query_states(0.4, 6, generator=gen(27))
    w = w_hat(flow.velocity, x, 0.4, eps=1e-3 * flow.rho_t_exact(0.4),
              num_probes=4, h=1 / 64, generator=gen(28))
    assert w.shape == (6,) and torch.all(w > 0) and torch.isfinite(w).all()


def test_w_hat_is_reproducible_and_seed_sensitive():
    x = randn(16, 32, seed=29)
    kw = dict(eps=1e-3, num_probes=4, h=1 / 128)
    a = w_hat(linear_field(0.3), x, 0.5, generator=gen(30), **kw)
    b = w_hat(linear_field(0.3), x, 0.5, generator=gen(30), **kw)
    c = w_hat(linear_field(0.3), x, 0.5, generator=gen(31), **kw)
    assert torch.equal(a, b) and not torch.equal(a, c)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_w_hat_dtype_follows_the_input(dtype):
    x = randn(8, 16, seed=32).to(dtype)
    assert w_hat(linear_field(0.3), x, 0.5, eps=1e-3, num_probes=3,
                 h=1 / 64, generator=gen()).dtype == dtype


# --------------------------------------------------------------------------- #
# w_hat, central difference
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sigma", [0.01, 0.3, 1.0])
@pytest.mark.parametrize("t", [0.0, 0.4, 0.9])
def test_central_reproduces_the_closed_form(sigma, t):
    x = randn(64, 256, seed=34)
    w = w_hat(linear_field(sigma), x, t, eps=1e-3, num_probes=16, h=1 / 512,
              generator=gen(35), scheme="central")
    assert float(w.mean()) == pytest.approx(schedule.w_exact(t, sigma), rel=0.02)


@pytest.mark.parametrize("t", [0.2, 0.6, 0.9])
def test_central_and_one_sided_agree_exactly_on_a_linear_field(t):
    """Both schemes are exact when the flow map is linear, and both draw the same
    directions from the same seed -- so they must agree to rounding.

    This is the factor-of-two test: the central divisor is 2 eps, not eps, and
    getting it wrong yields a w four times too small. That would be a plausible
    curve of the right shape, which nothing downstream would question.
    """
    x = randn(32, 64, seed=36)
    kw = dict(eps=1e-3, num_probes=8, h=1 / 256)
    one = w_hat(linear_field(0.3), x, t, generator=gen(37), scheme="one_sided", **kw)
    two = w_hat(linear_field(0.3), x, t, generator=gen(37), scheme="central", **kw)
    assert torch.allclose(one, two, rtol=1e-9, atol=1e-12)


def test_central_is_independent_of_eps_on_a_linear_field():
    x = randn(32, 64, seed=38)
    ws = [w_hat(linear_field(0.3), x, 0.4, eps=e, num_probes=8, h=1 / 256,
                generator=gen(39), scheme="central")
          for e in (1e-4, 1e-3, 1e-2, 3e-2)]
    for w in ws[1:]:
        assert torch.allclose(w, ws[0], rtol=1e-9, atol=1e-12)


def test_central_has_less_truncation_bias_on_a_curved_field():
    """The reason central exists at all.

    On a nonlinear flow map one-sided carries an O(eps) bias and central an
    O(eps^2) one. Measured against the eps -> 0 limit at a deliberately large
    eps, central must land closer. Both schemes see identical directions here,
    so the only difference is the scheme.
    """
    g = torch.Generator().manual_seed(40)
    flow = MarginalFlow(torch.randn((300, 8), generator=g, dtype=torch.float64) * 4.0,
                        0.15, chunk=64)
    x = flow.sample_query_states(0.35, 24, generator=gen(41))
    kw = dict(num_probes=32, h=1 / 256)
    scale = flow.rho_t_exact(0.35)

    limit = w_hat(flow.velocity, x, 0.35, eps=1e-6 * scale,
                  generator=gen(42), scheme="central", **kw)
    big = 0.3 * scale
    one = w_hat(flow.velocity, x, 0.35, eps=big, generator=gen(42),
                scheme="one_sided", **kw)
    two = w_hat(flow.velocity, x, 0.35, eps=big, generator=gen(42),
                scheme="central", **kw)

    bias_one = float((one - limit).abs().mean())
    bias_two = float((two - limit).abs().mean())
    assert bias_two < bias_one, f"central {bias_two:.3e} should beat one-sided {bias_one:.3e}"


def test_central_costs_two_solves_per_probe_and_one_sided_costs_k_plus_one():
    """The trade the scheme choice is making, pinned as a number."""
    calls = {"one_sided": 0, "central": 0}

    def counting(scheme):
        def f(y, s):
            calls[scheme] += y.shape[0]
            return schedule.k_t(s, 0.3) * y
        return f

    x = randn(10, 4, seed=43)
    for scheme in ("one_sided", "central"):
        w_hat(counting(scheme), x, 0.5, eps=1e-3, num_probes=4, h=1 / 4,
              generator=gen(44), scheme=scheme)
    assert calls["one_sided"] / calls["central"] == pytest.approx((4 + 1) / (2 * 4))


def test_the_integrator_is_swappable():
    """w_hat takes the integrator as a parameter; nothing had ever passed one.

    Euler needs a far smaller step for the same accuracy, so this also shows the
    estimator is measuring the FLOW rather than the integrator: both must land on
    the same w once each has converged.
    """
    from gtmf.integrate import euler
    x = randn(32, 64, seed=45)
    kw = dict(eps=1e-3, num_probes=8)
    with_rk4 = w_hat(linear_field(0.3), x, 0.4, h=1 / 256, generator=gen(46), **kw)
    with_euler = w_hat(linear_field(0.3), x, 0.4, h=1 / 262144, generator=gen(46),
                       integrator=euler, **kw)
    assert torch.allclose(with_rk4, with_euler, rtol=1e-4, atol=1e-8)


def test_a_too_coarse_integrator_shows_up_in_w():
    """Sanity on the above: if the integrator were irrelevant, an obviously
    under-resolved run would still agree, and the test would prove nothing."""
    from gtmf.integrate import euler
    x = randn(32, 64, seed=47)
    kw = dict(eps=1e-3, num_probes=8, generator=None)
    good = w_hat(linear_field(0.3), x, 0.4, h=1 / 256, num_probes=8, eps=1e-3,
                 generator=gen(48))
    coarse = w_hat(linear_field(0.3), x, 0.4, h=1 / 4, num_probes=8, eps=1e-3,
                   generator=gen(48), integrator=euler)
    assert not torch.allclose(coarse, good, rtol=1e-2)
