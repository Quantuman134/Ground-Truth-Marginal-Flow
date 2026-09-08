"""Tests for gtmf/marginal_flow.py.

Two kinds of test here:

  * the closed-form oracles, run against EVERY implementation via ORACLE_IMPLS,
    so each new step inherits the whole set automatically;
  * per-step tests, checking that a step agrees with the one before it and does
    the specific thing it was added for.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtmf import schedule                                          # noqa: E402
from gtmf.marginal_flow import velocity_naive, MarginalFlow  # noqa: E402

def chunked(mu, sigma, x, t, chunk=65536):
    """Call the chunked field (step 5), which now lives on the class."""
    return MarginalFlow(mu, sigma, chunk=chunk).velocity(x, t)


def velocity_via_class(mu, sigma, x, t):
    """Adapter so MarginalFlow.velocity inherits every oracle below."""
    return chunked(mu, sigma, x, t, chunk=64)


# Every implementation must satisfy every oracle. Append new steps here.
ORACLE_IMPLS = [velocity_naive, velocity_via_class]

SIGMAS = [0.01, 0.3, 1.0]
TIMES = [0.0, 0.25, 0.6, 0.9, 1.0]


def randn(n, d, seed, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n, d), generator=g, dtype=dtype) * scale


# --------------------------------------------------------------------------- #
# closed-form oracles -- every implementation, every time
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("velocity", ORACLE_IMPLS)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("t", TIMES)
def test_one_component_at_origin_gives_k_times_x(velocity, sigma, t):
    """With N=1 and mu=0 there is nothing to mix, so v = k_t * x.

    This is the single-Gaussian field the phase-4 sanity check will use.
    """
    mu = torch.zeros((1, 5), dtype=torch.float64)
    x = randn(6, 5, seed=1)
    assert torch.allclose(velocity(mu, sigma, x, t), schedule.k_t(t, sigma) * x)


@pytest.mark.parametrize("velocity", ORACLE_IMPLS)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("t", TIMES)
def test_one_component_off_origin_is_eq_16(velocity, sigma, t):
    """With N=1 the responsibility is 1, so v is exactly Eq. 16."""
    mu = randn(1, 5, seed=2)
    x = randn(6, 5, seed=3)
    assert torch.allclose(velocity(mu, sigma, x, t),
                          mu + schedule.k_t(t, sigma) * (x - t * mu))


@pytest.mark.parametrize("velocity", ORACLE_IMPLS)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("n", [1, 4, 50])
def test_at_t_zero_the_field_is_mean_mu_minus_x(velocity, sigma, n):
    """At t=0 the query says nothing about which component it came from, so every
    responsibility is 1/N and v = E[y] - x.

    Derived independently: x_t = x_0 at t=0, and y is drawn independently of x_0,
    so E[y - x_0 | x_0 = x] = E[y] - x.
    """
    mu = randn(n, 5, seed=4, scale=3.0)
    x = randn(6, 5, seed=5)
    assert torch.allclose(velocity(mu, sigma, x, 0.0), mu.mean(dim=0) - x)


@pytest.mark.parametrize("velocity", ORACLE_IMPLS)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("n", [1, 4, 50])
def test_at_t_one_the_field_is_the_identity(velocity, sigma, n):
    """At t=1, x_1 = y and x_0 is independent of it, so
    v = E[y - x_0 | x_1 = x] = x - 0 = x."""
    mu = randn(n, 5, seed=6, scale=3.0)
    x = randn(6, 5, seed=7)
    assert torch.allclose(velocity(mu, sigma, x, 1.0), x)


@pytest.mark.parametrize("velocity", ORACLE_IMPLS)
@pytest.mark.parametrize("t", [0.2, 0.5, 0.8])
def test_midpoint_of_two_centers_gives_their_mean(velocity, t):
    """A query equidistant from two centers splits its responsibility evenly, so
    mu_bar is their mean; sitting at x = t*mu_bar also kills the k(x - t mu) term,
    leaving v = mean(mu)."""
    mu = torch.tensor([[2.0, 0.0], [-2.0, 0.0]], dtype=torch.float64)
    x = (t * mu.mean(dim=0)).unsqueeze(0)
    assert torch.allclose(velocity(mu, 0.5, x, t), mu.mean(dim=0, keepdim=True))


@pytest.mark.parametrize("velocity", ORACLE_IMPLS)
def test_output_shape_follows_the_batch(velocity):
    mu = randn(7, 5, seed=8)
    for batch in (1, 3, 20):
        assert velocity(mu, 0.3, randn(batch, 5, seed=9), 0.4).shape == (batch, 5)


@pytest.mark.parametrize("velocity", ORACLE_IMPLS)
def test_inputs_are_not_modified(velocity):
    mu, x = randn(7, 5, seed=10), randn(4, 5, seed=11)
    mu_before, x_before = mu.clone(), x.clone()
    velocity(mu, 0.3, x, 0.6)
    assert torch.equal(mu, mu_before) and torch.equal(x, x_before)


# --------------------------------------------------------------------------- #
# the max shift
# --------------------------------------------------------------------------- #




@pytest.mark.parametrize("sigma", [0.01, 0.3])
@pytest.mark.parametrize("t", [0.3, 0.7, 0.98])
def test_the_shift_is_what_keeps_this_finite(sigma, t):
    """Without the shift every weight underflows and r would be 0/0 = NaN.

    Uses d = 256 and fp32 -- the production configuration. The best achievable
    score is about -d/2 = -128 because a d-dimensional Gaussian puts its mass at
    radius ~sqrt(d)*c_t, so even the nearest component is that far away.
    exp(-128) ~ 1e-56 underflows fp32, whose smallest normal is ~1e-38.
    """
    mu = randn(200, 256, seed=14, scale=0.589, dtype=torch.float32)   # real latent scale
    x = t * mu[:4] + randn(4, 256, seed=15, dtype=torch.float32) * schedule.c_t(t, sigma)

    c2 = schedule.c_t_sq(t, sigma)
    logits = -((x[:, None, :] - t * mu[None, :, :]) ** 2).sum(-1) / (2 * c2)

    unshifted = logits.exp().sum(dim=1)                  # what a naive version does
    assert torch.all(unshifted == 0), "expected underflow; the test case is stale"

    got = chunked(mu, sigma, x, t)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, velocity_naive(mu, sigma, x, t), rtol=1e-5, atol=1e-5)


def test_shifted_weights_put_the_largest_at_exactly_one():
    """The point of the shift: the winning weight is 1, so nothing can underflow
    and the denominator is always >= 1."""
    mu = randn(60, 256, seed=16, scale=0.589)
    x = randn(5, 256, seed=17, scale=0.589)
    c2 = schedule.c_t_sq(0.7, 0.01)
    logits = -((x[:, None, :] - 0.7 * mu[None, :, :]) ** 2).sum(-1) / (2 * c2)
    weights = (logits - logits.max(dim=1, keepdim=True).values).exp()
    assert torch.allclose(weights.max(dim=1).values, torch.ones(5, dtype=torch.float64))
    assert torch.all(weights.sum(dim=1) >= 1.0)


# --------------------------------------------------------------------------- #
# the collapsed sum, against an independent loop
# --------------------------------------------------------------------------- #

def velocity_python_loop(mu, sigma, x, t):
    """Eq. 17 with an explicit Python loop -- no broadcasting, no matmul, no
    collapse. Slow and obviously correct; exists only to check step 3's algebra
    against something that shares none of its machinery."""
    c2 = schedule.c_t_sq(t, sigma)
    k = schedule.k_t(t, sigma)
    rows = []
    for b in range(x.shape[0]):
        scores = torch.stack([-((x[b] - t * mu[i]) ** 2).sum() / (2 * c2)
                              for i in range(mu.shape[0])])
        r = torch.softmax(scores, dim=0)
        v = sum(r[i] * (mu[i] + k * (x[b] - t * mu[i])) for i in range(mu.shape[0]))
        rows.append(v)
    return torch.stack(rows)





@pytest.mark.parametrize("sigma", [0.01, 0.3, 1.0])
@pytest.mark.parametrize("t", [0.0, 0.4, 0.85, 1.0])
def test_agrees_with_an_independent_python_loop(sigma, t):
    """Checks (1 - kt) mu_bar + k x against a term-by-term sum written without
    any of the same operations. If the collapse dropped a factor of t or flipped
    a sign, only this and the step-2 comparison would catch it."""
    mu = randn(12, 4, seed=20, scale=2.0)
    x = randn(5, 4, seed=21, scale=2.0)
    assert torch.allclose(chunked(mu, sigma, x, t),
                          velocity_python_loop(mu, sigma, x, t), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("t", [0.3, 0.7])
def test_mu_bar_is_a_convex_combination_of_the_centers(t):
    """mu_bar is a weighted average with weights summing to 1, so it cannot leave
    the coordinate-wise bounding box of the centres. Recovered from the output,
    which also checks the (1 - kt) coefficient is what the docstring claims."""
    mu = randn(80, 6, seed=22, scale=3.0)
    x = randn(7, 6, seed=23, scale=3.0)
    k = schedule.k_t(t, 0.25)
    mu_bar = (chunked(mu, 0.25, x, t) - k * x) / (1.0 - k * t)
    assert torch.all(mu_bar >= mu.min(dim=0).values - 1e-9)
    assert torch.all(mu_bar <= mu.max(dim=0).values + 1e-9)


# --------------------------------------------------------------------------- #
# the expanded distance, under cancellation
# --------------------------------------------------------------------------- #







@pytest.mark.parametrize("t", [0.3, 0.7, 0.98])
def test_expansion_survives_cancellation_at_production_scale(t):
    """The expansion subtracts two large numbers to get a small one, so it can
    lose precision where step 3's direct distance does not.

    Run in fp32 at d = 256 with the real latent scale, against step 3 in fp64.
    """
    mu64 = randn(300, 256, seed=28, scale=0.589)
    x64 = t * mu64[:6] + randn(6, 256, seed=29) * schedule.c_t(t, 0.01)
    exact = velocity_naive(mu64, 0.01, x64, t)
    got = chunked(mu64.float(), 0.01, x64.float(), t).double()
    rel = ((got - exact).norm(dim=1) / exact.norm(dim=1)).max()
    assert rel < 1e-3, f"fp32 expansion lost too much: relative error {rel:.2e}"


# --------------------------------------------------------------------------- #
# chunking over components
# --------------------------------------------------------------------------- #

def velocity_chunked_per_block_max(mu, sigma, x, t, chunk):
    """The bug spec 8.1 warns about: each block normalised against its OWN max,
    with no rescaling of what came before. Exists only so the test below can show
    that chunk-invariance actually discriminates."""
    c2 = schedule.c_t_sq(t, sigma)
    k = schedule.k_t(t, sigma)
    sq = (mu ** 2).sum(dim=1)
    s = torch.zeros((x.shape[0], 1), dtype=x.dtype)
    acc = torch.zeros((x.shape[0], mu.shape[1]), dtype=x.dtype)
    for lo in range(0, mu.shape[0], chunk):
        hi = min(lo + chunk, mu.shape[0])
        logits = (t * (x @ mu[lo:hi].T) - 0.5 * t * t * sq[lo:hi]) / c2
        w = (logits - logits.max(dim=1, keepdim=True).values).exp()   # local max only
        s = s + w.sum(dim=1, keepdim=True)
        acc = acc + w @ mu[lo:hi]
    return (1.0 - k * t) * (acc / s) + k * x


@pytest.mark.parametrize("chunk", [1, 3, 17, 64, 200, 199, 1000])
@pytest.mark.parametrize("t", [0.0, 0.35, 0.8, 1.0])
def test_chunking_never_changes_the_answer(chunk, t):
    """Chunking is a memory decision, so it must not touch the answer -- at sizes
    that divide N, sizes that do not, one component at a time, and all at once."""
    mu = randn(200, 6, seed=30, scale=3.0)
    x = randn(9, 6, seed=31, scale=3.0)
    assert torch.allclose(chunked(mu, 0.2, x, t, chunk=chunk),
                          velocity_naive(mu, 0.2, x, t), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("t", [0.4, 0.9])
def test_a_per_block_max_really_does_break_it(t):
    """Proves the invariance test above is not vacuous: the wrong version, which
    normalises inside each block, gives a different and finite answer -- exactly
    the failure that would otherwise pass unnoticed."""
    mu = randn(200, 6, seed=32, scale=4.0)
    x = randn(5, 6, seed=33, scale=4.0)
    right = velocity_naive(mu, 0.15, x, t)
    wrong = velocity_chunked_per_block_max(mu, 0.15, x, t, chunk=32)
    assert torch.isfinite(wrong).all(), "the bug is silent, not a crash"
    assert not torch.allclose(wrong, right, rtol=1e-3, atol=1e-3)


def test_chunk_larger_than_n_is_just_one_block():
    mu = randn(50, 6, seed=34, scale=2.0)
    x = randn(4, 6, seed=35, scale=2.0)
    assert torch.equal(chunked(mu, 0.3, x, 0.6, chunk=10_000),
                       chunked(mu, 0.3, x, 0.6, chunk=50))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accumulators_follow_the_input_dtype(dtype):
    """A bare torch.zeros would be fp32 and would silently downcast an fp64 run."""
    mu = randn(80, 6, seed=36, scale=2.0).to(dtype)
    x = randn(5, 6, seed=37, scale=2.0).to(dtype)
    assert chunked(mu, 0.25, x, 0.5, chunk=16).dtype == dtype


REAL_CENTERS = Path("/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/"
                    "latents_8_mean_fp16/centers_fp16.npy")


@pytest.mark.skipif(not REAL_CENTERS.exists(), reason="production centers not built")
@pytest.mark.parametrize("t,sigma", [(0.0, 0.01), (1.0, 0.01), (0.0, 0.6), (1.0, 0.6)])
def test_endpoint_oracles_on_the_real_mixture(t, sigma):
    """The t=0 and t=1 oracles hold for any N, so they can run against all
    1,281,167 centres. Step 5 is the first version that fits."""
    from gtmf.data import load_centers
    centers = load_centers(REAL_CENTERS, dtype=torch.float32)
    x = torch.randn((2, centers.d), generator=torch.Generator().manual_seed(38))
    got = MarginalFlow(centers.mu, sigma, chunk=1 << 18).velocity(x, t)
    expected = (centers.mu.mean(dim=0) - x) if t == 0.0 else x
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# sampling and rho_t
# --------------------------------------------------------------------------- #

def test_class_velocity_agrees_with_the_unchunked_reference():
    mu = randn(300, 6, seed=39, scale=2.0)
    x = randn(7, 6, seed=40, scale=2.0)
    flow = MarginalFlow(mu, 0.25, chunk=64)
    for t in (0.0, 0.3, 0.9, 1.0):
        assert torch.allclose(flow.velocity(x, t),
                              velocity_naive(mu, 0.25, x, t), rtol=1e-11, atol=1e-11)


def test_squared_norms_are_computed_once_and_correctly():
    mu = randn(40, 6, seed=41, scale=2.0)
    flow = MarginalFlow(mu, 0.3)
    assert torch.allclose(flow.sq, (mu ** 2).sum(dim=1))
    assert flow.n == 40 and flow.d == 6


@pytest.mark.parametrize("t", [0.2, 0.5, 0.9])
def test_query_states_have_the_moments_p_t_says_they_should(t):
    """x_t = (1-t) x_0 + t (mu_i + sigma eps) with i uniform, so

        E[x_t]   = t * mean(mu)
        Var(x_t) = (1-t)^2 + t^2 sigma^2 + t^2 Var(mu) = c_t^2 + t^2 Var(mu)

    Both derived from the definition, independently of the sampling code.
    """
    g0 = torch.Generator().manual_seed(42)
    mu = torch.randn((400, 5), generator=g0, dtype=torch.float64) * 2.0
    sigma = 0.3
    flow = MarginalFlow(mu, sigma)
    xs = flow.sample_query_states(t, 400_000, generator=torch.Generator().manual_seed(43))

    assert torch.allclose(xs.mean(dim=0), t * mu.mean(dim=0), atol=0.02)
    want_var = schedule.c_t_sq(t, sigma) + t * t * mu.var(dim=0, unbiased=False)
    assert torch.allclose(xs.var(dim=0, unbiased=False), want_var, rtol=0.03)


@pytest.mark.parametrize("t", [0.1, 0.6, 0.98])
def test_empirical_rho_t_matches_its_closed_form(t):
    g0 = torch.Generator().manual_seed(44)
    mu = torch.randn((500, 5), generator=g0, dtype=torch.float64) * 1.5
    flow = MarginalFlow(mu, 0.25)
    xs = flow.sample_query_states(t, 400_000, generator=torch.Generator().manual_seed(45))
    assert flow.rho_t(xs) == pytest.approx(flow.rho_t_exact(t), rel=0.01)


def test_rho_t_is_not_c_t():
    """rho_t includes the spread of the centres; c_t does not. Mixing them up
    would size the perturbation wrongly -- by ~27x at t=0.98, sigma=0.01."""
    mu = randn(300, 256, seed=46, scale=0.589)
    flow = MarginalFlow(mu, 0.01)
    assert flow.rho_t_exact(0.98) / schedule.c_t(0.98, 0.01) > 20


def test_sampling_is_reproducible_and_seed_sensitive():
    flow = MarginalFlow(randn(50, 6, seed=47), 0.3)
    a = flow.sample_query_states(0.4, 64, generator=torch.Generator().manual_seed(9))
    b = flow.sample_query_states(0.4, 64, generator=torch.Generator().manual_seed(9))
    c = flow.sample_query_states(0.4, 64, generator=torch.Generator().manual_seed(10))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_sampling_does_not_mutate_the_centers():
    mu = randn(40, 6, seed=48)
    flow = MarginalFlow(mu, 0.5)
    before = mu.clone()
    flow.sample_query_states(0.5, 256, generator=torch.Generator().manual_seed(11))
    assert torch.equal(mu, before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_samples_follow_the_center_dtype(dtype):
    flow = MarginalFlow(randn(30, 6, seed=49).to(dtype), 0.3)
    xs = flow.sample_query_states(0.5, 16, generator=torch.Generator().manual_seed(12))
    assert xs.dtype == dtype


# --------------------------------------------------------------------------- #
# responsibility pruning
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("t", [0.3, 0.75, 0.98])
@pytest.mark.parametrize("tau", [30.0, 60.0, 120.0])
def test_pruning_does_not_change_the_answer(t, tau):
    """tau is a compute knob, not a modelling choice. At the spec's tau=30 the
    largest discarded weight is exp(-30) ~ 9e-14 of the winner, so the curve must
    not move."""
    mu = randn(400, 8, seed=50, scale=5.0)
    x = randn(8, 8, seed=51, scale=5.0)
    plain = MarginalFlow(mu, 0.12, chunk=64).velocity(x, t)
    pruned = MarginalFlow(mu, 0.12, chunk=64, tau=tau).velocity(x, t)
    assert torch.allclose(pruned, plain, rtol=1e-9, atol=1e-9)


def top_weight(mu, sigma, x, t):
    """Largest responsibility per query -- how committed the mixture is."""
    c2 = schedule.c_t_sq(t, sigma)
    logits = -((x[:, None, :] - t * mu[None, :, :]) ** 2).sum(-1) / (2 * c2)
    shifted = logits - logits.max(dim=1, keepdim=True).values
    return (shifted.exp() / shifted.exp().sum(dim=1, keepdim=True)).max(dim=1).values


def test_pruning_actually_prunes():
    """Proves the test above is not vacuous: with an absurd tau the cutoff bites
    and the answer DOES move, so the mechanism is really wired in.

    Only possible where the mixture is genuinely blended. The precondition below
    is part of the test: past the commitment transition the winning component
    already carries ~all the mass, so discarding the rest changes nothing and
    this would pass for the wrong reason.
    """
    mu = randn(400, 8, seed=52, scale=5.0)
    x = randn(8, 8, seed=53, scale=5.0)
    t = 0.3
    assert top_weight(mu, 0.12, x, t).mean() < 0.95, "fixture is no longer blended"

    plain = MarginalFlow(mu, 0.12, chunk=64).velocity(x, t)
    savage = MarginalFlow(mu, 0.12, chunk=64, tau=0.05).velocity(x, t)
    assert not torch.allclose(savage, plain, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("t", [0.75, 0.9])
def test_past_the_transition_pruning_cannot_matter(t):
    """The other half of the same finding, pinned as a property.

    Once the mixture has committed -- t beyond ~0.5-0.7, the transition recorded
    in CLAUDE.md -- one component holds essentially all the responsibility, so
    even a cutoff severe enough to keep ONLY the winner leaves the field
    unchanged. This is why tau buys nothing at the t values the experiment cares
    most about.
    """
    mu = randn(400, 8, seed=52, scale=5.0)
    x = randn(8, 8, seed=53, scale=5.0)
    assert top_weight(mu, 0.12, x, t).mean() > 0.999, "fixture is no longer committed"

    plain = MarginalFlow(mu, 0.12, chunk=64).velocity(x, t)
    savage = MarginalFlow(mu, 0.12, chunk=64, tau=0.05).velocity(x, t)
    assert torch.allclose(savage, plain, rtol=1e-3, atol=1e-3)


def test_pruning_keeps_the_dtype():
    """w * (w >= floor) multiplies by a bool mask; the result must stay float."""
    for dtype in (torch.float32, torch.float64):
        mu = randn(100, 6, seed=54, scale=3.0).to(dtype)
        x = randn(4, 6, seed=55, scale=3.0).to(dtype)
        assert MarginalFlow(mu, 0.2, chunk=32, tau=30.0).velocity(x, 0.5).dtype == dtype


@pytest.mark.parametrize("tau", [5.0, 30.0])
@pytest.mark.parametrize("chunk", [17, 400])
def test_retained_counts_match_a_direct_computation(tau, chunk):
    """Counted against the global max computed in one shot on a small mixture."""
    mu = randn(400, 6, seed=56, scale=4.0)
    x = randn(5, 6, seed=57, scale=4.0)
    flow = MarginalFlow(mu, 0.2, chunk=chunk)

    c2 = schedule.c_t_sq(0.6, 0.2)
    logits = -((x[:, None, :] - 0.6 * mu[None, :, :]) ** 2).sum(-1) / (2 * c2)
    expected = (logits - logits.max(dim=1, keepdim=True).values >= -tau).sum(dim=1)
    assert torch.equal(flow.retained_counts(x, 0.6, tau=tau), expected)


def test_retained_counts_work_with_pruning_switched_off():
    """The whole point of the tau argument: we run with tau=None, but spec 10.4
    still wants the count logged per timestep."""
    mu = randn(300, 6, seed=58, scale=4.0)
    x = randn(4, 6, seed=59, scale=4.0)
    off = MarginalFlow(mu, 0.2)                       # tau=None, no pruning
    on = MarginalFlow(mu, 0.2, tau=30.0)
    assert torch.equal(off.retained_counts(x, 0.4, tau=30.0),
                       on.retained_counts(x, 0.4, tau=30.0))
    assert torch.all(off.retained_counts(x, 0.4, tau=30.0) < 300)


def test_dropped_mass_bound_is_n_times_exp_minus_tau():
    mu = randn(1000, 6, seed=60)
    assert MarginalFlow(mu, 0.3).dropped_mass_bound() == 0.0
    assert MarginalFlow(mu, 0.3, tau=30.0).dropped_mass_bound() == pytest.approx(
        1000 * math.exp(-30.0))
    # explicit tau overrides the object's, so it works with pruning off
    assert MarginalFlow(mu, 0.3).dropped_mass_bound(tau=30.0) == pytest.approx(
        1000 * math.exp(-30.0))


# --------------------------------------------------------------------------- #
# precomputed sq, responsibilities, TF32
# --------------------------------------------------------------------------- #

def test_precomputed_sq_gives_identical_results():
    """load_centers already builds ||mu_i||^2; reusing it must not change a thing."""
    mu = randn(300, 6, seed=61, scale=3.0)
    x = randn(6, 6, seed=62, scale=3.0)
    sq = (mu ** 2).sum(dim=1)
    for t in (0.2, 0.85):
        assert torch.equal(MarginalFlow(mu, 0.25, chunk=64, sq=sq).velocity(x, t),
                           MarginalFlow(mu, 0.25, chunk=64).velocity(x, t))


def test_bad_sq_is_rejected_rather_than_silently_used():
    mu = randn(50, 6, seed=63)
    with pytest.raises(ValueError):
        MarginalFlow(mu, 0.3, sq=torch.ones(49, dtype=torch.float64))
    with pytest.raises(TypeError):
        MarginalFlow(mu, 0.3, sq=torch.ones(50, dtype=torch.float32))


@pytest.mark.parametrize("t", [0.0, 0.4, 0.9])
def test_responsibilities_sum_to_one_and_match_the_naive_softmax(t):
    mu = randn(200, 6, seed=64, scale=3.0)
    x = randn(5, 6, seed=65, scale=3.0)
    r = MarginalFlow(mu, 0.2).responsibilities(x, t)

    c2 = schedule.c_t_sq(t, 0.2)
    logits = -((x[:, None, :] - t * mu[None, :, :]) ** 2).sum(-1) / (2 * c2)
    assert torch.allclose(r, torch.softmax(logits, dim=1), rtol=1e-10, atol=1e-10)
    assert torch.allclose(r.sum(dim=1), torch.ones(5, dtype=torch.float64))


def test_responsibilities_refuse_to_blow_up_memory():
    flow = MarginalFlow(randn(100_000, 6, seed=66), 0.3)
    with pytest.raises(ValueError, match="exceeds the"):
        flow.responsibilities(randn(1000, 6, seed=67), 0.5)


def test_tf32_flag_is_inert_on_cpu():
    """Centres on CPU: the constructor must not touch global CUDA state."""
    before = torch.backends.cuda.matmul.allow_tf32
    MarginalFlow(randn(20, 6, seed=68), 0.3, allow_tf32=True)
    assert torch.backends.cuda.matmul.allow_tf32 == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU on this machine")
@pytest.mark.parametrize("flag", [False, True])
def test_tf32_flag_is_applied_on_cuda(flag):
    mu = randn(20, 6, seed=69).float().cuda()
    MarginalFlow(mu, 0.3, allow_tf32=flag)
    assert torch.backends.cuda.matmul.allow_tf32 is flag
