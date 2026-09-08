"""Finite-difference estimator for the amplification factor.

Nudge a query state by a small amount, carry both the original and the nudged
state to t = 1 through the marginal ODE, and see how much the gap between them
grew. That ratio is Phi; w is its square, averaged over directions.
"""

import torch

from .integrate import rk4


def perturbed_batch(x, eps, num_probes, generator=None, scheme="one_sided"):
    """Stack the states whose endpoints the estimator needs into ONE tensor.

        x           (B, d)  query states
        eps         float   perturbation size, alpha * rho_t
        num_probes  int     K, random directions per query
        scheme      "one_sided" (Eq. 27) or "central" (Eq. 29)

    Returns (batch, num_blocks) where batch is (num_blocks * B, d), laid out in
    contiguous blocks of B rows so the caller can recover it with

        endpoints.reshape(num_blocks, B, d)

    Block layout:

        one_sided   [ x , x + eps u_1 , ... , x + eps u_K ]        1 + K blocks
        central     [ x + eps u_1..K  , x - eps u_1..K   ]         2K blocks

    Blocks rather than interleaved rows because unpacking is then a reshape, and
    an off-by-one that pairs the wrong perturbation with the wrong query would
    look like noise rather than like a bug.

    Everything goes in one tensor because spec 8.5 requires the perturbed and
    unperturbed trajectories to share an identical time grid -- the integrator
    guarantees that within a batch, and only within a batch.

    Note the central scheme carries no unperturbed row: (F(x+eps u) - F(x-eps u))
    never references F(x), so integrating it would be wasted work.
    """
    if scheme not in ("one_sided", "central"):
        raise ValueError(f"scheme must be 'one_sided' or 'central', got {scheme!r}")
    if num_probes < 1:
        raise ValueError(f"num_probes must be >= 1, got {num_probes}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if x.dim() != 2:
        raise ValueError(f"x must be (B, d), got {tuple(x.shape)}")

    # Follow x: a bare torch.randn would be fp32 on the CPU and would silently
    # downcast an fp64 run, or land the probes on the wrong device.
    u = torch.randn((num_probes, *x.shape), generator=generator,
                    dtype=x.dtype, device=x.device)

    if scheme == "one_sided":
        blocks = torch.cat([x.unsqueeze(0), x + eps * u])          # (1 + K, B, d)
    else:
        # The same u for both signs: the difference has to be along one direction.
        blocks = torch.cat([x + eps * u, x - eps * u])             # (2K, B, d)

    return blocks.reshape(-1, x.shape[1]), blocks.shape[0]


def w_hat(field, x, t, eps, num_probes, h, generator=None,
          scheme="one_sided", integrator=rk4):
    """Per-query amplification w(x_t, t), one value per row of x.   Eq. 27

        field       (x, t) -> v, the marginal velocity
        x           (B, d) query states drawn from p_t
        t           float, the time they were drawn at
        eps         perturbation size; the caller sets it, normally alpha * rho_t
        num_probes  K random directions per query
        h           integrator step size

    Nudge each query in K random directions, carry the original and all the
    nudged copies to t = 1 through the SAME batched solve, and measure how the
    gap grew:

        w = 1 / (K d eps^2) * sum_k || F(x + eps u_k) - F(x) ||^2

    Since E[u u^T] = I, averaging over directions turns that into ||Phi||_F^2 / d
    -- the quantity spec Eq. 3 defines.

    `eps` is passed in rather than derived from alpha here: the caller already
    knows rho_t for this batch, and keeping the perturbation size explicit is what
    makes the alpha sweep a plain loop over eps.

    Two schemes, both estimating the same Phi u:

        one_sided   ( F(x + eps u) - F(x)         ) / eps      Eq. 27, K+1 solves
        central     ( F(x + eps u) - F(x - eps u) ) / (2 eps)  Eq. 29, 2K solves

    One-sided leaves an O(eps) truncation bias; central cancels it and leaves
    O(eps^2), for roughly double the integration cost. Neither difference shows up
    on a linear field, where both are exact -- which is what makes the linear case
    a clean test of the arithmetic and a useless test of the schemes.
    """
    batch, n_blocks = perturbed_batch(x, eps, num_probes, generator, scheme)
    ends = integrator(field, batch, t, 1.0, h).reshape(n_blocks, *x.shape)

    if scheme == "one_sided":
        # ends[0] is F(x); ends[1:] are the K perturbed endpoints.
        gaps = (ends[1:] - ends[0]) / eps                    # (K, B, d) ~ Phi u_k
    else:
        # First K blocks are the + side, last K the - side, probe for probe.
        # The divisor is 2 eps, not eps: the two endpoints straddle x, so the
        # gap between them spans twice the perturbation.
        gaps = (ends[:num_probes] - ends[num_probes:]) / (2.0 * eps)

    return gaps.pow(2).sum(dim=-1).mean(dim=0) / x.shape[1]
