"""Marginal velocity field of the shared-sigma GMM.

Built in steps -- see documents/marginal_flow_steps.md. Each version is a refactor
of the one before it and must give the same answer.

velocity_naive is the equations written out literally -- slow, toy-sized, and
readable. MarginalFlow.velocity computes the same thing in a form that fits at
N = 1.28M. The tests check the two against each other directly.
"""

import math

import torch

from . import schedule
from .rng import check_generator


def velocity_naive(mu, sigma, x, t):
    """Marginal velocity v(x, t). Literal translation of spec Eqs. 9, 10, 16, 17.

        mu    (N, d)  component centers
        sigma float   the one shared component std
        x     (B, d)  query states
        t     float   time in [0, 1]
        ->    (B, d)

    Builds every responsibility and every component velocity explicitly. That
    needs a (B, N, d) tensor, so this is a toy-size function forever -- at
    N = 1.28M it would want 1.3 GB per query point. It stays in the file as the
    readable reference that the faster versions are tested against.
    """
    c2 = schedule.c_t_sq(t, sigma)               # Eq. 9
    k = schedule.k_t(t, sigma)                   # Eq. 16, the velocity coefficient

    # x - t*mu_i for every (query, component) pair.
    diff = x[:, None, :] - t * mu[None, :, :]                  # (B, N, d)

    # Eq. 10. Every component has the same variance c2 and the same prior weight
    # 1/N, so the Gaussian normalisation and the prior both cancel in the softmax
    # and only the squared distance matters.
    logits = -(diff ** 2).sum(-1) / (2 * c2)                   # (B, N)
    r = torch.softmax(logits, dim=1)                           # (B, N)

    v_per_component = mu[None, :, :] + k * diff                # (B, N, d)  Eq. 16
    return (r[:, :, None] * v_per_component).sum(dim=1)        # (B, d)     Eq. 17


class MarginalFlow:
    """Step 6: the centres, their squared norms and sigma, kept together.

    `velocity` is the field the experiment actually runs -- the only form that
    fits at N = 1.28M. It lives here rather than as a loose function because it
    needs mu, sigma, sq and chunk together, and sq is computed once at
    construction instead of on every call.

    Pruning (`tau`) defaults to off. It was meant to stop the exponentials
    underflowing, which the max-shift already prevents; it saves no compute,
    since the matmul covers every component either way; and past t ~ 0.7 one
    component holds essentially all the mass, so there is nothing to drop. It is
    implemented and configurable for spec compliance, and proven by test not to
    change the answer at tau = 30.

    Also holds the two pieces that belong to the distribution rather than the
    field: drawing query states from p_t, and the scale those states have.
    """

    def __init__(self, mu, sigma, chunk=65536, tau=None, sq=None, allow_tf32=False):
        self.mu = mu                                 # (N, d) component centres
        self.sigma = sigma
        self.chunk = chunk
        self.tau = tau                               # None = no pruning

        # load_centers already computed these; passing them in avoids a second
        # pass over 1.28M x 256 and keeps one definition of ||mu_i||^2.
        self.sq = (mu ** 2).sum(dim=1) if sq is None else sq
        if self.sq.shape != (mu.shape[0],):
            raise ValueError(f"sq must be ({mu.shape[0]},), got {tuple(self.sq.shape)}")
        if self.sq.dtype != mu.dtype:
            raise TypeError(f"sq dtype {self.sq.dtype} != centres dtype {mu.dtype}")

        # TF32 rounds mantissas to 10 bits. The scores here are a difference of
        # large numbers divided by a small c_t^2, so that perturbs the exponent
        # by ~200 against gaps of ~1e4 -- worst exactly where sigma is smallest.
        # This is process-global CUDA state, not per-object.
        self.allow_tf32 = allow_tf32
        if mu.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    @property
    def n(self):
        return self.mu.shape[0]

    @property
    def d(self):
        return self.mu.shape[1]

    def velocity(self, x, t):
        """Marginal velocity v(x, t) -- the right-hand side of the ODE.

        The same quantity velocity_naive computes, rearranged until it fits at
        N = 1.28M. Four exact transformations separate them, and the tests check
        this against velocity_naive directly rather than trusting the chain.

        1. SUBTRACT THE LARGEST SCORE before exponentiating (spec 8.1).
           A per-row constant cancels in the ratio, but it moves the largest
           exponential to exactly 1. Not a nicety: in d = 256 a Gaussian puts its
           mass at radius ~sqrt(d)*c_t, so even the NEAREST component sits that
           far away and the best score is only ever about -d/2 = -128. exp(-128)
           is ~1e-56, which underflows fp32, so without the shift every weight is
           0 and the responsibilities come out 0/0 = NaN. Measured: this happens
           at every sigma and t in our sweep, not just the extremes.

        2. COLLAPSE THE SUM. Every responsibility multiplies the same k, so with
           mu_bar = sum_i r_i mu_i,

               v = sum_i r_i [ mu_i + k (x - t mu_i) ]
                 = mu_bar + k x - k t mu_bar            (sum_i r_i = 1)
                 = (1 - k t) mu_bar + k x

           The per-component velocities never have to be built -- one weighted
           average of the centres is enough. That drops an (N, d) tensor per
           query point, 1.3 GB at production size.

        3. EXPAND THE SQUARED DISTANCE.

               ||x - t mu_i||^2 = ||x||^2 - 2t <x,mu_i> + t^2 ||mu_i||^2

           ||x||^2 is the same for every component, so it shifts a whole row by a
           constant and cancels in the softmax exactly like the shift in (1).
           What is left is one matrix product against a precomputed ||mu_i||^2:

               l_i = ( t <x, mu_i> - 0.5 t^2 ||mu_i||^2 ) / c_t^2

           Memory drops from O(B N d) to O(B N).

        4. CHUNK OVER COMPONENTS. Even (B, N) is too large for a real batch, so
           the components are visited self.chunk at a time and never all held at
           once. That rules out torch.softmax, which needs a whole row to find
           its max and its sum, so the shift of (1) becomes a running quantity.
           Three accumulators are carried, all relative to `m`:

               m    largest score seen SO FAR
               s    sum of exp(score - m) over components seen so far
               acc  sum of exp(score - m) * mu_i over components seen so far

           When a block holds a bigger score, m rises to m_new and everything
           already accumulated is rescaled by exp(m - m_new) < 1 to put it on the
           new footing. At the end acc / s is exactly sum_i r_i mu_i, because the
           common factor exp(-m) cancels between numerator and denominator.

           The maximum must be over every component seen, NOT the maximum within
           each block. Normalising per block is the one bug in this module that
           produces a smooth, plausible, completely wrong curve instead of an
           error -- it is what spec 8.1 warns about, and what the
           chunk-invariance test exists to catch.

        Pruning (self.tau) drops components whose weight is below exp(-tau). It
        is applied against the RUNNING maximum, before the global one is known,
        which is safe in the only direction that matters: the final maximum can
        only be larger, so anything dropped here would also have been dropped by
        the exact rule. It can never discard a component the global criterion
        would have kept. It is off by default -- see the class docstring.
        """
        c2 = schedule.c_t_sq(t, self.sigma)          # Eq. 9
        k = schedule.k_t(t, self.sigma)              # Eq. 16, velocity coefficient

        # Match the inputs: a bare torch.zeros would silently be fp32 and would
        # quietly downcast the whole computation when mu and x are fp64.
        kw = {"dtype": x.dtype, "device": x.device}
        m = torch.full((x.shape[0], 1), float("-inf"), **kw)       # (B, 1)
        s = torch.zeros((x.shape[0], 1), **kw)                     # (B, 1)
        acc = torch.zeros((x.shape[0], self.d), **kw)              # (B, d)

        for lo in range(0, self.n, self.chunk):
            hi = min(lo + self.chunk, self.n)
            block = self.mu[lo:hi]                                 # (C, d)
            logits = (t * (x @ block.T)
                      - 0.5 * t * t * self.sq[lo:hi]) / c2         # (B, C)

            m_new = torch.maximum(m, logits.max(dim=1, keepdim=True).values)
            # First block: m is -inf, so rescale is exp(-inf) = 0 and the empty
            # accumulators are zeroed rather than turning into NaN.
            rescale = (m - m_new).exp()                            # (B, 1)
            w = (logits - m_new).exp()                             # (B, C)
            if self.tau is not None:
                # Step 7, spec 8.2. Multiplying by the bool mask keeps w's dtype;
                # comparing then indexing would need a second temporary.
                w = w * (w >= math.exp(-self.tau))

            s = s * rescale + w.sum(dim=1, keepdim=True)
            acc = acc * rescale + w @ block
            m = m_new

        mu_bar = acc / s                                           # (B, d)
        return (1.0 - k * t) * mu_bar + k * x                      # (B, d)  Eq. 17

    def retained_counts(self, x, t, tau=30.0):
        """Diagnostic: how many components would survive a cutoff of `tau`.

        Spec 10.4 asks for this per timestep. Deliberately independent of
        self.tau, so the count can be logged with pruning switched off -- which
        is how we run, since pruning changes nothing (see the class docstring).

        Two passes -- find the global maximum, then count against it -- because
        unlike `velocity` this is not on the hot path and clarity is worth more.
        """
        c2 = schedule.c_t_sq(t, self.sigma)
        peak = torch.full((x.shape[0], 1), float("-inf"),
                          dtype=x.dtype, device=x.device)
        for lo in range(0, self.n, self.chunk):
            hi = min(lo + self.chunk, self.n)
            logits = (t * (x @ self.mu[lo:hi].T)
                      - 0.5 * t * t * self.sq[lo:hi]) / c2
            peak = torch.maximum(peak, logits.max(dim=1, keepdim=True).values)

        kept = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for lo in range(0, self.n, self.chunk):
            hi = min(lo + self.chunk, self.n)
            logits = (t * (x @ self.mu[lo:hi].T)
                      - 0.5 * t * t * self.sq[lo:hi]) / c2
            kept += (logits - peak >= -tau).sum(dim=1)
        return kept

    def dropped_mass_bound(self, tau=None):
        """Worst-case posterior mass a cutoff can discard: N * exp(-tau).

        The denominator is at least the largest component's weight, which the
        cutoff always keeps, so each dropped component carries at most exp(-tau)
        of it and there are fewer than N of them (spec 8.2).
        """
        tau = self.tau if tau is None else tau
        return 0.0 if tau is None else self.n * math.exp(-tau)

    def responsibilities(self, x, t, max_elements=1 << 24):
        """Dense (B, N) responsibilities. Diagnostics and tests only.

        `velocity` is written specifically to avoid holding this matrix, so it is
        capped -- at N = 1.28M a single query row is already 5 MB.
        """
        if x.shape[0] * self.n > max_elements:
            raise ValueError(f"{x.shape[0]} x {self.n} responsibilities exceeds the "
                             f"{max_elements:,}-element cap; use velocity() instead")
        c2 = schedule.c_t_sq(t, self.sigma)
        logits = (t * (x @ self.mu.T) - 0.5 * t * t * self.sq) / c2
        return torch.softmax(logits, dim=1)

    def sample_query_states(self, t, m, generator=None):
        """Draw m states from the marginal p_t   (spec Eqs. 20-21).

        Pick a component uniformly, draw y from it, draw noise, interpolate:

            i ~ Uniform{1..N},  y = mu_i + sigma * eps,  x_0 ~ N(0, I)
            x_t = (1 - t) x_0 + t y

        This is used ONLY to place query states. It must never be used to carry a
        state to the endpoint -- that is what the marginal ODE is for. See the
        Critical Invariant in CLAUDE.md.

        The draws follow the centres onto their device, so `generator` has to live
        there too -- see gtmf.rng.
        """
        check_generator(generator, self.mu.device, "sample_query_states generator")
        kw = {"dtype": self.mu.dtype, "device": self.mu.device}
        i = torch.randint(self.n, (m,), generator=generator, device=self.mu.device)
        eps = torch.randn((m, self.d), generator=generator, **kw)
        x0 = torch.randn((m, self.d), generator=generator, **kw)
        y = self.mu[i] + self.sigma * eps            # (m, d)  out of place: mu is shared
        return (1.0 - t) * x0 + t * y                # (m, d)

    def rho_t(self, x_t):
        """Marginal RMS scale sqrt(E[||x_t||^2 / d]) measured from a sample (Eq. 34).

        This is the scale the perturbation is sized against, eps = alpha * rho_t.
        Not the same as c_t, which leaves out the spread of the centres.
        """
        return float(x_t.pow(2).mean().sqrt())

    def rho_t_exact(self, t):
        """Closed form of rho_t: sqrt(c_t^2 + t^2 E||mu||^2 / d).

        From x_t = (1-t) x_0 + t (mu_i + sigma eps), whose per-coordinate second
        moment is (1-t)^2 + t^2 sigma^2 + t^2 E[mu_j^2] = c_t^2 + t^2 E||mu||^2/d.
        """
        mean_sq = float(self.sq.mean()) / self.d
        return math.sqrt(schedule.c_t_sq(t, self.sigma) + t * t * mean_sq)
