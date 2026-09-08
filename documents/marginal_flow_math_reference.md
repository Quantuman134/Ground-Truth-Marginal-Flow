# Phase 6 — Marginal Flow: Implementation Plan

The analytic marginal velocity field induced by the shared-sigma GMM (spec §3).
Every trajectory in the experiment is integrated through this field, and it is the
one place where a bug produces a smooth, plausible, completely meaningless curve.
Nothing downstream can detect that; only the tests in §7 can.

Deliverables: `gtmf/data.py` (centers loader), `gtmf/marginal_flow.py` (the field),
`tests/test_data.py`, `tests/test_marginal_flow.py`.

---

## 1. What is being computed

Target distribution, all components sharing one isotropic sigma and equal weight:

$$
p_1(y) \;=\; \frac{1}{N}\sum_{i=1}^{N}\mathcal{N}\!\left(y;\ \mu_i,\ \sigma^2 I\right)
$$

Conditional path $x_t = (1-t)x_0 + t\,y$ with $x_0 \sim \mathcal{N}(0,I)$ gives the
per-component time-$t$ marginal $x_t \mid i \sim \mathcal{N}(t\mu_i,\ c_t^2 I)$ with

$$
c_t^2 = (1-t)^2 + t^2\sigma^2
\qquad\text{(Eq. 9)}
$$

$$
k_t = \frac{t\sigma^2 - (1-t)}{c_t^2}
\qquad\text{(Eq. 16 coefficient)}
$$

Responsibilities and the component velocity:

$$
r_i(x,t) = \frac{\mathcal{N}(x;\ t\mu_i,\ c_t^2 I)}
                 {\sum_{j}\mathcal{N}(x;\ t\mu_j,\ c_t^2 I)}
\qquad\text{(Eq. 10)}
$$

$$
v_i(x,t) = \mu_i + k_t\,(x - t\mu_i)
\qquad\text{(Eq. 16)}
$$

$$
v(x,t) = \sum_{i=1}^{N} r_i(x,t)\, v_i(x,t)
\qquad\text{(Eq. 17)}
$$

$c_t$ and $k_t$ come from `gtmf/schedule.py` (phase 1) — never re-derived here, so
the single-Gaussian path and the mixture path cannot drift apart.

## 2. The sum collapses — no per-component velocity is ever built

Substituting Eq. 16 into Eq. 17 and using $\sum_i r_i = 1$:

$$
\begin{aligned}
v(x,t) &= \sum_i r_i\left[\mu_i + k_t\,(x - t\mu_i)\right] \\[2pt]
       &= \left(1 - k_t t\right)\bar\mu(x,t) \;+\; k_t\,x,
\qquad
\bar\mu(x,t) \equiv \sum_i r_i(x,t)\,\mu_i
\end{aligned}
$$

So the whole field needs only the **responsibility-weighted mean of the centers**.
An $(N, d)$ per-component velocity tensor is never materialised — at
$N = 1.28\,\mathrm{M}$ and $d = 256$ that would be 1.3 GB **per query point**.

Two matmuls per evaluation: `X @ MU.T` for the scores, and `P @ MU` for $\bar\mu$.

## 3. The query norm cancels

$$
\lVert x - t\mu_i\rVert^2
= \lVert x\rVert^2 - 2t\,\langle x, \mu_i\rangle + t^2\lVert\mu_i\rVert^2
$$

$\lVert x\rVert^2$ is identical across $i$, so it cancels in the softmax.
Precompute $\mathrm{SQ}_i = \lVert\mu_i\rVert^2$ once ($N$ floats, 5 MB) and reuse
it for the entire run:

$$
\ell_i \;=\; \frac{t\,\langle x, \mu_i\rangle - \tfrac12 t^2\,\mathrm{SQ}_i}{c_t^2}
$$

This is the *relative* log score of §8.1 — the Gaussian normalisation constant is
common to all components and cancels too, so it is never computed.

## 4. Streaming (online) log-sum-exp

The score matrix is $(B, N)$ and the centers are 1.3 GB in fp32, so both are
chunked over components. **The maximum must be global.** Normalising per chunk is
silently wrong and produces a plausible curve — spec §8.1 calls this out
explicitly.

Two passes over 1.3 GB per evaluation is avoidable: carry running state and rescale
when a chunk raises the max. Per query row, maintain $m$ (running max, shape $(B,)$),
$s$ (running sum, $(B,)$) and $\mathrm{acc}$ (running weighted center sum, $(B,d)$).
The recurrence, for each chunk $C$:

$$
\begin{aligned}
m' &= \max\!\left(m,\ \max_{i \in C} \ell_i\right), &
s  &\leftarrow s\,e^{\,m - m'} + \sum_{i \in C} e^{\,\ell_i - m'}, \\[2pt]
\mathrm{acc} &\leftarrow \mathrm{acc}\,e^{\,m - m'}
              + \sum_{i \in C} e^{\,\ell_i - m'}\mu_i, &
m &\leftarrow m'
\end{aligned}
$$

then $\bar\mu = \mathrm{acc}/s$. In code:

```
for each chunk C:
    S      = (t/c2) * X @ MU[C].T - (t*t/(2*c2)) * SQ[C]        # (B, |C|)
    m_new  = maximum(m, S.max(dim=1))
    scale  = exp(m - m_new)                                      # (B,)
    P      = exp(S - m_new[:, None])                             # (B, |C|), in place
    s      = s * scale + P.sum(dim=1)
    acc    = acc * scale[:, None] + P @ MU[C]
    m      = m_new
mu_bar = acc / s[:, None]
v      = (1 - k*t) * mu_bar + k * X
```

$m$ initialises to $-\infty$; guard $e^{-\infty - (-\infty)}$ by special-casing the
first chunk. This is the standard online-softmax recurrence and is exact up to fp32
rounding, independent of chunk size — which §7 tests directly.

## 5. Pruning, and what it is actually for

$\tau = 30$ cutoff, applied only after the global max is known, then renormalise.
Report the worst-case dropped mass $N e^{-\tau} = 3.8\times10^{-8}$ at
$N = 1.28\,\mathrm{M}$, and the retained-component count per $t$.

**Be clear about its role.** The spec motivates $\tau$ as an underflow safeguard,
but the online log-sum-exp above already removes that hazard — the largest
exponential is exactly $1$ by construction. Here $\tau$ only saves work in the
$\mathrm{acc}$ accumulation, and only if survivors are compacted. Keep it
configurable, verify that raising it to $60$ does not move the curve, and **do not
let it become load-bearing**: correctness must not depend on the cutoff.

The measured structure in CLAUDE.md says survivors number $\sim\!10^3$ at $t = 0.3$
and exactly $1$ for $t \ge 0.7$, so a top-$k$ compaction would help — but it is an
optimisation to add *after* the exact path passes its gates, guarded by a test that
compacted and uncompacted results agree.

## 6. Precision: disable TF32 for the score matmul

This deserves care, because the scores are a difference of large numbers divided by
a small one. At $\sigma = 0.01$, $t = 0.98$: $c_t^2 = 4.96\times10^{-4}$,
$t\langle x,\mu_i\rangle \approx 90$, $\tfrac12 t^2\mathrm{SQ}_i \approx 45$, so
$\ell_i \approx 9\times10^4$ — while the *meaningful gaps* between components are
$\sim\!10^4$, the $\lVert\Delta\mu\rVert^2 / 2c_t^2$ scale.

- **fp16 inputs are unusable** — $\ell_i$ overflows fp16's 65504 range outright.
- **TF32** rounds mantissas to 10 bits: $\sim\!10^{-3}$ relative error on
  $\langle x,\mu_i\rangle$, so $\sim\!0.1$ absolute, so $\sim\!200$ in $\ell$ units
  against gaps of $\sim\!10^4$. A 2% perturbation of the responsibility exponent —
  too close to mattering.
- **Full fp32** gives $\sim\!10^{-5}$ absolute on the dot product,
  $\sim\!0.02$ in $\ell$ units.

So set `torch.backends.cuda.matmul.allow_tf32 = False` for this matmul. It costs
throughput (67 vs ~495 TFLOPS) but §8 shows the run is minutes either way, so
precision wins outright. Make it a config flag and have the test in §7 bound the
error against an fp64 reference.

Centers are stored fp16 on disk (656 MB) and **upcast once to fp32 on the GPU**
(1.31 GB) at load — not per evaluation.

## 7. Tests — the oracle list

Requirement 4 demands each test assert against something known independently. This
field is unusually well supplied with exact oracles; use them all.

### Exact closed forms

| # | Case | Expected | Why it is exact |
|---|---|---|---|
| 1 | $N=1$, $\mu = 0$ | $v = k_t\,x$ | The phase-4 single-Gaussian field. **Run this first.** |
| 2 | $N=1$, $\mu = \mu_0$ | $v = \mu_0 + k_t(x - t\mu_0)$ | Eq. 16 with $r_1 = 1$ |
| 3 | $t = 0$, any $N$ | $v = \bar\mu_{\text{global}} - x$ | At $t=0$ every score is $0$, so $r$ is uniform and $\bar\mu$ is the global mean. Independently, $\mathbb{E}[y - x_0 \mid x_0 = x] = \mathbb{E}[y] - x$ |
| 4 | $t = 1$, any $N$ | $v = x$ | $k_1 = 1$ so $(1 - k_1 t) = 0$. Independently, $x_1 = y$ and $x_0$ is independent of it, so $\mathbb{E}[y - x_0 \mid x_1 = x] = x$ |
| 5 | $N=2$, $x$ at the midpoint $\tfrac{t}{2}(\mu_1+\mu_2)$ | $r = (\tfrac12,\tfrac12)$, $\bar\mu = \tfrac12(\mu_1+\mu_2)$ | Symmetry |

Cases 3 and 4 are the strongest routine tests: they hold for *any* $N$ and any
centers, so they can run against the real 1.28 M-center array, not just toys.

### Reference implementation

| # | Check |
|---|---|
| 6 | At $N = 64$, $d = 8$: a naive float64 loop computing $r_i$ and $\sum_i r_i v_i$ term by term must match the chunked fp32 path to $\sim\!10^{-5}$ relative. Tests §2's algebra and §4's recurrence together. |
| 7 | Same reference, but computing distances directly as $\lVert x - t\mu\rVert^2$ with no expansion — bounds the §3/§6 cancellation error. |

### Invariants

| # | Invariant |
|---|---|
| 8 | $\sum_i r_i = 1$ to fp32 tolerance, at every $t$ including the extremes |
| 9 | $\bar\mu$ lies within the coordinate-wise convex hull of the centers |
| 10 | **Chunk invariance** — several chunk sizes (and one single-chunk run) agree to fp32 tolerance |
| 11 | **Translation equivariance** — $\mu_i \mapsto \mu_i + c$ with $x \mapsto x + tc$ leaves $r$ unchanged and gives $v \mapsto v + c$ |
| 12 | **Scale equivariance** — $\mu \mapsto \lambda\mu$, $x \mapsto \lambda x$, $\sigma \mapsto \lambda\sigma$ gives $v \mapsto \lambda v$ |
| 13 | **Pruning is not load-bearing** — $\tau = 30$, $\tau = 60$ and no pruning all agree |
| 14 | Perplexity of $r$ reproduces the measured table in CLAUDE.md ($\sim\!10^3$ at $t=0.3$, $\approx 1$ at $t \ge 0.7$) — a regression guard on the whole path |

### Sampling (§6e)

| # | Check |
|---|---|
| 15 | $x_t$ sample mean approaches $t\,\bar\mu_{\text{global}}$; per-coordinate variance approaches $c_t^2 + t^2\operatorname{Var}(\mu)$ |
| 16 | Empirical $\rho_t$ matches the closed form $\sqrt{c_t^2 + t^2\,\mathbb{E}\lVert\mu\rVert^2/d}$ |
| 17 | Same seed reproduces the same states; different seeds do not |

Sampling draws $i$ uniform, $y = \mu_i + \sigma\varepsilon_y$,
$x_0 \sim \mathcal{N}(0,I)$, $x_t = (1-t)x_0 + t y$ (Eqs. 20–21). This is exactly
$\mathcal{N}(t\mu_i,\ c_t^2 I)$, so the one-step form is equivalent and cheaper; the two-step form is kept for auditability
against the spec, and test 15 checks they agree in distribution.

## 8. Compute budget — **corrected**

Per batched evaluation with $B = M(K+1) = 2560$ rows:

$$
\underbrace{2BdN}_{X\,\mu^\top} + \underbrace{2BNd}_{P\,\mu}
= 4BNd
= 4 \cdot 2560 \cdot 1{,}281{,}167 \cdot 256
\approx 3.36\times10^{12}\ \text{FLOPs}
$$

The whole production sweep is $72{,}192$ batched evaluations (the two-tier
step-size figure, all four $\sigma$, 50 $t$ points):

$$
72{,}192 \times 3.36\times10^{12} \;\approx\; 2.4\times10^{17}\ \text{FLOPs}
$$

At fp32 (TF32 off), 8 $\times$ H200 at 67 TFLOPS peak and $\sim$40% achieved gives
$\sim\!2\times10^{14}$ FLOP/s, so **roughly 20 minutes**. With TF32 on it would be
minutes. Add the $(B,N)$ exponential — $2.4\times10^{14}$ of them — and memory
traffic, and a realistic figure is **tens of minutes, not days**.

> **This supersedes the ~16 h / ~50 h estimates recorded earlier in CLAUDE.md,
> which were wrong by about 1000x through an arithmetic slip.** The correct
> conclusion is the opposite of what those numbers implied: compute is *not* the
> binding constraint, so `M = 512` is far more conservative than it needs to be.
> Monte-Carlo error is the real limit — raising
> $M$ to 16,384 costs $32\times$ (still hours at most) and cuts the standard error
> $5.7\times$ (it falls as $1/\sqrt{M}$).
> Revisit `M` and `K` in phase 8 with this budget, and treat the step-size cost
> comparison as a ratio, not an absolute.

Peak memory is the $(B, N)$ score chunk: at $B = 2560$ with a 262,144-column chunk
that is 2.7 GB in fp32, against 140 GB available. A single full-$N$ chunk would be
13 GB — also feasible, but chunking stays configurable so $B$ can grow.

## 9. API surface

```python
# gtmf/data.py
load_centers(path, apply_sd_scale=True, device="cuda", dtype=torch.float32)
    -> (MU (N,d), SQ (N,), meta)      # mmap, upcast once, precompute ||mu_i||^2

# gtmf/marginal_flow.py
class MarginalFlow:
    def __init__(self, MU, SQ, sigma, tau=30.0, chunk=262_144, allow_tf32=False)
    def velocity(self, x, t) -> (B, d)                  # the field, Eq. 17
    def log_scores(self, x, t, lo, hi) -> (B, hi-lo)    # exposed so §7 can test it
    def responsibilities(self, x, t) -> (B, N)          # diagnostics only, small N
    def sample_query_states(self, t, m, generator) -> (m, d)
    def rho_t(self, x_t) -> float                       # Eq. 34, empirical
    def stats(self) -> dict                             # retained counts, dropped-mass bound
```

`velocity` must be the *only* entry point the integrator uses, and it must be pure:
no cached state that depends on `t`, so the RK4 stages cannot contaminate each
other.

## 10. Gates before phase 7

1. Oracles 1–5 pass, cases 3 and 4 against the **real** 1.28 M centers.
2. Reference implementation agreement (6, 7) at small N.
3. Chunk invariance (10) across at least three chunk sizes including single-chunk.
4. Pruning shown not to be load-bearing (13).
5. Perplexity regression (14) matches the values already measured in CLAUDE.md.
6. One timed batched evaluation at production `B` and `N`, recorded — this is the
   number that replaces every remaining cost estimate.

## 11. Failure modes to watch

- **Per-chunk normalisation** instead of a global max. Produces a smooth, wrong
  curve. Caught only by test 10.
- **Reusing the conditional interpolation to propagate.** Not in this module, but
  the field is what makes the mistake invisible — see the Critical Invariant in
  CLAUDE.md.
- **TF32 left on.** Degrades responsibilities near $t \to 1$ where $c_t^2$ is
  smallest; would show as a subtly wrong curve at exactly the sigma we care most
  about.
- **Pruning becoming load-bearing.** If $\tau = 60$ moves the curve, the cutoff is
  discarding real mass, not numerical noise.
- **$\sum_i r_i$ drifting from 1** after rescaling — the online recurrence is easy
  to get subtly wrong; test 8 at every `t`.
