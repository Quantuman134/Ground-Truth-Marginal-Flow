# Marginal Flow — Step-by-Step Build Order

Seven small steps. Each one is a short, readable function you can check against the
equations by eye, and each is verified against the step before it.

The idea: **step 1 is the honest, obvious translation of the maths.** It is slow
and it only works for a few thousand centers, but anyone can read it beside the
spec and confirm it is right. Every later step is a *refactor that must not change
the answer* — so the test for step $k$ is "agrees with step $k-1$", and correctness
propagates forward from a version you personally verified.

That is why the fast version can be trusted at the end: not because it looks
careful, but because a chain of equality tests ties it back to the readable one.

## Style for this project

This is experiment code, not a library.

- Plain functions taking tensors. One small class at the end, only to avoid passing
  `mu, sq, sigma` around everywhere.
- Comments name the equation being implemented (`# Eq. 16`), not the obvious.
- No defensive validation beyond what catches a real mistake, no error-message
  polish, no diagnostics machinery.
- **No performance tricks before step 5.** In-place ops, buffer reuse and fused
  kernels get added only where a measurement says they are needed, and only after
  the readable version exists to compare against.

Each step: I write it, run its test, and show you the diff. You review before the
next one.

---

## Step 1 — `velocity_naive`: the equations, written out

The literal translation. Build every responsibility, build every component
velocity, sum them.

```python
def velocity_naive(mu, sigma, x, t):
    c2 = (1 - t)**2 + t*t*sigma*sigma          # Eq. 9
    k  = (t*sigma*sigma - (1 - t)) / c2        # Eq. 16
    diff = x[:, None, :] - t * mu[None, :, :]  # (B, N, d)
    logits = -(diff**2).sum(-1) / (2*c2)       # Eq. 10, unnormalised
    r = torch.softmax(logits, dim=1)           # Eq. 10
    v_i = mu[None, :, :] + k * diff            # Eq. 16
    return (r[:, :, None] * v_i).sum(dim=1)    # Eq. 17
```

Memory is $O(BNd)$, so this is a toy-size function forever. That is fine — it
becomes the oracle.

**Tests:** $N=1$ gives $k_t x$ · $t=0$ gives $\mathbb{E}[y]-x$ · $t=1$ gives $x$ ·
two symmetric centers split $\tfrac12/\tfrac12$.

## Step 2 — subtract the max before exponentiating

`torch.softmax` already does this internally, so step 2 makes it explicit in *our*
code, because from step 4 onward we compute the exponentials ourselves and the
shift stops being free.

**Test:** identical to step 1 on ordinary data, and still finite where a naive
`exp` would overflow (centers pushed 50× apart).

## Step 3 — collapse the sum

Since $\sum_i r_i = 1$, substituting Eq. 16 into Eq. 17 gives

$$
v(x,t) = (1 - k_t t)\,\bar\mu + k_t\,x, \qquad \bar\mu = \sum_i r_i \mu_i
$$

so only the responsibility-weighted **mean** of the centers is needed. This is what
removes the $(B,N,d)$ tensor — 1.3 GB per query point at production size.

**Test:** exact agreement with step 2. This is the step where a sign or factor
error is most likely, and the test catches it directly.

## Step 4 — expand the squared distance

$\lVert x - t\mu_i\rVert^2 = \lVert x\rVert^2 - 2t\langle x,\mu_i\rangle + t^2\lVert\mu_i\rVert^2$,
and $\lVert x\rVert^2$ is the same for every component so it cancels in the softmax:

$$
\ell_i = \frac{t\langle x,\mu_i\rangle - \tfrac12 t^2\lVert\mu_i\rVert^2}{c_t^2}
$$

The distance computation becomes one matrix multiply plus a precomputed
$\lVert\mu_i\rVert^2$. Memory drops to $O(BN)$.

**Test:** agreement with step 3, including with far-apart centers where the
expansion could lose precision to cancellation.

## Step 5 — chunk over components

$(B,N)$ is still too big at $N = 1.28\,\mathrm{M}$ with a large batch, so process
the centers in blocks, carrying a running max, running sum, and running weighted
sum. **The max must be over all components, not per block** — a per-block max is
the one bug in this whole module that produces a smooth, plausible, wrong curve.

**Test:** several chunk sizes (including one chunk, and a size that does not divide
$N$) all agree with step 4.

## Step 6 — sampling and $\rho_t$

Draw query states from $p_t$ (Eqs. 20–21) and measure the marginal RMS scale
$\rho_t = \sqrt{\mathbb{E}[\lVert x_t\rVert^2/d]}$ that the perturbation is sized
against. Wrap `mu`, `sq`, `sigma` in a small class so the pieces stop being passed
around by hand.

**Test:** sample mean $\to t\bar\mu$, per-coordinate variance
$\to c_t^2 + t^2\operatorname{Var}(\mu)$, and empirical $\rho_t$ against its closed
form.

## Step 7 — pruning, and only if we want it

Drop components whose responsibility is below $e^{-\tau}$ (spec §8.2).

Worth saying plainly: after step 5 the running max already prevents underflow, and
in a dense implementation pruning saves no compute — the matmul runs over every
component regardless. So this step is for spec compliance and as a knob, not for
speed. **Test: $\tau \in \{30, 60, \infty\}$ must all agree**, i.e. the cutoff must
not affect the answer.

If that is the whole benefit, it is reasonable to skip step 7 and record why.

---

## Where the earlier attempt went wrong

The first version jumped straight to the step-5 form with pruning, buffer reuse and
diagnostics folded in. Every individual piece was defensible, but the result was a
60-line function doing six things at once, and there was no readable version to
check it against — only tests, which tell you *that* it agrees with the maths, not
*why*.

This order fixes that: the readable version is kept, and it stays the reference.

## Verification summary

| Step | Verified against |
|---|---|
| 1 | closed forms and hand-checkable symmetry |
| 2 | step 1, plus an overflow case step 1 fails |
| 3 | step 2, exactly |
| 4 | step 3, including a precision-stress case |
| 5 | step 4, across chunk sizes |
| 6 | analytic moments of $p_t$ |
| 7 | itself at three values of $\tau$ |

Steps 1–4 stay in the codebase. They cost nothing to keep, they are the oracles the
fast path is tested against, and they are what makes the module reviewable a month
from now.
