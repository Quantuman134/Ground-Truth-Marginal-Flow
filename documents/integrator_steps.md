# RK4 Integrator — Step-by-Step Build Order

Phase 2. Three small code steps, then the verification that spans them.

Deliverable: `gtmf/integrate.py`, `tests/test_integrate.py`.

The integrator is generic over the field — it takes a `(x, t) -> v` callable and
knows nothing else. That is what lets one integrator serve both the single-Gaussian
sanity check (phase 4) and the real mixture, and it is what makes the exact test
below possible at all.

## The oracle this part is lucky to have

For the single-Gaussian field `v(x, t) = k_t · x`, the ODE is linear and solves in
closed form. Since `k_t = ½ d/dt log c_t²`,

$$
\frac{dx}{ds} = k_s\,x
\quad\Longrightarrow\quad
x(t_1) = \frac{c_{t_1}}{c_{t_0}}\; x(t_0)
$$

So we know the exact answer for any `t0`, `t1` and any starting point, from
`schedule.py` alone. Everything below is checked against that, not against another
integrator.

---

## Step 1 — `time_grid(t0, t1, h)`

The step-count policy, on its own, before either integrator exists.

`(t1 - t0)/h` does not divide evenly, so per CLAUDE.md: take `S = ceil((t1-t0)/h)`
and then shrink the step to `(t1-t0)/S`, giving **uniform steps within a
trajectory** rather than a short final one. Returns `(S, actual_h)`.

**Tests:** the steps land exactly on `t1` · `actual_h <= h` always · `S` is the
smallest count that achieves that · a `t0 == t1` interval is zero steps · both
integrators later share this, so the policy is defined once.

## Step 2 — `euler(field, x, t0, t1, h)`

One field evaluation per step, `x += h·v(x, t)`. Not for production — it is here as
the thing RK4 is measured against, and because a first-order method is the clearest
possible statement of what "order" means.

**Tests:** matches `c_t1/c_t0` as `h` shrinks · observed convergence order ≈ 1.

## Step 3 — `rk4(field, x, t0, t1, h)`

Four evaluations per step: the slope at the start, twice at the midpoint, once at
the end, combined as `(k1 + 2k2 + 2k3 + k4)/6`.

**Tests:** matches `c_t1/c_t0` · observed convergence order ≈ 4 · beats Euler at
matched *evaluation* budget, which is the comparison that decided the solver
choice in the first place.

## Verification spanning all three

| Check | Why it can fail |
|---|---|
| Exact against `c_t1/c_t0` | wrong stage weights, wrong step size, wrong direction |
| **Observed order** from the error-vs-`h` slope: 1 for Euler, 4 for RK4 | a mis-weighted RK4 silently degrades to order 2 and still looks convergent |
| **Batched == individual** — integrating a batch gives the same rows as integrating each alone | spec 8.5 requires the perturbed and unperturbed trajectories to see identical arithmetic; any per-row decision breaks it |
| **Composition** — `t0→t1` then `t1→t2` matches `t0→t2` | holds for the real GMM field too, where no closed form exists |
| Euler and RK4 agree as `h → 0` | both converge to the same solution, so a systematic disagreement means one is solving a different ODE |

The order measurement is the one that matters most. An RK4 with a mistyped
coefficient still converges, still looks smooth, and is simply less accurate than
it should be — only the slope reveals it.

## Not in this phase

No adaptive stepping, no dense output, no trajectory history: the estimator needs
endpoints only, so nothing accumulates across steps. No `torchdiffeq` — we need no
gradients, and both trajectories must share an identical grid, which an adaptive
solver actively fights.
