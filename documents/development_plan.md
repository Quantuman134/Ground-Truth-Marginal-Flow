# GTMF Development Plan

Build order for the reference `w_avg(t)` experiment. The authoritative
specification is [reference_wavg_forward_plan_v5.html](reference_wavg_forward_plan_v5.html);
settled parameters live in [../CLAUDE.md](../CLAUDE.md). This file records *what
gets built, in what order, and what must pass before moving on*.

Ordering principle: **every phase is verifiable on its own, and each one gates the
next.** Nothing touches the GPU cluster until phase 10; nothing touches the full
1.28 M-component mixture until phase 8.

Notation follows CLAUDE.md: `c_t` is the per-component scale, `rho_t` the marginal
RMS, `d = 256`, `N` the number of centers.

---

## Phase 0 — Scaffolding and configuration

**Deliverable:** `configs/`, `gtmf/config.py`, repo skeleton.

YAML loader producing a resolved config object, dumped verbatim into every output
directory. Every control in spec §10.2 settable without touching source:
`data.latent_root`, `data.num_target_samples`, `data.apply_sd_scale`,
`gmm.component_sigma` (a **list**), `time.{num_points,t_min,t_max}`,
`monte_carlo.{num_query_states,num_probes,epsilon_alpha,responsibility_log_cutoff}`,
`ode.{step_size,scheme}`, `distributed.num_gpus`, `output.output_dir`, seeds.

`ode.step_size` is a **mapping**, not a scalar — `{0.01: 1/512, default: 1/64}`.

**Gate:** a config round-trips (load → dump → reload → identical); a missing key
fails loudly with the key named, never silently defaults.

---

## Phase 1 — Interpolant schedule and primitives

**Deliverable:** `gtmf/schedule.py`

Pure functions, no state:

```
c_t(t, sigma)      -> sqrt((1-t)^2 + t^2 sigma^2)          # Eq. 9
k_t(t, sigma)      -> (t sigma^2 - (1-t)) / c_t^2          # velocity coefficient, Eq. 16
phi_exact(t, sigma)-> sigma / c_t(t, sigma)                # single-Gaussian amplification
w_exact(t, sigma)  -> sigma^2 / c_t(t, sigma)^2            # §7.1 closed form
t_peak(sigma)      -> 1 / (1 + sigma^2)
```

`k_t` is the whole of the dynamics; deriving it once here means the single-Gaussian
path and the mixture path cannot drift apart.

**Gate:** `k_t` equals `0.5 * d/dt log c_t^2` to numerical tolerance — this identity
is what makes the closed form exact, so it is the real test. Sign of `k_t` flips at
`t_peak`.

---

## Phase 2 — RK4 integrator

**Deliverable:** `gtmf/integrate.py`

```
rk4(field, x, t0, t1, h) -> x1
```

`x` is `(B, d)`; `field(x, t)` returns `(B, d)`. Steps: `S = ceil((t1-t0)/h)`, then
the actual step is `(t1-t0)/S` so steps are uniform *within* a trajectory. The whole
batch shares one time grid — no per-row step decisions, ever.

Generic over `field`, so the single-Gaussian case and the mixture use identical
integration code. Also ship `euler` behind a config switch: not for production, but
so the phase-4 convergence-order test can measure both and confirm 1 and 4.

**Gate:** on `dx/dt = k_t(t) x` the returned amplification matches
`c_t1 / c_t0` to ~1e-6; measured convergence order is ~4 for RK4 and ~1 for Euler.

---

## Phase 3 — Perturbation and the finite-difference estimator

**Deliverable:** `gtmf/estimator.py`

```
rho_t(x_t)                    -> sqrt(mean(||x_t||^2) / d)     # Eq. 34, empirical
w_hat(field, x_t, t, K, alpha, scheme) -> (B,) per-query w
```

Draw `u ~ N(0, I)`, set `eps = alpha * rho_t` with `rho_t` measured from the batch.
Stack the unperturbed state and all `K` perturbed states into **one** `(B*(K+1), d)`
tensor and integrate once (spec §8.5 — shared grid is mandatory, not an
optimisation). Then

```
w_hat = (1 / (K d eps^2)) * sum_k || F(x_t + eps u_k) - F(x_t) ||^2      # Eq. 27
```

Central difference (Eq. 29) as a config switch: `2K+1` rows instead of `K+1`.

**Gate:** on a linear field, `w_hat` is independent of `alpha` across the whole
sweep range (the field is exactly linear, so there is no truncation bias) — any
alpha-dependence here is a bug in the estimator, not physics.

---

## Phase 4 — Single-Gaussian sanity check

**Deliverable:** `sanity_gaussian.py`

Integration test of phases 1–3 against the one case with a closed form (spec §7.1).
No data, no GPU, no distributed setup; runs in seconds. Field is
`v(x,t) = k_t(t) * x` for `p_1 = N(0, sigma^2 I)`.

**Gate:** reproduces `w(t) = sigma^2/c_t^2` across the 50-point grid to the accuracy
budget; peak lands at `t = 1/(1+sigma^2)`; RK4 order ~4. **A failure here is a bug
in the foundation, not in the mixture** — do not proceed past it.

Produces the §9 sanity-check figure (numerical vs analytic, with curve error and
peak-location error reported).

---

## Phase 5 — Consolidated centers file

**Deliverable:** `build_centers.py`, `remote_bash_script/build_centers.sh`

One pass over the 1,281,167 `.pt` files → a single flat fp16 array `(N, 4, 8, 8)`
(~656 MB) plus a row → source-path index, written into `latents_8_mean_fp16/` with
that directory's README updated. Worker pool writing into a preallocated memmap;
rank-0 progress bar with count, percentage, elapsed, ETA and files/s (§10.5).

Stored **raw** — the `0.18215` scale is applied at load time, per config flag.

**Gate:** row count exactly 1,281,167; a random sample of rows byte-matches the
original `.pt` contents; a full read takes seconds against the ~33 min the
small-file tree needs (measured 651 files/s).

Independent of phases 0–4; can run in parallel with them.

---

## Phase 6 — Marginal flow

**Deliverable:** `gtmf/marginal_flow.py`

The heart of the project: the analytic marginal velocity field of spec §3. This is
the field every trajectory in the experiment is integrated through, and the single
place where an error produces a plausible-looking but meaningless curve.

### 6a. Algebraic reduction

Substituting Eq. 16 into Eq. 17 collapses the whole sum. With
`k = k_t(t, sigma)` and `mu_bar(x,t) = sum_i r_i(x,t) mu_i`:

```
v(x,t) = sum_i r_i [ mu_i + k (x - t mu_i) ]
       = (1 - k t) * mu_bar(x,t)  +  k * x
```

So **no per-component velocity is ever materialised.** One needs only the
responsibility-weighted mean of the centers. Two matmuls per field evaluation:
`x @ mu.T` for the scores, and `r @ mu` for `mu_bar`.

### 6b. Scores without the query norm

`||x - t mu_i||^2 = ||x||^2 - 2t (x . mu_i) + t^2 ||mu_i||^2`, and `||x||^2` is
identical across `i`, so it cancels in the softmax. Precompute `||mu_i||^2` once
(`N` floats, reused for the whole run):

```
l_i = ( t (x . mu_i) - 0.5 t^2 ||mu_i||^2 ) / c_t^2
```

### 6c. Streaming (online) log-sum-exp over chunks

`N * d` at fp32 is 1.3 GB and the score matrix is `B x N` — both demand chunking
over components. The global maximum must be **global**; a per-chunk normalisation
is silently wrong (spec §8.1). Use the one-pass online update rather than two
passes over the centers: carry running `m` (max), `s` (sum), `acc` (weighted
center sum); on a new chunk with max `m'`,

```
if m' > m:  s *= exp(m - m');  acc *= exp(m - m');  m = m'
s   += sum(exp(l_chunk - m))
acc += exp(l_chunk - m).T @ mu_chunk
mu_bar = acc / s
```

### 6d. Pruning

`tau = 30` cutoff applied only after the global max is known, then renormalise.
Report the worst-case dropped mass `N * exp(-tau)` and the retained-component
count per `t`.

*Note:* the online log-sum-exp already removes the underflow hazard that motivates
pruning in §8.2, so here `tau` is mainly a compute saver on the `acc` accumulation
and a safeguard. Keep it configurable and verify that raising it does not move the
curve — do not let it become load-bearing.

### 6e. Query-state sampling

`x_t ~ p_t` per Eqs. 20–21: draw `i` uniform, `y = mu_i + sigma * eps_y`,
`x_0 ~ N(0,I)`, `x_t = (1-t) x_0 + t y`. Note this is exactly `N(t mu_i, c_t^2 I)`,
so the direct draw is equivalent and cheaper; either is acceptable, the two-step
form is kept for auditability against the spec.

**Gates — all three must pass:**

1. **Reduction to phase 4.** With `N = 1` and `mu_1 = 0`, the field must equal
   `k_t * x` *exactly*. This is the strongest correctness test available and it
   costs nothing. Run it first.
2. **Chunking invariance.** Responsibilities and `mu_bar` from chunked and
   unchunked evaluation agree to fp32 tolerance, for several chunk sizes.
3. **Responsibility sanity.** `r` sums to 1; `mu_bar` lies inside the convex hull
   of the retained centers; the perplexity of `r` reproduces the measured table in
   CLAUDE.md (~10^3 at `t=0.3`, ~1 at `t>=0.7`).

---

## Phase 7 — Single-GPU `w_avg` pipeline

**Deliverable:** `wavg.py`

Wires phases 3, 6 and the config together at reduced scale (`N ~ 50 k`, small
`M`/`K`): for each `t`, sample `M` query states, run the batched perturbation
estimator through the marginal flow, aggregate to `w_avg(t)` and its standard
error, persist **per-query raw estimates** alongside the aggregate.

**Gate:** the **critical invariant** — conditional interpolation is used only to
draw `x_t`, never to propagate. Assert it structurally (the sampling function and
the integration function share no code path). Curve is finite and smooth. Record
the wall-clock for one `t` point: this is the first real number behind every cost
estimate in CLAUDE.md, which are currently unbenchmarked guesses.

---

## Phase 8 — Convergence studies

**Deliverable:** `sweeps.py`, still at reduced `N`

Three studies, in order:

1. **alpha sweep** (§7.2) over `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`, for **every**
   sigma → the plateau plot. Alpha is then chosen **by hand**, never automatically.
   Watch the low end: at `alpha = 1e-4` the perturbation is close to the fp32 floor,
   and a collapsing plateau there is the signal to escalate that sigma to fp64.
2. **M / K convergence** — vary independently; expect raising `M` to beat raising
   `K` in `d = 256`. Sets both against a measured time budget.
3. **halve-`h` validation** — re-run several `t` points at half the configured step
   and confirm the curve does not move. The two-tier step size rests on the *smooth*
   single-Gaussian field; this is the only test that exercises it across the
   mixture's commitment layer near `t ≈ 0.5–0.7`. **Expect to revise `h` here.**

**Gate:** alpha fixed per sigma, `h` confirmed or revised, `M`/`K` set.

---

## Phase 9 — Distributed production run

**Deliverable:** `remote_bash_script/run_wavg_8xh200.sh`

`torchrun --standalone --nproc_per_node=$NUM_GPUS`, GPU count from the config.
Query states sharded across the 8 H200s with global reductions for the mean and
standard error. Centers are **replicated, not sharded** — 656 MB fits on one GPU,
which removes distributed log-sum-exp from the problem entirely. Full `N`, full
sigma list. Rank-0 progress with stage, ETA and throughput, mirrored periodically
into the log file so it survives an SSH disconnect (§10.6). Per-`t` intermediate
results so a long run resumes without recomputing.

**Gate:** an 8-GPU run reproduces the single-GPU phase-7 curve at matched settings,
within Monte-Carlo error. If it does not, the reduction is wrong — not the physics.

---

## Phase 10 — Outputs and figures

**Deliverable:** `plots.py`

Per §9: raw curve; normalized curve (Eq. 42, integral over the **actual** `[0, 0.98]`
grid, labelled as such); mean ±1.96 SE band; peak location reported and marked; the
phase-4 sanity-check comparison; the phase-8 convergence figures. Overlay the
closed form `sigma^2/((1-t)^2 + sigma^2 t^2)` on every curve — in the committed
regime the two should coincide, so the plot shows directly what the run bought
beyond analysis.

PNG and PDF, with the underlying numbers written as CSV/JSON next to every figure.
Reads only persisted raw estimates, so replotting never triggers a re-run.

---

## Sequencing notes

- **Phases 0–4 and phase 5 are independent** and can proceed in parallel.
  Phases 6 → 10 are strictly sequential.
- **Phase 6 is the one to slow down on.** It is where the project's only silent
  failure mode lives: a wrong field yields a smooth, plausible, meaningless curve.
  Its three gates are cheap; run all of them.
- **`M = 512`, `K = 4` are placeholders**, not decisions — set in phase 8.
- **Every cost estimate in CLAUDE.md is unbenchmarked.** Phase 7 replaces them.
