# w_avg Pipeline — Step-by-Step Build Order

Phase 7. Five steps. Deliverable: `wavg.py`, plus `mean_and_stderr` in
`gtmf/estimator.py` and `build_target` wherever it lands.

Everything it needs already exists and is tested. This phase is wiring, output
layout, and one structural guarantee.

---

## Step 1 — `build_target(cfg, sigma)` -> MarginalFlow

Two sources, chosen by `data.source`:

- **`latents`** -- `load_centers` on the consolidated array, honouring
  `num_target_samples` and `apply_sd_scale`, passing `sq` straight through so it
  is not recomputed.
- **`synthetic`** -- `num_components` centres built in code. With
  `num_components: 1, centers: zeros` the mixture *is* `N(0, sigma^2 I)`, which is
  what lets phase 11 be a config rather than a script.

**Tests:** synthetic N=1 reproduces the closed-form field · the latents path
loads the real array and gets N = 1,281,167 · `sq` is reused, not recomputed ·
device and dtype follow the config · an unknown source raises.

## Step 2 — `mean_and_stderr(w)` in `gtmf/estimator.py`

`w_avg(t)` and its Monte-Carlo uncertainty, spec Eq. 43:

    SE(t) = Std_m[w_hat_m(t)] / sqrt(M)

Lives with the estimator because it is the estimator's own error bar, and phase 9
needs the same formula after its cross-rank reduction.

**Tests:** against a hand-computed case · SE falls as `1/sqrt(M)` · a single
query gives a defined mean and an undefined SE, reported as such rather than as
zero.

## Step 3 — `run_timepoint(flow, cfg, sigma, t, generator)`

The core. Sample `M` query states, size the perturbation, integrate, estimate,
aggregate. Returns the aggregate, the **per-query** values, and the timing.

    rho_t  <- flow.rho_t_exact(t)         (or empirical, per config)
    eps    <- alpha * rho_t
    x_t    <- flow.sample_query_states(t, M, generator)
    w      <- w_hat(flow.velocity, x_t, t, eps, K, h, scheme, integrator)
    ->  w_avg, stderr, w_per_query, rho_t, eps, seconds

**The phase gate lives here.** The critical invariant is that the conditional
interpolation `x_t = (1-t) x_0 + t y` may be used ONLY to place a query state,
never to reach the endpoint. Reaching for it instead of integrating produces a
smooth, plausible, meaningless curve that nothing downstream can detect.

Guaranteed structurally: `sample_query_states` returns states and nothing else,
the only route to an endpoint is `integrator(flow.velocity, ...)`, and the two
share no call path.

**Tested by counting field evaluations.** A correct run calls the velocity field
exactly `4S` times per step of RK4; a shortcut through the interpolant would call
it zero times and still return numbers. Counting is the cheap way to prove the
ODE was actually solved.

Other tests: single-Gaussian target reproduces `sigma^2/c_t^2` · the same seed
reproduces the run · `t = 0` and `t = 1` behave.

## Step 4 — `run_sigma(cfg, sigma, out_dir)`

Walk the time grid, collect, write. Progress with ETA on the way (spec §10.6),
mirrored into the log so it survives an SSH disconnect.

Writes into `sigma_<value>/`:

| file | contents |
|---|---|
| `w_avg.csv` | `t, w_avg, stderr, rho_t, eps, seconds` -- one row per timepoint |
| `raw.npz` | the per-query `w`, `(num_points, M)` -- what phase 10 redraws from |
| `summary.json` | scalars: peak location, totals, timings, the sigma's settings |

Per-timepoint results are written as they complete, so a long run can resume
rather than restart.

**Peak location needs the caveat from CLAUDE.md**: `t_peak = 1/(1+sigma^2)` sits
outside `[0, 0.98]` for sigma <= 0.1, so report *peak not in range* rather than
naming the sampled argmax.

## Step 5 — CLI and the sweep

    python wavg.py --config configs/wavg_imagenet.yaml [--output PATH] [--force]

Create the run directory (Layout B), dump `resolved_config.yaml`, open `run.log`,
then walk `component_sigma` writing one subdirectory each.

**Tests:** the sanity config runs end to end and lands on the closed form · the
output tree matches Layout B · an existing `--output` is refused without
`--force` · the resolved config round-trips from inside the run directory.

---

## What this phase produces that nothing else has

**The first real timing.** Every wall-clock figure in CLAUDE.md is an unverified
estimate; step 3 returns measured seconds per timepoint. On this machine that is
CPU-only and at reduced scale, so it constrains the production number without
settling it -- the cluster still has to confirm.

## Not in this phase

No distribution (phase 9), no figures (phase 10), no convergence sweeps
(phase 8). The pipeline runs one config through to numbers on disk; everything
else reads those numbers.
