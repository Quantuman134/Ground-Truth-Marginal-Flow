# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code
in this repository.

## Coding Requirements

0. Before executing any of the instructions below, say in the chat box which one
   you are about to execute. If none apply, say so explicitly.

1. After each large update (adding a function, changing the working logic), execute:
   "Check if this change introduces any logical error. If so, fix it."

2. After each update involving a variable type transform, or that changes the type
   of a variable relative to the original version, check for mistakes.

3. Write experiment outputs to `results/` inside this project — never to `/tmp` or
   another scratch directory.

4. **Every function gets a test that verifies it is implemented correctly.** Write
   the test alongside the function, not later — a function without one is not
   finished. Tests live in `tests/`, mirroring the module layout, and must be
   runnable as a suite.

   A test has to be able to *fail*: assert against an independently known value —
   a closed form, a hand-worked case, a slow-but-obvious reference implementation,
   a limiting case with a known answer, or an invariant the result must satisfy
   (sums to 1, symmetry, correct sign, correct scaling under a change of units).
   Re-running the function and comparing it to itself proves nothing.

   Where a closed form exists, use it: the single-Gaussian case gives exact values
   for the velocity coefficient, the amplification factor and `w(t)`, and the
   marginal flow must reduce to it exactly at `N = 1`. The per-phase gates in
   [documents/development_plan.md](documents/development_plan.md) are the
   integration-level version of this rule, not a substitute for it.

## What This Is

**GTMF (Ground-Truth Marginal Flow).** Computes a *reference* `w_avg(t)` — the
average squared amplification that the marginal flow applies to a small state
perturbation injected at time `t`:

    w_avg(t) = E_{x_t ~ p_t} [ (1/d) ||Phi(1,t)||_F^2 ]

It is a reference, not ground truth: exact only relative to the chosen GMM target,
the induced marginal flow, the ODE solve, and the infinitesimal-perturbation limit.
Purpose is to give Path-Weighted Flow Matching a target curve against which simpler
analytical timestep weights can be judged. No neural network is trained anywhere.

**Full specification:** [documents/reference_wavg_forward_plan_v5.html](documents/reference_wavg_forward_plan_v5.html).
That document is authoritative; this file records decisions made on top of it.

This is a standalone project. It is conceptually related to
`../Path-Weighted_Flow_Matching` (PWFM), but is a clean slate: do not import from
or modify that repo. PWFM contains earlier attempts at this same quantity
(`exp_true_marginal_wavg*.py`, 2026-08-03) — do not consult them unless asked.

## Critical Invariant

The conditional interpolation `x_t = (1-t)x_0 + t*y` is used **only** to sample a
query state from `p_t`. It must never be used to propagate a state to the endpoint.
Both the unperturbed and perturbed trajectories are integrated through the marginal
velocity field `v(x,t)` (Eq. 17). Getting this wrong silently produces a plausible
but meaningless curve.

## Notation

- `t = 0` is the noise side (`p_0 = N(0,I)`), `t = 1` is the data side (`p_1` = GMM).
- `d = 256` (latents are 4x8x8, flattened).
- The spec overloads `s_t` for two different scales. Disambiguated here:
  - **`c_t`** = sqrt((1-t)^2 + t^2*sigma^2) — per-component scale, Eq. 9. Used in
    responsibilities and in the velocity coefficient.
  - **`rho_t`** = sqrt(E[||x_t||^2 / d]) — marginal RMS, Eq. 34. Includes the spread
    of the centers, so `rho_t^2 = c_t^2 + t^2 * E||mu||^2/d`.
- **The perturbation uses `rho_t`**: `eps = alpha * rho_t`, with `rho_t` measured
  empirically from the sampled query states at each `t`. (At t=0.98 with
  sigma=1e-2 these differ by ~27x, so the distinction matters.)

## Decisions

| Item | Decision |
|---|---|
| `component_sigma` | Config takes a **list** for sweeping. Production list `[0.01, 0.1, 0.3, 0.6]`; `0.01` is the anchor/default. Upper bound is set by the data's own per-coord std (0.589) — beyond it components are broader than the data and only a single Gaussian is being measured. |
| N (GMM centers) | All 1,281,167 latents in production; integer override for dev. |
| Finite difference | Implement both one-sided (Eq. 27) and central (Eq. 29); default one-sided. |
| ODE solver | Fixed-step **rk4**, hand-rolled (~15 lines). Not `torchdiffeq` — we need no gradients, and both trajectories must share an identical time grid per Sec. 8.5. |
| M / K | Defaults `M=512`, `K=4`; both config-settable, tune later. |
| Time grid | **Uniform**, 50 points on `[0, 0.98]`. Explicitly not concentrated near the transition. |
| Solver steps | **Fixed step size** `h` (one step advances time by `h`), so `S = ceil((1-t)/h)` varies with `t` — uniform accuracy across the curve, since values are compared across `t`. Validate by halving `h` on a few timesteps. |
| Step size `h` | **Two tiers**, keyed on sigma: `0.01 -> 1/512` (0.001953125), **everything else -> 1/64** (0.015625). Only sigma=0.01 genuinely needs the fine step; the rest sit at 1.3e-6 or better at 1/64. Costs 10% more than a fully per-sigma table and far less config. Do NOT use one global `h`: 1/512 everywhere is 3.2x the cost for accuracy already 7 orders past the target at sigma=0.6, and 1/256 everywhere leaves sigma=0.01 with only 17x margin. |
| Step remainder | `(1-t)/h` does not divide evenly: take `S = ceil((1-t)/h)`, then shrink the step to `(1-t)/S` so steps are uniform *within* a trajectory. Not "keep h exact, add a short final step". |
| Batched solve | The unperturbed trajectory and all `K` perturbed ones are integrated in **one batched solve** sharing an identical time grid — a hard requirement of Sec. 8.5, not an optimisation. |
| Precision | **FP32** for the field evaluation and the ODE, per Sec. 8.5. The two trajectories are nearly identical, so FP32 roundoff is highly correlated and largely cancels in the difference; what survives is differential roundoff, far below the ~6e-8 single-trajectory scale. The alpha sweep is what detects a problem — if the plateau collapses at small alpha, escalate to FP64 for that sigma only. Full FP64 (including the distance computation) costs 2-7x on H100/H200 — FP64 34 TFLOPS vs FP32 67 vs TF32 tensor ~495 — and doubles the centers array to 2.6 GB. FP64 for state accumulation alone is <5% overhead but buys almost nothing: the floor is the FP32 field evaluation, not the accumulation. |
| alpha sweep | Runs for **every** sigma; the chosen alpha per sigma is set by the user in the config, never auto-selected. |
| Sweep execution | One invocation walks the whole sigma list sequentially, one output subdir per sigma. |
| Raw data | Always persist per-query raw estimates, not just aggregates, so plots can be reformatted without re-running. |
| alpha selection | From the Sec. 7.2 sweep, chosen by hand off the plotted plateau. |
| Component sharding | Not needed — all centers fit on one GPU (656 MB). Replicate per rank. |
| Logging | No W&B. CSV/JSON + PNG/PDF per Sec. 9. |
| Normalization | Eq. 42 divides by the integral over the **actual grid** `[0, 0.98]`, not `[0,1]`. Never extrapolate into a region that was not measured; label plots with the real range. |
| Sanity check | Separate `sanity_gaussian.py` — needs no data, no GPU, no distributed setup. |
| Output dir | `results/<experiment>_<sigma>_<timestamp>/`, with the fully resolved config dumped inside. |

### Measured structure of the mixture (why sigma matters the way it does)

Collapse to winner-take-all is driven by **t**, not sigma; sigma only sets *when*
it happens. Effective contributing components (responsibility perplexity, median,
N=130k centers):

| sigma | t=0.3 | t=0.5 | t=0.7 | t=0.9 |
|---|---|---|---|---|
| 0.01 | 812 | 1.0 | 1.0 | 1.0 |
| 0.1  | 717 | 1.0 | 1.0 | 1.0 |
| 0.3  | 798 | 1.0 | 1.0 | 1.0 |
| 0.6  | 1016 | 1.0 | 1.0 | 1.0 |
| 1.0  | 1483 | 3.5 | 1.0 | 1.0 |

Early on the `(1-t)` noise smears components together (~10^3 contribute); a sharp
commitment transition near t≈0.5-0.7 follows, after which the field is exactly
nearest-center. **In the committed regime the marginal velocity reduces to the
single-component form, so `w_avg` there is just the closed form
`sigma^2/((1-t)^2 + sigma^2 t^2)`.** Everything the experiment adds beyond analysis
lives in the blended region and the transition — keep that in mind when choosing
the t grid and when interpreting results.

Scale reference (scaled latents, per coordinate): data std **0.589**,
nearest-neighbour distance **0.372** at production N, random-pair distance
**0.768**; `E||mu||^2/d = 0.3516`. NN scales as `N^-0.03` (effective dimension ~34), so 10x more centers
pulls neighbours only 7% closer.

### Measured RK4 accuracy vs step size (single-Gaussian case)

Worst-case relative error in `Phi` over the 50-point t grid. **Bold = the chosen
tier**; the target is 1e-5, i.e. 100x margin under the 1e-3 accuracy goal, because
the real GMM's commitment layer is rougher than this smooth test field.

| sigma | h=1/16 | h=1/32 | h=1/64 | h=1/128 | h=1/256 | h=1/512 |
|---|---|---|---|---|---|---|
| 0.01 | 2.8e-1 | 3.4e-2 | 2.1e-2 | 1.6e-3 | 5.9e-5 | **3.0e-6** |
| 0.1  | 5.2e-4 | 2.3e-5 | **1.3e-6** | 9.2e-8 | 5.7e-9 | 3.9e-10 |
| 0.3  | 3.8e-6 | 2.5e-7 | **1.5e-8** | 9.0e-10 | 5.6e-11 | 3.5e-12 |
| 0.6  | 8.6e-7 | 6.4e-8 | **4.1e-9** | 4.1e-9 | 1.6e-11 | 1.0e-12 |

The error is **flat in t** at fixed `h`, which confirms fixed-`h`-not-fixed-`S`
delivers the intended uniform accuracy across the curve — no graded/non-uniform
step schedule near t->1 is needed.

Cost of one full t-sweep, velocity evaluations per trajectory summed over all four
sigmas: global 1/512 = 209,280 · global 1/256 = 104,832 · **two-tier = 72,192** ·
fully per-sigma = 65,856. Those **ratios** are what drove the step-size choice and
they stand.

**Corrected wall-clock.** An earlier note here put the production run at ~16 h
(two-tier) / ~50 h (global 1/512). Those absolute numbers were wrong by ~1000x
through an arithmetic slip. Redone: at `B = M(K+1) = 2560`, one batched field
evaluation is `2 * 2 * B * d * N = 3.36e12` FLOPs, and the whole four-sigma sweep is
`72,192 * 3.36e12 = 2.4e17` FLOPs — **tens of minutes on 8x H200**, not days. See
[documents/marginal_flow_plan.md](documents/marginal_flow_plan.md) §8.

The consequence matters: **compute is not the binding constraint, Monte-Carlo error
is.** `M = 512` is far more conservative than necessary; SE falls as `1/sqrt(M)`, so
`M = 16,384` costs 32x (hours at most) and cuts it 5.7x. Revisit `M`/`K` in phase 8
against this budget. Still benchmark one timed evaluation before trusting it.

sigma=0.01 is the stiffest case near t->1 and needs the most rk4 steps.

## Data

`/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/latents_8_mean_fp16/train`

- `train/<wnid>/<image>.pt` -> `{'mean': fp16 (4,8,8)}`, 1000 classes,
  1,281,167 files, ~3.0 GB on disk. See that directory's README.md.
- **Values are raw — the `0.18215` SD scale is NOT applied on disk.** Raw std is
  3.28, range ~[-15.7, 18.3]. Apply `0.18215` after loading (config flag); scaled
  std is 0.60 and `E||mu||^2/d` is **0.3516** (exact, over all 1,281,167 centers;
  earlier 0.359 was a 40k sample).
- Do not pool or downsample further; these are already 4x8x8.
- **Consolidated centers file** (to be built): a single flat fp16 array of shape
  (N,4,8,8), ~656 MB, plus a row->source index, written into
  `latents_8_mean_fp16/` with that README updated. Reading the 1.28M small files
  takes ~33 min single-threaded (measured 651 files/s); the flat file is seconds.
  Keep it raw (unscaled) to stay faithful to the source.

## Environment and Execution

Conda env `SiT`, torch 2.11.0+cu128. Available: numpy 2.4.3, pyyaml, tqdm,
matplotlib, scipy, torchdiffeq 0.2.5 (unused by choice).

```bash
source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
```

All experiments run **remotely on 8x H200 (~140 GB VRAM each)** via `torchrun`.
Claude has no GPU access from this machine — write launchers, do not run them.
The GPU count must be settable from the config file.

## Layout

Mirrors PWFM: scripts + configs + bash launchers.

- `configs/` — YAML, one per experiment. All controls configurable without source edits.
- `remote_bash_script/` — `torchrun` launchers. Follow the style of
  `../Path-Weighted_Flow_Matching/remote_bash_script/run_true_marginal_wavg_8xh200.sh`:
  `set -euo pipefail`, conda activate, randomized `MASTER_PORT`, banner, then
  `torchrun --standalone --nproc_per_node=$NUM_GPUS`.
- `documents/` — spec, notes, derivations.
- `results/` — experiment outputs (gitignored).

## Reporting Requirements

Long runs need live progress with ETA (Sec. 10.5-10.6): rank 0 only for the main
progress bar, mirrored periodically into the log file so progress survives an SSH
disconnect. Save intermediate per-timestep results so runs can resume.

## Open / Deferred

Decided later with the user — do not guess these:

- Development order (build sequence).
- Consolidated centers file: **one flat file** (decided); when to build it is deferred.
