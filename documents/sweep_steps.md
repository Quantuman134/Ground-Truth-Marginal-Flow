# Convergence Studies — What Was Built

Phase 8. This file was originally a five-step plan for a separate `sweeps.py`
with three studies. Both of those changed, and this records what exists instead.

## What changed from the plan

**Two of the three studies were settled by decision, not by measurement.**

| knob | plan | what happened |
|---|---|---|
| `h` | halve-h study across the commitment layer | set to **one global 1/512** — the finest tier, everywhere. The two-tier scheme is deferred, not overturned; see CLAUDE.md. |
| `M`, `K` | vary independently, set against a measured error target | set by hand to **M=2048, K=16**, tuned by hand from here |
| `alpha` | sweep spec Eq. 36, choose from the plateau | **unchanged** — this is the one that still has to be measured |

**There is no `sweeps.py`.** A separate sweep driver duplicated `run_sigma`'s
grid walking, incremental writing and resume, and collided by name with
`gtmf/sweeps.py`. The sweep is now part of the ordinary pipeline.

## How the alpha sweep works

`monte_carlo.epsilon_alpha` takes two shapes, and the shape is the request:

```yaml
epsilon_alpha: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]   # a SWEEP (spec Eq. 36)
epsilon_alpha: {default: 1e-3, by_sigma: {0.01: 3e-4}} # a single run, per sigma
```

A list makes each alpha its own experiment directory holding a full sigma sweep:

```
results/<experiment>_<timestamp>/
    alpha_0.0001/  sigma_0.01/ ... sigma_0.6/
    alpha_0.001/   ...
    figures/plateau.png            <- the figure alpha is chosen from
```

Alpha is the outer level in the directories **and** in the running order, so a
run killed part way leaves complete experiments rather than every alpha half
done. The loop still builds each sigma's target once and caches it, or a
6-alpha, 4-sigma sweep would read the centres 24 times instead of 4.

`derive(cfg, **overrides)` in `gtmf/pipeline.py` is what changes the knob: a
copy of the config, re-validated, so a sweep point is an ordinary `run_timepoint`
call carrying the same settings record production runs carry.

## What the sweep does NOT do

It never chooses alpha, never prints a recommendation, and never writes to a
config. That is a CLAUDE.md decision: a plateau is a judgement about the shape of
a curve, and an argmin would happily pick a point on a slope. There is a test
asserting the log contains no recommendation.

## Reading the plateau

`figures/plateau.png` plots `w_avg` against alpha, one line per `t`, one panel
per sigma. A usable alpha sits on a **flat** stretch: to the right the finite
difference is biased (on the real mixture the nudged state crosses into another
component's basin), to the left fp32 cancellation eats it.

Two measured warnings:

* the bias explodes between alpha 1e-2 and 3e-2 on a curved field, so the
  plateau's upper edge is well below the spec's largest alpha;
* at alpha = 1e-4 the perturbation is near the fp32 floor. A plateau that
  collapses there means escalate **that sigma** to fp64 — not pick a larger alpha.

## Still to do

Run it, read the plateau, and write the per-sigma values into
`configs/wavg_imagenet.yaml` by hand. Until then `epsilon_alpha` is a list of
candidates, not a result, and the phase 8 gate is not met.

One measured observation from a partial run at N=130k, sigma=0.3: a clean
plateau at t=0.98, but **none at t=0.30**, where `w_avg` fell 200x across the
alpha range with standard errors the size of the values. Either the perturbation
crosses basins even at 1e-4, or `w` is heavy-tailed over `x_t` there and M was
too small to resolve the mean. That has to be settled before an alpha can be
chosen for the blended regime.
