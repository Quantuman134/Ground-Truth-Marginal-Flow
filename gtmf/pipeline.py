"""One point of the w_avg curve.

`run_timepoint` is where the pieces finally meet: draw query states from p_t,
size the perturbation against the marginal scale, carry everything to t = 1
through the marginal ODE, and collapse the result to one mean and one error bar.

**The critical invariant lives here** (CLAUDE.md, spec 8.5). The conditional
interpolation `x_t = (1-t) x_0 + t y` may be used ONLY to place a query state --
never to reach the endpoint. Reaching for it instead of integrating returns a
smooth, plausible, meaningless curve that nothing downstream can detect.

It is guaranteed structurally rather than by a check: `sample_query_states`
returns states and nothing else, the only route to an endpoint is
`integrator(flow.velocity, ...)` inside `w_hat`, and the two share no call path.
What proves it at runtime is counting field evaluations -- a real solve calls the
velocity field `4S` times for RK4, and the interpolant shortcut would call it
zero times while still returning numbers.
"""

import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from . import schedule
from .estimator import mean_and_stderr, w_hat
from .integrate import euler, rk4
from .rng import make_generator
from .target import build_target

INTEGRATORS = {"rk4": rk4, "euler": euler}
RHO_SOURCES = ("exact", "empirical")


@dataclass(frozen=True)
class TimepointResult:
    """Everything one timepoint produces. One row of `w_avg.csv`, plus the raw.

    `w_per_query` is kept because the figures are redrawn from it without
    re-running (CLAUDE.md, "Raw data"), and it is moved to the CPU so step 4 can
    write it straight out and so a long sweep does not pin M floats per timepoint
    on the GPU.
    """

    t: float
    w_avg: float
    stderr: float
    w_per_query: torch.Tensor       # (M,) on the CPU
    rho_t: float
    eps: float
    seconds: float


def _sync(device):
    """Finish outstanding GPU work so a wall-clock reading means something.

    CUDA kernels launch asynchronously, so without this the timer measures how
    long it took to *queue* the solve. Phase 7 exists partly to replace the
    unbenchmarked cost estimates in CLAUDE.md, and an un-synchronised number
    would replace them with a smaller fiction.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_timepoint(flow, cfg, sigma, t, generator=None, probe_generator=None):
    """Estimate w_avg at one time t.   Returns a :class:`TimepointResult`.

        flow       MarginalFlow for this sigma (from build_target)
        cfg        Config -- supplies M, K, scheme, alpha, h, integrator
        sigma      the swept sigma; selects alpha and h from their by_sigma maps
        t          the time to inject the perturbation at
        generator  seeded, and on the same device as the centres (see gtmf.rng)
        probe_generator  optional second stream for the probe directions;
                   defaults to `generator`. Kept separate because the estimator
                   needs u independent of x_t -- E[u u^T] = I is what turns the
                   direction average into ||Phi||_F^2 -- and because phase 8
                   varies M and K independently.

    The perturbation is sized as `eps = alpha * rho_t`, against the marginal RMS
    rho_t rather than the per-component scale c_t. The two differ by ~27x at
    t = 0.98 with sigma = 0.01, so this is not interchangeable.

    `alpha` has no default anywhere: `for_sigma` raises when it is null, which is
    how the production config ships, so a run without a hand-chosen alpha stops
    instead of quietly guessing one.
    """
    m = cfg["monte_carlo.num_query_states"]
    k = cfg["monte_carlo.num_probes"]
    scheme = cfg["monte_carlo.scheme"]
    alpha = cfg.for_sigma("monte_carlo.epsilon_alpha", sigma)
    h = cfg.for_sigma("ode.step_size", sigma)

    name = cfg["ode.integrator"]
    if name not in INTEGRATORS:
        raise ValueError(f"ode.integrator must be one of {sorted(INTEGRATORS)}, "
                         f"got {name!r}")
    rho_source = cfg.get("monte_carlo.rho_source", default="exact")
    if rho_source not in RHO_SOURCES:
        raise ValueError(f"monte_carlo.rho_source must be one of "
                         f"{list(RHO_SOURCES)}, got {rho_source!r}")

    device = flow.mu.device
    _sync(device)
    started = time.perf_counter()

    # Sampling only. This state is the START of a trajectory, never the end.
    x_t = flow.sample_query_states(t, m, generator=generator)

    # "exact" is the default: the closed form costs nothing and keeps sampling
    # noise out of the perturbation size, so eps depends on t alone and not on
    # which M states happened to be drawn.
    rho_t = flow.rho_t_exact(t) if rho_source == "exact" else flow.rho_t(x_t)
    eps = alpha * rho_t

    # The ONLY route to t = 1: the marginal field, integrated.
    w = w_hat(flow.velocity, x_t, t, eps, k, h,
              generator=generator if probe_generator is None else probe_generator,
              scheme=scheme, integrator=INTEGRATORS[name])

    w_avg, stderr = mean_and_stderr(w)
    _sync(device)
    seconds = time.perf_counter() - started

    return TimepointResult(t=float(t), w_avg=w_avg, stderr=stderr,
                           w_per_query=w.detach().to("cpu"),
                           rho_t=float(rho_t), eps=float(eps), seconds=seconds)


# --------------------------------------------------------------------------- #
# Step 4 -- one sigma: walk the time grid, write as you go
# --------------------------------------------------------------------------- #

CSV_NAME, RAW_NAME, SUMMARY_NAME = "w_avg.csv", "raw.npz", "summary.json"
CSV_COLUMNS = ("t", "w_avg", "stderr", "rho_t", "eps", "seconds")

QUERY_STREAM, PROBE_STREAM = 0, 1


def measurement_times(cfg):
    """The t values to measure w_avg at, as plain Python floats.

    Not to be confused with `integrate.time_grid`, which is the SOLVER's step
    policy inside a single trajectory. These are the points of the output curve
    -- CLAUDE.md calls this "the time grid" and what integrate computes "solver
    steps", and the two are one import apart in this file.

    Uniform on [t_min, t_max] (CLAUDE.md: explicitly NOT concentrated near the
    transition). Converted off numpy deliberately -- a np.float64 would reach the
    CSV as `np.float64(0.3)` and the integrator's step count through a different
    type than every test uses.
    """
    return [float(t) for t in np.linspace(cfg["time.t_min"], cfg["time.t_max"],
                                          int(cfg["time.num_points"]))]


def _stream(base_seed, stream, index, device):
    """An independent generator for (base seed, stream id, timepoint index).

    Two properties this has to have, and a plain `base + index` does not:

    * The query-state and probe streams must be independent even when the config
      gives them the SAME seed -- both shipped configs say 0, and seeding two
      generators with 0 would make the probe directions a copy of the draws that
      produced the query states. w_hat leans on E[u u^T] = I, so that correlation
      would bias every w it returns.
    * Each timepoint must be reproducible on its own, so a resumed run continues
      the same experiment rather than a differently-seeded one. Indexing the
      stream does that; advancing one long-lived generator would not.
    """
    (entropy,) = np.random.SeedSequence([int(base_seed), int(stream), int(index)]
                                        ).generate_state(1, dtype=np.uint64)
    return make_generator(int(entropy), device)


def settings_for(cfg, sigma):
    """Everything that determines the numbers for ONE sigma, already resolved.

    Written into summary.json and compared on resume. Per-sigma settings are
    resolved rather than compared raw, so adding a sigma to the sweep does not
    invalidate the sigmas already computed -- only a change that would alter
    THIS curve counts as a change.
    """
    return {
        "sigma": float(sigma),
        "time": {"num_points": int(cfg["time.num_points"]),
                 "t_min": float(cfg["time.t_min"]),
                 "t_max": float(cfg["time.t_max"])},
        "monte_carlo": {
            "num_query_states": cfg["monte_carlo.num_query_states"],
            "num_probes": cfg["monte_carlo.num_probes"],
            "scheme": cfg["monte_carlo.scheme"],
            "epsilon_alpha": cfg.for_sigma("monte_carlo.epsilon_alpha", sigma),
            "rho_source": cfg.get("monte_carlo.rho_source", default="exact")},
        "ode": {"integrator": cfg["ode.integrator"],
                "step_size": cfg.for_sigma("ode.step_size", sigma)},
        "compute": {"device": cfg.get("compute.device", default="cpu"),
                    "dtype": cfg.get("compute.dtype", default="float32")},
        "data": cfg.get("data", default={}),
        "seeds": cfg.get("seeds", default={}),
    }


def _differences(stored, current, prefix=""):
    """Dotted paths where two settings dicts disagree, for the error message."""
    out = []
    for key in sorted(set(stored) | set(current)):
        here = f"{prefix}{key}"
        a, b = stored.get(key, "<absent>"), current.get(key, "<absent>")
        if isinstance(a, dict) and isinstance(b, dict):
            out += _differences(a, b, here + ".")
        elif a != b:
            out.append(f"{here}: {a!r} -> {b!r}")
    return out


def _hms(seconds):
    return f"{int(seconds) // 3600:d}:{int(seconds) // 60 % 60:02d}:{int(seconds) % 60:02d}"


def _read_done(out_dir, grid, settings=None):
    """Rows already on disk, for resuming. Returns (rows, per_query list).

    A resumed run must be the SAME experiment. Two things are checked:

    * the SETTINGS that determine the numbers -- alpha, h, K, the scheme, the
      seeds, the target. Checking only the grid would let a run resumed after an
      edit produce one curve whose early points used one alpha and whose later
      points used another, with resolved_config.yaml recording only the second.
      The per-row `eps` column would show it, but nothing would say so.
    * every stored t against the grid this config produces, so rows from one
      grid are never appended to rows from another.

    Both raise rather than continuing: the result would be smooth, plausible,
    and not an experiment anyone performed.
    """
    csv_path, raw_path = out_dir / CSV_NAME, out_dir / RAW_NAME
    if not csv_path.exists():
        return [], []

    summary_path = out_dir / SUMMARY_NAME
    if settings is not None and summary_path.exists():
        stored = json.loads(summary_path.read_text()).get("settings")
        if stored is not None and stored != settings:
            changed = "\n  ".join(_differences(stored, settings)) or "<unknown>"
            raise ValueError(
                f"{out_dir} was computed with different settings; refusing to "
                f"resume and mix them into one curve:\n  {changed}\n"
                f"Start a fresh run directory, or delete this sigma's files to "
                f"recompute it.")
    with csv_path.open() as fh:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]
    if not rows:
        return [], []

    if len(rows) > len(grid):
        raise ValueError(f"{csv_path} holds {len(rows)} rows but the grid has "
                         f"{len(grid)}; the config changed since that run")
    for i, row in enumerate(rows):
        if not math.isclose(row["t"], grid[i], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{csv_path} row {i} is t={row['t']}, but this config "
                             f"puts t={grid[i]} there; refusing to resume a "
                             f"different time grid")

    raw = np.load(raw_path) if raw_path.exists() else None
    if raw is None or len(raw["w"]) != len(rows):
        raise ValueError(f"{raw_path} does not match {csv_path} row for row; "
                         f"delete the directory to start this sigma over")
    return rows, [w for w in raw["w"]]


def _write_progress(out_dir, rows, per_query):
    """Rewrite both output files. Called after every timepoint, so a killed run
    resumes from the last completed t rather than restarting the sigma.

    raw.npz is rewritten whole rather than appended: at 50 x 512 it is ~100 KB,
    and npz has no append, so this trades nothing for a file that is always
    internally consistent with the CSV beside it.
    """
    with (out_dir / CSV_NAME).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    np.savez(out_dir / RAW_NAME,
             t=np.array([r["t"] for r in rows], dtype=np.float64),
             w=np.stack(per_query) if per_query else np.zeros((0, 0)))


def _peak(cfg, sigma, rows):
    """Where w(t) peaks -- with the caveat CLAUDE.md insists on.

    t_peak = 1/(1+sigma^2) is OUTSIDE the [0, 0.98] grid for sigma <= 0.1. There
    the curve rises across the whole grid and its sampled argmax is just the
    right-hand edge, which is not the peak. Report that it is out of range
    instead of naming a maximum the run never saw.
    """
    t_min, t_max = float(cfg["time.t_min"]), float(cfg["time.t_max"])
    t_peak = schedule.t_peak(sigma)
    out = {"t_peak_closed_form": t_peak, "grid": [t_min, t_max],
           "in_grid_range": bool(t_min <= t_peak <= t_max)}
    if not out["in_grid_range"]:
        out["note"] = (f"peak not in range: t_peak = {t_peak:.4f} lies outside "
                       f"[{t_min}, {t_max}], so the sampled argmax is the grid "
                       f"edge, not the peak")
        return out
    best = max(rows, key=lambda r: r["w_avg"])
    out["sampled_argmax_t"] = best["t"]
    out["sampled_max_w_avg"] = best["w_avg"]
    return out


def _write_summary(out_dir, cfg, sigma, flow, rows, settings, seconds, complete):
    """Build summary.json and write it. Returns the dict.

    `complete` says whether every timepoint on the grid is present. Phase 10
    needs it: a partially-written sigma has a real curve and a real peak block,
    both computed over fewer points than the run asked for.
    """
    summary = {
        "experiment": cfg.get("experiment", default=None),
        "sigma": float(sigma),
        "complete": bool(complete),
        "num_timepoints": len(rows),
        "num_centers": int(flow.n), "dim": int(flow.d),
        "dtype": str(flow.mu.dtype), "device": str(flow.mu.device),
        "settings": settings,
        "peak": _peak(cfg, sigma, rows),
        "w_avg_range": [min(r["w_avg"] for r in rows), max(r["w_avg"] for r in rows)],
        "timing": {"total_seconds_this_invocation": seconds,
                   "sum_of_timepoint_seconds": sum(r["seconds"] for r in rows),
                   "mean_seconds_per_timepoint":
                       sum(r["seconds"] for r in rows) / len(rows)},
    }
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_sigma(cfg, sigma, out_dir, flow=None, log=print):
    """Walk the time grid for one sigma, writing results as they complete.

        cfg      the run's config          out_dir  sigma_<value>/
        sigma    the swept sigma           flow     prebuilt target, or None
        log      where progress goes; step 5 mirrors it into run.log

    Writes `w_avg.csv` (one row per t), `raw.npz` (the per-query estimates phase
    10 redraws from) and `summary.json`. The first two are rewritten after every
    timepoint so a run killed by an SSH drop resumes at the next t.

    Returns the summary dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = measurement_times(cfg)

    # Resolved from the config alone, so a settings mismatch stops the run before
    # build_target reads 1.3 GB of centres.
    settings = settings_for(cfg, sigma)
    rows, per_query = _read_done(out_dir, grid, settings)
    start = len(rows)                       # first index THIS invocation computes
    if start:
        log(f"  resuming at t index {start} of {len(grid)} ({start} already done)")
    if flow is None:
        flow = build_target(cfg, sigma)

    device = flow.mu.device
    query_seed = cfg.get("seeds.query_states", default=0)
    probe_seed = cfg.get("seeds.probes", default=0)
    started = time.perf_counter()

    for i in range(start, len(grid)):
        t = grid[i]
        result = run_timepoint(
            flow, cfg, sigma, t,
            generator=_stream(query_seed, QUERY_STREAM, i, device),
            probe_generator=_stream(probe_seed, PROBE_STREAM, i, device))

        row = asdict(result)
        per_query.append(row.pop("w_per_query").numpy())
        rows.append(row)
        _write_progress(out_dir, rows, per_query)
        # After every timepoint, not just at the end: this is the file a resumed
        # run reads its settings back from, and a killed run leaves one that says
        # how far it got.
        _write_summary(out_dir, cfg, sigma, flow, rows, settings,
                       time.perf_counter() - started,
                       complete=len(rows) == len(grid))

        # Spec 10.6: progress with an ETA, emitted per timepoint so it reaches
        # the log file and survives losing the terminal. The rate is measured
        # over THIS invocation only -- timepoints restored from disk cost no
        # time here, and counting them would report an ETA that keeps sliding.
        elapsed = time.perf_counter() - started
        done = i - start + 1
        eta = (len(grid) - 1 - i) * elapsed / done
        log(f"  [{i + 1:>3}/{len(grid)}] t={t:.4f}  "
            f"w_avg={result.w_avg:.6g} +/- {result.stderr:.2g}  "
            f"{result.seconds:.1f}s  elapsed {_hms(elapsed)}  ETA {_hms(eta)}")

    return _write_summary(out_dir, cfg, sigma, flow, rows, settings,
                          time.perf_counter() - started,
                          complete=len(rows) == len(grid))


# --------------------------------------------------------------------------- #
# Step 5 -- the sweep: one invocation, one directory
# --------------------------------------------------------------------------- #

LOG_NAME, RESOLVED_NAME = "run.log", "resolved_config.yaml"


class RunLog:
    """Progress to stdout and to `run.log` at once.

    Every line is flushed as it is written. Spec 10.6 asks for progress that
    survives losing the terminal, and a buffered log file loses exactly the last
    few lines -- the ones that say where a run had got to when it died.
    """

    def __init__(self, path, echo=True):
        self.handle = Path(path).open("a", buffering=1)      # line buffered
        self.echo = echo

    def __call__(self, message=""):
        if self.echo:
            print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def prepare_run_dir(run_dir, force=False):
    """Create the run directory, refusing to walk into an existing one.

    CLAUDE.md: one invocation, one directory, timestamped so a re-run cannot
    destroy the previous one -- production runs take hours and their raw
    per-query data is what every figure is redrawn from. Only an explicit
    `--output` can collide, and then only deliberately.

    `force` does not delete anything: it allows the run to CONTINUE into the
    directory, and run_sigma then resumes each sigma from its completed
    timepoints. A changed config still raises there rather than stitching two
    runs together.
    """
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise FileExistsError(
            f"{run_dir} already exists and is not empty. Pass --force to "
            f"continue into it (completed timepoints are resumed, not redone), "
            f"choose another --output, or drop --output for a timestamped "
            f"directory that cannot collide.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_sweep(cfg, run_dir=None, force=False, echo=True):
    """Walk every sigma in `gmm.component_sigma`, one subdirectory each.

        results/<experiment>_<timestamp>/
            resolved_config.yaml     what was actually run
            run.log                  progress, mirrored from stdout
            sigma_0.01/ ... sigma_0.6/

    `resolved_config.yaml` and `run.log` sit at the top because they belong to
    the invocation, not to any one sigma. Sigmas are walked sequentially, which
    is the decision in CLAUDE.md -- each one already saturates the GPU.

    Returns a dict with the run directory and one summary per sigma.
    """
    run_dir = prepare_run_dir(cfg.run_dir() if run_dir is None else run_dir, force)
    cfg.dump(run_dir / RESOLVED_NAME)
    sigmas = list(cfg["gmm.component_sigma"])
    started = time.perf_counter()

    with RunLog(run_dir / LOG_NAME, echo=echo) as log:
        rule = "=" * 70
        log(rule)
        log(" GTMF -- reference w_avg(t)")
        log(f" experiment : {cfg.get('experiment', default='?')}")
        log(f" config     : {cfg.source if cfg.source else '<in memory>'}")
        log(f" run dir    : {run_dir}")
        log(f" sigmas     : {sigmas}")
        log(f" started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(rule)

        summaries = {}
        for sigma in sigmas:
            out_dir = cfg.sigma_dir(run_dir, sigma)
            log("")
            log(f"sigma = {sigma}  ->  {out_dir.name}/   "
                f"h={cfg.for_sigma('ode.step_size', sigma)}  "
                f"alpha={cfg.for_sigma('monte_carlo.epsilon_alpha', sigma)}  "
                f"M={cfg['monte_carlo.num_query_states']}  "
                f"K={cfg['monte_carlo.num_probes']}")
            at = time.perf_counter()
            summaries[str(sigma)] = run_sigma(cfg, sigma, out_dir, log=log)
            log(f"  sigma {sigma} done in {_hms(time.perf_counter() - at)}")

        total = time.perf_counter() - started
        log("")
        log(rule)
        log(f" all {len(sigmas)} sigma(s) done in {_hms(total)}  ->  {run_dir}")
        log(rule)

    return {"run_dir": run_dir, "sigmas": sigmas, "summaries": summaries,
            "seconds": total}
