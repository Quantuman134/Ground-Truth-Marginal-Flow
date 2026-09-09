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

import copy
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
from . import dist
from .config import Config, ConfigError
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

    `w_per_probe` is the same estimate before averaging over the K directions.
    It costs K times as much -- 6.25 MB per sigma at production size against
    0.39 MB -- and it is the only thing that separates the variance WITHIN a
    query state (which K reduces) from the variance ACROSS query states (which
    only M reduces). Without it, choosing M and K is guesswork.
    """

    t: float
    w_avg: float
    stderr: float
    w_per_query: torch.Tensor       # (M,) on the CPU
    w_per_probe: torch.Tensor       # (M, K) on the CPU, before the K average
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


def run_timepoint(flow, cfg, sigma, t, generator=None, probe_generator=None,
                  shard=True):
    """Estimate w_avg at one time t.   Returns a :class:`TimepointResult`.

        flow       MarginalFlow for this sigma (from build_target)
        cfg        Config -- supplies M, K, scheme, alpha, h, integrator
        sigma      the swept sigma; selects alpha and h from their by_sigma maps
        t          the time to inject the perturbation at
        generator  seeded, and on the same device as the centres (see gtmf.rng)
        shard      split the M query states across ranks (default). Every rank
                   draws the SAME full sample and keeps a contiguous slice, so
                   the union is exactly what one process would have drawn and
                   the gathered result is identical at any world size. Pass
                   False to force a whole-sample run on this rank.
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
    # The FULL sample is drawn on every rank; the split happens after, so the
    # draws never depend on how many ranks there are.
    x_full = flow.sample_query_states(t, m, generator=generator)

    # Drawn here rather than inside perturbed_batch for the same reason: same
    # generator, same (K, M, d) shape, same order, so the single-process numbers
    # are unchanged -- and a shard can be taken from the full set.
    u_full = torch.randn(
        (k, m, flow.d),
        generator=generator if probe_generator is None else probe_generator,
        dtype=x_full.dtype, device=x_full.device)

    # "exact" is the default: the closed form costs nothing and keeps sampling
    # noise out of the perturbation size, so eps depends on t alone and not on
    # which M states happened to be drawn. Measured from the FULL sample either
    # way, so eps does not depend on how the work was divided.
    rho_t = flow.rho_t_exact(t) if rho_source == "exact" else flow.rho_t(x_full)
    eps = alpha * rho_t

    lo, hi = dist.shard_bounds(m) if shard else (0, m)
    x_t, u = x_full[lo:hi], u_full[:, lo:hi]

    # The ONLY route to t = 1: the marginal field, integrated.
    w_local, probe_local = w_hat(flow.velocity, x_t, t, eps, k, h,
                                 scheme=scheme, integrator=INTEGRATORS[name],
                                 probes=u, per_probe=True)

    # Back to the full per-query vectors, in the original order, on every rank,
    # so mean_and_stderr sees exactly what a single process would have.
    w = dist.gather_rows(w_local) if shard else w_local
    per_probe = dist.gather_rows(probe_local) if shard else probe_local
    w_avg, stderr = mean_and_stderr(w)
    _sync(device)
    seconds = time.perf_counter() - started

    return TimepointResult(t=float(t), w_avg=w_avg, stderr=stderr,
                           w_per_query=w.detach().to("cpu"),
                           w_per_probe=per_probe.detach().to("cpu"),
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


def derive(cfg, **overrides):
    """A copy of `cfg` with dotted overrides applied, re-validated.

        derive(cfg, **{"monte_carlo.epsilon_alpha": 3e-4})

    How a sweep changes a knob. Routing every study through here means a sweep
    point is an ordinary run -- same code, same settings record -- rather than a
    special path, and an override cannot smuggle in a setting the loader would
    have rejected.

    A missing SECTION raises rather than being created: a typo like
    `montecarlo.num_probes` would otherwise sit in the config doing nothing while
    the real setting kept its old value. A missing leaf is fine.
    """
    data = copy.deepcopy(cfg.data)
    for dotted, value in overrides.items():
        node = data
        *parents, leaf = dotted.split(".")
        for i, part in enumerate(parents):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"cannot override '{dotted}': no section "
                                  f"'{'.'.join(parents[:i + 1])}' in this config")
            node = node[part]
        if not isinstance(node, dict):
            raise ConfigError(f"cannot override '{dotted}': "
                              f"'{'.'.join(parents)}' is not a section")
        node[leaf] = value
    return Config(data, source=cfg.source).validate()


# Spec Eq. 36 -- the alpha sweep of section 7.2.
ALPHAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def alpha_dir(run_dir, alpha):
    """One directory per swept alpha, each holding a full sigma sweep."""
    return Path(run_dir) / f"alpha_{float(alpha):g}"


def swept_alphas(cfg):
    """The alphas to sweep, or None for a single run.

    `monte_carlo.epsilon_alpha` takes two shapes, and the shape IS the request:

        [1e-4, 3e-4, 1e-3]      a sweep -- one experiment per alpha
        {default:, by_sigma:}   a single run, resolved per sigma
        3e-3                    the same, one value for every sigma

    The list form mirrors `gmm.component_sigma`; the mapping form is what a
    production config uses once the alphas have been chosen by hand, since
    CLAUDE.md wants a different alpha per sigma there.
    """
    node = cfg.get("monte_carlo.epsilon_alpha", default=None)
    return [float(a) for a in node] if isinstance(node, list) else None


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
    """Rows already on disk, for resuming. Returns (rows, per_query, per_probe).

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
        return [], [], []

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
        return [], [], []

    if len(rows) > len(grid):
        raise ValueError(f"{csv_path} holds {len(rows)} rows but the grid has "
                         f"{len(grid)}; the config changed since that run")
    for i, row in enumerate(rows):
        if not math.isclose(row["t"], grid[i], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{csv_path} row {i} is t={row['t']}, but this config "
                             f"puts t={grid[i]} there; refusing to resume a "
                             f"different time grid")

    if not raw_path.exists():
        raise ValueError(f"{csv_path} has rows but {raw_path} is missing; "
                         f"delete the directory to start this sigma over")
    stored = np.load(raw_path)
    if "w_probe" not in stored:
        raise ValueError(f"{raw_path} predates per-probe raw data and cannot be "
                         f"resumed into; delete the directory to recompute it")
    raw, probe = list(stored["w"]), list(stored["w_probe"])
    if not len(raw) == len(probe) == len(rows):
        raise ValueError(f"{raw_path} does not match {csv_path} row for row; "
                         f"delete the directory to start this sigma over")
    return rows, raw, probe


def _write_progress(out_dir, rows, per_query, per_probe):
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
             w=np.stack(per_query) if per_query else np.zeros((0, 0)),
             # (num_points, M, K) -- ~6 MB per sigma at production size, and the
             # only record of how much of the spread K can actually remove.
             w_probe=np.stack(per_probe) if per_probe else np.zeros((0, 0, 0)))


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
    if dist.is_main():
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
    if dist.is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()                       # nobody reads before rank 0 has created it
    log = log if dist.is_main() else (lambda *_: None)
    grid = measurement_times(cfg)

    # Resolved from the config alone, so a settings mismatch stops the run before
    # build_target reads 1.3 GB of centres.
    settings = settings_for(cfg, sigma)
    rows, per_query, per_probe = _read_done(out_dir, grid, settings)
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
        per_probe.append(row.pop("w_per_probe").numpy())
        rows.append(row)
        _write_progress(out_dir, rows, per_query, per_probe)
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

    def __init__(self, path, echo=True, write=True):
        # Rank 0 alone opens the file: eight ranks appending the same lines would
        # interleave them into an unreadable log.
        self.handle = Path(path).open("a", buffering=1) if write else None
        self.echo = echo

    def __call__(self, message=""):
        if self.echo:
            print(message, flush=True)
        if self.handle is not None:
            self.handle.write(message + "\n")
            self.handle.flush()

    def close(self):
        if self.handle is not None:
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


def _finished_summary(dest):
    """The summary of an already-complete run at `dest`, or None.

    Checked before the target is built, so a resumed sweep with nothing left to
    do does not read 1.3 GB of centres to discover that. A PARTIAL run returns
    None on purpose -- run_sigma resumes those.
    """
    path = Path(dest) / SUMMARY_NAME
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    return summary if summary.get("complete") else None


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
    run_dir = Path(cfg.run_dir() if run_dir is None else run_dir)
    if dist.is_main():
        prepare_run_dir(run_dir, force)
        cfg.dump(run_dir / RESOLVED_NAME)
    dist.barrier()
    sigmas = list(cfg["gmm.component_sigma"])
    alphas = swept_alphas(cfg)
    # Alpha is the outer level, in the directories AND in the running order: one
    # alpha's full sigma sweep finishes before the next begins, so a run killed
    # part way leaves complete experiments rather than every alpha half done.
    plan = [(None, run_dir)] if alphas is None else \
        [(a, alpha_dir(run_dir, a)) for a in alphas]
    started = time.perf_counter()

    with RunLog(run_dir / LOG_NAME, echo=echo and dist.is_main(),
                write=dist.is_main()) as log:
        rule = "=" * 70
        log(rule)
        log(" GTMF -- reference w_avg(t)")
        log(f" experiment : {cfg.get('experiment', default='?')}")
        log(f" config     : {cfg.source if cfg.source else '<in memory>'}")
        log(f" run dir    : {run_dir}")
        log(f" sigmas     : {sigmas}")
        log(f" ranks      : {dist.world_size()}")
        log(f" alphas     : {alphas if alphas else '(single run, from the config)'}")
        log(f" started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(rule)

        summaries = {}
        # Targets are cached across the whole sweep, because alpha is now the
        # outer loop: without this, each sigma's centres would be re-read once
        # per alpha -- 24 loads of 1.3 GB for a 6-alpha, 4-sigma sweep instead of
        # 4. The centres are the same array for every sigma, so what is really
        # bought is one load; holding all four flows costs ~5.3 GB at full N,
        # which is nothing against 140 GB.
        flows = {}
        for alpha, base in plan:
            for sigma in sigmas:
                dest = cfg.sigma_dir(base, sigma)
                key = str(sigma) if alpha is None else f"{sigma}/{alpha:g}"
                finished = _finished_summary(dest)
                if finished is not None:
                    summaries[key] = finished
                    log("")
                    log(f"sigma = {sigma}  ->  {dest.relative_to(run_dir)}/   "
                        f"already complete, skipping")
                    continue
                derived = cfg if alpha is None else derive(
                    cfg, **{"monte_carlo.epsilon_alpha": float(alpha)})
                log("")
                log(f"sigma = {sigma}  ->  {dest.relative_to(run_dir)}/   "
                    f"h={derived.for_sigma('ode.step_size', sigma)}  "
                    f"alpha={derived.for_sigma('monte_carlo.epsilon_alpha', sigma)}  "
                    f"M={derived['monte_carlo.num_query_states']}  "
                    f"K={derived['monte_carlo.num_probes']}")
                at = time.perf_counter()
                if sigma not in flows:
                    flows[sigma] = build_target(cfg, sigma)
                summaries[key] = run_sigma(derived, sigma, dest,
                                           flow=flows[sigma], log=log)
                log(f"  {key} done in {_hms(time.perf_counter() - at)}")

        total = time.perf_counter() - started
        log("")
        log(rule)
        what = (f"{len(sigmas)} sigma(s)" if alphas is None else
                f"{len(alphas) * len(sigmas)} run(s) "
                f"({len(alphas)} alphas x {len(sigmas)} sigmas)")
        log(f" all {what} done in {_hms(total)}  ->  {run_dir}")
        log(rule)

    return {"run_dir": run_dir, "sigmas": sigmas, "alphas": alphas,
            "summaries": summaries, "seconds": total}
