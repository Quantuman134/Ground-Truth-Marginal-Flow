"""Figures and their underlying numbers -- phase 10, spec section 9.

Reads only what a run persisted, so replotting never triggers a re-run: that is
why phase 7 keeps the per-query raw estimates alongside every aggregate.

Every figure is written as PNG and PDF with a CSV of exactly the numbers drawn,
so a figure in a paper can be traced back to the values behind it without
re-deriving them from the run.
"""

import csv
import json
from pathlib import Path

import numpy as np

from . import schedule

CSV_NAME, SUMMARY_NAME = "w_avg.csv", "summary.json"

# numpy renamed trapz in 2.0; the project runs on both sides of that.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def discover(run_dir):
    """Every finished curve under a run directory.

    Handles both layouts a run can have -- `sigma_<s>/` for a single run and
    `alpha_<a>/sigma_<s>/` for a sweep -- by looking for the CSV rather than by
    assuming a depth.

    Returns a list of dicts sorted by (alpha, sigma), each with `alpha` (None
    when not swept), `sigma`, `dir`, and the run's own `summary`.
    """
    run_dir = Path(run_dir)
    found = []
    for csv_path in sorted(run_dir.rglob(CSV_NAME)):
        d = csv_path.parent
        summary_path = d / SUMMARY_NAME
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        parent = d.parent.name
        found.append({
            "dir": d,
            "sigma": float(summary.get("sigma", d.name.replace("sigma_", ""))),
            "alpha": float(parent.replace("alpha_", "")) if parent.startswith("alpha_")
                     else None,
            "summary": summary,
        })
    return sorted(found, key=lambda r: (r["alpha"] if r["alpha"] is not None else -1,
                                        r["sigma"]))


def load_curve(directory):
    """The columns of one `w_avg.csv`, as float arrays."""
    with (Path(directory) / CSV_NAME).open() as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def normalize(t, w):
    """Eq. 42, divided by the integral over the ACTUAL grid.

    The spec writes the denominator as an integral over [0, 1], but the grid
    stops at t_max (0.98 in production). Dividing by an integral over [0, 1]
    would mean extrapolating across a region the run never measured -- and the
    integrand rises steeply exactly there. So the divisor is the trapezoid over
    [t[0], t[-1]], and every caller is expected to label the plot with that
    range rather than implying [0, 1].
    """
    area = float(_trapezoid(w, t))
    if area <= 0:
        raise ValueError(f"w_avg integrates to {area}; cannot normalize")
    return w / area, area


def closed_form(t, sigma):
    """sigma^2 / c_t^2 -- exact for a single Gaussian, and what the mixture
    reduces to in the committed regime. Overlaid on every curve so a figure
    shows directly what the run bought beyond analysis."""
    return np.array([schedule.w_exact(float(x), sigma) for x in t])


def peak_note(summary):
    """What to write on the plot about the peak.

    t_peak = 1/(1+sigma^2) falls outside [0, 0.98] for sigma <= 0.1. There the
    sampled argmax is the right-hand grid edge, which is not the peak -- so the
    figure has to say so rather than mark a maximum the run never saw.
    """
    peak = summary.get("peak", {})
    if not peak:
        return None, "peak: unknown (no summary)"
    if not peak.get("in_grid_range"):
        return None, (f"peak not in range (t_peak = "
                      f"{peak['t_peak_closed_form']:.4f} > grid)")
    return peak.get("sampled_argmax_t"), (
        f"t_peak = {peak['t_peak_closed_form']:.4f}, "
        f"sampled argmax {peak.get('sampled_argmax_t'):.4f}")


def with_ext(stem, ext):
    """Append an extension to a stem that may contain dots.

    Path.with_suffix would read the "0.0001" in `raw_alpha0.0001` as an
    extension and replace it, so every alpha in a sweep would write to
    `raw_alpha0.png` -- one file, silently overwritten five times.
    """
    stem = Path(stem)
    return stem.parent / f"{stem.name}.{ext}"


def write_table(path, columns):
    """The numbers behind a figure, as CSV. `columns` is {name: array}."""
    path = Path(path)
    keys = list(columns)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        w.writerows(zip(*(np.asarray(columns[k]).tolist() for k in keys)))
    return path


def save(fig, stem, formats=("png", "pdf")):
    """PNG for reading, PDF for including. Returns the paths written."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        path = with_ext(stem, ext)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        out.append(path)
    return out


# --------------------------------------------------------------------------- #
# the figures themselves
# --------------------------------------------------------------------------- #

def _pyplot():
    """Import matplotlib with a headless backend. Deferred so importing this
    module costs nothing on a compute node that will never draw anything."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def curve_figure(runs, out_stem, normalized=False):
    """w_avg(t) for every sigma in `runs`, with its uncertainty band.

    Drawn per spec 9: the mean, a shaded mean +/- 1.96 SE band, the closed form
    overlaid, and the peak marked when it lies inside the grid. Log y, because
    w spans four orders of magnitude between sigma = 0.01 and 0.6.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    table = {}

    for run in runs:
        c = load_curve(run["dir"])
        t, w, se = c["t"], c["w_avg"], c["stderr"]
        exact = closed_form(t, run["sigma"])
        label = f"sigma={run['sigma']:g}"
        if normalized:
            w, area = normalize(t, w)
            se = se / area
            exact = exact / float(_trapezoid(exact, t))

        line, = ax.plot(t, w, marker="o", ms=3, lw=1.5, label=label)
        # 1.96 SE is Monte-Carlo uncertainty ONLY -- not the finite-difference
        # bias, not the ODE error. The band is narrower than the true error bar.
        ax.fill_between(t, w - 1.96 * se, w + 1.96 * se, alpha=0.25,
                        color=line.get_color(), lw=0)
        ax.plot(t, exact, ls="--", lw=1, color=line.get_color(), alpha=0.65)

        argmax_t, note = peak_note(run["summary"])
        if argmax_t is not None:
            ax.axvline(argmax_t, color=line.get_color(), ls=":", lw=1, alpha=0.7)
        table[f"t_sigma{run['sigma']:g}"] = t
        table[f"w_sigma{run['sigma']:g}"] = w
        table[f"se_sigma{run['sigma']:g}"] = se
        table[f"closed_form_sigma{run['sigma']:g}"] = exact

    grid = f"[{runs[0]['summary'].get('settings', {}).get('time', {}).get('t_min', 0)}, " \
           f"{load_curve(runs[0]['dir'])['t'][-1]:g}]"
    ax.set_xlabel("t   (0 = noise, 1 = data)")
    ax.set_ylabel("normalized w(t)" if normalized else r"$w_{avg}(t)$")
    ax.set_yscale("log")
    ax.set_title(("Normalized " if normalized else "") + r"$w_{avg}(t)$"
                 + (f"   -- normalized over the measured grid {grid}, not [0, 1]"
                    if normalized else "")
                 + "\ndashed = closed form $\\sigma^2/c_t^2$, band = mean $\\pm$ 1.96 SE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    paths = save(fig, out_stem)
    plt.close(fig)
    write_table(with_ext(out_stem, "csv"), table)
    return paths


def plateau_figure(runs, out_stem):
    """w_avg against alpha, one line per t, one panel per sigma.

    THE figure for choosing alpha. A usable alpha sits where the line is flat:
    to the right the finite difference is biased (on the real mixture the nudged
    state crosses into another component's basin), to the left it is eaten by
    fp32 cancellation. This draws the evidence and names nothing -- CLAUDE.md
    requires the value be read off by hand.
    """
    plt = _pyplot()
    sigmas = sorted({r["sigma"] for r in runs})
    alphas = sorted({r["alpha"] for r in runs})
    fig, axes = plt.subplots(1, len(sigmas), figsize=(5 * len(sigmas), 4.2),
                             squeeze=False)
    table = {"alpha": np.array(alphas)}

    for ax, sigma in zip(axes[0], sigmas):
        curves = {r["alpha"]: load_curve(r["dir"])
                  for r in runs if r["sigma"] == sigma}
        times = curves[alphas[0]]["t"]
        for i, t in enumerate(times):
            w = np.array([curves[a]["w_avg"][i] for a in alphas])
            se = np.array([curves[a]["stderr"][i] for a in alphas])
            ax.errorbar(alphas, w, yerr=1.96 * se, marker="o", ms=4, lw=1.3,
                        capsize=2, label=f"t={t:g}")
            table[f"w_sigma{sigma:g}_t{t:g}"] = w
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha$   ($\epsilon = \alpha\,\rho_t$)")
        ax.set_ylabel(r"$w_{avg}$")
        ax.set_title(f"sigma = {sigma:g}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)

    fig.suptitle("Perturbation-size sweep (spec 7.2). A usable alpha sits on a "
                 "FLAT stretch;\nchoose it by hand -- this figure recommends "
                 "nothing.", fontsize=10)
    fig.tight_layout()
    paths = save(fig, out_stem)
    plt.close(fig)
    write_table(with_ext(out_stem, "csv"), table)
    return paths


def accuracy_table(runs):
    """How far each curve sits from the closed form, per spec 9's sanity figure.

    Meaningful for a single-Gaussian target, where the closed form is exact; on
    the real mixture it is only expected to match in the committed regime, so
    the max is reported alongside the committed-regime max.
    """
    out = []
    for run in runs:
        c = load_curve(run["dir"])
        exact = closed_form(c["t"], run["sigma"])
        rel = np.abs(c["w_avg"] - exact) / exact
        committed = c["t"] >= 0.7
        out.append({
            "sigma": run["sigma"], "alpha": run["alpha"],
            "max_rel_error": float(rel.max()),
            "max_rel_error_committed": float(rel[committed].max())
                if committed.any() else float("nan"),
            "median_rel_error": float(np.median(rel)),
        })
    return out


def draw_run(run_dir, out=None):
    """Every figure for a finished run directory. Returns (paths, num_curves).

    The one place that knows which figures a run gets, so `plots.py` and the end
    of `wavg.py` cannot drift into drawing different things.

    Handles both layouts, and gives a sweep one pair of curve figures PER alpha
    rather than overlaying six versions of the same sigma on one axis.
    """
    run_dir = Path(run_dir)
    runs = discover(run_dir)
    if not runs:
        raise FileNotFoundError(f"no {CSV_NAME} found under {run_dir}")

    out = Path(out) if out else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    alphas = sorted({r["alpha"] for r in runs},
                    key=lambda a: (a is not None, a))
    written = []
    for alpha in alphas:
        group = [r for r in runs if r["alpha"] == alpha]
        tag = "" if alpha is None else f"_alpha{alpha:g}"
        written += curve_figure(group, out / f"raw{tag}")
        written += curve_figure(group, out / f"normalized{tag}", normalized=True)

    if len([a for a in alphas if a is not None]) > 1:
        written += plateau_figure([r for r in runs if r["alpha"] is not None],
                                  out / "plateau")

    accuracy = out / "accuracy.json"
    accuracy.write_text(json.dumps(accuracy_table(runs), indent=2) + "\n")
    return written + [accuracy], len(runs)
