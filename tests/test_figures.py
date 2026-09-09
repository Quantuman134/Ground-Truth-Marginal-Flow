"""Tests for gtmf/figures.py and plots.py -- phase 10.

The oracles are analytic: a constant curve has a normalization anyone can do by
hand, and the closed form is the same one the rest of the project checks against.
Figures are checked for being written and for carrying their numbers, not for
looking a particular way.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plots as cli                                        # noqa: E402
from gtmf import figures, schedule                         # noqa: E402

COLUMNS = ("t", "w_avg", "stderr", "rho_t", "eps", "seconds")


def fake_curve(directory, sigma, t, w=None, alpha=None, in_range=True):
    """A results directory holding a plausible curve, written by hand.

    Deliberately not produced by running the pipeline: phase 10 must depend only
    on what a run PERSISTED, so a test that had to compute first would be
    testing the wrong contract.
    """
    directory.mkdir(parents=True, exist_ok=True)
    w = np.array([schedule.w_exact(float(x), sigma) for x in t]) if w is None else w
    with (directory / "w_avg.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(COLUMNS)
        for i, ti in enumerate(t):
            wr.writerow([ti, w[i], 0.01 * w[i], 0.5, 5e-4, 1.0])
    peak = {"t_peak_closed_form": 1 / (1 + sigma ** 2), "in_grid_range": in_range}
    if in_range:
        peak["sampled_argmax_t"] = float(t[int(np.argmax(w))])
        peak["sampled_max_w_avg"] = float(np.max(w))
    else:
        peak["note"] = "peak not in range: past the grid edge"
    (directory / "summary.json").write_text(json.dumps({
        "sigma": sigma, "complete": True, "peak": peak,
        "settings": {"time": {"t_min": float(t[0]), "t_max": float(t[-1])},
                     "monte_carlo": {"epsilon_alpha": alpha}}}))
    return directory


@pytest.fixture
def flat_run(tmp_path):
    """A single run: sigma_<s>/ directly under the run directory."""
    t = np.linspace(0.0, 0.98, 6)
    for sigma, in_range in ((0.1, False), (0.6, True)):
        fake_curve(tmp_path / f"sigma_{sigma:g}", sigma, t, in_range=in_range)
    return tmp_path


@pytest.fixture
def swept_run(tmp_path):
    """A sweep: alpha_<a>/sigma_<s>/."""
    t = np.linspace(0.0, 0.98, 6)
    for alpha in (1e-4, 1e-3, 1e-2):
        for sigma in (0.1, 0.6):
            fake_curve(tmp_path / f"alpha_{alpha:g}" / f"sigma_{sigma:g}",
                       sigma, t, alpha=alpha, in_range=(sigma == 0.6))
    return tmp_path


# --- discovery --------------------------------------------------------------- #

def test_discover_finds_the_flat_layout(flat_run):
    runs = figures.discover(flat_run)
    assert [(r["alpha"], r["sigma"]) for r in runs] == [(None, 0.1), (None, 0.6)]


def test_discover_finds_the_swept_layout(swept_run):
    runs = figures.discover(swept_run)
    assert len(runs) == 6
    assert {r["alpha"] for r in runs} == {1e-4, 1e-3, 1e-2}
    assert [r["alpha"] for r in runs] == sorted(r["alpha"] for r in runs)


def test_discover_returns_nothing_for_an_empty_directory(tmp_path):
    assert figures.discover(tmp_path) == []


# --- normalization (Eq. 42) --------------------------------------------------- #

def test_a_constant_curve_normalizes_to_one_over_the_range():
    """Hand-workable oracle: for w = c on [a, b] the integral is c(b-a), so the
    normalized curve is the constant 1/(b-a) whatever c was."""
    t = np.linspace(0.2, 0.9, 40)
    for c in (1e-3, 1.0, 250.0):
        norm, area = figures.normalize(t, np.full_like(t, c))
        assert area == pytest.approx(c * 0.7)
        assert norm == pytest.approx(np.full_like(t, 1 / 0.7))


def test_normalization_uses_the_measured_grid_not_zero_to_one():
    """CLAUDE.md: never extrapolate into a region that was not measured. A curve
    on [0, 0.5] must be divided by ITS integral, not by one over [0, 1]."""
    t = np.linspace(0.0, 0.5, 21)
    _, area = figures.normalize(t, np.ones_like(t))
    assert area == pytest.approx(0.5)          # not 1.0


def test_the_normalized_curve_integrates_to_one():
    t = np.linspace(0.0, 0.98, 50)
    w = np.array([schedule.w_exact(float(x), 0.3) for x in t])
    norm, _ = figures.normalize(t, w)
    assert float(figures._trapezoid(norm, t)) == pytest.approx(1.0)


def test_a_degenerate_curve_will_not_normalize():
    t = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="cannot normalize"):
        figures.normalize(t, np.zeros_like(t))


# --- the closed form and the peak caveat -------------------------------------- #

@pytest.mark.parametrize("sigma", [0.01, 0.3, 0.6])
def test_closed_form_matches_the_schedule(sigma):
    t = np.linspace(0.0, 0.98, 7)
    assert figures.closed_form(t, sigma) == pytest.approx(
        [schedule.w_exact(float(x), sigma) for x in t])


def test_a_peak_outside_the_grid_is_not_marked(flat_run):
    """sigma=0.1 has t_peak = 0.9901, past the grid. Marking the sampled argmax
    would put a line on the right-hand edge and call it the peak."""
    runs = {r["sigma"]: r for r in figures.discover(flat_run)}
    argmax, note = figures.peak_note(runs[0.1]["summary"])
    assert argmax is None and "peak not in range" in note
    argmax, note = figures.peak_note(runs[0.6]["summary"])
    assert argmax is not None and "t_peak" in note


def test_peak_note_survives_a_missing_summary():
    assert figures.peak_note({}) == (None, "peak: unknown (no summary)")


# --- file naming: the dot trap ------------------------------------------------ #

@pytest.mark.parametrize("stem,ext,expected", [
    ("raw_alpha0.0001", "png", "raw_alpha0.0001.png"),
    ("raw_alpha0.01", "pdf", "raw_alpha0.01.pdf"),
    ("plateau", "csv", "plateau.csv"),
])
def test_extensions_are_appended_not_substituted(tmp_path, stem, ext, expected):
    """Path.with_suffix would read the "0.0001" as an extension and replace it,
    so every alpha in a sweep wrote to raw_alpha0.png -- one file, overwritten
    once per alpha, silently."""
    assert figures.with_ext(tmp_path / stem, ext).name == expected


# --- the figures -------------------------------------------------------------- #

def test_curve_figure_writes_both_formats_and_its_numbers(flat_run, tmp_path):
    out = tmp_path / "fig" / "raw"
    paths = figures.curve_figure(figures.discover(flat_run), out)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    with figures.with_ext(out, "csv").open() as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 6
    assert "w_sigma0.6" in table[0] and "closed_form_sigma0.6" in table[0]


def test_the_normalized_figure_writes_normalized_numbers(flat_run, tmp_path):
    out = tmp_path / "fig" / "normalized"
    figures.curve_figure(figures.discover(flat_run), out, normalized=True)
    with figures.with_ext(out, "csv").open() as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r["t_sigma0.6"]) for r in rows])
    w = np.array([float(r["w_sigma0.6"]) for r in rows])
    assert float(figures._trapezoid(w, t)) == pytest.approx(1.0)


def test_plateau_figure_writes_one_column_per_sigma_and_t(swept_run, tmp_path):
    out = tmp_path / "fig" / "plateau"
    figures.plateau_figure(figures.discover(swept_run), out)
    with figures.with_ext(out, "csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [float(r["alpha"]) for r in rows] == [1e-4, 1e-3, 1e-2]
    assert sum(k.startswith("w_sigma0.6") for k in rows[0]) == 6      # 6 timepoints


def test_accuracy_table_measures_distance_from_the_closed_form(flat_run):
    """These fake curves ARE the closed form, so every error must be zero -- a
    table that reported anything else would be measuring itself wrong."""
    for row in figures.accuracy_table(figures.discover(flat_run)):
        assert row["max_rel_error"] == pytest.approx(0.0, abs=1e-12)
        assert row["median_rel_error"] == pytest.approx(0.0, abs=1e-12)


def test_accuracy_table_sees_a_curve_that_is_wrong(tmp_path):
    t = np.linspace(0.0, 0.98, 6)
    w = np.array([schedule.w_exact(float(x), 0.3) for x in t]) * 1.05
    fake_curve(tmp_path / "sigma_0.3", 0.3, t, w=w)
    row = figures.accuracy_table(figures.discover(tmp_path))[0]
    assert row["max_rel_error"] == pytest.approx(0.05, rel=1e-6)


# --- the CLI ------------------------------------------------------------------ #

def test_the_cli_draws_a_flat_run(flat_run, capsys):
    assert cli.main(["--run", str(flat_run)]) == 0
    out = flat_run / "figures"
    for name in ("raw.png", "raw.pdf", "raw.csv", "normalized.png", "accuracy.json"):
        assert (out / name).is_file(), name
    assert not (out / "plateau.png").exists()      # nothing was swept
    assert "2 curve(s)" in capsys.readouterr().out


def test_the_cli_draws_a_sweep_with_one_figure_per_alpha(swept_run):
    cli.main(["--run", str(swept_run)])
    out = swept_run / "figures"
    for alpha in ("0.0001", "0.001", "0.01"):
        assert (out / f"raw_alpha{alpha}.png").is_file()
        assert (out / f"normalized_alpha{alpha}.png").is_file()
    assert (out / "plateau.png").is_file()


def test_the_cli_refuses_a_directory_with_no_results(tmp_path):
    with pytest.raises(SystemExit, match="no w_avg.csv"):
        cli.main(["--run", str(tmp_path)])


# --- draw_run: one definition, shared by plots.py and wavg.py ---------------- #

def test_draw_run_produces_the_flat_set(flat_run):
    written, count = figures.draw_run(flat_run)
    assert count == 2
    names = {p.name for p in written}
    assert {"raw.png", "raw.pdf", "normalized.png", "normalized.pdf",
            "accuracy.json"} <= names
    assert "plateau.png" not in names          # nothing was swept


def test_draw_run_gives_a_sweep_one_pair_per_alpha(swept_run):
    written, count = figures.draw_run(swept_run)
    assert count == 6
    names = {p.name for p in written}
    for alpha in ("0.0001", "0.001", "0.01"):
        assert f"raw_alpha{alpha}.png" in names
    assert "plateau.png" in names


def test_draw_run_honours_an_output_directory(flat_run, tmp_path):
    out = tmp_path / "elsewhere"
    written, _ = figures.draw_run(flat_run, out=out)
    assert all(p.parent == out for p in written)
    assert not (flat_run / "figures").exists()


def test_draw_run_says_so_when_there_is_nothing_to_draw(tmp_path):
    with pytest.raises(FileNotFoundError, match="no w_avg.csv"):
        figures.draw_run(tmp_path)
