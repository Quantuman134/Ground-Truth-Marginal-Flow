"""Tests for wavg.py and gtmf.pipeline.run_sweep -- phase 7, step 5.

The end-to-end test drives the SHIPPED configs/sanity_gaussian.yaml, trimmed only
in size (points, M, K), so what is exercised is the real file phase 11 will run
rather than a fixture invented to pass.
"""

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wavg                                                    # noqa: E402
from gtmf import schedule                                      # noqa: E402
from gtmf.config import Config, ConfigError                    # noqa: E402
from gtmf.pipeline import prepare_run_dir, run_sweep           # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SANITY = REPO / "configs/sanity_gaussian.yaml"


def trimmed_sanity(tmp_path, **over):
    """The shipped sanity config, cut down so a test finishes in seconds.

    Only the sizes change: source, dtype, the per-sigma step tiers and the sigma
    list are the file's own.
    """
    d = yaml.safe_load(SANITY.read_text())
    d["time"]["num_points"] = 3
    d["monte_carlo"]["num_query_states"] = 128
    d["monte_carlo"]["num_probes"] = 8
    d["gmm"]["component_sigma"] = [0.1, 0.6]
    # The shipped file now sweeps spec Eq. 36 (six alphas). Most tests here are
    # about the single-run layout, so pin one alpha by default; the sweep tests
    # pass their own list through **over.
    d["monte_carlo"]["epsilon_alpha"] = 0.001
    d["ode"]["step_size"] = 0.015625                # 1/64: a SIZE trim like the
                                                    # ones above. The shipped file
                                                    # runs 1/512 everywhere, which
                                                    # is 8x the steps for no extra
                                                    # coverage of what step 5 does.
    d["output"]["output_dir"] = str(tmp_path / "results")

    # Dropping a sigma orphans its by_sigma entries, and Config.validate rightly
    # refuses a by_sigma key naming an unswept sigma -- that guard is what stops
    # a typo'd key from silently running a sigma on the default. So pruning here
    # is a consequence of trimming the sweep, not an extra liberty with the file.
    swept = set(d["gmm"]["component_sigma"])
    for section, key in (("ode", "step_size"), ("monte_carlo", "epsilon_alpha")):
        node = d[section][key]
        if isinstance(node, dict) and node.get("by_sigma"):
            node["by_sigma"] = {k: v for k, v in node["by_sigma"].items()
                                if k in swept}
    for dotted, value in over.items():
        node = d
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    path = tmp_path / "sanity.yaml"
    path.write_text(yaml.safe_dump(d))
    return path


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    """One sweep, shared by every test that only READS its output.

    Tests about collisions, --force and the CLI each need their own run and take
    it; the rest would just be paying for the same computation ten times.
    """
    tmp = tmp_path_factory.mktemp("swept")
    cfg = Config.load(trimmed_sanity(tmp))
    return cfg, run_sweep(cfg, echo=False)


# --- layout B ---------------------------------------------------------------- #

def test_the_output_tree_is_layout_b(swept):
    """resolved_config.yaml and run.log belong to the RUN; each sigma gets a
    subdirectory of its own."""
    run = swept[1]["run_dir"]

    assert (run / "resolved_config.yaml").is_file()
    assert (run / "run.log").is_file()
    assert sorted(p.name for p in run.iterdir() if p.is_dir()) == \
        ["sigma_0.1", "sigma_0.6"]
    for sub in ("sigma_0.1", "sigma_0.6"):
        for name in ("w_avg.csv", "raw.npz", "summary.json"):
            assert (run / sub / name).is_file(), f"{sub}/{name} missing"


def test_the_run_directory_is_named_experiment_and_timestamp(swept):
    run = swept[1]["run_dir"]
    assert run.name.startswith("sanity_gaussian_")
    assert run.parent.name == "results"


def test_two_runs_never_collide(tmp_path):
    """The timestamp exists so a re-run cannot destroy hours of raw data."""
    cfg = Config.load(trimmed_sanity(tmp_path))
    a = run_sweep(cfg, echo=False)["run_dir"]
    b = run_sweep(cfg, run_dir=a.parent / (a.name + "b"), echo=False)["run_dir"]
    assert a != b and a.is_dir() and b.is_dir()


# --- the resolved config ------------------------------------------------------ #

def test_the_resolved_config_round_trips_from_inside_the_run(swept):
    """Reloading the dumped copy must give the same settings -- it is the record
    of what was run, and a figure is only reproducible if it is faithful."""
    source, run = swept[0], swept[1]["run_dir"]
    reloaded = Config.load(run / "resolved_config.yaml")
    assert reloaded.data == source.data
    for sigma in source["gmm.component_sigma"]:
        assert reloaded.for_sigma("ode.step_size", sigma) == \
            source.for_sigma("ode.step_size", sigma)


# --- the closed form, end to end ---------------------------------------------- #

def test_the_sanity_config_lands_on_the_closed_form(swept):
    """Spec 7.1 through the production pipeline: one component at the origin
    makes p_1 exactly N(0, sigma^2 I), so w(t) = sigma^2/c_t^2 everywhere."""
    run = swept[1]["run_dir"]
    for sigma in (0.1, 0.6):
        with (run / f"sigma_{sigma:g}" / "w_avg.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3
        for row in rows:
            t = float(row["t"])
            assert float(row["w_avg"]) == \
                pytest.approx(schedule.w_exact(t, sigma), rel=0.03)


def test_every_swept_sigma_gets_its_own_summary(swept):
    result = swept[1]
    assert sorted(result["summaries"]) == ["0.1", "0.6"]
    for key, summary in result["summaries"].items():
        on_disk = json.loads(
            (result["run_dir"] / f"sigma_{float(key):g}" / "summary.json").read_text())
        assert on_disk == summary and summary["sigma"] == float(key)


# --- the log ------------------------------------------------------------------ #

def test_the_log_mirrors_progress_and_is_flushed(swept):
    """Spec 10.6 -- what is on disk has to be readable after an SSH drop, so the
    per-timepoint lines must reach the file, not sit in a buffer."""
    text = (swept[1]["run_dir"] / "run.log").read_text()
    assert "GTMF -- reference w_avg(t)" in text
    assert "sigma = 0.1" in text and "sigma = 0.6" in text
    assert text.count("ETA") == 6                       # 3 timepoints x 2 sigmas
    assert "all 2 sigma(s) done in" in text


# --- refusing to overwrite ---------------------------------------------------- #

def test_an_existing_output_is_refused(tmp_path):
    pinned = tmp_path / "pinned"
    run_sweep(Config.load(trimmed_sanity(tmp_path)), run_dir=pinned, echo=False)
    with pytest.raises(FileExistsError, match="--force"):
        run_sweep(Config.load(trimmed_sanity(tmp_path)), run_dir=pinned, echo=False)


def test_force_continues_into_it_without_deleting(tmp_path):
    """--force resumes; it must not throw away the raw data already there."""
    pinned = tmp_path / "pinned"
    cfg = Config.load(trimmed_sanity(tmp_path))
    first = run_sweep(cfg, run_dir=pinned, echo=False)
    before = (pinned / "sigma_0.1" / "raw.npz").read_bytes()
    again = run_sweep(cfg, run_dir=pinned, force=True, echo=False)
    assert again["run_dir"] == first["run_dir"]
    assert (pinned / "sigma_0.1" / "raw.npz").read_bytes() == before


def test_an_empty_directory_is_not_a_collision(tmp_path):
    """A launcher that mkdir -p's the path first must still work."""
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    assert prepare_run_dir(pinned) == pinned


# --- the CLI ------------------------------------------------------------------ #

def test_the_cli_runs_a_config_end_to_end(tmp_path, capsys):
    out = tmp_path / "cli_run"
    assert wavg.main(["--config", str(trimmed_sanity(tmp_path)),
                      "--output", str(out), "--quiet"]) == 0
    assert (out / "resolved_config.yaml").is_file()
    assert (out / "sigma_0.6" / "w_avg.csv").is_file()
    assert f"wrote {out}" in capsys.readouterr().out


def test_quiet_keeps_progress_out_of_stdout_but_not_the_log(tmp_path, capsys):
    out = tmp_path / "q"
    wavg.main(["--config", str(trimmed_sanity(tmp_path)), "--output", str(out),
               "--quiet"])
    assert "ETA" not in capsys.readouterr().out
    assert "ETA" in (out / "run.log").read_text()


def test_the_cli_refuses_an_existing_output_without_force(tmp_path):
    args = ["--config", str(trimmed_sanity(tmp_path)), "--output",
            str(tmp_path / "twice"), "--quiet"]
    wavg.main(args)
    with pytest.raises(FileExistsError):
        wavg.main(args)
    assert wavg.main(args + ["--force"]) == 0


def test_a_bad_config_fails_before_any_directory_is_made(tmp_path):
    """Load first, create second: a typo must not leave an empty run directory
    behind that a later --force would then happily continue into."""
    bad = trimmed_sanity(tmp_path, **{"ode.integrator": "midpoint"})
    out = tmp_path / "never"
    with pytest.raises(Exception):
        wavg.main(["--config", str(bad), "--output", str(out), "--quiet"])
    assert not out.exists()


def test_config_is_required(tmp_path):
    with pytest.raises(SystemExit):
        wavg.main([])


# =========================================================================== #
# --alphas: the sweep, run through the same pipeline
# =========================================================================== #

import numpy as np                                              # noqa: E402

from gtmf import schedule as sched                              # noqa: E402
from gtmf.pipeline import run_sigma                             # noqa: E402


@pytest.fixture(scope="module")
def swept_alphas(tmp_path_factory):
    """One two-alpha sweep, shared by the tests that only read its output."""
    tmp = tmp_path_factory.mktemp("alphas")
    cfg = Config.load(trimmed_sanity(
        tmp, **{"monte_carlo.epsilon_alpha": [1e-3, 1e-2]}))
    return run_sweep(cfg, run_dir=tmp / "r", echo=False)["run_dir"]


def test_alphas_give_one_subdirectory_each(swept_alphas):
    run = swept_alphas
    assert sorted(p.name for p in run.iterdir() if p.is_dir()) == \
        ["alpha_0.001", "alpha_0.01"]
    for a in ("alpha_0.001", "alpha_0.01"):
        assert sorted(p.name for p in (run / a).iterdir()) == \
            ["sigma_0.1", "sigma_0.6"]
        for name in ("w_avg.csv", "raw.npz", "summary.json"):
            assert (run / a / "sigma_0.6" / name).is_file()


def test_each_sweep_point_records_its_own_alpha(swept_alphas):
    """The win over a bespoke sweep: every point carries the full settings
    block, so the resume guard protects sweep points too."""
    run = swept_alphas
    for a in (1e-3, 1e-2):
        s = json.loads((run / f"alpha_{a:g}" / "sigma_0.6" / "summary.json").read_text())
        assert s["settings"]["monte_carlo"]["epsilon_alpha"] == a
        assert s["complete"] is True


def test_eps_scales_with_alpha_and_rho_t_does_not(swept_alphas):
    run = swept_alphas
    rows = {}
    for a in (1e-3, 1e-2):
        with (run / f"alpha_{a:g}" / "sigma_0.6" / "w_avg.csv").open() as fh:
            rows[a] = list(csv.DictReader(fh))
    for lo, hi in zip(rows[1e-3], rows[1e-2]):
        assert float(lo["rho_t"]) == pytest.approx(float(hi["rho_t"]))
        assert float(hi["eps"]) / float(lo["eps"]) == pytest.approx(10.0)


def test_a_linear_flow_map_is_alpha_independent(swept_alphas):
    """The machinery test: a single Gaussian has no finite-difference bias, so
    the sweep must come out flat. Structure here would be the sweep's own, and
    would read as physics on the real target."""
    run = swept_alphas
    curves = []
    for a in (1e-3, 1e-2):
        with (run / f"alpha_{a:g}" / "sigma_0.6" / "w_avg.csv").open() as fh:
            curves.append([float(r["w_avg"]) for r in csv.DictReader(fh)])
    for point in zip(*curves):
        assert max(point) / min(point) == pytest.approx(1.0, rel=1e-3)


def test_every_alpha_sees_the_same_query_states(swept_alphas):
    """The comparison must be PAIRED -- seeds come from the timepoint index, not
    from alpha -- or differences between alphas are just sampling noise."""
    run = swept_alphas
    raw = [np.load(run / f"alpha_{a:g}" / "sigma_0.6" / "raw.npz")["w"]
           for a in (1e-3, 1e-2)]
    assert np.allclose(raw[0], raw[1], rtol=1e-6)


def test_a_completed_sweep_does_not_even_build_the_target(tmp_path, monkeypatch):
    cfg = Config.load(trimmed_sanity(
        tmp_path, **{"monte_carlo.epsilon_alpha": [1e-3]}))
    run_sweep(cfg, run_dir=tmp_path / "r", echo=False)
    import gtmf.pipeline as mod
    monkeypatch.setattr(mod, "build_target",
                        lambda *a: pytest.fail("rebuilt a finished sweep"))
    out = run_sweep(cfg, run_dir=tmp_path / "r", force=True, echo=False)
    assert set(out["summaries"]) == {"0.1/0.001", "0.6/0.001"}


def test_no_alphas_keeps_the_flat_layout(tmp_path):
    """Existing behaviour is untouched when the flag is not given."""
    run = run_sweep(Config.load(trimmed_sanity(tmp_path)), run_dir=tmp_path / "r",
                    echo=False)["run_dir"]
    assert (run / "sigma_0.6" / "w_avg.csv").is_file()
    assert not any(p.name.startswith("alpha_") for p in (run / "sigma_0.6").iterdir())


def test_the_cli_sweeps_when_the_config_lists_alphas(tmp_path):
    """No flag: the config is the single source of truth, and the resolved copy
    in the run directory therefore records the sweep that was run."""
    out = tmp_path / "cli"
    cfg = trimmed_sanity(tmp_path, **{"monte_carlo.epsilon_alpha": [1e-3, 1e-2]})
    assert wavg.main(["--config", str(cfg), "--output", str(out), "--quiet"]) == 0
    assert (out / "alpha_0.001" / "sigma_0.6" / "w_avg.csv").is_file()
    assert Config.load(out / "resolved_config.yaml")[
        "monte_carlo.epsilon_alpha"] == [1e-3, 1e-2]


def test_a_list_alpha_cannot_be_resolved_as_a_single_value(tmp_path):
    """for_sigma must refuse a sweep list rather than hand back the list."""
    cfg = Config.load(trimmed_sanity(
        tmp_path, **{"monte_carlo.epsilon_alpha": [1e-3, 1e-2]}))
    with pytest.raises(ConfigError, match="SWEEP"):
        cfg.for_sigma("monte_carlo.epsilon_alpha", 0.6)


@pytest.mark.parametrize("bad", [[], [1e-3, 0], [1e-3, None]])
def test_a_bad_alpha_list_is_rejected_at_load(tmp_path, bad):
    with pytest.raises(ConfigError):
        Config.load(trimmed_sanity(tmp_path, **{"monte_carlo.epsilon_alpha": bad}))


def test_alpha_is_the_outer_loop(tmp_path, monkeypatch):
    """Running order, not just directory order: one alpha's whole sigma sweep
    finishes before the next starts, so a run killed part way leaves complete
    experiments rather than every alpha half done."""
    import gtmf.pipeline as mod
    order, real = [], mod.run_sigma
    monkeypatch.setattr(mod, "run_sigma",
                        lambda cfg, sigma, out, **kw: (
                            order.append((out.parent.name, sigma)),
                            real(cfg, sigma, out, **kw))[1])
    cfg = Config.load(trimmed_sanity(
        tmp_path, **{"monte_carlo.epsilon_alpha": [1e-3, 1e-2]}))
    run_sweep(cfg, run_dir=tmp_path / "r", echo=False)
    assert order == [("alpha_0.001", 0.1), ("alpha_0.001", 0.6),
                     ("alpha_0.01", 0.1), ("alpha_0.01", 0.6)]


def test_targets_are_cached_across_the_outer_alpha_loop(tmp_path, monkeypatch):
    """With alpha outermost, an uncached build would re-read each sigma's
    centres once per alpha -- 1.3 GB a time at production N."""
    import gtmf.pipeline as mod
    calls, real = [], mod.build_target
    monkeypatch.setattr(mod, "build_target",
                        lambda c, s: (calls.append(s), real(c, s))[1])
    cfg = Config.load(trimmed_sanity(
        tmp_path, **{"monte_carlo.epsilon_alpha": [1e-3, 1e-2, 1e-1]}))
    run_sweep(cfg, run_dir=tmp_path / "r", echo=False)
    assert calls == [0.1, 0.6]                    # 2 sigmas, not 3 x 2


def test_the_closing_line_counts_runs_not_sigmas_when_sweeping(tmp_path):
    """With 3 alphas x 2 sigmas the run did 6 experiments; reporting "2 sigmas"
    would understate it by 3x in the log a long run is read back from."""
    cfg = Config.load(trimmed_sanity(
        tmp_path, **{"monte_carlo.epsilon_alpha": [1e-3, 1e-2, 1e-1]}))
    run = run_sweep(cfg, run_dir=tmp_path / "r", echo=False)["run_dir"]
    assert "all 6 run(s) (3 alphas x 2 sigmas) done in" in (run / "run.log").read_text()


# =========================================================================== #
# figures drawn at the end of a run
# =========================================================================== #

def test_a_run_draws_its_own_figures(tmp_path):
    """One invocation, one directory, results AND figures."""
    out = tmp_path / "run"
    assert wavg.main(["--config", str(trimmed_sanity(tmp_path)),
                      "--output", str(out), "--quiet"]) == 0
    for name in ("raw.png", "raw.pdf", "raw.csv", "normalized.png", "accuracy.json"):
        assert (out / "figures" / name).is_file(), name


def test_a_swept_run_draws_the_plateau(tmp_path):
    out = tmp_path / "run"
    cfg = trimmed_sanity(tmp_path, **{"monte_carlo.epsilon_alpha": [1e-3, 1e-2]})
    wavg.main(["--config", str(cfg), "--output", str(out), "--quiet"])
    assert (out / "figures" / "plateau.png").is_file()
    assert (out / "figures" / "raw_alpha0.001.png").is_file()


def test_no_plots_skips_them(tmp_path):
    out = tmp_path / "run"
    wavg.main(["--config", str(trimmed_sanity(tmp_path)), "--output", str(out),
               "--quiet", "--no-plots"])
    assert (out / "sigma_0.6" / "w_avg.csv").is_file()
    assert not (out / "figures").exists()


def test_a_failing_figure_does_not_lose_the_run(tmp_path, monkeypatch, capsys):
    """Hours of compute must not be thrown away because a plot raised. The
    numbers are on disk and plots.py can redraw them."""
    import gtmf.figures as fig
    monkeypatch.setattr(fig, "curve_figure",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = tmp_path / "run"
    assert wavg.main(["--config", str(trimmed_sanity(tmp_path)),
                      "--output", str(out), "--quiet"]) == 0
    assert (out / "sigma_0.6" / "w_avg.csv").is_file()      # results survived
    printed = capsys.readouterr().out
    assert "figures failed" in printed and "plots.py --run" in printed
