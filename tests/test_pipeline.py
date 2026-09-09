"""Tests for gtmf/pipeline.py.

The oracles are the single-Gaussian closed form w = sigma^2/c_t^2, and a direct
count of how many times the velocity field was evaluated -- which is what proves
the ODE was actually solved rather than the interpolant shortcut being taken.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtmf import schedule                                    # noqa: E402
from gtmf.config import Config, ConfigError                  # noqa: E402
from gtmf.integrate import time_grid                         # noqa: E402
from gtmf.marginal_flow import MarginalFlow                   # noqa: E402
from gtmf.pipeline import TimepointResult, run_timepoint      # noqa: E402
from gtmf.rng import make_generator                          # noqa: E402

H = 1 / 64

BASE = {
    "experiment": "t", "gmm": {"component_sigma": [0.3]},
    "time": {"num_points": 50, "t_min": 0.0, "t_max": 0.98},
    "monte_carlo": {"num_query_states": 8, "num_probes": 2, "scheme": "central",
                    "epsilon_alpha": {"default": 1e-3}, "rho_source": "exact"},
    "ode": {"integrator": "rk4", "step_size": H},
    "data": {"source": "synthetic",
             "synthetic": {"num_components": 1, "dim": 8, "centers": "zeros"}},
    "compute": {"device": "cpu", "dtype": "float64"},
    "output": {"output_dir": "results"},
}


def cfg_with(tmp_path, **overrides):
    """A config with `monte_carlo.x=1`-style dotted overrides applied."""
    d = yaml.safe_load(yaml.safe_dump(BASE))
    for dotted, value in overrides.items():
        node = d
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return Config.load(p)


def single_gaussian(sigma, d=8, dtype=torch.float64):
    """N = 1 at the origin: the mixture IS N(0, sigma^2 I), so w has a closed form."""
    return MarginalFlow(torch.zeros(1, d, dtype=dtype), sigma)


class CountingFlow:
    """Wraps a MarginalFlow and counts velocity evaluations.

    The whole point of the phase gate: a real solve calls the field 4S times for
    RK4 and S times for Euler. Reaching the endpoint through the interpolant
    instead would call it ZERO times and still return a plausible number, so the
    count is what separates the two.
    """

    def __init__(self, flow):
        self.flow = flow
        self.calls = 0

    def velocity(self, x, t):
        self.calls += 1
        return self.flow.velocity(x, t)

    def __getattr__(self, name):
        return getattr(self.flow, name)


# --- the phase gate: the ODE was actually solved ----------------------------- #

@pytest.mark.parametrize("t", [0.0, 0.3, 0.75, 0.98])
def test_rk4_evaluates_the_field_four_times_per_step(tmp_path, t):
    """4S evaluations, S from the integrator's own step policy. Zero would mean
    the endpoint was reached without solving anything."""
    flow = CountingFlow(single_gaussian(0.3))
    run_timepoint(flow, cfg_with(tmp_path), 0.3, t, make_generator(0))
    steps, _ = time_grid(t, 1.0, H)
    assert flow.calls == 4 * steps > 0


@pytest.mark.parametrize("t", [0.0, 0.5, 0.9])
def test_euler_evaluates_the_field_once_per_step(tmp_path, t):
    """The same gate for the other integrator -- S, not 4S, so the count is
    tracking the method rather than a coincidence."""
    flow = CountingFlow(single_gaussian(0.3))
    run_timepoint(flow, cfg_with(tmp_path, **{"ode.integrator": "euler"}),
                  0.3, t, make_generator(0))
    steps, _ = time_grid(t, 1.0, H)
    assert flow.calls == steps > 0


def test_at_t_equals_one_there_is_nothing_to_integrate(tmp_path):
    """Phi(1,1) = I, so w = ||I||_F^2/d = 1 and the field is never evaluated."""
    flow = CountingFlow(single_gaussian(0.3))
    r = run_timepoint(flow, cfg_with(tmp_path, **{"monte_carlo.num_query_states": 4096,
                                                  "monte_carlo.num_probes": 16}),
                      0.3, 1.0, make_generator(0))
    assert flow.calls == 0
    assert r.w_avg == pytest.approx(1.0, rel=0.02)


# --- the closed form --------------------------------------------------------- #

@pytest.mark.parametrize("sigma", [0.01, 0.1, 0.3, 0.6])
@pytest.mark.parametrize("t", [0.0, 0.3, 0.7, 0.98])
def test_the_single_gaussian_target_reproduces_the_closed_form(tmp_path, sigma, t):
    """w = sigma^2 / c_t^2, spec Eq. 32 -- an oracle that knows nothing about
    this code path. Step size follows the production tier for sigma = 0.01."""
    h = 1 / 512 if sigma == 0.01 else H
    cfg = cfg_with(tmp_path, **{"gmm.component_sigma": [sigma], "ode.step_size": h,
                                "monte_carlo.num_query_states": 2048,
                                "monte_carlo.num_probes": 8})
    r = run_timepoint(single_gaussian(sigma), cfg, sigma, t, make_generator(1))
    assert r.w_avg == pytest.approx(schedule.w_exact(t, sigma), rel=0.02)


def test_the_error_bar_shrinks_as_one_over_sqrt_m(tmp_path):
    """SE is the estimator's own Monte-Carlo error: 64x the draws, ~8x tighter."""
    def se(m):
        cfg = cfg_with(tmp_path, **{"monte_carlo.num_query_states": m})
        return run_timepoint(single_gaussian(0.3), cfg, 0.3, 0.5,
                             make_generator(2)).stderr
    assert se(64) / se(4096) == pytest.approx(8.0, rel=0.4)


# --- the perturbation size --------------------------------------------------- #

@pytest.mark.parametrize("t", [0.2, 0.9])
def test_eps_is_alpha_times_rho_t_not_c_t(tmp_path, t):
    """rho_t includes the spread of the centres; c_t does not. With centres off
    the origin the two differ, and eps must follow rho_t."""
    mu = torch.full((4, 8), 2.0, dtype=torch.float64)
    flow = MarginalFlow(mu, 0.3)
    cfg = cfg_with(tmp_path, **{"monte_carlo.epsilon_alpha": {"default": 1e-3}})
    r = run_timepoint(flow, cfg, 0.3, t, make_generator(3))
    assert r.rho_t == pytest.approx(flow.rho_t_exact(t))
    assert r.eps == pytest.approx(1e-3 * flow.rho_t_exact(t))
    assert r.rho_t != pytest.approx(schedule.c_t(t, 0.3))     # the two differ here


def test_empirical_rho_tracks_the_exact_one(tmp_path):
    """Same quantity, one measured and one closed form, so they must agree --
    and the exact one must not depend on the draw."""
    flow = MarginalFlow(torch.full((16, 8), 1.5, dtype=torch.float64), 0.3)
    kw = {"monte_carlo.num_query_states": 8192}
    exact = run_timepoint(flow, cfg_with(tmp_path, **kw), 0.3, 0.4, make_generator(4))
    emp = run_timepoint(flow, cfg_with(tmp_path, **{**kw,
                        "monte_carlo.rho_source": "empirical"}), 0.3, 0.4,
                        make_generator(5))
    assert emp.rho_t == pytest.approx(exact.rho_t, rel=0.02)
    assert exact.rho_t == pytest.approx(flow.rho_t_exact(0.4))


# --- reproducibility and shape ----------------------------------------------- #

def test_the_same_seed_reproduces_the_run(tmp_path):
    cfg = cfg_with(tmp_path)
    a = run_timepoint(single_gaussian(0.3), cfg, 0.3, 0.5, make_generator(6))
    b = run_timepoint(single_gaussian(0.3), cfg, 0.3, 0.5, make_generator(6))
    c = run_timepoint(single_gaussian(0.3), cfg, 0.3, 0.5, make_generator(7))
    assert torch.equal(a.w_per_query, b.w_per_query) and a.w_avg == b.w_avg
    assert not torch.equal(a.w_per_query, c.w_per_query)


def test_the_result_carries_what_step_4_has_to_write(tmp_path):
    """w_avg.csv needs t/w_avg/stderr/rho_t/eps/seconds; raw.npz needs the
    per-query values on the CPU."""
    cfg = cfg_with(tmp_path, **{"monte_carlo.num_query_states": 16})
    r = run_timepoint(single_gaussian(0.3), cfg, 0.3, 0.5, make_generator(8))
    assert isinstance(r, TimepointResult)
    assert r.w_per_query.shape == (16,) and r.w_per_query.device.type == "cpu"
    for field in (r.t, r.w_avg, r.stderr, r.rho_t, r.eps, r.seconds):
        assert isinstance(field, float)
    assert r.t == 0.5 and r.seconds > 0.0
    assert r.w_avg == pytest.approx(float(r.w_per_query.mean()))


@pytest.mark.parametrize("scheme", ["one_sided", "central"])
def test_both_schemes_land_on_the_closed_form(tmp_path, scheme):
    cfg = cfg_with(tmp_path, **{"monte_carlo.scheme": scheme,
                                "monte_carlo.num_query_states": 2048,
                                "monte_carlo.num_probes": 8})
    r = run_timepoint(single_gaussian(0.3), cfg, 0.3, 0.6, make_generator(9))
    assert r.w_avg == pytest.approx(schedule.w_exact(0.6, 0.3), rel=0.02)


# --- the settings that must not be guessed ----------------------------------- #

def test_a_null_alpha_stops_the_run(tmp_path):
    """How the production config ships: no hand-chosen alpha, so no run."""
    cfg = cfg_with(tmp_path, **{"monte_carlo.epsilon_alpha": {"default": None}})
    with pytest.raises(ConfigError, match="epsilon_alpha"):
        run_timepoint(single_gaussian(0.3), cfg, 0.3, 0.5, make_generator(10))


def test_alpha_and_h_are_taken_per_sigma(tmp_path):
    """by_sigma must win over default, or a sweep silently runs one sigma on
    another's settings."""
    cfg = cfg_with(tmp_path, **{
        "gmm.component_sigma": [0.01, 0.3],
        "monte_carlo.epsilon_alpha": {"default": 1e-3, "by_sigma": {0.01: 5e-3}},
        "ode.step_size": {"default": H, "by_sigma": {0.01: 1 / 512}}})
    flow = CountingFlow(single_gaussian(0.01))
    r = run_timepoint(flow, cfg, 0.01, 0.5, make_generator(11))
    assert r.eps == pytest.approx(5e-3 * flow.rho_t_exact(0.5))
    assert flow.calls == 4 * time_grid(0.5, 1.0, 1 / 512)[0]


@pytest.mark.parametrize("key,value,match", [
    ("ode.integrator", "midpoint", "ode.integrator"),
    ("monte_carlo.rho_source", "emprical", "rho_source"),
])
def test_an_unknown_choice_is_caught_at_load(tmp_path, key, value, match):
    """Both are rejected by Config.validate, i.e. before the centres are read --
    which is where a typo should stop, not after 1.3 GB of loading."""
    with pytest.raises(ConfigError, match=match):
        cfg_with(tmp_path, **{key: value})


@pytest.mark.parametrize("key,value,match", [
    ("integrator", "midpoint", "ode.integrator"),
    ("rho_source", "emprical", "rho_source"),
])
def test_run_timepoint_guards_them_too(tmp_path, key, value, match):
    """Second line of defence. Config(data) does not validate -- only
    Config.load does -- so an unvalidated config must not reach the solver and
    silently fall back to a default."""
    data = yaml.safe_load(yaml.safe_dump(BASE))
    if key == "integrator":
        data["ode"]["integrator"] = value
    else:
        data["monte_carlo"]["rho_source"] = value
    with pytest.raises(ValueError, match=match):
        run_timepoint(single_gaussian(0.3), Config(data), 0.3, 0.5,
                      make_generator(12))


# --- GPU --------------------------------------------------------------------- #

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU on this machine")
def test_it_runs_on_cuda_and_still_hits_the_closed_form(tmp_path):
    """fp32 on the GPU -- the arrangement the production config actually asks
    for, including the device-matched generator."""
    cfg = cfg_with(tmp_path, **{"compute.device": "cuda", "compute.dtype": "float32",
                                "monte_carlo.num_query_states": 4096,
                                "monte_carlo.num_probes": 8})
    flow = MarginalFlow(torch.zeros(1, 256, device="cuda"), 0.3)
    r = run_timepoint(flow, cfg, 0.3, 0.7, make_generator(13, "cuda"))
    assert r.w_avg == pytest.approx(schedule.w_exact(0.7, 0.3), rel=0.02)
    assert r.w_per_query.device.type == "cpu"       # moved off the GPU for step 4


# =========================================================================== #
# Step 4 -- run_sigma
# =========================================================================== #

import csv as _csv                                          # noqa: E402
import json                                                 # noqa: E402

import numpy as np                                          # noqa: E402

from gtmf.pipeline import (CSV_COLUMNS, run_sigma,          # noqa: E402
                           measurement_times, _stream)


def sigma_cfg(tmp_path, **overrides):
    """A config whose target build_target can construct on its own."""
    return cfg_with(tmp_path, **{"time.num_points": 5, "time.t_max": 0.98,
                                 "monte_carlo.num_query_states": 8, **overrides})


# --- the time grid ----------------------------------------------------------- #

def test_the_grid_is_uniform_and_hits_both_ends(tmp_path):
    grid = measurement_times(sigma_cfg(tmp_path, **{"time.num_points": 5}))
    assert grid == [0.0, 0.245, 0.49, 0.735, 0.98]
    assert all(isinstance(t, float) and not isinstance(t, np.floating) for t in grid)


# --- independent random streams ---------------------------------------------- #

def test_query_and_probe_streams_differ_even_at_the_same_seed():
    """Both shipped configs say query_states: 0 AND probes: 0. Seeding the two
    generators with that number directly would make the probe directions a copy
    of the query-state draws, and w_hat needs u independent of x_t."""
    a = torch.randn(4, 3, generator=_stream(0, 0, 7, torch.device("cpu")))
    b = torch.randn(4, 3, generator=_stream(0, 1, 7, torch.device("cpu")))
    assert not torch.equal(a, b)


def test_each_timepoint_gets_its_own_reproducible_stream():
    """Same (seed, stream, index) repeats; a different index does not. This is
    what makes a resumed run continue the same experiment."""
    dev = torch.device("cpu")
    same = [torch.randn(3, generator=_stream(5, 0, 2, dev)) for _ in range(2)]
    assert torch.equal(*same)
    assert not torch.equal(same[0], torch.randn(3, generator=_stream(5, 0, 3, dev)))


# --- the output tree --------------------------------------------------------- #

def test_it_writes_the_three_files_step_10_reads(tmp_path):
    out = tmp_path / "sigma_0.3"
    summary = run_sigma(sigma_cfg(tmp_path), 0.3, out, log=lambda *_: None)

    with (out / "w_avg.csv").open() as fh:
        rows = list(_csv.DictReader(fh))
    assert [r for r in rows[0]] == list(CSV_COLUMNS)
    assert len(rows) == 5

    raw = np.load(out / "raw.npz")
    assert raw["w"].shape == (5, 8)                      # (num_points, M)
    assert np.allclose(raw["t"], measurement_times(sigma_cfg(tmp_path)))

    assert json.loads((out / "summary.json").read_text()) == summary


def test_the_csv_agrees_with_the_raw_it_sits_beside(tmp_path):
    """w_avg must be the mean of the per-query values stored for that row, or
    the figures and the table are telling different stories."""
    out = tmp_path / "s"
    run_sigma(sigma_cfg(tmp_path), 0.3, out, log=lambda *_: None)
    with (out / "w_avg.csv").open() as fh:
        rows = list(_csv.DictReader(fh))
    raw = np.load(out / "raw.npz")
    for row, w in zip(rows, raw["w"]):
        assert float(row["w_avg"]) == pytest.approx(float(w.mean()), rel=1e-6)


def test_the_curve_lands_on_the_closed_form_at_every_t(tmp_path):
    """End to end for a whole sigma, against sigma^2/c_t^2."""
    cfg = sigma_cfg(tmp_path, **{"monte_carlo.num_query_states": 2048,
                                 "monte_carlo.num_probes": 8})
    run_sigma(cfg, 0.3, tmp_path / "s", log=lambda *_: None)
    with (tmp_path / "s" / "w_avg.csv").open() as fh:
        for row in _csv.DictReader(fh):
            t = float(row["t"])
            assert float(row["w_avg"]) == pytest.approx(schedule.w_exact(t, 0.3),
                                                        rel=0.03)


# --- the peak caveat --------------------------------------------------------- #

@pytest.mark.parametrize("sigma", [0.01, 0.1])
def test_a_peak_outside_the_grid_is_reported_as_such(tmp_path, sigma):
    """t_peak = 1/(1+sigma^2) is 0.9999 and 0.9901 -- both past t_max = 0.98.
    The sampled argmax would be the grid edge, which is not the peak."""
    cfg = sigma_cfg(tmp_path, **{"gmm.component_sigma": [sigma],
                                 "ode.step_size": 1 / 64})
    peak = run_sigma(cfg, sigma, tmp_path / "s", log=lambda *_: None)["peak"]
    assert peak["in_grid_range"] is False
    assert "peak not in range" in peak["note"]
    assert "sampled_argmax_t" not in peak


@pytest.mark.parametrize("sigma", [0.3, 0.6])
def test_a_peak_inside_the_grid_is_reported_with_its_location(tmp_path, sigma):
    """t_peak = 0.9174 and 0.7353 -- both inside, so an argmax is meaningful."""
    cfg = sigma_cfg(tmp_path, **{"gmm.component_sigma": [sigma]})
    peak = run_sigma(cfg, sigma, tmp_path / "s", log=lambda *_: None)["peak"]
    assert peak["in_grid_range"] is True
    assert peak["t_peak_closed_form"] == pytest.approx(1 / (1 + sigma ** 2))
    assert peak["sampled_argmax_t"] in measurement_times(cfg)


# --- resuming ---------------------------------------------------------------- #

def test_a_killed_run_resumes_where_it_stopped(tmp_path):
    """Truncate the outputs to 2 of 5 rows, rerun, and the finished curve must
    be identical to one computed in a single pass -- the point of seeding per
    timepoint rather than advancing one long-lived generator."""
    cfg = sigma_cfg(tmp_path)
    whole = tmp_path / "whole"
    run_sigma(cfg, 0.3, whole, log=lambda *_: None)
    reference = np.load(whole / "raw.npz")["w"]

    part = tmp_path / "part"
    run_sigma(cfg, 0.3, part, log=lambda *_: None)
    with (part / "w_avg.csv").open() as fh:
        rows = list(_csv.DictReader(fh))[:2]
    with (part / "w_avg.csv").open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader(); writer.writerows(rows)
    kept = np.load(part / "raw.npz")
    np.savez(part / "raw.npz", **{k: kept[k][:2] for k in kept})

    messages = []
    run_sigma(cfg, 0.3, part, log=messages.append)
    assert any("resuming at t index 2" in m for m in messages)
    assert np.allclose(np.load(part / "raw.npz")["w"], reference)


def test_a_completed_sigma_recomputes_nothing(tmp_path):
    cfg = sigma_cfg(tmp_path)
    out = tmp_path / "s"
    run_sigma(cfg, 0.3, out, log=lambda *_: None)
    flow = CountingFlow(single_gaussian(0.3))
    run_sigma(cfg, 0.3, out, flow=flow, log=lambda *_: None)
    assert flow.calls == 0


def test_resuming_onto_a_different_grid_refuses(tmp_path):
    """The failure this prevents is a curve stitched from two different runs --
    smooth, plausible, and not an experiment anyone performed."""
    out = tmp_path / "s"
    run_sigma(sigma_cfg(tmp_path, **{"time.num_points": 5}), 0.3, out,
              log=lambda *_: None)
    with pytest.raises(ValueError, match="different settings"):
        run_sigma(sigma_cfg(tmp_path, **{"time.num_points": 7}), 0.3, out,
                  log=lambda *_: None)


def test_the_grid_check_still_guards_when_there_is_no_summary(tmp_path):
    """Second line of defence: settings are read from summary.json, so without
    one the stored t values are all there is to check against."""
    out = tmp_path / "s"
    run_sigma(sigma_cfg(tmp_path, **{"time.num_points": 5}), 0.3, out,
              log=lambda *_: None)
    (out / "summary.json").unlink()
    with pytest.raises(ValueError, match="different time grid|config changed"):
        run_sigma(sigma_cfg(tmp_path, **{"time.num_points": 7}), 0.3, out,
                  log=lambda *_: None)


def test_a_raw_file_out_of_step_with_the_csv_refuses(tmp_path):
    out = tmp_path / "s"
    run_sigma(sigma_cfg(tmp_path), 0.3, out, log=lambda *_: None)
    kept = np.load(out / "raw.npz")
    np.savez(out / "raw.npz", **{k: kept[k][:3] for k in kept})
    with pytest.raises(ValueError, match="row for row"):
        run_sigma(sigma_cfg(tmp_path), 0.3, out, log=lambda *_: None)


# --- progress and summary ---------------------------------------------------- #

def test_progress_carries_an_eta_for_every_timepoint(tmp_path):
    """Spec 10.6 -- the line has to survive being read from a log file."""
    messages = []
    run_sigma(sigma_cfg(tmp_path), 0.3, tmp_path / "s", log=messages.append)
    assert len(messages) == 5
    assert all("ETA" in m and "elapsed" in m for m in messages)
    assert "[  5/5]" in messages[-1]


def test_the_summary_records_the_settings_the_run_actually_used(tmp_path):
    """resolved_config.yaml says what was asked for; this says what was used --
    including the per-sigma alpha and h, already resolved."""
    cfg = sigma_cfg(tmp_path, **{
        "gmm.component_sigma": [0.01, 0.3],
        "monte_carlo.epsilon_alpha": {"default": 1e-3, "by_sigma": {0.01: 5e-3}},
        "ode.step_size": {"default": 1 / 64, "by_sigma": {0.01: 1 / 512}}})
    s = run_sigma(cfg, 0.01, tmp_path / "s", log=lambda *_: None)
    assert s["sigma"] == 0.01
    assert s["settings"]["monte_carlo"]["epsilon_alpha"] == 5e-3
    assert s["settings"]["ode"]["step_size"] == 1 / 512
    assert s["num_centers"] == 1 and s["dim"] == 8
    assert s["timing"]["mean_seconds_per_timepoint"] > 0
    assert s["w_avg_range"][0] <= s["w_avg_range"][1]


# --- resuming must not mix two sets of settings ------------------------------ #
#
# Checking only the time grid let a run resumed after a config edit produce ONE
# curve whose early points used one alpha and whose later points used another,
# with resolved_config.yaml recording only the second. Measured before the fix:
# eps went 0.001 -> 0.0047 -> 0.0026 down a single w_avg.csv.

from gtmf.pipeline import settings_for                        # noqa: E402


class Killed(Exception):
    """Stands in for the SSH drop / scheduler kill a resume exists to survive."""


def half_finished(tmp_path, cfg, sigma=0.3, after=1):
    """Genuinely interrupt a sigma after `after` timepoints.

    `log` is called once per timepoint, after that timepoint has been written,
    so raising there leaves exactly the state a killed process would: `after`
    rows in the CSV and raw file, and a summary.json saying complete=False.
    Truncating the files afterwards instead would leave summary.json claiming
    the run had finished, which is not a state that can actually occur.
    """
    out = tmp_path / "s"
    seen = []

    def log(message):
        seen.append(message)
        if len(seen) >= after:
            raise Killed()

    with pytest.raises(Killed):
        run_sigma(cfg, sigma, out, log=log)
    return out


@pytest.mark.parametrize("key,value", [
    ("monte_carlo.epsilon_alpha", {"default": 9e-3}),
    ("ode.step_size", 1 / 32),
    ("monte_carlo.num_probes", 3),
    ("monte_carlo.scheme", "one_sided"),
    ("monte_carlo.rho_source", "empirical"),
    ("ode.integrator", "euler"),
    ("seeds.query_states", 11),
])
def test_resuming_after_a_settings_change_refuses(tmp_path, key, value):
    """Every setting that moves the numbers must block a resume."""
    out = half_finished(tmp_path, sigma_cfg(tmp_path))
    with pytest.raises(ValueError, match="different settings"):
        run_sigma(sigma_cfg(tmp_path, **{key: value}), 0.3, out,
                  log=lambda *_: None)


def test_the_refusal_names_what_changed(tmp_path):
    out = half_finished(tmp_path, sigma_cfg(tmp_path))
    with pytest.raises(ValueError) as exc:
        run_sigma(sigma_cfg(tmp_path, **{"monte_carlo.epsilon_alpha":
                                         {"default": 9e-3}}), 0.3, out,
                  log=lambda *_: None)
    assert "monte_carlo.epsilon_alpha: 0.001 -> 0.009" in str(exc.value)


def test_an_unchanged_config_still_resumes(tmp_path):
    """The guard must not block the case it exists to protect."""
    cfg = sigma_cfg(tmp_path)
    out = half_finished(tmp_path, cfg)
    messages = []
    run_sigma(cfg, 0.3, out, log=messages.append)
    assert any("resuming at t index 1" in m for m in messages)
    assert len(list(_csv.DictReader((out / "w_avg.csv").open()))) == 5


def test_adding_a_sigma_to_the_sweep_does_not_block_the_others(tmp_path):
    """Settings are RESOLVED per sigma, so extending component_sigma leaves the
    sigmas already computed resumable."""
    base = sigma_cfg(tmp_path, **{"gmm.component_sigma": [0.3]})
    out = half_finished(tmp_path, base)
    wider = sigma_cfg(tmp_path, **{"gmm.component_sigma": [0.3, 0.6]})
    run_sigma(wider, 0.3, out, log=lambda *_: None)
    assert len(list(_csv.DictReader((out / "w_avg.csv").open()))) == 5


# --- the summary is written as it goes ---------------------------------------- #

def test_a_killed_run_still_leaves_a_summary(tmp_path):
    """summary.json used to appear only at the end, so an interrupted sigma left
    no record of its settings -- which is what the resume check reads."""
    out = half_finished(tmp_path, sigma_cfg(tmp_path))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["settings"] == settings_for(sigma_cfg(tmp_path), 0.3)
    assert summary["complete"] is False and summary["num_timepoints"] == 1


def test_complete_is_true_only_when_the_grid_is_finished(tmp_path):
    out = tmp_path / "s"
    summary = run_sigma(sigma_cfg(tmp_path), 0.3, out, log=lambda *_: None)
    assert summary["complete"] is True and summary["num_timepoints"] == 5
    assert json.loads((out / "summary.json").read_text())["complete"] is True


# =========================================================================== #
# derive -- how a sweep changes a knob
# =========================================================================== #

from gtmf.config import Config                                 # noqa: E402
from gtmf.pipeline import ALPHAS, alpha_dir, derive            # noqa: E402


def test_derive_applies_the_override(tmp_path):
    cfg = sigma_cfg(tmp_path)
    assert derive(cfg, **{"monte_carlo.num_probes": 16})["monte_carlo.num_probes"] == 16


def test_derive_replaces_a_mapping_with_a_scalar(tmp_path):
    """How the alpha sweep sets one alpha for every sigma."""
    out = derive(sigma_cfg(tmp_path), **{"monte_carlo.epsilon_alpha": 3e-4})
    assert out.for_sigma("monte_carlo.epsilon_alpha", 0.3) == 3e-4


def test_derive_does_not_mutate_the_source(tmp_path):
    cfg = sigma_cfg(tmp_path)
    before = yaml.safe_dump(cfg.data)
    derive(cfg, **{"monte_carlo.num_probes": 99})
    assert yaml.safe_dump(cfg.data) == before


def test_derive_revalidates(tmp_path):
    """An override must not smuggle past the loader's checks."""
    cfg = sigma_cfg(tmp_path)
    with pytest.raises(ConfigError, match="ode.integrator"):
        derive(cfg, **{"ode.integrator": "midpoint"})
    with pytest.raises(ConfigError, match="at least 1"):
        derive(cfg, **{"monte_carlo.num_probes": 0})


@pytest.mark.parametrize("dotted", ["montecarlo.num_probes", "nope.deep.key"])
def test_derive_refuses_to_invent_a_section(tmp_path, dotted):
    """A typo'd section would sit in the config doing nothing while the real
    setting kept its old value."""
    with pytest.raises(ConfigError, match="no section"):
        derive(sigma_cfg(tmp_path), **{dotted: 1})


def test_derive_can_set_a_leaf_that_did_not_exist(tmp_path):
    assert derive(sigma_cfg(tmp_path), **{"compute.allow_tf32": True})[
        "compute.allow_tf32"] is True


def test_the_spec_alpha_list_is_equation_36():
    assert ALPHAS == (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def test_alpha_directories_sort_and_do_not_collide():
    names = [alpha_dir(Path("s"), a).name for a in ALPHAS]
    assert len(set(names)) == len(ALPHAS)
    assert names[0] == "alpha_0.0001"
