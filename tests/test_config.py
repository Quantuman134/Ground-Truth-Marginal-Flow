"""Tests for gtmf/config.py.

The failure this module exists to prevent is a silently-defaulted setting, which
produces a complete, plausible run of the wrong experiment. So most of these
tests check that something RAISES.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtmf.config import Config, ConfigError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SHIPPED = sorted((REPO / "configs").glob("*.yaml"))

MINIMAL = {
    "experiment": "unit_test",
    "data": {"source": "synthetic"},
    "gmm": {"component_sigma": [0.01, 0.3]},
    "time": {"num_points": 50, "t_min": 0.0, "t_max": 0.98},
    "monte_carlo": {"num_query_states": 8, "num_probes": 2, "scheme": "central",
                    "epsilon_alpha": {"default": None, "by_sigma": {0.01: 1e-3}}},
    "ode": {"integrator": "rk4",
            "step_size": {"default": 0.015625, "by_sigma": {0.01: 0.001953125}}},
    "output": {"output_dir": "results"},
}


def write(tmp_path, **overrides):
    data = yaml.safe_load(yaml.safe_dump(MINIMAL))     # deep copy via yaml
    for dotted, value in overrides.items():
        node = data
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        if value is ...:
            del node[parts[-1]]
        else:
            node[parts[-1]] = value
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


# --- the shipped configs must actually be valid ----------------------------- #

@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_every_shipped_config_loads_and_validates(path):
    """These are the files the experiment is run from; a typo here is a broken
    run, not a broken test."""
    cfg = Config.load(path)
    assert cfg["experiment"]
    assert len(cfg["gmm.component_sigma"]) >= 1


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_every_shipped_sigma_resolves_a_step_size(path):
    cfg = Config.load(path)
    for sigma in cfg["gmm.component_sigma"]:
        assert cfg.for_sigma("ode.step_size", sigma) > 0


def test_the_two_step_size_tiers_resolve_as_recorded():
    """sigma=0.01 gets 1/512; everything else gets 1/64 (CLAUDE.md)."""
    cfg = Config.load(REPO / "configs/wavg_imagenet.yaml")
    assert cfg.for_sigma("ode.step_size", 0.01) == pytest.approx(1 / 512)
    for sigma in (0.1, 0.3, 0.6):
        assert cfg.for_sigma("ode.step_size", sigma) == pytest.approx(1 / 64)


# --- round trip -------------------------------------------------------------- #

def test_dump_reload_is_identical(tmp_path):
    """resolved_config.yaml has to be a faithful record of the run, not an
    approximation of one."""
    cfg = Config.load(REPO / "configs/wavg_imagenet.yaml")
    out = cfg.dump(tmp_path / "resolved_config.yaml")
    assert Config.load(out).data == cfg.data


# --- missing settings must raise, not default -------------------------------- #

@pytest.mark.parametrize("missing", [
    "experiment", "gmm.component_sigma", "time.num_points", "time.t_max",
    "monte_carlo.num_probes", "monte_carlo.scheme", "ode.integrator",
    "ode.step_size", "output.output_dir",
])
def test_a_missing_setting_raises_and_names_itself(tmp_path, missing):
    path = write(tmp_path, **{missing: ...})
    with pytest.raises(ConfigError, match=missing.split(".")[-1]):
        Config.load(path)


def test_lookup_without_a_default_raises(tmp_path):
    cfg = Config.load(write(tmp_path))
    with pytest.raises(ConfigError, match="nonexistent"):
        cfg["gmm.nonexistent"]


def test_an_explicit_default_is_honoured(tmp_path):
    cfg = Config.load(write(tmp_path))
    assert cfg.get("gmm.nonexistent", default="fallback") == "fallback"
    assert cfg.get("gmm.component_sigma", default="unused") == [0.01, 0.3]


# --- per-sigma resolution ------------------------------------------------------ #

def test_by_sigma_wins_and_default_fills_the_rest(tmp_path):
    cfg = Config.load(write(tmp_path))
    assert cfg.for_sigma("ode.step_size", 0.01) == pytest.approx(1 / 512)
    assert cfg.for_sigma("ode.step_size", 0.3) == pytest.approx(1 / 64)


def test_a_null_value_raises_when_it_is_used(tmp_path):
    """epsilon_alpha.default is null on purpose: a production run without a
    hand-chosen alpha must stop rather than guess one."""
    cfg = Config.load(write(tmp_path))
    assert cfg.for_sigma("monte_carlo.epsilon_alpha", 0.01) == pytest.approx(1e-3)
    with pytest.raises(ConfigError, match="null for sigma=0.3"):
        cfg.for_sigma("monte_carlo.epsilon_alpha", 0.3)


def test_by_sigma_keys_are_matched_numerically_not_textually(tmp_path):
    """YAML gives 1e-2 and 0.01 as the same float; the lookup must agree."""
    path = write(tmp_path, **{"ode.step_size": {"default": 1.0, "by_sigma": {1e-2: 0.5}}})
    assert Config.load(path).for_sigma("ode.step_size", 0.01) == 0.5


def test_a_by_sigma_key_for_an_unswept_sigma_raises(tmp_path):
    """The quiet one. A typo'd key is ignored otherwise, and the sigma it was
    meant for runs on the default -- a plausible result at the wrong step size."""
    path = write(tmp_path,
                 **{"ode.step_size": {"default": 1.0, "by_sigma": {0.011: 0.5}}})
    with pytest.raises(ConfigError, match="not in gmm.component_sigma"):
        Config.load(path)


def test_a_bare_scalar_means_the_same_value_for_every_sigma(tmp_path):
    """Shorthand for quick runs. Explicit, so there is nothing silent to guard:
    one number was written, one number is used."""
    cfg = Config.load(write(tmp_path, **{"ode.step_size": 0.5}))
    assert all(cfg.for_sigma("ode.step_size", s) == 0.5 for s in (0.01, 0.3))


def test_a_mapping_without_a_default_raises(tmp_path):
    path = write(tmp_path, **{"ode.step_size": {"by_sigma": {0.01: 0.5}}})
    with pytest.raises(ConfigError, match="needs a 'default' key"):
        Config.load(path).for_sigma("ode.step_size", 0.3)


# --- validation ------------------------------------------------------------------ #

@pytest.mark.parametrize("override,message", [
    ({"gmm.component_sigma": []}, "non-empty list"),
    ({"gmm.component_sigma": [0.1, -0.3]}, "must be positive"),
    ({"time.t_min": 0.99}, "below t_max"),
    ({"time.num_points": 1}, "at least 2"),
    ({"monte_carlo.num_probes": 0}, "at least 1"),
    ({"monte_carlo.num_query_states": 0}, "at least 1"),
    ({"monte_carlo.scheme": "two_sided"}, "one_sided"),
    ({"ode.integrator": "dopri5"}, "rk4"),
    ({"data.source": "hdf5"}, "latents"),
])
def test_validation_rejects(tmp_path, override, message):
    with pytest.raises(ConfigError, match=message):
        Config.load(write(tmp_path, **override))


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "nope.yaml")


def test_a_non_mapping_file_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="mapping"):
        Config.load(path)


# --- run layout -------------------------------------------------------------------- #

def test_run_and_sigma_directories_follow_layout_b(tmp_path):
    """One invocation, one directory; the config and log live at the top and each
    sigma gets a subdirectory."""
    cfg = Config.load(write(tmp_path))
    run = cfg.run_dir(stamp="20260908-1432")
    assert run == Path("results/unit_test_20260908-1432")
    assert Config.sigma_dir(run, 0.01) == run / "sigma_0.01"
    assert Config.sigma_dir(run, 0.6) == run / "sigma_0.6"


def test_the_output_root_can_be_overridden(tmp_path):
    cfg = Config.load(write(tmp_path))
    assert cfg.run_dir(root="/scratch/elsewhere", stamp="x") == \
        Path("/scratch/elsewhere/unit_test_x")


def test_two_runs_a_second_apart_do_not_collide(tmp_path):
    cfg = Config.load(write(tmp_path))
    assert cfg.run_dir(stamp="20260908-143200") != cfg.run_dir(stamp="20260908-143201")


def test_contains_treats_a_null_as_absent(tmp_path):
    cfg = Config.load(write(tmp_path))
    assert "gmm.component_sigma" in cfg
    assert "gmm.nonexistent" not in cfg
