"""Tests for gtmf/target.py."""

import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtmf import schedule                      # noqa: E402
from gtmf.config import Config                 # noqa: E402
from gtmf.target import build_target           # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REAL_CENTERS = Path("/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/"
                    "latents_8_mean_fp16/centers_fp16.npy")

BASE = {
    "experiment": "t", "gmm": {"component_sigma": [0.3]},
    "time": {"num_points": 50, "t_min": 0.0, "t_max": 0.98},
    "monte_carlo": {"num_query_states": 4, "num_probes": 2, "scheme": "central"},
    "ode": {"integrator": "rk4", "step_size": 0.015625},
    "output": {"output_dir": "results"},
}


def cfg_with(tmp_path, data, compute=None, gmm=None):
    d = yaml.safe_load(yaml.safe_dump(BASE))
    d["data"] = data
    d["compute"] = compute or {"device": "cpu", "dtype": "float64"}
    if gmm:
        d["gmm"].update(gmm)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d))
    return Config.load(p)


# --- synthetic --------------------------------------------------------------- #

@pytest.mark.parametrize("sigma", [0.01, 0.3, 1.0])
@pytest.mark.parametrize("t", [0.0, 0.4, 1.0])
def test_a_single_component_at_the_origin_is_the_single_gaussian(tmp_path, sigma, t):
    """The whole reason the synthetic source exists: this config, and no special
    code path, gives the spec 7.1 field v = k_t x."""
    flow = build_target(cfg_with(tmp_path, {
        "source": "synthetic",
        "synthetic": {"num_components": 1, "dim": 8, "centers": "zeros"}}), sigma)
    x = torch.randn((5, 8), generator=torch.Generator().manual_seed(0),
                    dtype=torch.float64)
    assert torch.allclose(flow.velocity(x, t), schedule.k_t(t, sigma) * x)


def test_the_shipped_sanity_config_builds_the_closed_form_field():
    """Not a hand-made config -- the actual file phase 11 will run."""
    cfg = Config.load(REPO / "configs/sanity_gaussian.yaml")
    for sigma in cfg["gmm.component_sigma"]:
        flow = build_target(cfg, sigma)
        assert flow.n == 1 and flow.d == 256
        x = torch.randn((3, 256), generator=torch.Generator().manual_seed(1),
                        dtype=torch.float64)
        assert torch.allclose(flow.velocity(x, 0.5), schedule.k_t(0.5, sigma) * x)


def test_synthetic_normal_centres_are_seed_reproducible(tmp_path):
    data = {"source": "synthetic",
            "synthetic": {"num_components": 16, "dim": 6, "centers": "normal"}}
    a = build_target(cfg_with(tmp_path, data), 0.3)
    b = build_target(cfg_with(tmp_path, data), 0.3)
    assert torch.equal(a.mu, b.mu)
    assert a.n == 16 and a.d == 6


def test_squared_norms_are_correct_however_the_target_was_built(tmp_path):
    flow = build_target(cfg_with(tmp_path, {
        "source": "synthetic",
        "synthetic": {"num_components": 32, "dim": 6, "centers": "normal"}}), 0.3)
    assert torch.allclose(flow.sq, flow.mu.pow(2).sum(dim=1))


# --- config plumbing ----------------------------------------------------------- #

@pytest.mark.parametrize("dtype_name,dtype", [("float32", torch.float32),
                                              ("float64", torch.float64)])
def test_dtype_comes_from_the_config(tmp_path, dtype_name, dtype):
    flow = build_target(cfg_with(
        tmp_path,
        {"source": "synthetic", "synthetic": {"num_components": 4, "dim": 6}},
        compute={"device": "cpu", "dtype": dtype_name}), 0.3)
    assert flow.mu.dtype == dtype and flow.sq.dtype == dtype


def test_pruning_cutoff_is_passed_through(tmp_path):
    data = {"source": "synthetic", "synthetic": {"num_components": 4, "dim": 6}}
    assert build_target(cfg_with(tmp_path, data), 0.3).tau is None
    assert build_target(cfg_with(tmp_path, data,
                                 gmm={"responsibility_log_cutoff": 30.0}), 0.3).tau == 30.0


def test_chunk_size_auto_leaves_the_default_alone(tmp_path):
    data = {"source": "synthetic", "synthetic": {"num_components": 4, "dim": 6}}
    auto = build_target(cfg_with(tmp_path, data,
                                 compute={"device": "cpu", "dtype": "float64",
                                          "chunk_size": "auto"}), 0.3)
    fixed = build_target(cfg_with(tmp_path, data,
                                  compute={"device": "cpu", "dtype": "float64",
                                           "chunk_size": 7}), 0.3)
    assert auto.chunk != 7 and fixed.chunk == 7


def test_an_unknown_source_is_caught_at_load(tmp_path):
    """Config.validate() rejects it before build_target ever runs, so the run
    stops at load rather than after the centres have been read."""
    from gtmf.config import ConfigError
    with pytest.raises(ConfigError, match="data.source"):
        cfg_with(tmp_path, {"source": "hdf5"})


def test_build_target_also_refuses_an_unknown_source(tmp_path):
    """Belt and braces: a Config built in code bypasses validate()."""
    cfg = cfg_with(tmp_path, {"source": "synthetic",
                              "synthetic": {"num_components": 2, "dim": 4}})
    cfg.data["data"]["source"] = "hdf5"
    with pytest.raises(ValueError, match="data.source"):
        build_target(cfg, 0.3)


def test_an_unknown_synthetic_centre_kind_raises(tmp_path):
    with pytest.raises(ValueError, match="zeros"):
        build_target(cfg_with(tmp_path, {
            "source": "synthetic",
            "synthetic": {"num_components": 2, "dim": 4, "centers": "uniform"}}), 0.3)


def test_an_unknown_dtype_raises(tmp_path):
    with pytest.raises(ValueError, match="compute.dtype"):
        build_target(cfg_with(
            tmp_path, {"source": "synthetic", "synthetic": {"num_components": 2, "dim": 4}},
            compute={"device": "cpu", "dtype": "bfloat16"}), 0.3)


# --- the real centres ------------------------------------------------------------ #

@pytest.mark.skipif(not REAL_CENTERS.exists(), reason="production centers not built")
def test_the_shipped_production_config_loads_every_centre():
    cfg = Config.load(REPO / "configs/wavg_imagenet.yaml")
    # The shipped config says device: cuda, which is right for the cluster. This
    # test is about the data path, so run it where the tests run.
    cfg.data["compute"]["device"] = "cpu"
    flow = build_target(cfg, 0.01)
    assert flow.n == 1_281_167 and flow.d == 256
    assert flow.mu.dtype == torch.float32
    # scaled latents: per-coordinate std 0.589, E||mu||^2/d = 0.3516 (CLAUDE.md)
    assert float(flow.mu.std()) == pytest.approx(0.589, rel=0.02)
    assert float(flow.sq.mean()) / flow.d == pytest.approx(0.3516, rel=0.02)


@pytest.mark.skipif(not REAL_CENTERS.exists(), reason="production centers not built")
def test_a_development_subset_is_honoured(tmp_path):
    cfg = cfg_with(tmp_path, {
        "source": "latents",
        "latent_root": str(REAL_CENTERS.parent),
        "num_target_samples": 5000,
        "apply_sd_scale": True}, compute={"device": "cpu", "dtype": "float32"})
    assert build_target(cfg, 0.3).n == 5000
