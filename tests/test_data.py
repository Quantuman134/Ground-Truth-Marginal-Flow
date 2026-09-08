"""Tests for gtmf/data.py.

Assertions compare against arrays this file constructed, so every expected value is
known independently of the loader.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtmf.data import SD_SCALE, CentersStore, load_centers  # noqa: E402


@pytest.fixture
def centers_dir(tmp_path):
    """A (40, 4, 8, 8) fp16 array whose row r is filled with the value r."""
    raw = np.stack([np.full((4, 8, 8), float(r), dtype=np.float16) for r in range(40)])
    np.save(tmp_path / "centers_fp16.npy", raw)
    (tmp_path / "centers_meta.json").write_text(json.dumps({"num_centers": 40}))
    return tmp_path, raw


def test_shape_is_flattened_to_n_by_d(centers_dir):
    d, raw = centers_dir
    store = load_centers(d)
    assert store.mu.shape == (40, 256)
    assert store.n == 40 and store.d == 256
    assert isinstance(store, CentersStore)


def test_sd_scale_is_applied_at_load(centers_dir):
    d, raw = centers_dir
    store = load_centers(d, apply_sd_scale=True)
    for r in (0, 7, 39):
        assert store.mu[r].allclose(torch.full((256,), r * SD_SCALE), atol=1e-5)


def test_raw_values_survive_when_scaling_is_off(centers_dir):
    d, raw = centers_dir
    store = load_centers(d, apply_sd_scale=False)
    for r in (0, 7, 39):
        assert store.mu[r].allclose(torch.full((256,), float(r)))
    assert store.meta["sd_scale_applied_at_load"] is None


def test_scale_happens_in_fp32_not_fp16(tmp_path):
    """Raw latents reach |18|; scaling must not round-trip through fp16."""
    raw = np.full((1, 4, 8, 8), 18.265625, dtype=np.float16)   # exactly representable
    np.save(tmp_path / "centers_fp16.npy", raw)
    store = load_centers(tmp_path)
    expected = np.float32(18.265625) * np.float32(SD_SCALE)
    assert float(store.mu[0, 0]) == pytest.approx(float(expected), rel=1e-6)


def test_squared_norms_match_a_direct_computation(centers_dir):
    d, _ = centers_dir
    store = load_centers(d)
    assert torch.allclose(store.sq, store.mu.pow(2).sum(dim=1), rtol=1e-6)
    # row r is constant r*SD_SCALE over 256 coords, so ||mu_r||^2 = 256 (r*s)^2
    for r in (3, 17, 39):
        assert float(store.sq[r]) == pytest.approx(256 * (r * SD_SCALE) ** 2, rel=1e-5)


def test_mean_sq_norm_matches_definition(centers_dir):
    d, _ = centers_dir
    store = load_centers(d)
    assert store.mean_sq_norm() == pytest.approx(
        float(store.mu.pow(2).sum(dim=1).mean()) / 256, rel=1e-6)


def test_limit_is_reproducible_sorted_and_seed_sensitive(centers_dir):
    d, _ = centers_dir
    a = load_centers(d, limit=10, seed=1)
    b = load_centers(d, limit=10, seed=1)
    c = load_centers(d, limit=10, seed=2)
    assert a.n == 10
    assert torch.equal(a.mu, b.mu)
    assert not torch.equal(a.mu, c.mu)
    rows = a.mu[:, 0] / SD_SCALE          # recover the row ids from the values
    assert torch.all(rows[1:] > rows[:-1]), "subset must stay in ascending row order"


def test_limit_above_total_loads_everything(centers_dir):
    d, _ = centers_dir
    assert load_centers(d, limit=10_000).n == 40


def test_dtype_and_device_are_honoured(centers_dir):
    d, _ = centers_dir
    store = load_centers(d, dtype=torch.float64)
    assert store.mu.dtype == torch.float64 and store.sq.dtype == torch.float64
    assert store.device.type == "cpu"


def test_meta_records_provenance(centers_dir):
    d, _ = centers_dir
    store = load_centers(d, limit=5, seed=3)
    assert store.meta["num_loaded"] == 5
    assert store.meta["num_available"] == 40
    assert store.meta["subset_seed"] == 3
    assert store.meta["sd_scale_applied_at_load"] == SD_SCALE


def test_accepts_the_array_path_directly(centers_dir):
    d, _ = centers_dir
    assert load_centers(d / "centers_fp16.npy").n == 40


def test_missing_array_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_centers(tmp_path)


def test_result_is_writable_and_does_not_alias_the_file(centers_dir):
    """The source is a read-only memmap; the returned tensor must be independent.

    Without an explicit copy, the no-scale/no-cast path can hand back a tensor
    backed by the mmap, so mutating it would corrupt the dataset on disk.
    """
    d, _ = centers_dir
    store = load_centers(d, apply_sd_scale=False, dtype=torch.float16)
    before = float(store.mu[0, 0])
    store.mu[0, 0] = 999.0                      # must not raise, must not persist
    reloaded = load_centers(d, apply_sd_scale=False, dtype=torch.float16)
    assert float(reloaded.mu[0, 0]) == before
