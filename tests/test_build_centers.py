"""Tests for build_centers.py.

Every assertion here compares against something known independently of the code
under test: a hand-constructed synthetic tree whose contents we chose, or an
invariant the output must satisfy. Several tests deliberately corrupt state and
assert that verification *fails* -- a verifier that cannot fail proves nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_centers as bc  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _make_tree(root, classes=("n00000003", "n00000001", "n00000002"), per_class=7):
    """Build a synthetic latent tree with known, distinguishable contents.

    Row (c, i) is filled with the constant value ``c * 100 + i``, so any row can
    be identified from its contents alone -- which is what lets the ordering
    tests below check placement rather than merely shape.

    Class names are deliberately out of alphabetical order so that a test can
    detect an implementation that relies on filesystem order.
    """
    root = Path(root)
    expected = {}
    for c, wnid in enumerate(classes):
        (root / wnid).mkdir(parents=True)
        for i in range(per_class):
            value = float(c * 100 + i)
            t = torch.full(bc.LATENT_SHAPE, value, dtype=torch.float16)
            rel = f"{wnid}/{wnid}_{i:04d}.pt"
            torch.save({"mean": t}, root / rel)
            expected[rel] = value
    return expected


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "train"
    expected = _make_tree(root)
    return root, expected


# --------------------------------------------------------------------------- #
# discover_latents
# --------------------------------------------------------------------------- #

def test_discover_finds_every_file_and_class(tree):
    root, expected = tree
    rel_paths, wnids = bc.discover_latents(root)
    assert len(rel_paths) == len(expected) == 21
    assert set(rel_paths) == set(expected)
    assert wnids == ["n00000001", "n00000002", "n00000003"]


def test_discover_order_is_sorted_not_filesystem_order(tree):
    """Ordering must be sorted-by-wnid-then-filename, independent of creation order.

    The tree is created with wnids in the order 3, 1, 2; a correct implementation
    returns them 1, 2, 3.
    """
    root, _ = tree
    rel_paths, _ = bc.discover_latents(root)
    assert rel_paths == sorted(rel_paths)
    assert rel_paths[0].startswith("n00000001/")
    assert rel_paths[-1].startswith("n00000003/")


def test_discover_ignores_non_pt_files(tree):
    root, expected = tree
    (root / "n00000001" / "notes.txt").write_text("ignore me")
    (root / "n00000001" / "checkpoint.pth").write_text("also ignore me")
    rel_paths, _ = bc.discover_latents(root)
    assert len(rel_paths) == len(expected)


def test_discover_limit_is_reproducible_and_sorted(tree):
    root, _ = tree
    a, _ = bc.discover_latents(root, limit=5, seed=1)
    b, _ = bc.discover_latents(root, limit=5, seed=1)
    c, _ = bc.discover_latents(root, limit=5, seed=2)
    assert a == b, "same seed must give the same subset"
    assert len(a) == 5 and a == sorted(a)
    assert a != c, "different seeds must give different subsets"


def test_discover_limit_above_total_returns_everything(tree):
    root, expected = tree
    rel_paths, _ = bc.discover_latents(root, limit=10_000)
    assert len(rel_paths) == len(expected)


def test_discover_missing_root_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        bc.discover_latents(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# load_latent
# --------------------------------------------------------------------------- #

def test_load_latent_returns_exact_values(tree):
    root, expected = tree
    for rel, value in expected.items():
        arr = bc.load_latent(root / rel)
        assert arr.shape == bc.LATENT_SHAPE
        assert arr.dtype == np.float16
        assert np.array_equal(arr, np.full(bc.LATENT_SHAPE, value, dtype=np.float16))


def test_load_latent_result_survives_source_deletion(tree):
    """The returned array must own its memory, not alias the loaded tensor."""
    root, expected = tree
    rel = sorted(expected)[0]
    arr = bc.load_latent(root / rel)
    before = arr.copy()
    import gc
    gc.collect()
    assert np.array_equal(arr, before)
    assert arr.flags.owndata


@pytest.mark.parametrize("payload, exc", [
    ({"std": torch.zeros(bc.LATENT_SHAPE, dtype=torch.float16)}, KeyError),
    ({"mean": torch.zeros((4, 32, 32), dtype=torch.float16)}, ValueError),
    ({"mean": torch.zeros(bc.LATENT_SHAPE, dtype=torch.float32)}, TypeError),
])
def test_load_latent_rejects_malformed_files(tmp_path, payload, exc):
    p = tmp_path / "bad.pt"
    torch.save(payload, p)
    with pytest.raises(exc):
        bc.load_latent(p)


# --------------------------------------------------------------------------- #
# build_centers
# --------------------------------------------------------------------------- #

def test_build_places_every_row_at_the_right_index(tree, tmp_path):
    """Each row must hold the value of the file the index names for that row."""
    root, expected = tree
    out = tmp_path / "out"
    meta = bc.build_centers(root, out, workers=2, chunk_size=4)

    arr = np.load(out / bc.ARRAY_NAME, mmap_mode="r")
    rel_paths = (out / bc.INDEX_NAME).read_text().splitlines()

    assert meta["num_centers"] == len(expected)
    assert arr.shape == (len(expected), *bc.LATENT_SHAPE)
    assert arr.dtype == np.float16
    assert len(rel_paths) == len(expected)
    for row, rel in enumerate(rel_paths):
        assert np.array_equal(np.asarray(arr[row]),
                              np.full(bc.LATENT_SHAPE, expected[rel], dtype=np.float16))


def test_build_labels_match_the_wnid_of_each_row(tree, tmp_path):
    root, _ = tree
    out = tmp_path / "out"
    bc.build_centers(root, out, workers=2, chunk_size=4)

    labels = np.load(out / bc.LABELS_NAME)
    rel_paths = (out / bc.INDEX_NAME).read_text().splitlines()
    _, wnids = bc.discover_latents(root)

    assert labels.dtype == np.int16
    for row, rel in enumerate(rel_paths):
        assert wnids[labels[row]] == rel.split("/", 1)[0]


def test_build_stores_values_raw_without_sd_scale(tree, tmp_path):
    """The 0.18215 scale must NOT be baked in -- downstream applies it at load."""
    root, expected = tree
    out = tmp_path / "out"
    meta = bc.build_centers(root, out, workers=2, chunk_size=4)

    arr = np.load(out / bc.ARRAY_NAME, mmap_mode="r")
    rel_paths = (out / bc.INDEX_NAME).read_text().splitlines()
    first = expected[rel_paths[0]]

    assert meta["sd_scale_applied"] is False
    assert np.asarray(arr[0]).flat[0] == np.float16(first)
    assert not np.isclose(float(np.asarray(arr[0]).flat[0]), first * 0.18215)


def test_build_is_chunk_size_and_worker_invariant(tree, tmp_path):
    """Parallel layout must not depend on how work was divided."""
    root, _ = tree
    a, b = tmp_path / "a", tmp_path / "b"
    bc.build_centers(root, a, workers=1, chunk_size=64)
    bc.build_centers(root, b, workers=4, chunk_size=2)
    assert np.array_equal(np.load(a / bc.ARRAY_NAME), np.load(b / bc.ARRAY_NAME))
    assert (a / bc.INDEX_NAME).read_text() == (b / bc.INDEX_NAME).read_text()


def test_build_reports_the_declared_byte_size(tree, tmp_path):
    root, expected = tree
    out = tmp_path / "out"
    meta = bc.build_centers(root, out, workers=2, chunk_size=4)
    assert meta["bytes"] == len(expected) * 4 * 8 * 8 * 2
    assert meta["flat_dim"] == 256
    assert (out / bc.ARRAY_NAME).stat().st_size >= meta["bytes"]


def test_build_raises_on_a_corrupt_source_rather_than_zero_filling(tree, tmp_path):
    """A silently zero-filled row would be a GMM component at the origin."""
    root, _ = tree
    (root / "n00000002" / "n00000002_0003.pt").write_bytes(b"not a torch file")
    with pytest.raises(RuntimeError, match="failed to load"):
        bc.build_centers(root, tmp_path / "out", workers=2, chunk_size=4)


def test_build_empty_tree_raises(tmp_path):
    root = tmp_path / "train"
    (root / "n00000001").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no .pt files"):
        bc.build_centers(root, tmp_path / "out", workers=1)


# --------------------------------------------------------------------------- #
# verify_centers -- must be able to fail
# --------------------------------------------------------------------------- #

def test_verify_passes_on_a_good_build(tree, tmp_path):
    root, _ = tree
    out = tmp_path / "out"
    bc.build_centers(root, out, workers=2, chunk_size=4)
    assert bc.verify_centers(out, root, num_samples=21) is True


def test_verify_detects_a_corrupted_row(tree, tmp_path):
    root, _ = tree
    out = tmp_path / "out"
    bc.build_centers(root, out, workers=2, chunk_size=4)

    arr = np.load(out / bc.ARRAY_NAME, mmap_mode="r+")
    arr[3] += np.float16(1.0)
    arr.flush()
    del arr

    with pytest.raises(AssertionError, match="does not match source"):
        bc.verify_centers(out, root, num_samples=21)


def test_verify_detects_a_truncated_index(tree, tmp_path):
    root, _ = tree
    out = tmp_path / "out"
    bc.build_centers(root, out, workers=2, chunk_size=4)

    lines = (out / bc.INDEX_NAME).read_text().splitlines()
    (out / bc.INDEX_NAME).write_text("\n".join(lines[:-1]) + "\n")

    with pytest.raises(AssertionError, match="index has"):
        bc.verify_centers(out, root, num_samples=5)


def test_verify_detects_a_shuffled_index(tree, tmp_path):
    """Right rows, wrong order -- shapes all agree, contents do not."""
    root, _ = tree
    out = tmp_path / "out"
    bc.build_centers(root, out, workers=2, chunk_size=4)

    lines = (out / bc.INDEX_NAME).read_text().splitlines()
    (out / bc.INDEX_NAME).write_text("\n".join(lines[::-1]) + "\n")

    with pytest.raises(AssertionError, match="does not match source"):
        bc.verify_centers(out, root, num_samples=21)


# --------------------------------------------------------------------------- #
# metadata and README
# --------------------------------------------------------------------------- #

def test_meta_records_subset_provenance(tree, tmp_path):
    root, _ = tree
    full = bc.build_centers(root, tmp_path / "full", workers=2)
    sub = bc.build_centers(root, tmp_path / "sub", limit=5, seed=7, workers=2)
    assert full["subset_limit"] is None and full["subset_seed"] is None
    assert sub["subset_limit"] == 5 and sub["subset_seed"] == 7
    assert sub["num_centers"] == 5


def test_meta_is_valid_json_on_disk(tree, tmp_path):
    root, _ = tree
    out = tmp_path / "out"
    meta = bc.build_centers(root, out, workers=2)
    assert json.loads((out / bc.META_NAME).read_text()) == meta


def test_update_readme_is_idempotent_and_preserves_surrounding_text(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Existing\n\nKeep this line.\n")
    meta = {"num_centers": 1_281_167, "bytes": 656_277_504,
            "ordering": "sorted by wnid, then by filename",
            "built_utc": "2026-09-03T00:00:00+00:00",
            "sd_scale_to_apply_at_load": 0.18215}

    once = bc.update_readme(readme, meta)
    twice = bc.update_readme(readme, meta)

    assert once == twice, "re-running must not append a second section"
    assert twice.count(bc.README_BEGIN) == 1
    assert "Keep this line." in twice
    assert "1,281,167" in twice


def test_update_readme_creates_the_file_when_absent(tmp_path):
    readme = tmp_path / "README.md"
    meta = {"num_centers": 10, "bytes": 5120,
            "ordering": "sorted by wnid, then by filename",
            "built_utc": "2026-09-03T00:00:00+00:00",
            "sd_scale_to_apply_at_load": 0.18215}
    bc.update_readme(readme, meta)
    assert readme.exists() and bc.README_BEGIN in readme.read_text()


# --------------------------------------------------------------------------- #
# regression tests for the review fixes
# --------------------------------------------------------------------------- #

def test_build_labels_rejects_more_classes_than_int16_holds():
    """int16 wraps silently past 32,767 -- the guard must fire instead."""
    wnids = [f"n{i:08d}" for i in range(40_000)]
    with pytest.raises(ValueError, match="exceeds int16"):
        bc.build_labels([f"{wnids[0]}/a.pt"], wnids)


def test_build_labels_accepts_imagenet_scale():
    wnids = [f"n{i:08d}" for i in range(1000)]
    labels = bc.build_labels([f"{wnids[999]}/a.pt", f"{wnids[0]}/b.pt"], wnids)
    assert labels.dtype == np.int16
    assert labels.tolist() == [999, 0]


def test_failed_build_leaves_no_partial_array(tree, tmp_path):
    """A half-filled array must not survive to block a rerun or look valid."""
    root, _ = tree
    out = tmp_path / "out"
    (root / "n00000002" / "n00000002_0003.pt").write_bytes(b"not a torch file")
    with pytest.raises(RuntimeError, match="partial array removed"):
        bc.build_centers(root, out, workers=2, chunk_size=4)
    assert not (out / bc.ARRAY_NAME).exists()


def test_update_readme_refuses_a_half_written_marker_pair(tmp_path):
    """One marker without its partner means append would duplicate the section."""
    readme = tmp_path / "README.md"
    readme.write_text(f"# Doc\n\n{bc.README_BEGIN}\ntruncated\n")
    meta = {"num_centers": 10, "bytes": 5120,
            "ordering": "sorted by wnid, then by filename",
            "built_utc": "2026-09-03T00:00:00+00:00",
            "sd_scale_to_apply_at_load": 0.18215}
    with pytest.raises(ValueError, match="without its pair"):
        bc.update_readme(readme, meta)
