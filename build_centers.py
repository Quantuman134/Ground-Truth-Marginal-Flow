#!/usr/bin/env python
"""Merge the per-image ImageNet latent files into one flat array of GMM centers.

The source tree holds 1,281,167 files of 2,322 bytes each for 512 bytes of real
payload, so a cold pass over it costs ~33 minutes (measured 651 files/s). This
writes the same data once as a contiguous fp16 array that reads back in seconds.

Output (into --out-dir, default the parent of --latent-root):
    centers_fp16.npy      (N, 4, 8, 8) fp16, RAW -- the 0.18215 SD scale is NOT applied
    centers_index.txt     N lines, "<wnid>/<file>.pt", row-aligned with the array
    centers_labels.npy    (N,) int16, index into the sorted wnid list
    centers_meta.json     provenance: N, shape, dtype, source, ordering, seed

Values are stored raw to stay faithful to the source; apply 0.18215 at load time.

Usage:
    python build_centers.py --latent-root /path/to/latents_8_mean_fp16/train
    python build_centers.py --limit 50000 --workers 8        # development subset
    python build_centers.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm

LATENT_SHAPE = (4, 8, 8)
ARRAY_NAME = "centers_fp16.npy"
INDEX_NAME = "centers_index.txt"
LABELS_NAME = "centers_labels.npy"
META_NAME = "centers_meta.json"
README_BEGIN = "<!-- BEGIN centers_fp16 -->"
README_END = "<!-- END centers_fp16 -->"


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def discover_latents(root, limit=None, seed=0):
    """List latent files under ``root`` in a deterministic order.

    Returns ``(rel_paths, wnids)``: paths relative to ``root`` as
    ``"<wnid>/<file>.pt"``, and the sorted list of class directories.

    Ordering is sorted-by-wnid then sorted-by-filename, so the row index of a
    given image is stable across runs and machines. When ``limit`` is smaller
    than the discovered count, a seeded random subset is drawn and then
    re-sorted -- reproducible per spec 10.3, and still sequential on disk.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"latent root does not exist: {root}")

    wnids = sorted(p.name for p in root.iterdir() if p.is_dir())
    rel_paths = []
    for wnid in wnids:
        for name in sorted(os.listdir(root / wnid)):
            if name.endswith(".pt"):
                rel_paths.append(f"{wnid}/{name}")

    if limit is not None and limit < len(rel_paths):
        rel_paths = sorted(random.Random(seed).sample(rel_paths, limit))

    return rel_paths, wnids


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_latent(path):
    """Load one ``.pt`` latent as a ``(4, 8, 8)`` fp16 numpy array.

    The stored tensor is already fp16 and already 4x8x8; this does not pool,
    rescale or cast it. ``.numpy()`` shares memory with the loaded tensor, which
    is about to go out of scope, so the result is copied.
    """
    obj = torch.load(path, weights_only=True, map_location="cpu")
    if "mean" not in obj:
        raise KeyError(f"{path}: expected key 'mean', got {sorted(obj)}")
    t = obj["mean"]
    if tuple(t.shape) != LATENT_SHAPE:
        raise ValueError(f"{path}: expected shape {LATENT_SHAPE}, got {tuple(t.shape)}")
    if t.dtype != torch.float16:
        raise TypeError(f"{path}: expected torch.float16, got {t.dtype}")
    return t.numpy().copy()


# --------------------------------------------------------------------------- #
# parallel fill
# --------------------------------------------------------------------------- #

_WORKER = {}


def _init_worker(array_path, root):
    # Opened once per process; workers touch disjoint row ranges, which is safe
    # on a shared memmap.
    _WORKER["array"] = open_memmap(array_path, mode="r+")
    _WORKER["root"] = Path(root)


def _fill_chunk(job):
    """Write rows [start, start+len(rel_paths)) of the memmap. Returns failures."""
    start, rel_paths = job
    arr, root = _WORKER["array"], _WORKER["root"]
    failures = []
    for offset, rel in enumerate(rel_paths):
        try:
            arr[start + offset] = load_latent(root / rel)
        except Exception as exc:                                  # noqa: BLE001
            failures.append((start + offset, rel, f"{type(exc).__name__}: {exc}"))
    return len(rel_paths), failures


def build_labels(rel_paths, wnids):
    """Map each row to the index of its wnid in the sorted class list.

    int16 is ample for ImageNet's 1,000 classes but wraps silently past 32,767,
    which would corrupt labels without any visible error -- so check it.
    """
    if len(wnids) > np.iinfo(np.int16).max:
        raise ValueError(f"{len(wnids)} classes exceeds int16; widen LABELS dtype")
    wnid_to_label = {w: i for i, w in enumerate(wnids)}
    return np.fromiter((wnid_to_label[p.split("/", 1)[0]] for p in rel_paths),
                       dtype=np.int16, count=len(rel_paths))


def build_centers(root, out_dir, limit=None, seed=0, workers=16, chunk_size=512):
    """Merge every discovered latent into one contiguous fp16 array.

    A load failure is collected and raised at the end rather than skipped: a
    zero-filled row would become a GMM component sitting at the origin and would
    quietly distort every downstream result.
    """
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"scanning {root} ...", flush=True)
    rel_paths, wnids = discover_latents(root, limit=limit, seed=seed)
    n = len(rel_paths)
    if n == 0:
        raise RuntimeError(f"no .pt files found under {root}")
    print(f"discovered N = {n:,} latents in {len(wnids)} classes "
          f"({time.time() - t0:.1f}s)", flush=True)

    array_path = out_dir / ARRAY_NAME
    arr = open_memmap(array_path, mode="w+", dtype=np.float16, shape=(n, *LATENT_SHAPE))
    arr.flush()
    del arr  # reopened per worker; the header is on disk now

    jobs = [(i, rel_paths[i:i + chunk_size]) for i in range(0, n, chunk_size)]
    failures = []
    ctx = get_context("forkserver")  # never fork(): torch holds threads in the parent
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(str(array_path), str(root))) as pool:
        with tqdm(total=n, unit="latent", unit_scale=True, smoothing=0.05,
                  desc="merging", file=sys.stdout) as bar:
            for done, chunk_failures in pool.imap_unordered(_fill_chunk, jobs):
                failures.extend(chunk_failures)
                bar.update(done)

    if failures:
        # Leave nothing behind: a partially filled array would block a rerun and,
        # worse, could later be mistaken for a good build.
        array_path.unlink(missing_ok=True)
        head = "\n".join(f"  row {r}  {p}  {m}" for r, p, m in failures[:20])
        raise RuntimeError(f"{len(failures)} latent(s) failed to load "
                           f"(partial array removed):\n{head}")

    labels = build_labels(rel_paths, wnids)
    np.save(out_dir / LABELS_NAME, labels)
    (out_dir / INDEX_NAME).write_text("\n".join(rel_paths) + "\n")

    elapsed = time.time() - t0
    meta = {
        "num_centers": n,
        "latent_shape": list(LATENT_SHAPE),
        "flat_dim": int(np.prod(LATENT_SHAPE)),
        "dtype": "float16",
        "sd_scale_applied": False,
        "sd_scale_to_apply_at_load": 0.18215,
        "source_root": str(root),
        "num_classes": len(wnids),
        "ordering": "sorted by wnid, then by filename",
        "subset_limit": limit,
        "subset_seed": seed if limit is not None else None,
        "bytes": int(n * np.prod(LATENT_SHAPE) * 2),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_seconds": round(elapsed, 1),
    }
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nwrote {array_path}  ({meta['bytes'] / 2**20:.0f} MiB)")
    print(f"  N = {n:,} | shape {LATENT_SHAPE} | flat dim {meta['flat_dim']} | fp16, raw")
    print(f"  {elapsed:.0f}s total, {n / elapsed:,.0f} latents/s")
    return meta


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def verify_centers(out_dir, root, num_samples=200, seed=0):
    """Re-read random source files and assert the stored rows match exactly.

    Raises on any mismatch. fp16 is copied verbatim, so this is an exact
    bit-for-bit comparison, not a tolerance check.
    """
    out_dir, root = Path(out_dir), Path(root)
    meta = json.loads((out_dir / META_NAME).read_text())
    arr = open_memmap(out_dir / ARRAY_NAME, mode="r")
    rel_paths = (out_dir / INDEX_NAME).read_text().splitlines()
    labels = np.load(out_dir / LABELS_NAME)

    n = meta["num_centers"]
    if arr.shape != (n, *LATENT_SHAPE):
        raise AssertionError(f"array shape {arr.shape} != {(n, *LATENT_SHAPE)}")
    if len(rel_paths) != n:
        raise AssertionError(f"index has {len(rel_paths)} rows, array has {n}")
    if labels.shape != (n,):
        raise AssertionError(f"labels shape {labels.shape} != {(n,)}")
    if arr.dtype != np.float16:
        raise AssertionError(f"array dtype {arr.dtype} != float16")

    rows = random.Random(seed).sample(range(n), min(num_samples, n))
    for row in rows:
        expected = load_latent(root / rel_paths[row])
        if not np.array_equal(np.asarray(arr[row]), expected):
            raise AssertionError(f"row {row} ({rel_paths[row]}) does not match source")

    t0 = time.time()
    total = float(np.asarray(arr[:min(n, 100_000)], dtype=np.float64).sum())
    read_s = time.time() - t0
    print(f"verified {len(rows)} random rows against source -- all exact")
    print(f"  N = {n:,} | {arr.nbytes / 2**20:.0f} MiB | "
          f"read {min(n, 100_000):,} rows in {read_s:.2f}s (checksum {total:.1f})")
    return True


# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #

def render_readme_section(meta):
    scale = meta["sd_scale_to_apply_at_load"]
    return f"""{README_BEGIN}
## Consolidated centers array

`{ARRAY_NAME}` -- all {meta['num_centers']:,} latents as one contiguous
`({meta['num_centers']}, 4, 8, 8)` fp16 array ({meta['bytes'] / 2**20:.0f} MiB), built by
`Ground-Truth-Marginal-Flow/build_centers.py` on {meta['built_utc']}.

Reading the small-file tree costs ~33 min; this reads in seconds. Row order is
{meta['ordering']}. Values are **raw** -- the `{scale}` SD scale is NOT applied here
either, exactly as in the per-image files.

| file | contents |
|---|---|
| `{ARRAY_NAME}` | `(N, 4, 8, 8)` fp16 centers |
| `{INDEX_NAME}` | N lines, `<wnid>/<file>.pt`, row-aligned |
| `{LABELS_NAME}` | `(N,)` int16, index into the sorted wnid list |
| `{META_NAME}` | provenance and build parameters |

```python
import numpy as np
z = np.load("{ARRAY_NAME}", mmap_mode="r")      # (N, 4, 8, 8) fp16, raw
x = np.asarray(z[:1024], dtype=np.float32) * {scale}
```
{README_END}"""


def update_readme(readme_path, meta):
    """Insert or replace the delimited centers section. Idempotent."""
    readme_path = Path(readme_path)
    section = render_readme_section(meta)
    text = readme_path.read_text() if readme_path.exists() else ""
    if (README_BEGIN in text) != (README_END in text):
        raise ValueError(f"{readme_path}: found one centers marker without its pair; "
                         "fix the file by hand rather than risk duplicating the section")
    if README_BEGIN in text and README_END in text:
        head, rest = text.split(README_BEGIN, 1)
        _, tail = rest.split(README_END, 1)
        text = head + section + tail
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    readme_path.write_text(text)
    return text


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latent-root", type=Path,
                    default=Path("/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/"
                                 "latents_8_mean_fp16/train"))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: parent of --latent-root")
    ap.add_argument("--limit", type=int, default=None,
                    help="development subset; production uses all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--verify-samples", type=int, default=200)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-readme", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the array already exists")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or args.latent_root.parent

    if args.verify_only:
        verify_centers(out_dir, args.latent_root, args.verify_samples, args.seed)
        return 0

    if (out_dir / ARRAY_NAME).exists() and not args.force:
        print(f"{out_dir / ARRAY_NAME} already exists -- pass --force to rebuild.")
        return 1

    meta = build_centers(args.latent_root, out_dir, limit=args.limit, seed=args.seed,
                         workers=args.workers, chunk_size=args.chunk_size)
    verify_centers(out_dir, args.latent_root, args.verify_samples, args.seed)

    if not args.no_readme:
        readme = out_dir / "README.md"
        update_readme(readme, meta)
        print(f"updated {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
