"""Load the consolidated GMM centers.

The centers live as one flat fp16 array written by ``build_centers.py``. Values on
disk are RAW: the Stable-Diffusion 0.18215 scale is applied here, at load, not by
the builder.

This runs once per process. Everything it produces -- the fp32 centers and their
squared norms -- is reused for the whole run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SD_SCALE = 0.18215
ARRAY_NAME = "centers_fp16.npy"
META_NAME = "centers_meta.json"

__all__ = ["CentersStore", "load_centers", "SD_SCALE"]


@dataclass(frozen=True)
class CentersStore:
    """GMM component centers, flattened and ready for the field.

    ``mu`` is ``(N, d)`` and ``sq`` is ``(N,)`` holding ``||mu_i||^2``. ``sq`` is
    precomputed because the score expansion needs it on every field evaluation and
    it never changes.
    """

    mu: torch.Tensor
    sq: torch.Tensor
    meta: dict

    @property
    def n(self):
        return self.mu.shape[0]

    @property
    def d(self):
        return self.mu.shape[1]

    @property
    def device(self):
        return self.mu.device

    @property
    def dtype(self):
        return self.mu.dtype

    def mean_sq_norm(self):
        """E||mu||^2 / d -- the center-spread term in rho_t (Eq. 34)."""
        return float(self.sq.mean().item() / self.d)

    def __repr__(self):
        return (f"CentersStore(N={self.n:,}, d={self.d}, {self.dtype}, "
                f"{self.device}, scaled={self.meta.get('sd_scale_applied_at_load')})")


def load_centers(path, *, apply_sd_scale=True, device="cpu",
                 dtype=torch.float32, limit=None, seed=0):
    """Load centers as an ``(N, d)`` tensor plus their squared norms.

    ``path`` may be the array itself or the directory containing it. Rows are
    memory-mapped and read once; the fp16 -> ``dtype`` cast and the scale are
    applied in the target precision, never in fp16 (which would clip: raw values
    reach |18| and the scaled ones must stay exact).

    ``limit`` takes a seeded random subset for development, reproducibly, and
    keeps it in ascending row order so the subset is a well-defined slice of the
    same ordering the index file describes.
    """
    path = Path(path)
    directory = path.parent if path.suffix == ".npy" else path
    array_path = path if path.suffix == ".npy" else path / ARRAY_NAME
    if not array_path.exists():
        raise FileNotFoundError(f"centers array not found: {array_path}")

    raw = np.load(array_path, mmap_mode="r")
    if raw.ndim < 2:
        raise ValueError(f"{array_path}: expected at least 2 dims, got {raw.shape}")

    n_total = raw.shape[0]
    rows = None
    if limit is not None and limit < n_total:
        rows = np.sort(np.random.default_rng(seed).choice(n_total, limit, replace=False))

    block = np.ascontiguousarray(raw[rows] if rows is not None else raw)
    if not block.flags.writeable:
        # `raw` is a read-only memmap and ascontiguousarray returned it unchanged.
        # torch.from_numpy would then produce a non-writable tensor which .to() may
        # hand back un-copied, leaving the caller aliasing the file on disk.
        block = np.array(block)
    flat = torch.from_numpy(block.reshape(block.shape[0], -1))

    mu = flat.to(device=device, dtype=dtype)
    if apply_sd_scale:
        mu = mu * SD_SCALE
    sq = mu.pow(2).sum(dim=1)

    meta_path = directory / META_NAME
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta = dict(meta)
    meta.update({
        "loaded_from": str(array_path),
        "num_loaded": int(mu.shape[0]),
        "num_available": int(n_total),
        "subset_seed": seed if rows is not None else None,
        "sd_scale_applied_at_load": SD_SCALE if apply_sd_scale else None,
        "load_dtype": str(dtype),
    })
    return CentersStore(mu=mu, sq=sq, meta=meta)
