"""Build the target mixture a config describes.

Two sources. `latents` loads the consolidated ImageNet centres; `synthetic`
builds them in code, and with one component at the origin the mixture is exactly
N(0, sigma^2 I) -- which is what lets the spec 7.1 sanity check be a config file
rather than a separate script.
"""

import torch

from .data import load_centers
from .marginal_flow import MarginalFlow

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def build_target(cfg, sigma):
    """Return the MarginalFlow for one sigma of a config's sweep."""
    source = cfg.get("data.source", default="latents")
    device = cfg.get("compute.device", default="cpu")
    dtype_name = cfg.get("compute.dtype", default="float32")
    if dtype_name not in DTYPES:
        raise ValueError(f"compute.dtype must be one of {sorted(DTYPES)}, "
                         f"got {dtype_name!r}")
    dtype = DTYPES[dtype_name]

    if source == "latents":
        # "all" is the production default; an integer is a development subset.
        limit = cfg.get("data.num_target_samples", default="all")
        limit = None if limit in ("all", None) else int(limit)
        store = load_centers(cfg["data.latent_root"],
                             apply_sd_scale=cfg.get("data.apply_sd_scale", default=True),
                             device=device, dtype=dtype, limit=limit,
                             seed=cfg.get("seeds.centers_subset", default=0))
        # Pass sq through: load_centers already built it, and recomputing would be
        # a second pass over 1.28M x 256.
        mu, sq = store.mu, store.sq

    elif source == "synthetic":
        n = cfg["data.synthetic.num_components"]
        d = cfg["data.synthetic.dim"]
        kind = cfg.get("data.synthetic.centers", default="zeros")
        if kind == "zeros":
            mu = torch.zeros((n, d), device=device, dtype=dtype)
        elif kind == "normal":
            g = torch.Generator().manual_seed(cfg.get("seeds.centers_subset", default=0))
            mu = torch.randn((n, d), generator=g, dtype=dtype).to(device)
        else:
            raise ValueError(f"data.synthetic.centers must be 'zeros' or 'normal', "
                             f"got {kind!r}")
        sq = None                                  # trivial to compute for a toy N

    else:
        raise ValueError(f"data.source must be 'latents' or 'synthetic', got {source!r}")

    kwargs = {}
    chunk = cfg.get("compute.chunk_size", default="auto")
    if chunk != "auto":                            # "auto" = MarginalFlow's default
        kwargs["chunk"] = int(chunk)

    return MarginalFlow(mu, sigma, sq=sq,
                        tau=cfg.get("gmm.responsibility_log_cutoff", default=None),
                        allow_tf32=cfg.get("compute.allow_tf32", default=False),
                        **kwargs)
