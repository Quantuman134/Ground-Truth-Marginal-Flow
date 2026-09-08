#!/usr/bin/env python
"""Compute the reference w_avg(t) curve for every sigma a config sweeps.

One invocation writes one directory (CLAUDE.md, "Output layout"):

    results/<experiment>_<timestamp>/
        resolved_config.yaml     the config this run actually used
        run.log                  progress, mirrored from stdout
        sigma_0.01/              w_avg.csv, raw.npz, summary.json
        ...

The timestamp is generated, so a re-run can never destroy the previous one --
production runs take hours and their per-query raw data is what every figure is
redrawn from. `--output` pins a stable path when a launcher needs one, and then
refuses an existing directory unless `--force` says to continue into it.

Usage:
    python wavg.py --config configs/sanity_gaussian.yaml
    python wavg.py --config configs/wavg_imagenet.yaml --output results/prod
    python wavg.py --config configs/wavg_imagenet.yaml --output results/prod --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gtmf.config import Config
from gtmf.pipeline import run_sweep


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="experiment YAML, e.g. configs/wavg_imagenet.yaml")
    ap.add_argument("--output", type=Path, default=None,
                    help="pin the run directory; default is "
                         "<output_dir>/<experiment>_<timestamp>")
    ap.add_argument("--force", action="store_true",
                    help="continue into an existing --output directory. Nothing "
                         "is deleted: completed timepoints are resumed, not redone")
    ap.add_argument("--quiet", action="store_true",
                    help="write progress only to run.log, not to stdout")
    args = ap.parse_args(argv)

    # Load first, so a bad config fails before any directory is created.
    cfg = Config.load(args.config)
    result = run_sweep(cfg, run_dir=args.output, force=args.force,
                       echo=not args.quiet)
    print(f"\nwrote {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
