#!/usr/bin/env python
"""Draw the figures for a finished run (phase 10, spec section 9).

Reads only what the run persisted, so this never triggers a computation: point it
at a results directory as often as you like. `wavg.py` calls the same code at the
end of a run, so this exists for redrawing -- after a plotting change, or on a
run copied off the cluster.

Writes into <run>/figures/ :

    raw.png/pdf/csv          w_avg(t) per sigma, mean +/- 1.96 SE, closed form
    normalized.png/pdf/csv   Eq. 42, normalized over the MEASURED grid
    plateau.png/pdf/csv      only when the run swept alpha -- the phase 8 figure
    accuracy.json            distance from the closed form, overall and committed

Usage:
    python plots.py --run results/wavg_imagenet_20260908-1830
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gtmf import figures


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a results/ run directory")
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/figures")
    args = ap.parse_args(argv)

    try:
        written, count = figures.draw_run(args.run, args.out)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    print(f"{count} curve(s) -> {written[0].parent}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
