"""Tests for gtmf/dist.py and the distributed path -- phase 9.

The gate is that a sharded run reproduces the single-process curve. It is tested
by actually spawning ranks (gloo, CPU) and comparing numbers, not by inspecting
the plumbing -- a reduction can be wrong in ways that still look like a working
distributed run.
"""

import csv
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtmf import dist as gdist                              # noqa: E402
from gtmf.config import Config                              # noqa: E402
from gtmf.pipeline import run_sigma, run_timepoint          # noqa: E402
from gtmf.rng import make_generator                         # noqa: E402
from gtmf.target import build_target                        # noqa: E402

CFG = {
    "experiment": "dist_t",
    "data": {"source": "synthetic",
             "synthetic": {"num_components": 1, "dim": 8, "centers": "zeros"}},
    "gmm": {"component_sigma": [0.3]},
    "time": {"num_points": 3, "t_min": 0.3, "t_max": 0.9},
    "monte_carlo": {"num_query_states": 25, "num_probes": 4, "scheme": "central",
                    "epsilon_alpha": 1e-3, "rho_source": "exact"},
    "ode": {"integrator": "rk4", "step_size": 0.015625},
    "compute": {"device": "cpu", "dtype": "float64"},
    "output": {"output_dir": "results"},
    "seeds": {"query_states": 0, "probes": 0},
}


def write_cfg(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(CFG))
    return p


# --- shard_bounds, without any process group --------------------------------- #

@pytest.mark.parametrize("n,world", [(8, 1), (8, 2), (8, 3), (25, 4), (2, 4), (0, 3)])
def test_shards_partition_exactly(n, world):
    """Every item lands in exactly one shard, in order -- which is what makes
    concatenating them rebuild the original vector."""
    bounds = [gdist.shard_bounds(n, r, world) for r in range(world)]
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    for (_, hi), (lo, _) in zip(bounds, bounds[1:]):
        assert hi == lo                                  # no gap, no overlap
    assert sum(hi - lo for lo, hi in bounds) == n


@pytest.mark.parametrize("n,world", [(8, 3), (25, 4), (10, 4)])
def test_shards_differ_by_at_most_one(n, world):
    sizes = [hi - lo for lo, hi in
             (gdist.shard_bounds(n, r, world) for r in range(world))]
    assert max(sizes) - min(sizes) <= 1


def test_an_out_of_range_rank_raises():
    with pytest.raises(ValueError, match="outside world"):
        gdist.shard_bounds(8, 4, 4)


def test_helpers_are_inert_without_a_process_group():
    """The single-process path must not branch on being distributed."""
    assert (gdist.rank(), gdist.world_size(), gdist.is_main()) == (0, 1, True)
    gdist.barrier()                                      # must not hang or raise
    x = torch.arange(5.0)
    assert torch.equal(gdist.gather_rows(x), x)
    assert gdist.setup() == (0, 1)


# --- the real thing: spawn ranks and compare against one process ------------- #

def _worker(rank, world, tmp, port, what):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      MKL_THREADING_LAYER="GNU")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gtmf import dist as d
    from gtmf.config import Config as C
    from gtmf.pipeline import run_sigma as rs, run_timepoint as rt
    from gtmf.target import build_target as bt

    d.setup(backend="gloo")
    try:
        cfg = C.load(Path(tmp) / "c.yaml")
        flow = bt(cfg, 0.3)
        if what == "work":
            # Count the ROWS this rank actually integrates. Equality of results
            # cannot distinguish "sharded correctly" from "every rank redundantly
            # computed the whole thing" -- both give identical numbers. Only the
            # work each rank does tells them apart.
            seen = []
            real_v = flow.velocity
            flow_velocity = lambda x, tt: (seen.append(x.shape[0]), real_v(x, tt))[1]
            flow.velocity = flow_velocity
            from gtmf.rng import make_generator as mg
            rt(flow, cfg, 0.3, 0.5, generator=mg(1), probe_generator=mg(2))
            (Path(tmp) / f"work_{rank}.json").write_text(json.dumps(
                {"rows": seen[0], "rank": rank}))
        elif what == "timepoint":
            from gtmf.rng import make_generator as mg
            r = rt(flow, cfg, 0.3, 0.5, generator=mg(1), probe_generator=mg(2))
            if rank == 0:
                (Path(tmp) / "out.json").write_text(json.dumps(
                    {"w_avg": r.w_avg, "stderr": r.stderr,
                     "n": int(r.w_per_query.numel()),
                     "w": r.w_per_query.tolist()}))
        else:
            rs(cfg, 0.3, Path(tmp) / "run", flow=flow, log=lambda *_: None)
    finally:
        d.cleanup()


def run_ranks(tmp, world, what, port=29571):
    # Children inherit this. Without it, a child that imports numpy after the
    # parent has already loaded libgomp dies with "MKL_THREADING_LAYER=INTEL is
    # incompatible with libgomp.so.1" -- which is why these tests pass alone and
    # fail inside the full suite, where earlier tests have loaded MKL first.
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    mp.spawn(_worker, args=(world, str(tmp), port, what), nprocs=world, join=True)


@pytest.mark.parametrize("world", [2, 3])
def test_a_sharded_timepoint_equals_the_single_process_one(tmp_path, world):
    """THE PHASE GATE. Not "within Monte-Carlo error" -- every rank draws the
    same full sample and keeps a contiguous slice, so the gathered vector is the
    single-process vector, and the mean must match to floating-point noise."""
    write_cfg(tmp_path)
    cfg = Config.load(tmp_path / "c.yaml")
    solo = run_timepoint(build_target(cfg, 0.3), cfg, 0.3, 0.5,
                         generator=make_generator(1), probe_generator=make_generator(2))

    run_ranks(tmp_path, world, "timepoint", port=29571 + world)
    got = json.loads((tmp_path / "out.json").read_text())

    assert got["n"] == cfg["monte_carlo.num_query_states"] == 25
    assert got["w_avg"] == pytest.approx(solo.w_avg, rel=1e-12)
    assert got["stderr"] == pytest.approx(solo.stderr, rel=1e-12)
    # per-query values, in the original order
    assert got["w"] == pytest.approx(solo.w_per_query.tolist(), rel=1e-12)


@pytest.mark.parametrize("world", [2, 5])
def test_each_rank_integrates_only_its_own_shard(tmp_path, world):
    """The test that would have caught a no-op shard.

    M=25 with a central scheme and K=4 means the full batch is 25*2*4 = 200 rows.
    Split across `world` ranks, each should see only its slice -- and the slices
    must sum to the whole. A rank seeing all 200 means every rank redundantly
    recomputed the entire sample, which produces the RIGHT ANSWER and no speedup
    at all.
    """
    write_cfg(tmp_path)
    run_ranks(tmp_path, world, "work", port=29611 + world)

    blocks = 2 * CFG["monte_carlo"]["num_probes"]          # central: 2K
    m = CFG["monte_carlo"]["num_query_states"]
    rows = [json.loads((tmp_path / f"work_{r}.json").read_text())["rows"]
            for r in range(world)]
    assert sum(rows) == m * blocks                          # nothing lost
    assert max(rows) < m * blocks                           # nobody did it all
    for r, got in enumerate(rows):
        lo, hi = gdist.shard_bounds(m, r, world)
        assert got == (hi - lo) * blocks


def test_a_sharded_sigma_writes_the_single_process_curve(tmp_path):
    """End to end through run_sigma: same CSV, and only rank 0 wrote it."""
    write_cfg(tmp_path)
    cfg = Config.load(tmp_path / "c.yaml")
    run_sigma(cfg, 0.3, tmp_path / "solo", log=lambda *_: None)
    run_ranks(tmp_path, 2, "sigma", port=29591)

    for name in ("w_avg.csv", "raw.npz", "summary.json"):
        assert (tmp_path / "run" / name).is_file()
    rows = {}
    for which in ("solo", "run"):
        with (tmp_path / which / "w_avg.csv").open() as fh:
            rows[which] = [(float(r["t"]), float(r["w_avg"]), float(r["stderr"]))
                           for r in csv.DictReader(fh)]
    assert len(rows["run"]) == 3
    for solo, sharded in zip(rows["solo"], rows["run"]):
        assert sharded == pytest.approx(solo, rel=1e-12)


def test_only_one_run_log_is_written(tmp_path):
    """Eight ranks appending the same lines would interleave into nonsense."""
    write_cfg(tmp_path)
    run_ranks(tmp_path, 2, "sigma", port=29601)
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["complete"] is True and summary["num_timepoints"] == 3
