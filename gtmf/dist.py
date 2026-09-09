"""Distributed helpers -- phase 9.

The parallelism here is over QUERY STATES, not components. The centres are
replicated on every rank (656 MB of 140 GB), which removes distributed
log-sum-exp from the problem entirely: each rank evaluates the same field on its
own slice of the M query states, and only the per-query results are ever
exchanged.

The design goal is stronger than "the curves agree": every rank draws the SAME
full sample from the shared generator and keeps a contiguous slice of it, so the
union across ranks is exactly the sample a single process would have drawn. What
comes back from the gather is that same vector, in the same order, so
`mean_and_stderr` sees identical input at any world size.

Everything degrades to a no-op when torch.distributed is not initialised, so the
single-process path stays exactly what phase 7 tested.
"""

import os

import torch
import torch.distributed as dist

__all__ = ["setup", "cleanup", "rank", "world_size", "is_main", "barrier",
           "shard_bounds", "gather_rows"]


def setup(backend=None):
    """Join the process group torchrun created, if there is one.

    Returns (rank, world_size). Outside torchrun this is (0, 1) and no group is
    created, so nothing downstream has to branch on whether it is distributed.
    """
    if dist.is_initialized():
        return rank(), world_size()
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        # Pin this rank to its own GPU BEFORE the group forms, or every rank
        # lands on cuda:0 and NCCL deadlocks.
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend=backend)
    return rank(), world_size()


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main():
    """Only this rank writes files and prints progress (spec 10.5)."""
    return rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def shard_bounds(n, r=None, world=None):
    """Contiguous [lo, hi) slice of `n` items for rank `r`.

    Contiguous rather than strided so concatenating the shards in rank order
    rebuilds the original ordering exactly -- which is what makes the gathered
    vector identical to the single-process one rather than merely equivalent.

    Handles n not divisible by the world size: the first `n % world` ranks take
    one extra item, and no rank is left empty unless n < world.
    """
    r = rank() if r is None else r
    world = world_size() if world is None else world
    if not 0 <= r < world:
        raise ValueError(f"rank {r} outside world of {world}")
    return (n * r) // world, (n * (r + 1)) // world


def gather_rows(x):
    """Concatenate a per-query tensor across ranks along dim 0, in rank order.

    Works for the (B,) per-query vector and the (B, K) per-probe matrix alike;
    only the leading dimension is sharded.

    Shards may differ in length by one, which plain all_gather cannot express, so
    each is padded to the longest and trimmed back using the true lengths.

    Returns the full vector on EVERY rank: the aggregation is cheap and this
    keeps every rank holding the same numbers, so nothing downstream depends on
    which rank it is running on.
    """
    if not dist.is_initialized() or world_size() == 1:
        return x
    if x.dim() < 1:
        raise ValueError(f"expected at least 1 dimension, got {tuple(x.shape)}")

    sizes = torch.zeros(world_size(), dtype=torch.long, device=x.device)
    sizes[rank()] = x.shape[0]
    dist.all_reduce(sizes)

    longest = int(sizes.max())
    padded = torch.zeros((longest, *x.shape[1:]), dtype=x.dtype, device=x.device)
    padded[:x.shape[0]] = x
    buckets = [torch.zeros_like(padded) for _ in range(world_size())]
    dist.all_gather(buckets, padded)
    return torch.cat([b[:int(n)] for b, n in zip(buckets, sizes)])
