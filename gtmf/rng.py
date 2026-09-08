"""Seeded generators that live on the device they will draw onto.

torch requires a generator's device type to match the tensor it fills:
``torch.randn(..., generator=g, device="cuda")`` with a CPU ``g`` raises
``RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'``.

Every sampler here takes its device from the data it follows -- the centres, or
the query states -- so a config with ``compute.device: cuda`` needs a CUDA
generator and nothing in the config says so. This module is the one place that
knows the rule, so the pipeline builds generators correctly and the samplers say
something useful when it is handed a wrong one.

Mismatch is an error rather than a silent transfer: drawing on one device and
copying to another gives a different random stream from drawing on the target
device, and the phase-9 gate is "the 8-GPU run reproduces the single-GPU curve".
A generator that quietly re-homed itself would move that curve by an amount that
looks exactly like Monte-Carlo noise.
"""

import torch

__all__ = ["make_generator", "check_generator"]


def make_generator(seed, device="cpu"):
    """A ``torch.Generator`` seeded with ``seed`` and homed on ``device``.

        make_generator(cfg["seeds.query_states"], cfg.get("compute.device"))

    Use this rather than ``torch.Generator().manual_seed(...)``, which is always
    a CPU generator and fails the moment the run is on a GPU.
    """
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def check_generator(generator, device, what="generator"):
    """Raise unless ``generator`` can draw onto ``device``. ``None`` always passes.

    ``None`` means "use the global default generator", which torch homes on the
    right device by itself, so it needs no check.

    Only the device *type* is compared, which is the granularity torch itself
    enforces: one CUDA generator serves every CUDA device.
    """
    if generator is None:
        return generator
    want = torch.device(device).type
    got = generator.device.type
    if got != want:
        raise ValueError(
            f"{what} is on '{got}' but the draw targets '{want}'. Build it with "
            f"gtmf.rng.make_generator(seed, '{want}') -- torch.Generator() alone "
            f"is always a CPU generator.")
    return generator
