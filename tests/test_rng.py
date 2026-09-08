"""Tests for gtmf/rng.py.

The oracle is torch itself: `check_generator` must accept exactly the pairings
torch accepts and reject exactly the ones it raises on. So every test that says
"this combination is legal" also performs the draw, and every test that says
"this one is not" shows torch refusing the same thing.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtmf.rng import check_generator, make_generator  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU on this machine")


def test_make_generator_is_seeded_and_reproducible():
    """Same seed, same stream; different seed, different stream."""
    a = torch.randn(8, generator=make_generator(7))
    b = torch.randn(8, generator=make_generator(7))
    c = torch.randn(8, generator=make_generator(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("seed", [0, 3, 2**31 + 5])
def test_make_generator_reports_the_seed_it_was_given(seed):
    assert make_generator(seed).initial_seed() == seed


def test_make_generator_defaults_to_cpu():
    assert make_generator(0).device.type == "cpu"


def test_make_generator_accepts_a_device_object_or_a_string():
    assert make_generator(0, torch.device("cpu")).device.type == "cpu"
    assert make_generator(0, "cpu").device.type == "cpu"


@CUDA
def test_make_generator_on_cuda_actually_draws_on_cuda():
    """The pairing the whole module exists for: this is what used to raise."""
    g = make_generator(0, "cuda")
    assert g.device.type == "cuda"
    x = torch.randn(4, 3, generator=g, device="cuda")
    assert x.device.type == "cuda"


@CUDA
def test_make_generator_is_reproducible_on_cuda():
    a = torch.randn(8, generator=make_generator(11, "cuda"), device="cuda")
    b = torch.randn(8, generator=make_generator(11, "cuda"), device="cuda")
    assert torch.equal(a, b)


def test_check_generator_passes_none():
    """None means the global default generator, which torch homes correctly."""
    assert check_generator(None, "cpu") is None
    assert check_generator(None, "cuda") is None


def test_check_generator_returns_the_generator_on_a_match():
    g = make_generator(0, "cpu")
    assert check_generator(g, "cpu") is g


@CUDA
def test_check_generator_rejects_exactly_what_torch_rejects():
    """Both directions, each paired with torch refusing the same draw."""
    cpu_g, cuda_g = make_generator(0, "cpu"), make_generator(0, "cuda")

    with pytest.raises(ValueError, match="cpu.*targets 'cuda'"):
        check_generator(cpu_g, "cuda")
    with pytest.raises(RuntimeError):
        torch.randn(4, generator=cpu_g, device="cuda")

    with pytest.raises(ValueError, match="cuda.*targets 'cpu'"):
        check_generator(cuda_g, "cpu")
    with pytest.raises(RuntimeError):
        torch.randn(4, generator=cuda_g, device="cpu")


@CUDA
def test_check_generator_ignores_the_device_index():
    """One CUDA generator serves every CUDA device -- torch compares type only."""
    g = make_generator(0, "cuda")
    assert check_generator(g, "cuda:0") is g
    torch.randn(4, generator=g, device="cuda:0")


def test_check_generator_names_the_caller_and_the_fix():
    """The message has to say which generator and what to do about it."""
    g = make_generator(0, "cpu")
    with pytest.raises(ValueError) as exc:
        check_generator(g, "cuda", "sample_query_states generator")
    assert "sample_query_states generator" in str(exc.value)
    assert "make_generator" in str(exc.value)
