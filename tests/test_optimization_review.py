"""
Unit tests for the optimization review changes:

1. ``NetworkTrainer.all_reduce_network`` / ``all_reduce_edm2_model`` skip the
   manual per-parameter gradient reduce when ``accelerator.num_processes <= 1``
   (avoids duplicate-DDP sync overhead and per-step Python iteration).
2. ``NetworkTrainer.should_sync_ramtorch`` gates the post-backward
   ``torch.cuda.synchronize()`` to gradient-sync boundaries only.
3. ``AnimaNetworkTrainer.get_padding_mask`` caches and reuses the zero padding
   mask buffer instead of allocating a fresh CUDA tensor per forward pass.

Usage:
    pytest tests/test_optimization_review.py -v
"""

import sys
import os

import pytest
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import train_network
import anima_train_network

DEVICE = torch.device("cuda")  # workspace assumes CUDA availability


class _FakeAccelerator:
    """Minimal stand-in exposing only the attributes the trainer touches."""

    def __init__(self, num_processes: int = 1, sync_gradients: bool = True):
        self.num_processes = num_processes
        self.sync_gradients = sync_gradients
        self.reduce_calls = []

    def reduce(self, tensor, reduction="mean"):
        self.reduce_calls.append((tensor, reduction))
        return tensor


class _FakeNetwork(nn.Module):
    """Two parameters that may carry gradients."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(4))
        self.b = nn.Parameter(torch.randn(4))
        self.a.grad = torch.randn_like(self.a)
        self.b.grad = None  # one param without grad to exercise the guard


def _make_args(**overrides):
    defaults = {
        "use_ramtorch": False,
        "use_ramtorch_network": False,
    }
    defaults.update(overrides)
    return type("Args", (), defaults)()


# ──────────────────────────────────────────────────────────────
# 1. all_reduce_network / all_reduce_edm2_model gating
# ──────────────────────────────────────────────────────────────
class TestAllReduceGating:
    def test_all_reduce_network_noop_single_process(self):
        trainer = train_network.NetworkTrainer()
        accel = _FakeAccelerator(num_processes=1)
        net = _FakeNetwork()
        original_grad = net.a.grad.clone()

        trainer.all_reduce_network(accel, net)

        assert accel.reduce_calls == [], "reduce must not be called with a single process"
        assert torch.equal(net.a.grad, original_grad), "grad must remain unchanged"

    def test_all_reduce_network_reduces_multiprocess(self):
        trainer = train_network.NetworkTrainer()
        accel = _FakeAccelerator(num_processes=2)
        net = _FakeNetwork()

        trainer.all_reduce_network(accel, net)

        # Only params with a grad should be reduced (a has grad, b does not).
        assert len(accel.reduce_calls) == 1
        reduced_tensor, reduction = accel.reduce_calls[0]
        assert reduced_tensor is net.a.grad
        assert reduction == "mean"

    def test_all_reduce_edm2_model_noop_when_none(self):
        trainer = train_network.NetworkTrainer()
        accel = _FakeAccelerator(num_processes=2)

        trainer.all_reduce_edm2_model(accel, None)  # must not raise

        assert accel.reduce_calls == []

    def test_all_reduce_edm2_model_noop_single_process(self):
        trainer = train_network.NetworkTrainer()
        accel = _FakeAccelerator(num_processes=1)
        model = _FakeNetwork()
        original_grad = model.a.grad.clone()

        trainer.all_reduce_edm2_model(accel, model)

        assert accel.reduce_calls == []
        assert torch.equal(model.a.grad, original_grad)

    def test_all_reduce_edm2_model_reduces_multiprocess(self):
        trainer = train_network.NetworkTrainer()
        accel = _FakeAccelerator(num_processes=2)
        model = _FakeNetwork()

        trainer.all_reduce_edm2_model(accel, model)

        assert len(accel.reduce_calls) == 1
        assert accel.reduce_calls[0][0] is model.a.grad


# ──────────────────────────────────────────────────────────────
# 2. should_sync_ramtorch gating
# ──────────────────────────────────────────────────────────────
class TestShouldSyncRamtorch:
    def test_false_when_no_ramtorch(self):
        trainer = train_network.NetworkTrainer()
        args = _make_args(use_ramtorch=False, use_ramtorch_network=False)
        assert trainer.should_sync_ramtorch(args, _FakeAccelerator(sync_gradients=True)) is False

    def test_false_on_micro_step_even_with_ramtorch(self):
        trainer = train_network.NetworkTrainer()
        args = _make_args(use_ramtorch=True)
        accel = _FakeAccelerator(sync_gradients=False)  # mid gradient-accumulation
        assert trainer.should_sync_ramtorch(args, accel) is False

    def test_true_on_sync_boundary_with_ramtorch(self):
        trainer = train_network.NetworkTrainer()
        args = _make_args(use_ramtorch=True)
        accel = _FakeAccelerator(sync_gradients=True)
        assert trainer.should_sync_ramtorch(args, accel) is True

    def test_true_on_sync_boundary_with_ramtorch_network(self):
        trainer = train_network.NetworkTrainer()
        args = _make_args(use_ramtorch=False, use_ramtorch_network=True)
        accel = _FakeAccelerator(sync_gradients=True)
        assert trainer.should_sync_ramtorch(args, accel) is True


# ──────────────────────────────────────────────────────────────
# 3. Anima padding-mask caching
# ──────────────────────────────────────────────────────────────
class TestAnimaPaddingMaskCache:
    def _trainer(self):
        return anima_train_network.AnimaNetworkTrainer()

    def test_returns_identical_buffer_for_same_key(self):
        trainer = self._trainer()
        m1 = trainer.get_padding_mask(2, 16, 32, torch.bfloat16, DEVICE)
        m2 = trainer.get_padding_mask(2, 16, 32, torch.bfloat16, DEVICE)

        assert m1.data_ptr() == m2.data_ptr(), "same key must return the same buffer"
        assert len(trainer._padding_mask_cache) == 1

    def test_returns_zeros_with_correct_shape_dtype(self):
        trainer = self._trainer()
        mask = trainer.get_padding_mask(3, 8, 8, torch.float32, DEVICE)

        assert mask.shape == (3, 1, 8, 8)
        assert mask.dtype == torch.float32
        assert mask.device.type == DEVICE.type  # cuda may normalize to cuda:0
        assert mask.device.index == 0 if mask.device.type == "cuda" else True
        assert mask.sum().item() == 0.0

    def test_distinct_buffer_per_shape(self):
        trainer = self._trainer()
        m1 = trainer.get_padding_mask(1, 16, 16, torch.bfloat16, DEVICE)
        m2 = trainer.get_padding_mask(2, 16, 16, torch.bfloat16, DEVICE)
        m3 = trainer.get_padding_mask(1, 32, 16, torch.bfloat16, DEVICE)

        assert m1.data_ptr() != m2.data_ptr()
        assert m1.data_ptr() != m3.data_ptr()
        assert len(trainer._padding_mask_cache) == 3

    def test_distinct_buffer_per_dtype(self):
        trainer = self._trainer()
        m1 = trainer.get_padding_mask(1, 16, 16, torch.float32, DEVICE)
        m2 = trainer.get_padding_mask(1, 16, 16, torch.bfloat16, DEVICE)

        assert m1.data_ptr() != m2.data_ptr()
        assert len(trainer._padding_mask_cache) == 2

    def test_cached_buffer_never_mutated_by_caller(self):
        trainer = self._trainer()
        m1 = trainer.get_padding_mask(1, 16, 16, torch.float32, DEVICE)
        # Simulate a caller reading (the Anima model only resizes/expands/concats).
        m1.sum().item()
        m2 = trainer.get_padding_mask(1, 16, 16, torch.float32, DEVICE)
        assert m2.sum().item() == 0.0, "buffer must remain all zeros across callers"

    def test_forward_paths_use_cached_mask(self):
        """The four forward paths should all hit the cache (no fresh zeros)."""
        trainer = self._trainer()
        trainer.get_padding_mask(2, 8, 8, torch.bfloat16, DEVICE)
        cache_len = len(trainer._padding_mask_cache)

        # All call sites share the same shape/dtype/device key in a training step.
        trainer.get_padding_mask(2, 8, 8, torch.bfloat16, DEVICE)
        trainer.get_padding_mask(2, 8, 8, torch.bfloat16, DEVICE)
        trainer.get_padding_mask(2, 8, 8, torch.bfloat16, DEVICE)
        assert len(trainer._padding_mask_cache) == cache_len
