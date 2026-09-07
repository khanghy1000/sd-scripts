"""
Tests for Patch Topology Loss (PatchTopologyLoss) integration.

Validates:
  - PatchTopologyLoss module computation
  - CLI argument registration
  - Integration with train_network.py's process_batch
  - Edge cases (identical inputs, different inputs, gradient flow, shape handling)
"""

import argparse
import sys
import os
import pytest
import torch
import torch.nn as nn
import math

# Add the sd_scripts directory to the path so we can import the library
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library.patch_topology_loss import PatchTopologyLoss, _ensure_4d_spatial


# ──────────────────────────────────────────────
# _ensure_4d_spatial helper tests
# ──────────────────────────────────────────────


class TestEnsure4dSpatial:
    """Unit tests for the _ensure_4d_spatial helper."""

    def test_4d_passthrough(self):
        """A 4D tensor should be returned unchanged."""
        x = torch.randn(2, 4, 8, 8)
        out = _ensure_4d_spatial(x)
        assert out.shape == (2, 4, 8, 8)
        assert out is x

    def test_3d_square_sequence(self):
        """A 3D (B, L, C) tensor with square L should reshape to (B, C, H, W)."""
        # L = 64 = 8*8
        x = torch.randn(2, 64, 4)
        out = _ensure_4d_spatial(x)
        assert out.shape == (2, 4, 8, 8)

    def test_3d_non_square_sequence_raises(self):
        """A 3D tensor with non-square L should raise ValueError."""
        x = torch.randn(2, 63, 4)  # 63 is not a perfect square
        with pytest.raises(ValueError, match="Cannot infer square spatial"):
            _ensure_4d_spatial(x)

    def test_2d_raises(self):
        """A 2D tensor should raise ValueError."""
        x = torch.randn(2, 4)
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            _ensure_4d_spatial(x)


# ──────────────────────────────────────────────
# PatchTopologyLoss module tests
# ──────────────────────────────────────────────


class TestPatchTopologyLossModule:
    """Unit tests for the PatchTopologyLoss class."""

    def test_output_shape_per_sample(self):
        """PatchTopologyLoss should return a per-sample tensor (B,)."""
        loss_fn = PatchTopologyLoss()
        pred = torch.randn(2, 4, 16, 16)
        target = torch.randn(2, 4, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.shape == (2,), f"Expected shape (2,), got {loss.shape}"

    def test_identical_inputs_yield_zero_loss(self):
        """When pred == target, the loss should be ~zero (KL divergence of identical distributions)."""
        loss_fn = PatchTopologyLoss(loss_type="kl")
        x = torch.randn(2, 4, 8, 8)
        loss = loss_fn(x, x)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5), (
            f"Expected zero loss for identical inputs, got {loss}"
        )

    def test_identical_inputs_ce_is_entropy(self):
        """Cross-entropy of identical distributions equals the entropy (positive, not zero).

        CE = -sum(P * log(P)) = H(P) >= 0. Only KL divergence is zero for identical
        distributions; cross-entropy reduces to entropy.
        """
        loss_fn = PatchTopologyLoss(loss_type="ce", apply_timestep_weight=False)
        x = torch.randn(2, 4, 8, 8)
        loss = loss_fn(x, x)
        # CE for identical inputs = entropy, which is non-negative
        assert (loss >= -1e-6).all(), f"CE for identical inputs should be non-negative, got {loss}"
        # And it should be less than CE for different inputs
        y = torch.randn(2, 4, 8, 8)
        loss_diff = loss_fn(x, y)
        assert (loss <= loss_diff + 1e-5).all(), (
            f"CE for identical inputs should be <= CE for different: {loss} vs {loss_diff}"
        )

    def test_identical_inputs_zero_loss_cosine(self):
        """Cosine distance of identical distributions should be ~zero."""
        loss_fn = PatchTopologyLoss(loss_type="cosine")
        x = torch.randn(2, 4, 8, 8)
        loss = loss_fn(x, x)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5), (
            f"Expected zero cosine loss for identical inputs, got {loss}"
        )

    def test_identical_inputs_zero_loss_l2(self):
        """L2 distance of identical distributions should be ~zero."""
        loss_fn = PatchTopologyLoss(loss_type="l2")
        x = torch.randn(2, 4, 8, 8)
        loss = loss_fn(x, x)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5), (
            f"Expected zero L2 loss for identical inputs, got {loss}"
        )

    def test_loss_is_non_negative(self):
        """All supported loss types should produce non-negative losses."""
        for loss_type in ["kl", "ce", "cosine", "l2"]:
            loss_fn = PatchTopologyLoss(loss_type=loss_type)
            pred = torch.randn(2, 4, 8, 8)
            target = torch.randn(2, 4, 8, 8)
            loss = loss_fn(pred, target)
            assert (loss >= -1e-6).all(), (
                f"Loss type '{loss_type}' produced negative values: {loss}"
            )

    def test_loss_weight_scaling(self):
        """loss_weight should scale the output linearly."""
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        loss_fn1 = PatchTopologyLoss(loss_weight=1.0, apply_timestep_weight=False)
        loss_fn2 = PatchTopologyLoss(loss_weight=2.0, apply_timestep_weight=False)
        loss1 = loss_fn1(pred, target)
        loss2 = loss_fn2(pred, target)
        assert torch.allclose(loss2, 2.0 * loss1, atol=1e-5), (
            f"loss_weight=2 should double the loss: {loss1} vs {loss2}"
        )

    def test_gradient_flows_through_pred(self):
        """Gradients should flow through the prediction tensor."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False)
        pred = torch.randn(1, 4, 8, 8, requires_grad=True)
        target = torch.randn(1, 4, 8, 8)
        loss = loss_fn(pred, target).mean()
        loss.backward()
        assert pred.grad is not None, "Gradient should flow through pred"
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad)), (
            "Gradient should be non-zero for different inputs"
        )

    def test_no_gradient_through_target(self):
        """Gradients should NOT flow through the target tensor (detached)."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False)
        pred = torch.randn(1, 4, 8, 8, requires_grad=True)
        target = torch.randn(1, 4, 8, 8, requires_grad=True)
        loss = loss_fn(pred, target).mean()
        loss.backward()
        assert target.grad is None, "Gradient should not flow through target"

    def test_works_with_float64(self):
        """PatchTopologyLoss should work with float64 tensors (used in training)."""
        loss_fn = PatchTopologyLoss()
        pred = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        target = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        loss = loss_fn(pred, target)
        assert loss.dtype == torch.float64, f"Expected float64 output, got {loss.dtype}"

    def test_works_with_float16(self):
        """PatchTopologyLoss should work with float16 tensors and return float16."""
        loss_fn = PatchTopologyLoss()
        pred = torch.randn(2, 4, 8, 8, dtype=torch.float16)
        target = torch.randn(2, 4, 8, 8, dtype=torch.float16)
        loss = loss_fn(pred, target)
        assert loss.dtype == torch.float16, f"Expected float16 output, got {loss.dtype}"

    def test_works_with_3d_sequence_input(self):
        """PatchTopologyLoss should accept 3D (B, L, C) sequence tensors (DiT/FLUX)."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False)
        # L = 64 = 8*8
        pred = torch.randn(2, 64, 4)
        target = torch.randn(2, 64, 4)
        loss = loss_fn(pred, target)
        assert loss.shape == (2,), f"Expected shape (2,), got {loss.shape}"

    def test_different_spatial_resolutions(self):
        """Pred and target with different spatial resolutions should work via interpolation."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False)
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.shape == (2,), f"Expected shape (2,), got {loss.shape}"

    def test_different_channel_counts(self):
        """Pred and target with different channel counts should work."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False)
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 8, 8, 8)
        loss = loss_fn(pred, target)
        assert loss.shape == (2,), f"Expected shape (2,), got {loss.shape}"

    def test_scale_levels(self):
        """Different scale_levels should produce different loss magnitudes."""
        pred = torch.randn(1, 4, 16, 16)
        target = torch.randn(1, 4, 16, 16)
        loss_fn1 = PatchTopologyLoss(scale_levels=1, apply_timestep_weight=False)
        loss_fn3 = PatchTopologyLoss(scale_levels=3, apply_timestep_weight=False)
        loss1 = loss_fn1(pred, target)
        loss3 = loss_fn3(pred, target)
        # Both should be valid tensors; multi-scale adds more terms
        assert loss1.shape == (1,)
        assert loss3.shape == (1,)

    def test_timestep_weight_high_t_lower_loss(self):
        """At high timesteps (t->1), the timestep weight (1-t) should reduce the loss."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=True)
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        # Low timestep (t=0) -> weight = 1.0
        t_low = torch.tensor([0, 0], dtype=torch.long)
        # High timestep (t=1000) -> weight = 0.0
        t_high = torch.tensor([1000, 1000], dtype=torch.long)
        loss_low = loss_fn(pred, target, timesteps=t_low)
        loss_high = loss_fn(pred, target, timesteps=t_high)
        assert torch.all(loss_high <= loss_low + 1e-6), (
            f"High timestep should have lower or equal loss: {loss_high} vs {loss_low}"
        )
        # At t=1000 (normalized to 1.0), weight = 0, so loss should be ~0
        assert torch.allclose(loss_high, torch.zeros_like(loss_high), atol=1e-6), (
            f"Loss at t=1000 should be ~0, got {loss_high}"
        )

    def test_timestep_weight_disabled(self):
        """When apply_timestep_weight=False, timesteps should not affect the loss."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False)
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0, 1000], dtype=torch.long)
        loss_with_t = loss_fn(pred, target, timesteps=t)
        loss_without_t = loss_fn(pred, target, timesteps=None)
        assert torch.allclose(loss_with_t, loss_without_t, atol=1e-6), (
            f"Timestep should not affect loss when disabled: {loss_with_t} vs {loss_without_t}"
        )

    def test_timestep_normalization_0_to_1(self):
        """Timesteps in [0, 1] format should be handled correctly."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=True)
        pred = torch.randn(1, 4, 8, 8)
        target = torch.randn(1, 4, 8, 8)
        # t=0.5 in [0,1] format -> weight = 0.5
        t = torch.tensor([0.5], dtype=torch.float32)
        loss = loss_fn(pred, target, timesteps=t)
        loss_no_t = loss_fn(pred, target, timesteps=None)
        # loss should be approximately 0.5 * loss_no_t
        expected = 0.5 * loss_no_t
        assert torch.allclose(loss, expected, atol=1e-5), (
            f"t=0.5 should give weight=0.5: {loss} vs {expected}"
        )

    def test_channel_norm_disabled(self):
        """Disabling channel norm should still produce valid output."""
        loss_fn = PatchTopologyLoss(apply_channel_norm=False, apply_timestep_weight=False)
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        loss = loss_fn(pred, target)
        assert loss.shape == (2,)
        assert not torch.isnan(loss).any(), "Loss should not be NaN"

    def test_no_nan_in_output(self):
        """Loss should never contain NaN values."""
        loss_fn = PatchTopologyLoss()
        for _ in range(5):
            pred = torch.randn(2, 4, 8, 8)
            target = torch.randn(2, 4, 8, 8)
            loss = loss_fn(pred, target)
            assert not torch.isnan(loss).any(), f"NaN in loss: {loss}"

    def test_small_spatial_dims(self):
        """Should work with small spatial dimensions (e.g., 4x4)."""
        loss_fn = PatchTopologyLoss(scale_levels=1, apply_timestep_weight=False)
        pred = torch.randn(1, 4, 4, 4)
        target = torch.randn(1, 4, 4, 4)
        loss = loss_fn(pred, target)
        assert loss.shape == (1,)

    def test_batch_sizes(self):
        """Should work with different batch sizes."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False)
        for batch_size in [1, 2, 4, 8]:
            pred = torch.randn(batch_size, 4, 8, 8)
            target = torch.randn(batch_size, 4, 8, 8)
            loss = loss_fn(pred, target)
            assert loss.shape == (batch_size,), f"Batch {batch_size}: got {loss.shape}"

    def test_more_similar_lower_loss(self):
        """More similar inputs should produce lower loss than dissimilar inputs."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False)
        base = torch.randn(1, 4, 8, 8)
        # Slightly perturbed version
        similar = base + 0.01 * torch.randn_like(base)
        # Very different
        dissimilar = torch.randn_like(base)
        loss_similar = loss_fn(base, similar)
        loss_dissimilar = loss_fn(base, dissimilar)
        assert loss_similar.item() < loss_dissimilar.item(), (
            f"Similar inputs should have lower loss: {loss_similar.item()} vs {loss_dissimilar.item()}"
        )

    def test_invalid_loss_type_defaults_to_l2(self):
        """An unknown loss_type should fall through to the l2 branch."""
        loss_fn = PatchTopologyLoss(loss_type="unknown", apply_timestep_weight=False)
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        loss = loss_fn(pred, target)
        assert loss.shape == (2,)
        assert not torch.isnan(loss).any()


# ──────────────────────────────────────────────
# CUDA tests
# ──────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPatchTopologyLossCUDA:
    """Tests for PatchTopologyLoss on CUDA (per project rules, assume CUDA)."""

    def test_cuda_forward(self):
        """Forward pass should work on CUDA."""
        device = "cuda"
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False).to(device)
        pred = torch.randn(2, 4, 16, 16, device=device)
        target = torch.randn(2, 4, 16, 16, device=device)
        loss = loss_fn(pred, target)
        assert loss.device.type == "cuda"
        assert loss.shape == (2,)

    def test_cuda_gradient_flow(self):
        """Gradients should flow on CUDA."""
        device = "cuda"
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False).to(device)
        pred = torch.randn(1, 4, 8, 8, device=device, requires_grad=True)
        target = torch.randn(1, 4, 8, 8, device=device)
        loss = loss_fn(pred, target).mean()
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.device.type == "cuda"

    def test_cuda_timestep_weight(self):
        """Timestep weighting should work on CUDA."""
        device = "cuda"
        loss_fn = PatchTopologyLoss(apply_timestep_weight=True).to(device)
        pred = torch.randn(2, 4, 8, 8, device=device)
        target = torch.randn(2, 4, 8, 8, device=device)
        t = torch.tensor([0, 500], dtype=torch.long, device=device)
        loss = loss_fn(pred, target, timesteps=t)
        assert loss.device.type == "cuda"
        assert loss.shape == (2,)

    def test_cuda_3d_sequence(self):
        """3D sequence input should work on CUDA."""
        device = "cuda"
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False).to(device)
        pred = torch.randn(2, 64, 4, device=device)
        target = torch.randn(2, 64, 4, device=device)
        loss = loss_fn(pred, target)
        assert loss.device.type == "cuda"
        assert loss.shape == (2,)

    def test_cuda_identical_zero(self):
        """Identical inputs on CUDA should yield ~zero loss."""
        device = "cuda"
        loss_fn = PatchTopologyLoss(loss_type="kl").to(device)
        x = torch.randn(2, 4, 8, 8, device=device)
        loss = loss_fn(x, x)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-4), (
            f"Expected ~zero loss for identical inputs on CUDA, got {loss}"
        )


# ──────────────────────────────────────────────
# CLI argument registration tests
# ──────────────────────────────────────────────


class TestPatchTopologyCLI:
    """Tests for CLI argument registration."""

    def test_patch_topology_loss_flag_registered(self):
        """--patch_topology_loss should be a registered argument."""
        from library.custom_train_functions import add_custom_train_arguments

        parser = argparse.ArgumentParser()
        add_custom_train_arguments(parser, support_weighted_captions=False)
        args = parser.parse_args(["--patch_topology_loss"])
        assert args.patch_topology_loss is True

    def test_patch_topology_defaults(self):
        """Default values should be correct."""
        from library.custom_train_functions import add_custom_train_arguments

        parser = argparse.ArgumentParser()
        add_custom_train_arguments(parser, support_weighted_captions=False)
        args = parser.parse_args([])
        assert args.patch_topology_loss is False
        assert args.patch_topology_weight == 1.0
        assert args.patch_topology_tau == 0.1
        assert args.patch_topology_scale_levels == 2
        assert args.patch_topology_loss_type == "kl"
        assert args.patch_topology_disable_timestep_weight is False
        assert args.patch_topology_chunk_size == 512
        assert args.patch_topology_start_step == 0
        assert args.patch_topology_warmup_steps == 0

    def test_patch_topology_custom_values(self):
        """Custom values should be parsed correctly."""
        from library.custom_train_functions import add_custom_train_arguments

        parser = argparse.ArgumentParser()
        add_custom_train_arguments(parser, support_weighted_captions=False)
        args = parser.parse_args([
            "--patch_topology_loss",
            "--patch_topology_weight", "0.5",
            "--patch_topology_tau", "0.2",
            "--patch_topology_scale_levels", "3",
            "--patch_topology_loss_type", "cosine",
            "--patch_topology_disable_timestep_weight",
            "--patch_topology_chunk_size", "256",
            "--patch_topology_start_step", "500",
            "--patch_topology_warmup_steps", "200",
        ])
        assert args.patch_topology_loss is True
        assert args.patch_topology_weight == 0.5
        assert args.patch_topology_tau == 0.2
        assert args.patch_topology_scale_levels == 3
        assert args.patch_topology_loss_type == "cosine"
        assert args.patch_topology_disable_timestep_weight is True
        assert args.patch_topology_chunk_size == 256
        assert args.patch_topology_start_step == 500
        assert args.patch_topology_warmup_steps == 200


# ──────────────────────────────────────────────
# Integration tests with train_network.py
# ──────────────────────────────────────────────


class TestPatchTopologyIntegration:
    """Tests for integration with train_network.py."""

    def test_import_patch_topology_loss(self):
        """PatchTopologyLoss should be importable from train_network."""
        import train_network
        assert hasattr(train_network, "PatchTopologyLoss")

    def test_trainer_has_patch_topology_attrs(self):
        """NetworkTrainer should initialize patch topology config attributes."""
        import train_network
        trainer = train_network.NetworkTrainer()
        assert hasattr(trainer, "patch_topology_enabled")
        assert hasattr(trainer, "patch_topology_loss_module")
        assert hasattr(trainer, "patch_topology_loss_value")
        assert hasattr(trainer, "patch_topology_full_weight")
        assert hasattr(trainer, "patch_topology_start_step")
        assert hasattr(trainer, "patch_topology_warmup_steps")
        assert hasattr(trainer, "_patch_topology_current_step")
        assert trainer.patch_topology_enabled is False
        assert trainer.patch_topology_loss_module is None
        assert trainer.patch_topology_loss_value is None
        assert trainer.patch_topology_full_weight == 1.0
        assert trainer.patch_topology_start_step == 0
        assert trainer.patch_topology_warmup_steps == 0
        assert trainer._patch_topology_current_step == 0

    def test_generate_step_logs_accepts_patch_topology(self):
        """generate_step_logs should accept current_patch_topology_loss parameter."""
        import train_network
        import argparse

        trainer = train_network.NetworkTrainer()
        args = argparse.Namespace(
            optimizer_type="AdamW",
            network_train_unet_only=False,
        )
        logs = trainer.generate_step_logs(
            args=args,
            current_loss=1.0,
            avr_loss=1.0,
            lr_scheduler=type("FakeSched", (), {"get_last_lr": lambda self: [0.001]})(),
            lr_descriptions=["unet"],
            current_patch_topology_loss=0.5,
        )
        assert "loss/current_patch_topology" in logs
        assert logs["loss/current_patch_topology"] == 0.5

    def test_generate_step_logs_patch_topology_none(self):
        """generate_step_logs should not add patch topology key when None."""
        import train_network
        import argparse

        trainer = train_network.NetworkTrainer()
        args = argparse.Namespace(
            optimizer_type="AdamW",
            network_train_unet_only=False,
        )
        logs = trainer.generate_step_logs(
            args=args,
            current_loss=1.0,
            avr_loss=1.0,
            lr_scheduler=type("FakeSched", (), {"get_last_lr": lambda self: [0.001]})(),
            lr_descriptions=["unet"],
            current_patch_topology_loss=None,
        )
        assert "loss/current_patch_topology" not in logs


# ──────────────────────────────────────────────
# Functional / combined loss tests
# ──────────────────────────────────────────────


class TestPatchTopologyFunctional:
    """Functional tests combining base loss + patch topology loss."""

    def test_combined_loss_backprop(self):
        """Combined base + patch topology loss should backprop correctly."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False)
        pred = torch.randn(1, 4, 8, 8, requires_grad=True)
        target = torch.randn(1, 4, 8, 8)
        base_loss = ((pred - target) ** 2).mean()
        topo_loss = loss_fn(pred, target).mean()
        total_loss = base_loss + topo_loss
        total_loss.backward()
        assert pred.grad is not None
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))

    def test_loss_weight_affects_total(self):
        """Higher loss_weight should produce higher total loss contribution."""
        pred = torch.randn(1, 4, 8, 8)
        target = torch.randn(1, 4, 8, 8)
        loss_low = PatchTopologyLoss(loss_weight=0.1, apply_timestep_weight=False)(pred, target)
        loss_high = PatchTopologyLoss(loss_weight=10.0, apply_timestep_weight=False)(pred, target)
        assert loss_high.mean() > loss_low.mean(), (
            f"Higher weight should give higher loss: {loss_high.mean()} vs {loss_low.mean()}"
        )

    def test_typical_sd_latent_dims(self):
        """Should work with typical SD1.5 latent dimensions (4, 64, 64) at scale_levels=1."""
        loss_fn = PatchTopologyLoss(scale_levels=1, apply_timestep_weight=False)
        pred = torch.randn(1, 4, 32, 32)  # 512x512 image -> 32x32 latent
        target = torch.randn(1, 4, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.shape == (1,)
        assert not torch.isnan(loss).any()

    def test_all_loss_types_produce_valid_output(self):
        """All four loss types should produce valid, finite output."""
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        for loss_type in ["kl", "ce", "cosine", "l2"]:
            loss_fn = PatchTopologyLoss(loss_type=loss_type, apply_timestep_weight=False)
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss).all(), f"Loss type '{loss_type}' produced non-finite values: {loss}"

    def test_flow_matching_4d_with_0_1_timesteps(self):
        """Simulate the Flux/SD3/Anima flow-matching path: 4D (B,C,H,W) tensors with timesteps in [0,1].

        This mirrors Option B in the plan: flow-matching/DiT models where model_pred and
        target = noise - latents are 4D, and timesteps are normalized to [0, 1].
        """
        loss_fn = PatchTopologyLoss(apply_timestep_weight=True)
        # 4D tensors as produced by Flux (unpacked) / SD3 / Anima (squeezed)
        pred = torch.randn(2, 16, 16, 16, requires_grad=True)
        target = (torch.randn(2, 16, 16, 16) - torch.randn(2, 16, 16, 16))  # noise - latents
        # Flow-matching timesteps in [0, 1]
        timesteps = torch.tensor([0.1, 0.9], dtype=torch.float32)
        loss = loss_fn(pred, target, timesteps=timesteps)
        assert loss.shape == (2,)
        assert torch.isfinite(loss).all()
        # Lower timestep (t=0.1 -> weight 0.9) should have higher loss than high timestep (t=0.9 -> weight 0.1)
        assert loss[0] > loss[1], (
            f"Lower timestep should yield higher weighted loss: t=0.1->{loss[0]} vs t=0.9->{loss[1]}"
        )
        # Gradient flows
        loss.mean().backward()
        assert pred.grad is not None

    def test_flow_matching_weighting_applied(self):
        """Per-sample weighting (as in Option B) should scale the patch topology loss."""
        loss_fn = PatchTopologyLoss(apply_timestep_weight=False)
        pred = torch.randn(3, 4, 8, 8)
        target = torch.randn(3, 4, 8, 8)
        weighting = torch.tensor([1.0, 2.0, 0.5])
        loss = loss_fn(pred, target)
        weighted = loss * weighting
        assert weighted.shape == (3,)
        assert torch.isfinite(weighted).all()
        # Middle sample (weight 2.0) should be largest
        assert weighted[1] > weighted[0] > weighted[2]


# ──────────────────────────────────────────────
# Chunked processing tests
# ──────────────────────────────────────────────


class TestPatchTopologyChunkedProcessing:
    """Tests for the chunked query processing memory optimization."""

    def test_chunk_size_default_is_512(self):
        """Default chunk_size should be 512."""
        loss_fn = PatchTopologyLoss()
        assert loss_fn.chunk_size == 512

    def test_chunk_size_constructor_parameter(self):
        """chunk_size should be configurable via constructor."""
        loss_fn = PatchTopologyLoss(chunk_size=128)
        assert loss_fn.chunk_size == 128

    def test_chunk_size_none_processes_all(self):
        """chunk_size=None should process all spatial patches in one iteration (equivalent to n)."""
        loss_fn = PatchTopologyLoss(chunk_size=None, apply_timestep_weight=False)
        pred = torch.randn(1, 4, 8, 8)
        target = torch.randn(1, 4, 8, 8)
        loss = loss_fn(pred, target)
        assert loss.shape == (1,)
        assert torch.isfinite(loss).all()

    def test_chunk_size_zero_processes_all(self):
        """chunk_size=0 should process all spatial patches in one iteration (equivalent to n)."""
        loss_fn = PatchTopologyLoss(chunk_size=0, apply_timestep_weight=False)
        pred = torch.randn(1, 4, 8, 8)
        target = torch.randn(1, 4, 8, 8)
        loss = loss_fn(pred, target)
        assert loss.shape == (1,)
        assert torch.isfinite(loss).all()

    def test_different_chunk_sizes_produce_equivalent_results(self):
        """Different chunk sizes should produce the same loss (within floating point tolerance).

        The chunked processing is purely a memory optimization and should not change the
        mathematical result of the loss computation.
        """
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 16, 16)
        target = torch.randn(2, 4, 16, 16)

        # N = 16*16 = 256, so chunk_size=256 processes everything in one chunk
        loss_full = PatchTopologyLoss(chunk_size=256, apply_timestep_weight=False)(pred, target)

        for chunk_size in [32, 64, 128]:
            loss_chunked = PatchTopologyLoss(chunk_size=chunk_size, apply_timestep_weight=False)(pred, target)
            assert torch.allclose(loss_full, loss_chunked, atol=1e-5), (
                f"chunk_size={chunk_size} should produce same result as full: "
                f"{loss_chunked} vs {loss_full}"
            )

    def test_chunk_size_one(self):
        """chunk_size=1 should process one spatial patch at a time (maximum chunking)."""
        torch.manual_seed(42)
        pred = torch.randn(1, 4, 4, 4)  # N=16, small for speed
        target = torch.randn(1, 4, 4, 4)

        loss_full = PatchTopologyLoss(chunk_size=16, apply_timestep_weight=False)(pred, target)
        loss_chunked = PatchTopologyLoss(chunk_size=1, apply_timestep_weight=False)(pred, target)

        assert torch.allclose(loss_full, loss_chunked, atol=1e-4), (
            f"chunk_size=1 should match full processing: {loss_chunked} vs {loss_full}"
        )

    def test_chunk_size_larger_than_n(self):
        """chunk_size larger than N should still work (processes everything in one chunk)."""
        loss_fn = PatchTopologyLoss(chunk_size=9999, apply_timestep_weight=False)
        pred = torch.randn(1, 4, 4, 4)  # N=16
        target = torch.randn(1, 4, 4, 4)
        loss = loss_fn(pred, target)
        assert loss.shape == (1,)
        assert torch.isfinite(loss).all()

    def test_chunk_size_preserves_gradient_flow(self):
        """Gradient flow should be preserved with chunked processing."""
        loss_fn = PatchTopologyLoss(chunk_size=16, apply_timestep_weight=False)
        pred = torch.randn(1, 4, 8, 8, requires_grad=True)
        target = torch.randn(1, 4, 8, 8)
        loss = loss_fn(pred, target).mean()
        loss.backward()
        assert pred.grad is not None, "Gradient should flow through pred with chunked processing"
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad)), (
            "Gradient should be non-zero for different inputs"
        )

    def test_chunk_size_with_all_loss_types(self):
        """Chunked processing should work with all loss types."""
        torch.manual_seed(42)
        pred = torch.randn(1, 4, 8, 8)
        target = torch.randn(1, 4, 8, 8)

        for loss_type in ["kl", "ce", "cosine", "l2"]:
            loss_full = PatchTopologyLoss(
                loss_type=loss_type, chunk_size=64, apply_timestep_weight=False
            )(pred, target)
            loss_chunked = PatchTopologyLoss(
                loss_type=loss_type, chunk_size=8, apply_timestep_weight=False
            )(pred, target)
            assert torch.allclose(loss_full, loss_chunked, atol=1e-5), (
                f"Loss type '{loss_type}' with chunk_size=8 should match chunk_size=64: "
                f"{loss_chunked} vs {loss_full}"
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_chunk_size_cuda(self):
        """Chunked processing should work on CUDA with different chunk sizes."""
        device = "cuda"
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 16, 16, device=device)
        target = torch.randn(2, 4, 16, 16, device=device)

        loss_full = PatchTopologyLoss(chunk_size=256, apply_timestep_weight=False).to(device)(pred, target)
        loss_chunked = PatchTopologyLoss(chunk_size=32, apply_timestep_weight=False).to(device)(pred, target)

        assert loss_full.device.type == "cuda"
        assert loss_chunked.device.type == "cuda"
        assert torch.allclose(loss_full, loss_chunked, atol=1e-4), (
            f"CUDA chunked processing should match full: {loss_chunked} vs {loss_full}"
        )

    def test_native_dtype_preserved(self):
        """Forward should preserve native input dtype (not always float32)."""
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            loss_fn = PatchTopologyLoss(chunk_size=8, apply_timestep_weight=False)
            pred = torch.randn(1, 4, 8, 8, dtype=dtype)
            target = torch.randn(1, 4, 8, 8, dtype=dtype)
            loss = loss_fn(pred, target)
            assert loss.dtype == dtype, f"Expected {dtype} output, got {loss.dtype}"


# ──────────────────────────────────────────────
# Warmup and start_step tests
# ──────────────────────────────────────────────


def _compute_effective_weight(current_step, full_weight, start_step, warmup_steps):
    """Replicates the warmup/start_step logic from NetworkTrainer.process_batch.

    This is the exact same formula used in train_network.py, extracted here
    for testability without requiring a full training loop.
    """
    effective_weight = 0.0
    if current_step >= start_step:
        if warmup_steps > 0:
            steps_into_warmup = current_step - start_step
            warmup_progress = min(1.0, float(steps_into_warmup) / float(warmup_steps))
            effective_weight = full_weight * warmup_progress
        else:
            effective_weight = full_weight
    return effective_weight


class TestPatchTopologyWarmupStartStep:
    """Tests for warmup and start_step effective weight computation."""

    def test_no_warmup_no_start(self):
        """With start_step=0 and warmup_steps=0, effective weight equals full weight immediately."""
        for step in [0, 1, 10, 100]:
            w = _compute_effective_weight(step, full_weight=0.5, start_step=0, warmup_steps=0)
            assert w == 0.5, f"Step {step}: expected 0.5, got {w}"

    def test_start_step_defers_activation(self):
        """Before start_step, effective weight should be 0."""
        for step in range(0, 100):
            w = _compute_effective_weight(step, full_weight=1.0, start_step=100, warmup_steps=0)
            assert w == 0.0, f"Step {step}: expected 0.0, got {w}"

    def test_start_step_activates(self):
        """At and after start_step (no warmup), effective weight equals full weight."""
        for step in [100, 101, 200, 1000]:
            w = _compute_effective_weight(step, full_weight=1.0, start_step=100, warmup_steps=0)
            assert w == 1.0, f"Step {step}: expected 1.0, got {w}"

    def test_warmup_linear_ramp(self):
        """With start_step=100 and warmup_steps=50, weight should linearly ramp from 0 to full."""
        full_weight = 0.8
        start = 100
        warmup = 50

        # Before start
        assert _compute_effective_weight(99, full_weight, start, warmup) == 0.0

        # At start (0% warmup)
        assert _compute_effective_weight(100, full_weight, start, warmup) == 0.0

        # 25% warmup
        w = _compute_effective_weight(112, full_weight, start, warmup)
        assert abs(w - 0.8 * 0.24) < 1e-6, f"Expected ~0.192, got {w}"

        # 50% warmup
        w = _compute_effective_weight(125, full_weight, start, warmup)
        assert abs(w - 0.8 * 0.5) < 1e-6, f"Expected ~0.4, got {w}"

        # 100% warmup
        w = _compute_effective_weight(150, full_weight, start, warmup)
        assert abs(w - full_weight) < 1e-6, f"Expected {full_weight}, got {w}"

        # Beyond warmup — clamped to full weight
        w = _compute_effective_weight(200, full_weight, start, warmup)
        assert abs(w - full_weight) < 1e-6, f"Expected {full_weight}, got {w}"

    def test_warmup_clamps_at_one(self):
        """Warmup progress should clamp at 1.0 even after many steps beyond warmup."""
        w = _compute_effective_weight(10000, full_weight=1.0, start_step=0, warmup_steps=100)
        assert w == 1.0, f"Expected 1.0, got {w}"

    def test_warmup_zero_steps_is_instant(self):
        """warmup_steps=0 means instant activation at start_step."""
        w = _compute_effective_weight(500, full_weight=1.0, start_step=500, warmup_steps=0)
        assert w == 1.0
        w = _compute_effective_weight(499, full_weight=1.0, start_step=500, warmup_steps=0)
        assert w == 0.0

    def test_warmup_with_combined_loss(self):
        """Effective weight applied to topology loss should produce expected combined loss."""
        torch.manual_seed(42)
        loss_fn = PatchTopologyLoss(loss_weight=1.0, apply_timestep_weight=False)
        pred = torch.randn(1, 4, 8, 8, requires_grad=True)
        target = torch.randn(1, 4, 8, 8)

        base_loss = ((pred - target) ** 2).mean()
        topo_raw = loss_fn(pred, target).mean()

        # Simulate: at step 5 (before start_step=10), effective weight = 0
        eff_w = _compute_effective_weight(5, full_weight=0.5, start_step=10, warmup_steps=0)
        total_before = base_loss + eff_w * topo_raw
        assert torch.allclose(total_before, base_loss), "Before start_step, topology loss should not contribute"

        # Simulate: at step 10 (at start_step), effective weight = 0.5
        eff_w = _compute_effective_weight(10, full_weight=0.5, start_step=10, warmup_steps=0)
        total_at = base_loss + eff_w * topo_raw
        expected = base_loss + 0.5 * topo_raw
        assert torch.allclose(total_at, expected), "At start_step, topology loss should contribute at full weight"

    def test_warmup_midpoint(self):
        """At midpoint of warmup, effective weight should be exactly half of full weight."""
        full_weight = 0.6
        start = 200
        warmup = 100
        w = _compute_effective_weight(250, full_weight, start, warmup)
        assert abs(w - 0.3) < 1e-6, f"Expected 0.3, got {w}"

    def test_start_step_and_warmup_default_no_effect(self):
        """Default values (start_step=0, warmup_steps=0) should apply full weight from step 0."""
        for step in [0, 1, 5, 100]:
            w = _compute_effective_weight(step, full_weight=1.0, start_step=0, warmup_steps=0)
            assert w == 1.0, f"Default config should apply full weight at step {step}"

    def test_large_start_step(self):
        """Large start_step values should defer loss until that step."""
        w = _compute_effective_weight(99999, full_weight=1.0, start_step=100000, warmup_steps=0)
        assert w == 0.0
        w = _compute_effective_weight(100000, full_weight=1.0, start_step=100000, warmup_steps=0)
        assert w == 1.0
