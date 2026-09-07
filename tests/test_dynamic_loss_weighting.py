"""
Tests for dynamic multi-loss weighting (none/dwa/gradnorm), PatchTopologyLoss
mask support, mask extraction from batches, and FLUX metadata integration.

CUDA is assumed to be available; GPU-dependent tests are skipped if it is not.
"""

import argparse
import sys
import os
import pytest
import torch
import torch.nn as nn

# Add the sd_scripts directory to the path so we can import the library
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library.dynamic_loss_weighting import DynamicLossWeighter, build_weighter_from_args
from library.patch_topology_loss import PatchTopologyLoss, extract_spatial_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ──────────────────────────────────────────────
# DynamicLossWeighter: mode validation & none
# ──────────────────────────────────────────────


class TestWeighterBasics:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown dynamic weighting mode"):
            DynamicLossWeighter(mode="banana")

    def test_none_mode_passthrough(self):
        w = DynamicLossWeighter(mode="none", user_weight=0.7)
        base = torch.tensor(1.0, requires_grad=True)
        aux = torch.tensor(2.0, requires_grad=True)
        assert w.compute_weight(base, aux) == pytest.approx(0.7)
        # History should still be recorded for consistency
        assert len(w._history) == 1

    def test_build_weighter_none_returns_none(self):
        args = argparse.Namespace(patch_topology_dynamic_weighting="none")
        assert build_weighter_from_args(args, 1.0) is None

    def test_build_weighter_dwa(self):
        args = argparse.Namespace(
            patch_topology_dynamic_weighting="dwa",
            patch_topology_dwa_temperature=3.0,
            patch_topology_gradnorm_alpha=1.0,
            patch_topology_dynamic_max_weight=5.0,
        )
        w = build_weighter_from_args(args, 0.5)
        assert w is not None
        assert w.mode == "dwa"
        assert w.user_weight == 0.5
        assert w.dwa_temperature == 3.0
        assert w.max_weight == 5.0

    def test_build_weighter_missing_args_uses_defaults(self):
        args = argparse.Namespace(patch_topology_dynamic_weighting="dwa")
        w = build_weighter_from_args(args, 1.0)
        assert w.dwa_temperature == 2.0
        assert w.gradnorm_alpha == 1.5
        assert w.max_weight == 10.0


# ──────────────────────────────────────────────
# DWA behavior
# ──────────────────────────────────────────────


class TestDWA:
    def test_first_steps_return_user_weight(self):
        """DWA needs 2 history entries; until then it returns the user weight."""
        w = DynamicLossWeighter(mode="dwa", user_weight=0.5)
        assert w.compute_weight(torch.tensor(1.0), torch.tensor(1.0)) == pytest.approx(0.5)

    def test_faster_decreasing_aux_gets_lower_weight(self):
        """DWA up-weights slower-learning tasks: aux decreasing faster than
        base -> r_aux < r_base -> weight < user_weight."""
        w = DynamicLossWeighter(mode="dwa", user_weight=1.0, dwa_temperature=2.0)
        # Step 1: base=1.0, aux=1.0
        w.compute_weight(torch.tensor(1.0), torch.tensor(1.0))
        # Step 2: base barely decreases, aux halves
        weight = w.compute_weight(torch.tensor(0.99), torch.tensor(0.5))
        assert weight < 1.0

    def test_slower_decreasing_aux_gets_higher_weight(self):
        """Aux loss stagnating relative to base -> weight > user_weight."""
        w = DynamicLossWeighter(mode="dwa", user_weight=1.0, dwa_temperature=2.0)
        w.compute_weight(torch.tensor(1.0), torch.tensor(1.0))
        # base halves, aux barely decreases
        weight = w.compute_weight(torch.tensor(0.5), torch.tensor(0.99))
        assert weight > 1.0

    def test_weight_clamped_to_max(self):
        # Base collapses while aux stagnates -> r_aux >> r_base -> clamped to max.
        w = DynamicLossWeighter(mode="dwa", user_weight=1.0, dwa_temperature=0.1, max_weight=3.0)
        w.compute_weight(torch.tensor(1.0), torch.tensor(1.0))
        weight = w.compute_weight(torch.tensor(1e-6), torch.tensor(1.0))
        assert weight == pytest.approx(3.0)

    def test_zero_loss_history_no_nan(self):
        """Zero losses must not produce NaN/Inf."""
        w = DynamicLossWeighter(mode="dwa", user_weight=1.0)
        w.compute_weight(torch.tensor(0.0), torch.tensor(0.0))
        weight = w.compute_weight(torch.tensor(0.0), torch.tensor(0.0))
        assert weight == weight  # not NaN
        assert weight >= 0.0


# ──────────────────────────────────────────────
# GradNorm behavior (CUDA)
# ──────────────────────────────────────────────


@requires_cuda
class TestGradNorm:
    def _make_losses(self):
        """Tiny shared-parameter model: two losses on the same parameters."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(8, 8, device=DEVICE))
        x = torch.randn(8, 8, device=DEVICE)
        base = ((x @ p) ** 2).mean()
        aux = ((x @ p).sum() - 3.0) ** 2
        return p, base, aux

    def test_gradnorm_returns_finite_weight(self):
        p, base, aux = self._make_losses()
        w = DynamicLossWeighter(mode="gradnorm", user_weight=1.0)
        weight = w.compute_weight(base, aux, shared_params=[p])
        assert weight == weight  # not NaN
        assert 0.0 <= weight <= w.max_weight

    def test_gradnorm_balances_gradient_norms(self):
        """If aux gradient is much larger than base, weight should be < 1."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(8, 8, device=DEVICE))
        x = torch.randn(8, 8, device=DEVICE)
        base = ((x @ p) ** 2).mean()  # small gradients
        aux = 1000.0 * ((x @ p).sum()) ** 2  # huge gradients
        w = DynamicLossWeighter(mode="gradnorm", user_weight=1.0)
        weight = w.compute_weight(base, aux, shared_params=[p])
        assert weight < 1.0

    def test_gradnorm_no_params_falls_back(self):
        _, base, aux = self._make_losses()
        w = DynamicLossWeighter(mode="gradnorm", user_weight=0.3)
        assert w.compute_weight(base, aux, shared_params=[]) == pytest.approx(0.3)

    def test_gradnorm_zero_aux_grad_falls_back(self):
        """Aux loss with zero gradient -> fallback to user weight."""
        p = nn.Parameter(torch.randn(4, 4, device=DEVICE))
        base = (p**2).mean()
        aux = (p.detach() ** 2).mean() * 0.0  # no gradient path to p
        w = DynamicLossWeighter(mode="gradnorm", user_weight=0.8)
        assert w.compute_weight(base, aux, shared_params=[p]) == pytest.approx(0.8)

    def test_state_dict_roundtrip(self):
        w = DynamicLossWeighter(mode="dwa", user_weight=1.0)
        w.compute_weight(torch.tensor(1.0), torch.tensor(0.5))
        w.compute_weight(torch.tensor(0.9), torch.tensor(0.4))
        state = w.state_dict()

        w2 = DynamicLossWeighter(mode="dwa", user_weight=1.0)
        w2.load_state_dict(state)
        assert list(w2._history) == list(w._history)
        assert w2._initial_base == w._initial_base
        assert w2._initial_aux == w._initial_aux


# ──────────────────────────────────────────────
# extract_spatial_mask
# ──────────────────────────────────────────────


class TestExtractSpatialMask:
    def test_alpha_masks_3d(self):
        batch = {"alpha_masks": torch.ones(2, 16, 16)}
        m = extract_spatial_mask(batch, (8, 8), "cpu", torch.float32)
        assert m.shape == (2, 1, 8, 8)
        assert m.mean().item() == pytest.approx(1.0)

    def test_alpha_masks_4d(self):
        batch = {"alpha_masks": torch.ones(2, 1, 16, 16)}
        m = extract_spatial_mask(batch, (16, 16), "cpu", torch.float32)
        assert m.shape == (2, 1, 16, 16)

    def test_conditioning_images_remap(self):
        """conditioning_images in [-1, 1] should remap to [0, 1]."""
        batch = {"conditioning_images": torch.full((2, 3, 8, 8), -1.0)}
        m = extract_spatial_mask(batch, (8, 8), "cpu", torch.float32)
        assert m.shape == (2, 1, 8, 8)
        assert m.mean().item() == pytest.approx(0.0)

    def test_conditioning_images_priority_over_alpha(self):
        batch = {
            "conditioning_images": torch.full((2, 3, 8, 8), 1.0),
            "alpha_masks": torch.zeros(2, 8, 8),
        }
        m = extract_spatial_mask(batch, (8, 8), "cpu", torch.float32)
        assert m.mean().item() == pytest.approx(1.0)

    def test_no_mask_returns_none(self):
        assert extract_spatial_mask({}, (8, 8), "cpu", torch.float32) is None
        assert extract_spatial_mask({"alpha_masks": None}, (8, 8), "cpu", torch.float32) is None


# ──────────────────────────────────────────────
# PatchTopologyLoss mask support (CUDA)
# ──────────────────────────────────────────────


@requires_cuda
class TestPatchTopologyMask:
    def _loss(self, **kw):
        defaults = dict(loss_weight=1.0, tau_latent=0.1, tau_target=0.1, scale_levels=1,
                        loss_type="kl", apply_timestep_weight=False, chunk_size=64)
        defaults.update(kw)
        return PatchTopologyLoss(**defaults).to(DEVICE)

    def test_mask_none_matches_previous_behavior(self):
        torch.manual_seed(0)
        mod = self._loss()
        pred = torch.randn(2, 4, 8, 8, device=DEVICE, requires_grad=True)
        target = torch.randn(2, 4, 8, 8, device=DEVICE)
        out_nomask = mod(pred, target)
        out_none = mod(pred, target, mask=None)
        assert torch.allclose(out_nomask, out_none)

    def test_full_mask_equals_no_mask(self):
        torch.manual_seed(0)
        mod = self._loss()
        pred = torch.randn(2, 4, 8, 8, device=DEVICE)
        target = torch.randn(2, 4, 8, 8, device=DEVICE)
        mask = torch.ones(2, 1, 8, 8, device=DEVICE)
        out_masked = mod(pred, target, mask=mask)
        out_plain = mod(pred, target)
        assert torch.allclose(out_masked, out_plain, atol=1e-5)

    def test_zero_mask_gives_zero_loss(self):
        torch.manual_seed(0)
        mod = self._loss()
        pred = torch.randn(2, 4, 8, 8, device=DEVICE)
        target = torch.randn(2, 4, 8, 8, device=DEVICE)
        mask = torch.zeros(2, 1, 8, 8, device=DEVICE)
        out = mod(pred, target, mask=mask)
        assert torch.all(out == 0.0)
        assert not torch.isnan(out).any()

    def test_partial_mask_changes_loss(self):
        torch.manual_seed(0)
        mod = self._loss()
        pred = torch.randn(2, 4, 8, 8, device=DEVICE)
        target = torch.randn(2, 4, 8, 8, device=DEVICE)
        mask = torch.ones(2, 1, 8, 8, device=DEVICE)
        mask[:, :, :, 4:] = 0.0  # mask out right half
        out_masked = mod(pred, target, mask=mask)
        out_plain = mod(pred, target)
        assert not torch.allclose(out_masked, out_plain)
        assert torch.all(out_masked >= 0.0)

    def test_mask_gradient_flows(self):
        torch.manual_seed(0)
        mod = self._loss()
        pred = torch.randn(2, 4, 8, 8, device=DEVICE, requires_grad=True)
        target = torch.randn(2, 4, 8, 8, device=DEVICE)
        mask = torch.ones(2, 1, 8, 8, device=DEVICE)
        mask[:, :, :4, :] = 0.0
        loss = mod(pred, target, mask=mask).mean()
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_mask_3d_input_accepted(self):
        """(B, H, W) masks should be accepted and broadcast."""
        torch.manual_seed(0)
        mod = self._loss()
        pred = torch.randn(2, 4, 8, 8, device=DEVICE)
        target = torch.randn(2, 4, 8, 8, device=DEVICE)
        mask = torch.ones(2, 8, 8, device=DEVICE)
        out = mod(pred, target, mask=mask)
        assert out.shape == (2,)

    def test_mask_multiscale(self):
        """Mask should be correctly pooled across scale octaves."""
        torch.manual_seed(0)
        mod = self._loss(scale_levels=2)
        pred = torch.randn(2, 4, 8, 8, device=DEVICE)
        target = torch.randn(2, 4, 8, 8, device=DEVICE)
        mask = torch.zeros(2, 1, 8, 8, device=DEVICE)
        mask[:, :, :4, :4] = 1.0  # top-left quadrant only
        out = mod(pred, target, mask=mask)
        assert torch.all(out > 0.0)
        assert not torch.isnan(out).any()

    def test_all_loss_types_with_mask(self):
        torch.manual_seed(0)
        for lt in ("kl", "ce", "cosine", "l2"):
            mod = self._loss(loss_type=lt)
            pred = torch.randn(2, 4, 8, 8, device=DEVICE)
            target = torch.randn(2, 4, 8, 8, device=DEVICE)
            mask = torch.rand(2, 1, 8, 8, device=DEVICE)
            out = mod(pred, target, mask=mask)
            assert out.shape == (2,)
            assert not torch.isnan(out).any(), f"NaN with loss_type={lt}"


# ──────────────────────────────────────────────
# CLI argument registration
# ──────────────────────────────────────────────


class TestDynamicWeightingArgs:
    def _parse(self, argv):
        from library.custom_train_functions import add_custom_train_arguments

        parser = argparse.ArgumentParser()
        add_custom_train_arguments(parser, support_weighted_captions=False)
        return parser.parse_args(argv)

    def test_defaults(self):
        args = self._parse([])
        assert args.patch_topology_dynamic_weighting == "none"
        assert args.patch_topology_dwa_temperature == 2.0
        assert args.patch_topology_gradnorm_alpha == 1.5
        assert args.patch_topology_dynamic_max_weight == 10.0

    def test_choices_accepted(self):
        for mode in ("none", "dwa", "gradnorm"):
            args = self._parse(["--patch_topology_dynamic_weighting", mode])
            assert args.patch_topology_dynamic_weighting == mode

    def test_invalid_choice_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--patch_topology_dynamic_weighting", "pcgrad"])

    def test_custom_values(self):
        args = self._parse([
            "--patch_topology_dynamic_weighting", "gradnorm",
            "--patch_topology_dwa_temperature", "3.0",
            "--patch_topology_gradnorm_alpha", "0.5",
            "--patch_topology_dynamic_max_weight", "4.0",
        ])
        assert args.patch_topology_dwa_temperature == 3.0
        assert args.patch_topology_gradnorm_alpha == 0.5
        assert args.patch_topology_dynamic_max_weight == 4.0


# ──────────────────────────────────────────────
# Trainer integration
# ──────────────────────────────────────────────


class TestTrainerIntegration:
    def test_trainer_has_weighter_attrs(self):
        import train_network

        trainer = train_network.NetworkTrainer()
        assert hasattr(trainer, "patch_topology_weighter")
        assert trainer.patch_topology_weighter is None
        assert hasattr(trainer, "patch_topology_effective_weight")
        assert trainer.patch_topology_effective_weight is None

    def test_generate_step_logs_weight(self):
        import train_network

        trainer = train_network.NetworkTrainer()
        args = argparse.Namespace(optimizer_type="AdamW", network_train_unet_only=False)
        logs = trainer.generate_step_logs(
            args,
            current_loss=0.1,
            avr_loss=0.1,
            lr_scheduler=type("FakeSched", (), {"get_last_lr": lambda self: [0.001]})(),
            lr_descriptions=["unet"],
            current_patch_topology_loss=0.5,
            current_patch_topology_weight=0.25,
        )
        assert logs["loss/current_patch_topology"] == 0.5
        assert logs["loss/patch_topology_effective_weight"] == 0.25

    def test_generate_step_logs_weight_none(self):
        import train_network

        trainer = train_network.NetworkTrainer()
        logs = trainer.generate_step_logs(
            argparse.Namespace(optimizer_type="AdamW", network_train_unet_only=False),
            current_loss=0.1,
            avr_loss=0.1,
            lr_scheduler=type("FakeSched", (), {"get_last_lr": lambda self: [0.001]})(),
            lr_descriptions=["unet"],
            current_patch_topology_weight=None,
        )
        assert "loss/patch_topology_effective_weight" not in logs


class TestFluxMetadata:
    def test_flux_update_metadata_includes_patch_topology(self):
        import flux_train_network

        trainer = flux_train_network.FluxNetworkTrainer.__new__(flux_train_network.FluxNetworkTrainer)
        args = argparse.Namespace(
            model_type="flux",
            apply_t5_attn_mask=False,
            weighting_scheme="none",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            guidance_scale=1.0,
            timestep_sampling="shift",
            sigmoid_scale=1.0,
            model_prediction_type="raw",
            discrete_flow_shift=3.0,
            patch_topology_loss=True,
            patch_topology_weight=0.5,
            patch_topology_dynamic_weighting="dwa",
        )
        metadata = {}
        trainer.update_metadata(metadata, args)
        assert metadata["ss_patch_topology_loss"] is True
        assert metadata["ss_patch_topology_weight"] == 0.5
        assert metadata["ss_patch_topology_dynamic_weighting"] == "dwa"
        assert metadata["ss_patch_topology_chunk_size"] == 512
        assert metadata["ss_patch_topology_dynamic_max_weight"] == 10.0
