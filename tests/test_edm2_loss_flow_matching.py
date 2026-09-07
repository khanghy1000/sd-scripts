"""Tests for EDM2 adaptive loss weighting with flow-matching schedulers.

Flow-matching trainers (anima/flux/sd3/...) use FlowMatchEulerDiscreteScheduler,
which has no `alphas_cumprod` or `all_snr` attributes. The AdaptiveLossWeightMLP
must derive the alphas_cumprod analog ((1 - sigma)^2) and SNR (((1-sigma)/sigma)^2)
from a sigma grid, and must accept timesteps either as sigmas in [0, 1] (anima
returns timesteps / 1000) or as discrete timesteps in [0, num_train_timesteps].
"""

import argparse

import pytest
import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler

from library import edm2_loss, edm2_loss_utils
from library.custom_train_functions import prepare_scheduler_for_custom_training
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

DEVICE = "cuda"
NUM_TRAIN_TIMESTEPS = 1000


@pytest.fixture
def flow_scheduler():
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=3.0)


@pytest.fixture
def ddpm_scheduler():
    scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    prepare_scheduler_for_custom_training(scheduler, torch.device(DEVICE))
    return scheduler


class TestFlowMatchingConstruction:
    def test_constructs_without_alphas_cumprod(self, flow_scheduler):
        assert not hasattr(flow_scheduler, "alphas_cumprod")
        model, optimizer = edm2_loss.create_weight_MLP(flow_scheduler, device=DEVICE)
        assert isinstance(model, edm2_loss.AdaptiveLossWeightMLP)
        assert model.is_flow_matching
        assert optimizer is not None

    def test_alphas_cumprod_analog_is_signal_power(self, flow_scheduler):
        model = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE)
        expected_sigmas = (
            torch.arange(NUM_TRAIN_TIMESTEPS, device=DEVICE, dtype=torch.float32) / NUM_TRAIN_TIMESTEPS
        )
        expected = (1.0 - expected_sigmas) ** 2
        torch.testing.assert_close(model.alphas_cumprod, expected)
        # Signal power must be monotonically decreasing in timestep (more noise later)
        assert (model.alphas_cumprod[:-1] >= model.alphas_cumprod[1:]).all()

    def test_importance_weights_finite_and_at_least_one(self, flow_scheduler):
        model = edm2_loss.AdaptiveLossWeightMLP(
            flow_scheduler,
            device=DEVICE,
            use_importance_weights=True,
            importance_weights_max_weight=10.0,
            importance_weights_min_snr_gamma=1.0,
        )
        assert torch.isfinite(model.importance_weights).all()
        assert (model.importance_weights >= 1.0).all()
        # The min-SNR heuristic peaks where SNR == gamma. For flow matching with
        # gamma=1.0, SNR = ((1-sigma)/sigma)^2 == 1 at sigma=0.5 -> t=500.
        peak = model.importance_weights[500]
        assert peak > 1.0
        assert peak > model.importance_weights[0]
        assert peak > model.importance_weights[-1]

    def test_use_importance_weights_flag_is_bool_not_tuple(self, flow_scheduler):
        """Regression test: `self.use_importance_weights = flag,` created a truthy tuple."""
        model_off = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE, use_importance_weights=False)
        assert model_off.use_importance_weights is False
        model_on = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE, use_importance_weights=True)
        assert model_on.use_importance_weights is True


class TestFlowMatchingForward:
    def test_forward_with_unit_range_sigmas(self, flow_scheduler):
        """Anima-style timesteps: sigmas in [0, 1]."""
        model = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE)
        loss = torch.rand(8, device=DEVICE) + 0.5
        timesteps = torch.rand(8, device=DEVICE)  # sigmas in [0, 1]
        weighted, scaled = model(loss, timesteps)
        assert weighted.shape == loss.shape
        assert scaled.shape == loss.shape
        assert torch.isfinite(weighted).all()
        assert torch.isfinite(scaled).all()

    def test_forward_with_discrete_range_timesteps(self, flow_scheduler):
        """Flux-style timesteps: float timesteps in [0, 1000]."""
        model = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE)
        loss = torch.rand(8, device=DEVICE) + 0.5
        timesteps = torch.rand(8, device=DEVICE) * NUM_TRAIN_TIMESTEPS
        weighted, scaled = model(loss, timesteps)
        assert weighted.shape == loss.shape
        assert torch.isfinite(weighted).all()

    def test_sigma_and_discrete_timesteps_agree(self, flow_scheduler):
        """A sigma s and its discrete form s * num_train_timesteps must map to the same index."""
        model = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE)
        loss = torch.ones(4, device=DEVICE)
        sigmas = torch.tensor([0.1, 0.25, 0.5, 0.9], device=DEVICE)
        w_sigma, s_sigma = model(loss, sigmas)
        w_disc, s_disc = model(loss, sigmas * NUM_TRAIN_TIMESTEPS)
        torch.testing.assert_close(w_sigma, w_disc)
        torch.testing.assert_close(s_sigma, s_disc)

    def test_sigma_timesteps_are_not_all_floored_to_zero(self, flow_scheduler):
        """Regression: `.long()` on [0,1] sigmas collapses everything to index 0."""
        model = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE)
        sigmas = torch.tensor([0.001, 0.5, 0.999], device=DEVICE)
        indices = model._normalize_timesteps(sigmas)
        assert indices.tolist() == [1, 500, 999]

    def test_timestep_indices_clamped(self, flow_scheduler):
        model = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE)
        indices = model._normalize_timesteps(torch.tensor([1.0, 1.5, 2.0], device=DEVICE) * NUM_TRAIN_TIMESTEPS)
        assert (indices <= NUM_TRAIN_TIMESTEPS - 1).all()

    def test_gradient_flows_to_mlp(self, flow_scheduler):
        model, optimizer = edm2_loss.create_weight_MLP(flow_scheduler, device=DEVICE)
        loss = torch.rand(16, device=DEVICE, requires_grad=True) + 0.5
        timesteps = torch.rand(16, device=DEVICE)
        weighted, _ = model(loss, timesteps)
        weighted.mean().backward()
        assert model.logvar_linear.weight.grad is not None
        assert torch.isfinite(model.logvar_linear.weight.grad).all()
        assert model.logvar_linear.weight.grad.abs().sum() > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def test_mlp_output_varies_across_timesteps(self, flow_scheduler):
        model = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE)
        timesteps = torch.arange(NUM_TRAIN_TIMESTEPS, device=DEVICE, dtype=torch.float32)
        out = model._forward(model._normalize_timesteps(timesteps))
        assert out.std() > 0


class TestDDPMRegression:
    def test_ddpm_path_still_works(self, ddpm_scheduler):
        model, optimizer = edm2_loss.create_weight_MLP(ddpm_scheduler, device=DEVICE)
        assert not model.is_flow_matching
        loss = torch.rand(8, device=DEVICE) + 0.5
        timesteps = torch.randint(0, NUM_TRAIN_TIMESTEPS, (8,), device=DEVICE)
        weighted, scaled = model(loss, timesteps)
        assert weighted.shape == loss.shape
        assert torch.isfinite(weighted).all()

    def test_ddpm_without_all_snr_computes_snr_internally(self):
        scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
        assert not hasattr(scheduler, "all_snr")
        model = edm2_loss.AdaptiveLossWeightMLP(scheduler, device=DEVICE, use_importance_weights=True)
        assert torch.isfinite(model.importance_weights).all()
        assert (model.importance_weights >= 1.0).all()

    def test_ddpm_integer_timesteps_unchanged(self, ddpm_scheduler):
        model = edm2_loss.AdaptiveLossWeightMLP(ddpm_scheduler, device=DEVICE)
        timesteps = torch.tensor([0, 1, 500, 999], device=DEVICE)
        indices = model._normalize_timesteps(timesteps)
        assert indices.tolist() == [0, 1, 500, 999]


class TestSigmaGrid:
    def test_grid_matches_training_interpolation(self, flow_scheduler):
        sigmas = edm2_loss.build_flow_matching_sigma_grid(flow_scheduler, DEVICE, torch.float32)
        assert sigmas.shape == (NUM_TRAIN_TIMESTEPS,)
        assert sigmas[0].item() == 0.0
        assert sigmas[-1].item() == pytest.approx((NUM_TRAIN_TIMESTEPS - 1) / NUM_TRAIN_TIMESTEPS)
        # Grid must be unshifted even when the scheduler has shift != 1, because
        # training timesteps are defined as sigma * num_train_timesteps.
        assert sigmas[500].item() == pytest.approx(0.5)

    def test_is_flow_matching_detection(self, flow_scheduler, ddpm_scheduler):
        assert edm2_loss.is_flow_matching_scheduler(flow_scheduler)
        assert not edm2_loss.is_flow_matching_scheduler(ddpm_scheduler)


class TestSaveLoad:
    def test_save_and_load_weights_roundtrip(self, flow_scheduler, tmp_path):
        model, _ = edm2_loss.create_weight_MLP(flow_scheduler, device=DEVICE)
        ckpt = tmp_path / "edm2_weights.safetensors"
        model.save_weights(str(ckpt), torch.float32, None)
        assert ckpt.exists()

        model2, _ = edm2_loss.create_weight_MLP(flow_scheduler, device=DEVICE)
        info = model2.load_weights(str(ckpt))
        assert not info.missing_keys
        assert not info.unexpected_keys

        loss = torch.rand(4, device=DEVICE) + 0.5
        timesteps = torch.rand(4, device=DEVICE)
        model.eval()
        model2.eval()
        with torch.no_grad():
            w1, s1 = model(loss, timesteps)
            w2, s2 = model2(loss, timesteps)
        torch.testing.assert_close(w1, w2)
        torch.testing.assert_close(s1, s2)


def _make_args(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(
        edm2_loss_weighting=True,
        edm2_loss_weighting_optimizer="torch.optim.AdamW",
        edm2_loss_weighting_optimizer_args="{'weight_decay': 0, 'betas': (0.9, 0.99)}",
        edm2_loss_weighting_optimizer_lr=2e-2,
        edm2_loss_weighting_num_channels=128,
        edm2_loss_weighting_initial_weights=None,
        edm2_loss_weighting_lr_scheduler=False,
        edm2_loss_weighting_lr_scheduler_warmup_percent=None,
        edm2_loss_weighting_lr_scheduler_constant_percent=None,
        edm2_loss_weighting_lr_scheduler_decay_scaling=None,
        edm2_loss_weighting_importance_weighting=True,
        edm2_loss_weighting_importance_weighting_max=10.0,
        edm2_loss_weighting_importance_min_snr_gamma=1.0,
        edm2_loss_weighting_importance_weighting_safety_override=False,
        edm2_loss_weighting_generate_graph=True,
        edm2_loss_weighting_generate_graph_every_x_steps=1,
        edm2_loss_weighting_generate_graph_output_dir=str(tmp_path),
        edm2_loss_weighting_generate_graph_y_limit=None,
        max_train_steps=10,
        output_name="edm2_flow_test",
        min_snr_gamma=5.0,
        debiased_estimation_loss=True,
        deepspeed=False,
    )


class TestPreparePipeline:
    def test_prepare_edm2_loss_weighting_with_flow_scheduler(self, flow_scheduler, tmp_path):
        """Functional test replicating the anima_train_network traceback path."""
        args = _make_args(tmp_path)
        accelerator = Accelerator()

        edm2_model, edm2_optimizer, edm2_lr_scheduler = edm2_loss_utils.prepare_edm2_loss_weighting(
            args, flow_scheduler, accelerator
        )
        assert edm2_model is not None
        assert edm2_optimizer is not None
        assert edm2_lr_scheduler is not None

        # Conflicting options must have been disabled by the safety handler
        assert args.min_snr_gamma is None
        assert args.debiased_estimation_loss is False

        # Simulate an anima training step: loss + sigmas in [0, 1]
        loss = torch.rand(4, device=accelerator.device) + 0.5
        timesteps = torch.rand(4, device=accelerator.device)
        weighted, scaled = edm2_model(loss, timesteps)
        weighted.mean().backward()
        edm2_optimizer.step()
        edm2_lr_scheduler.step()
        edm2_optimizer.zero_grad(set_to_none=True)
        assert torch.isfinite(weighted).all()

    def test_plot_edm2_loss_weighting_with_flow_model(self, flow_scheduler, tmp_path):
        args = _make_args(tmp_path)
        model, _ = edm2_loss.create_weight_MLP(flow_scheduler, device=DEVICE)
        edm2_loss_utils.plot_edm2_loss_weighting(
            args, 1, model, num_timesteps=NUM_TRAIN_TIMESTEPS, device=DEVICE
        )
        expected = tmp_path / "edm2_flow_test" / "weighting_step_0000001.png"
        assert expected.exists()


class TestTorchCompileCompatibility:
    """Regression tests for dynamo graph breaks seen when accelerate compiles the
    EDM2 model (dynamo backend enabled):
    1. Data-dependent branching (`if ts.max() <= 1.5:`) in _normalize_timesteps.
    2. accelerate's ConvertOutputsToFp32 failing to trace `tuple(generator)` on
       tuple outputs, which is why forward returns a single stacked tensor.
    """

    def test_forward_returns_single_stacked_tensor(self, flow_scheduler):
        model = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE)
        loss = torch.rand(8, device=DEVICE) + 0.5
        timesteps = torch.rand(8, device=DEVICE)
        out = model(loss, timesteps)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 8)
        # Tuple-style unpacking must keep working for existing call sites
        weighted, scaled = model(loss, timesteps)
        torch.testing.assert_close(weighted, out[0])
        torch.testing.assert_close(scaled, out[1])

    def test_compiled_forward_matches_eager(self, flow_scheduler):
        model = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE)
        compiled = torch.compile(model)
        loss = torch.rand(16, device=DEVICE) + 0.5
        for timesteps in (
            torch.rand(16, device=DEVICE),  # sigmas in [0, 1]
            torch.rand(16, device=DEVICE) * NUM_TRAIN_TIMESTEPS,  # discrete [0, 1000]
        ):
            eager_out = model(loss, timesteps)
            compiled_out = compiled(loss, timesteps)
            torch.testing.assert_close(compiled_out, eager_out)

    def test_no_graph_breaks_in_forward(self, flow_scheduler):
        model = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE)
        loss = torch.rand(8, device=DEVICE) + 0.5
        timesteps = torch.rand(8, device=DEVICE)
        explanation = torch._dynamo.explain(model)(loss, timesteps)
        assert explanation.graph_break_count == 0, (
            f"Unexpected graph breaks in EDM2 forward: {explanation}"
        )

    def test_no_graph_breaks_through_accelerate_fp32_wrapper(self, flow_scheduler):
        """Simulate accelerate's mixed-precision wrapper: convert_to_fp32(model(...)).

        This is the exact path that broke dynamo with `tuple(generator)` when
        forward returned a tuple of tensors.
        """
        from accelerate.utils.operations import convert_to_fp32

        model = edm2_loss.AdaptiveLossWeightMLP(flow_scheduler, device=DEVICE)

        def accelerate_wrapped_forward(loss, timesteps):
            return convert_to_fp32(model(loss, timesteps))

        loss = torch.rand(8, device=DEVICE) + 0.5
        timesteps = torch.rand(8, device=DEVICE)
        explanation = torch._dynamo.explain(accelerate_wrapped_forward)(loss, timesteps)
        assert explanation.graph_break_count == 0, (
            f"Graph breaks through accelerate fp32 wrapper: {explanation}"
        )
        # And the compiled result must match eager execution
        compiled = torch.compile(accelerate_wrapped_forward)
        torch.testing.assert_close(compiled(loss, timesteps), accelerate_wrapped_forward(loss, timesteps))
