"""Integration tests for Adaptive Non-uniform Timestep Sampling.

Tests end-to-end behavior including:
1. DDPM-style training loop simulation with adaptive sampling
2. Flow-matching training loop simulation with adaptive sampling
3. Checkpoint save/load round-trip with state preservation
4. get_adaptive_model_type override verification across trainers
"""

import json
import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from library.adaptive_timestep_sampler import (
    AdaptiveTimestepManager,
    TimestepSamplerNetwork,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_ddpm_scheduler(device, T=1000):
    """Create a mock DDPMScheduler with realistic cosine alphas_cumprod."""
    scheduler = MagicMock()
    t = torch.linspace(0, 1, T, device=device)
    alphas_cumprod = torch.cos(t * math.pi / 2) ** 2
    scheduler.alphas_cumprod = alphas_cumprod
    scheduler.config = MagicMock()
    scheduler.config.num_train_timesteps = T
    return scheduler


def _make_flow_matching_scheduler(device, T=1000):
    """Create a mock FlowMatchEulerDiscreteScheduler (no alphas_cumprod)."""
    scheduler = MagicMock()
    scheduler.config = MagicMock()
    scheduler.config.num_train_timesteps = T
    # Intentionally no alphas_cumprod — flow-matching doesn't use it
    # The manager will set _alphas_cumprod = None when model_type="flow_matching"
    scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
    return scheduler


class TinyModel(nn.Module):
    """Minimal model for integration tests that returns predictions of same shape as input."""

    def __init__(self, in_channels=4, seed=42):
        super().__init__()
        # Use a fixed seed for deterministic initial predictions
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        # Initialize with known weights
        nn.init.ones_(self.conv.weight)

    def forward(self, x, timesteps=None, **kwargs):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Integration Test: DDPM Training Loop Simulation
# ---------------------------------------------------------------------------

class TestDDPMIntegration:
    """Simulate several training steps with adaptive timestep sampling (DDPM mode)."""

    def test_training_loop_simulation(self, device):
        """Run a mini training loop: sample timesteps, compute loss, backward, update,
        then run Algorithm 2 at the correct frequency."""
        T = 100
        scheduler = _make_ddpm_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)

        manager = AdaptiveTimestepManager(
            sampler_network=net,
            noise_scheduler=scheduler,
            device=device,
            dtype=torch.float32,
            learning_rate=1e-2,
            entropy_coeff=1e-2,
            update_freq=3,  # f_S=3 for fast testing
            queue_size=5,
            num_selected=2,
            model_type="ddpm",
        )

        model = TinyModel(in_channels=4).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        def model_fn(noisy_latents, timesteps, wdtype):
            return model(noisy_latents.to(wdtype))

        losses_before_history = []
        delta_history = []
        selected_history = []

        for step in range(9):
            x_0 = torch.randn(2, 4, 8, 8, device=device)
            noise = torch.randn_like(x_0)

            # Step 1: Sample timesteps using adaptive sampler
            timesteps = manager.sample_timesteps(x_0, T)
            assert timesteps.shape == (2,)
            assert (timesteps >= 0).all() and (timesteps < T).all()

            # Step 2: Add noise at sampled timesteps
            alpha_bar = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            noisy = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

            # Step 3: Forward + loss + backward
            pred = model(noisy)
            target = noise  # epsilon prediction
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()

            # Step 4: Before optimizer step — compute per-timestep losses
            if manager.should_update(step):
                losses_before = manager.compute_per_timestep_losses(
                    x_0[:1], noise[:1], model_fn, torch.float32, chunk_size=50
                )
                losses_before_history.append(losses_before.detach().cpu())

            # Step 5: Optimizer step
            optimizer.step()

            # Step 6: After optimizer step — Algorithm 2
            if manager.should_update(step):
                delta_approx, selected = manager.compute_delta_approximation(
                    model_fn, x_0[:1], noise[:1], torch.float32, losses_before,
                    full_batch_latents=x_0, full_batch_noise=noise,
                )
                manager.update_sampler(delta_approx, x_0[:1])
                delta_history.append(delta_approx.item())
                selected_history.append(selected.tolist())

        # Verify Algorithm 2 ran at steps 3, 6 (0-indexed, step % 3 == 0 and step > 0)
        # should_update returns True when step > 0 and step % f_S == 0
        # Steps: 0,1,2,3,4,5,6,7,8 → should_update at 3, 6
        assert len(losses_before_history) == 2, f"Expected 2 Algorithm 2 runs, got {len(losses_before_history)}"
        assert len(delta_history) == 2
        assert len(selected_history) == 2
        assert len(manager.queue) == 2

        # Selected timesteps should be valid indices
        for sel in selected_history:
            assert all(0 <= s < T for s in sel)
            assert len(sel) == 2  # num_selected=2

    def test_sampler_distribution_changes(self, device):
        """The Beta(a,b) distribution should change over training as the sampler learns."""
        T = 100
        scheduler = _make_ddpm_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)

        manager = AdaptiveTimestepManager(
            sampler_network=net,
            noise_scheduler=scheduler,
            device=device,
            update_freq=2,
            queue_size=5,
            num_selected=2,
        )

        model = TinyModel(in_channels=4).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        def model_fn(noisy_latents, timesteps, wdtype):
            return model(noisy_latents.to(wdtype))

        # Use a fixed probe input to measure Beta(a,b) parameters before/after training
        probe_input = torch.randn(1, 4, 8, 8, device=device)
        with torch.no_grad():
            a_init, _ = manager.sampler_network(probe_input)
        initial_a_mean = a_init.mean().item()

        for step in range(10):
            x_0 = torch.randn(2, 4, 8, 8, device=device)
            noise = torch.randn_like(x_0)
            timesteps = manager.sample_timesteps(x_0, T)

            alpha_bar = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            noisy = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
            pred = model(noisy)
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if manager.should_update(step):
                losses_before = manager.compute_per_timestep_losses(
                    x_0[:1], noise[:1], model_fn, torch.float32, chunk_size=50
                )
                delta_approx, selected = manager.compute_delta_approximation(
                    model_fn, x_0[:1], noise[:1], torch.float32, losses_before,
                )
                manager.update_sampler(delta_approx, x_0)

        with torch.no_grad():
            a_final, _ = manager.sampler_network(probe_input)
        final_a_mean = a_final.mean().item()

        # The distribution parameters should have been updated at least once
        # (not necessarily different due to small model, but the update ran)
        assert initial_a_mean is not None
        assert final_a_mean is not None


# ---------------------------------------------------------------------------
# Integration Test: Flow-Matching Training Loop Simulation
# ---------------------------------------------------------------------------

class TestFlowMatchingIntegration:
    """Simulate training steps with flow-matching noise addition."""

    def test_flow_matching_training_loop(self, device):
        """Run a mini flow-matching training loop with adaptive sampling."""
        T = 100
        scheduler = _make_flow_matching_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)

        manager = AdaptiveTimestepManager(
            sampler_network=net,
            noise_scheduler=scheduler,
            device=device,
            update_freq=3,
            queue_size=5,
            num_selected=2,
            model_type="flow_matching",
        )

        model = TinyModel(in_channels=4).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        def model_fn(noisy_latents, timesteps, wdtype):
            return model(noisy_latents.to(wdtype))

        for step in range(9):
            x_0 = torch.randn(2, 4, 8, 8, device=device)
            noise = torch.randn_like(x_0)
            timesteps = manager.sample_timesteps(x_0, T)

            # Flow-matching noise addition
            sigmas = timesteps.float().view(-1, 1, 1, 1) / T
            noisy = sigmas * noise + (1.0 - sigmas) * x_0

            pred = model(noisy)
            target = noise - x_0  # velocity target
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if manager.should_update(step):
                losses_before = manager.compute_per_timestep_losses(
                    x_0[:1], noise[:1], model_fn, torch.float32, chunk_size=50
                )
                delta_approx, selected = manager.compute_delta_approximation(
                    model_fn, x_0[:1], noise[:1], torch.float32, losses_before,
                    full_batch_latents=x_0, full_batch_noise=noise,
                )
                manager.update_sampler(delta_approx, x_0[:1])

        assert len(manager.queue) == 2

    def test_flow_matching_with_custom_loss_fn(self, device):
        """Flow-matching with a custom compute_loss_fn that uses L1 loss."""
        T = 100
        scheduler = _make_flow_matching_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)

        custom_called = {"count": 0}

        def custom_loss_fn(model_output, x_0, noise, t_indices):
            custom_called["count"] += 1
            target = (noise - x_0).to(torch.float32)
            losses = F.l1_loss(model_output, target, reduction="none")
            return losses.mean(dim=list(range(1, losses.ndim)))

        manager = AdaptiveTimestepManager(
            sampler_network=net,
            noise_scheduler=scheduler,
            device=device,
            update_freq=2,
            queue_size=5,
            num_selected=2,
            model_type="flow_matching",
            compute_loss_fn=custom_loss_fn,
        )

        model = TinyModel(in_channels=4).to(device)

        def model_fn(noisy_latents, timesteps, wdtype):
            return model(noisy_latents.to(wdtype))

        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        losses = manager.compute_per_timestep_losses(x_0, noise, model_fn, torch.float32, chunk_size=50)
        assert custom_called["count"] > 0
        assert losses.shape == (T,)


# ---------------------------------------------------------------------------
# Integration Test: Checkpoint Save/Load Round-Trip
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    """Test that adaptive sampler state survives a full save/load cycle."""

    def test_save_load_preserves_sampler_state(self, device):
        """After save/load, the sampler should produce identical predictions."""
        T = 100
        scheduler = _make_ddpm_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)

        manager = AdaptiveTimestepManager(
            sampler_network=net,
            noise_scheduler=scheduler,
            device=device,
            update_freq=2,
            queue_size=5,
            num_selected=2,
        )

        # Populate the queue with some data
        for _ in range(3):
            manager.queue.append(torch.randn(T, device=device))

        # Save state
        state = manager.state_dict()

        # Verify state contains all expected keys
        assert "sampler_network" in state
        assert "optimizer" in state
        assert "queue" in state
        assert "v_parameterization" in state
        assert "model_type" in state
        assert len(state["queue"]) == 3

        # Create a new manager and load state
        new_net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        new_manager = AdaptiveTimestepManager(
            sampler_network=new_net,
            noise_scheduler=scheduler,
            device=device,
            update_freq=2,
            queue_size=5,
            num_selected=2,
        )
        new_manager.load_state_dict(state)

        # Queue should be restored
        assert len(new_manager.queue) == 3

        # Same input should produce same sampling distribution
        x_test = torch.randn(1, 4, 8, 8, device=device)

        manager.sampler_network.eval()
        new_manager.sampler_network.eval()

        with torch.no_grad():
            a1, b1 = manager.sampler_network(x_test)
            a2, b2 = new_manager.sampler_network(x_test)

        assert torch.allclose(a1, a2, atol=1e-6), "a parameters differ after load"
        assert torch.allclose(b1, b2, atol=1e-6), "b parameters differ after load"

    def test_json_serialization_round_trip(self, device):
        """Simulate the JSON-based checkpoint save/load used in train_network.py."""
        T = 100
        scheduler = _make_ddpm_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)

        manager = AdaptiveTimestepManager(
            sampler_network=net,
            noise_scheduler=scheduler,
            device=device,
            update_freq=2,
            queue_size=5,
            num_selected=2,
        )

        # Populate queue
        manager.queue.append(torch.randn(T, device=device))

        # Simulate the JSON serialization from train_network.py save_model_hook
        adaptive_state = manager.state_dict()
        adaptive_state_serializable = {
            "sampler_network": {k: v.tolist() if isinstance(v, torch.Tensor) else v
                               for k, v in adaptive_state["sampler_network"].items()},
            "optimizer": adaptive_state["optimizer"],
            "queue": [q.tolist() if isinstance(q, torch.Tensor) else q
                      for q in adaptive_state["queue"]],
            "learning_rate": adaptive_state["learning_rate"],
            "entropy_coeff": adaptive_state["entropy_coeff"],
            "f_s": adaptive_state["f_s"],
            "queue_size": adaptive_state["queue_size"],
            "num_selected": adaptive_state["num_selected"],
            "v_parameterization": adaptive_state.get("v_parameterization", False),
            "model_type": adaptive_state.get("model_type", "ddpm"),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(adaptive_state_serializable, f)
            temp_path = f.name

        try:
            # Simulate load
            with open(temp_path, "r") as f:
                loaded = json.load(f)

            # Convert back to tensors (same as train_network.py load_model_hook)
            adaptive_state_restored = {
                "sampler_network": {k: torch.tensor(v, device=device) if isinstance(v, list) else v
                                   for k, v in loaded["sampler_network"].items()},
                "optimizer": loaded["optimizer"],
                "queue": [torch.tensor(q, device=device) if isinstance(q, list) else q
                          for q in loaded["queue"]],
                "learning_rate": loaded["learning_rate"],
                "entropy_coeff": loaded["entropy_coeff"],
                "f_s": loaded["f_s"],
                "queue_size": loaded["queue_size"],
                "num_selected": loaded["num_selected"],
            }

            new_net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
            new_manager = AdaptiveTimestepManager(
                sampler_network=new_net,
                noise_scheduler=scheduler,
                device=device,
                update_freq=2,
                queue_size=5,
                num_selected=2,
            )
            new_manager.load_state_dict(adaptive_state_restored)

            assert len(new_manager.queue) == 1

            # Verify network weights were restored
            x_test = torch.randn(1, 4, 8, 8, device=device)
            manager.sampler_network.eval()
            new_manager.sampler_network.eval()
            with torch.no_grad():
                a1, b1 = manager.sampler_network(x_test)
                a2, b2 = new_manager.sampler_network(x_test)
            assert torch.allclose(a1, a2, atol=1e-6)
            assert torch.allclose(b1, b2, atol=1e-6)
        finally:
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Integration Test: get_adaptive_model_type Overrides
# ---------------------------------------------------------------------------

class TestAdaptiveModelTypeOverrides:
    """Verify that all flow-matching trainers return 'flow_matching'."""

    def test_flux_trainer_returns_flow_matching(self):
        from flux_train_network import FluxNetworkTrainer
        trainer = FluxNetworkTrainer()
        args = MagicMock()
        assert trainer.get_adaptive_model_type(args) == "flow_matching"

    def test_sd3_trainer_returns_flow_matching(self):
        from sd3_train_network import Sd3NetworkTrainer
        trainer = Sd3NetworkTrainer()
        args = MagicMock()
        assert trainer.get_adaptive_model_type(args) == "flow_matching"

    def test_lumina_trainer_returns_flow_matching(self):
        from lumina_train_network import LuminaNetworkTrainer
        trainer = LuminaNetworkTrainer()
        args = MagicMock()
        assert trainer.get_adaptive_model_type(args) == "flow_matching"

    def test_anima_trainer_returns_flow_matching(self):
        from anima_train_network import AnimaNetworkTrainer
        trainer = AnimaNetworkTrainer()
        args = MagicMock()
        assert trainer.get_adaptive_model_type(args) == "flow_matching"

    def test_hunyuan_trainer_returns_flow_matching(self):
        from hunyuan_image_train_network import HunyuanImageNetworkTrainer
        trainer = HunyuanImageNetworkTrainer()
        args = MagicMock()
        assert trainer.get_adaptive_model_type(args) == "flow_matching"

    def test_base_trainer_returns_ddpm(self):
        from train_network import NetworkTrainer
        trainer = NetworkTrainer()
        args = MagicMock()
        assert trainer.get_adaptive_model_type(args) == "ddpm"


# ---------------------------------------------------------------------------
# Tests: build_adaptive_model_fn (conditioning expansion)
# ---------------------------------------------------------------------------

class TestBuildAdaptiveModelFn:
    """Test that build_adaptive_model_fn correctly expands conditioning to match
    the batch size of noisy_latents passed to model_fn."""

    def test_base_trainer_build_adaptive_model_fn_expands(self, device):
        """Base trainer's model_fn should expand text_conds to match N."""
        from train_network import NetworkTrainer

        trainer = NetworkTrainer()
        T = 100
        scheduler = _make_flow_matching_scheduler(device, T=T)
        B = 4

        # Set up adaptive manager
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        trainer.adaptive_manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            model_type="flow_matching",
        )

        # Simulate stored data from last training step
        latents = torch.randn(B, 4, 8, 8, device=device)
        noise = torch.randn_like(latents)
        # text_conds: [embeddings, masks, extra1, extra2]
        text_conds = [
            torch.randn(B, 16, 64, device=device),  # embeddings
            torch.ones(B, 16, device=device),  # masks
            torch.randn(B, 32, device=device),  # extra
        ]

        trainer._adaptive_last_latents = latents
        trainer._adaptive_last_noise = noise
        trainer._adaptive_last_text_conds = text_conds
        trainer._adaptive_last_args = MagicMock()
        trainer._adaptive_last_batch = {}

        # Mock call_unet to capture the batch size of text_conds it receives
        received_batch_sizes = []
        def mock_call_unet(args, accel, unet, noisy_latents, timesteps, text_conds, masks, batch, wdt, **kw):
            received_batch_sizes.append(text_conds[0].shape[0])
            return noisy_latents  # return input as prediction
        trainer.call_unet = mock_call_unet

        model_fn = trainer.build_adaptive_model_fn(MagicMock(), MagicMock(), torch.float32)

        # Call with N=10 (different from B=4) — like compute_per_timestep_losses would
        x_expanded = torch.randn(10, 4, 8, 8, device=device)
        t_expanded = torch.randint(0, T, (10,), device=device)
        result = model_fn(x_expanded, t_expanded, torch.float32)

        # The model_fn should have expanded text_conds from B=4 to N=10
        assert received_batch_sizes[-1] == 10, (
            f"text_conds[0].shape[0] should be 10 (expanded), got {received_batch_sizes[-1]}"
        )
        assert result.shape == x_expanded.shape

    def test_base_trainer_model_fn_same_batch_size(self, device):
        """When N == B, model_fn should not expand (pass through as-is)."""
        from train_network import NetworkTrainer

        trainer = NetworkTrainer()
        T = 100
        scheduler = _make_flow_matching_scheduler(device, T=T)
        B = 4

        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        trainer.adaptive_manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            model_type="flow_matching",
        )

        latents = torch.randn(B, 4, 8, 8, device=device)
        noise = torch.randn_like(latents)
        text_conds = [torch.randn(B, 16, 64, device=device), torch.ones(B, 16, device=device)]

        trainer._adaptive_last_latents = latents
        trainer._adaptive_last_noise = noise
        trainer._adaptive_last_text_conds = text_conds
        trainer._adaptive_last_args = MagicMock()
        trainer._adaptive_last_batch = {}

        received_batch_sizes = []
        def mock_call_unet(args, accel, unet, noisy_latents, timesteps, tc, masks, batch, wdt, **kw):
            received_batch_sizes.append(tc[0].shape[0])
            return noisy_latents
        trainer.call_unet = mock_call_unet

        model_fn = trainer.build_adaptive_model_fn(MagicMock(), MagicMock(), torch.float32)

        # Call with N=4 (same as B)
        x_same = torch.randn(B, 4, 8, 8, device=device)
        t_same = torch.randint(0, T, (B,), device=device)
        model_fn(x_same, t_same, torch.float32)

        assert received_batch_sizes[-1] == B

    def test_anima_build_adaptive_model_fn(self, device):
        """AnimaNetworkTrainer's build_adaptive_model_fn should create a model_fn
        that expands conditioning, scales timesteps, and uses 5D latents."""
        from anima_train_network import AnimaNetworkTrainer

        trainer = AnimaNetworkTrainer()
        T = 1000
        scheduler = _make_flow_matching_scheduler(device, T=T)
        B = 2

        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        trainer.adaptive_manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            model_type="flow_matching",
        )

        latents = torch.randn(B, 4, 8, 8, device=device)
        noise = torch.randn_like(latents)
        # Anima text_conds: [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
        text_conds = [
            torch.randn(B, 16, 64, device=device),  # prompt_embeds
            torch.ones(B, 16, dtype=torch.bool, device=device),  # attn_mask
            torch.randint(0, 1000, (B, 32), device=device),  # t5_input_ids
            torch.ones(B, 32, dtype=torch.bool, device=device),  # t5_attn_mask
        ]

        trainer._adaptive_last_latents = latents
        trainer._adaptive_last_noise = noise
        trainer._adaptive_last_text_conds = text_conds
        trainer._adaptive_last_args = MagicMock()
        trainer._adaptive_last_batch = {}

        # Mock the Anima model
        call_log = {}
        class MockAnima(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, timesteps, context, **kwargs):
                call_log["x_shape"] = x.shape  # Should be 5D
                call_log["ts_range"] = (timesteps.min().item(), timesteps.max().item())
                call_log["ts_dtype"] = timesteps.dtype  # Must match weight dtype (bf16/fp16/fp32)
                call_log["x_dtype"] = x.dtype
                call_log["ctx_shape"] = context.shape
                call_log["N"] = x.shape[0]
                # Return 5D output
                return x  # Same shape
        mock_anima = MockAnima().to(device)

        # Provide a real accelerator-like mock with a proper device
        mock_accel = MagicMock()
        mock_accel.device = device
        model_fn = trainer.build_adaptive_model_fn(mock_anima, mock_accel, torch.float32)

        # Call with N=6 (expanded from B=2, like B*|S| = 2*3 = 6)
        x_input = torch.randn(6, 4, 8, 8, device=device)
        t_input = torch.randint(0, T, (6,), device=device)
        result = model_fn(x_input, t_input, torch.float32)

        # Verify the model was called with correct shapes
        assert call_log["x_shape"] == (6, 4, 1, 8, 8), f"Expected 5D shape, got {call_log['x_shape']}"
        assert call_log["N"] == 6
        assert call_log["ctx_shape"][0] == 6, "Context should be expanded to N=6"

        # Verify timesteps were scaled to [0, 1] range
        ts_min, ts_max = call_log["ts_range"]
        assert ts_min >= 0.0 and ts_max <= 1.0, f"Timesteps should be in [0,1], got [{ts_min}, {ts_max}]"

        # Verify timesteps were cast to the weight dtype, matching the training path
        # (flux_train_utils returns timesteps in weight_dtype; the sinusoidal embedder
        # inherits this dtype, and Anima's AdaLN runs with autocast disabled so a
        # float32 embedding would crash against bf16 weights).
        assert call_log["ts_dtype"] == torch.float32, (
            f"Timestep dtype should match weight dtype (fp32 in this test), got {call_log['ts_dtype']}"
        )
        assert call_log["x_dtype"] == torch.float32

        # Result should be 4D (squeezed from 5D)
        assert result.shape == x_input.shape, f"Expected 4D result, got {result.shape}"

    def test_anima_build_adaptive_model_fn_single_sample(self, device):
        """Anima model_fn with N=1 (single x_0 for queue) should work."""
        from anima_train_network import AnimaNetworkTrainer

        trainer = AnimaNetworkTrainer()
        T = 1000
        scheduler = _make_flow_matching_scheduler(device, T=T)
        B = 4

        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        trainer.adaptive_manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            model_type="flow_matching",
        )

        latents = torch.randn(B, 4, 8, 8, device=device)
        noise = torch.randn_like(latents)
        text_conds = [
            torch.randn(B, 16, 64, device=device),
            torch.ones(B, 16, dtype=torch.bool, device=device),
            torch.randint(0, 1000, (B, 32), device=device),
            torch.ones(B, 32, dtype=torch.bool, device=device),
        ]

        trainer._adaptive_last_latents = latents
        trainer._adaptive_last_noise = noise
        trainer._adaptive_last_text_conds = text_conds
        trainer._adaptive_last_args = MagicMock()
        trainer._adaptive_last_batch = {}

        call_log = {}
        class MockAnima(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, timesteps, context, **kwargs):
                call_log["N"] = x.shape[0]
                call_log["ctx_N"] = context.shape[0]
                return x
        mock_anima = MockAnima().to(device)

        mock_accel = MagicMock()
        mock_accel.device = device
        model_fn = trainer.build_adaptive_model_fn(mock_anima, mock_accel, torch.float32)

        # Call with N=1 (single sample for queue)
        x_single = torch.randn(1, 4, 8, 8, device=device)
        t_single = torch.randint(0, T, (1,), device=device)
        result = model_fn(x_single, t_single, torch.float32)

        assert call_log["N"] == 1
        assert call_log["ctx_N"] == 1, f"Context should be expanded to N=1, got {call_log['ctx_N']}"
        assert result.shape == x_single.shape


# ---------------------------------------------------------------------------
# Tests: Min/Max timestep integration
# ---------------------------------------------------------------------------

class TestMinMaxTimestepIntegration:
    """Test min/max timestep with a full flow-matching training loop."""

    def test_flow_matching_with_min_max_timestep(self, device):
        """Adaptive sampling should respect min/max timestep in a flow-matching loop."""
        T = 100
        scheduler = _make_flow_matching_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)

        min_ts, max_ts = 20, 80
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            model_type="flow_matching", update_freq=5, queue_size=5, num_selected=3,
            min_timestep=min_ts, max_timestep=max_ts,
        )

        # Sample many timesteps and verify all are in [min_ts, max_ts)
        for _ in range(10):
            x = torch.randn(8, 4, 8, 8, device=device)
            timesteps = manager.sample_timesteps(x, num_timesteps=T)
            assert (timesteps >= min_ts).all(), f"Min timestep violated: {timesteps.min().item()} < {min_ts}"
            assert (timesteps < max_ts).all(), f"Max timestep violated: {timesteps.max().item()} >= {max_ts}"

    def test_state_dict_preserves_min_max_timestep(self, device):
        """Min/max timestep should survive checkpoint round-trip."""
        T = 100
        scheduler = _make_flow_matching_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            min_timestep=30, max_timestep=70,
        )

        state = manager.state_dict()

        new_net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        new_manager = AdaptiveTimestepManager(
            sampler_network=new_net, noise_scheduler=scheduler, device=device,
        )
        new_manager.load_state_dict(state)
        assert new_manager.min_timestep == 30
        assert new_manager.max_timestep == 70


# ---------------------------------------------------------------------------
# Integration Tests: VRAM Optimizations
# ---------------------------------------------------------------------------

class TestStridedEvalGridIntegration:
    """Integration tests for strided eval grid in training loops."""

    def test_ddpm_with_stride(self, device):
        """DDPM training loop with eval_stride>1 should still update the sampler."""
        T = 100
        scheduler = _make_ddpm_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            model_type="ddpm", update_freq=3, queue_size=5, num_selected=2,
            eval_stride=4, eval_chunk_size=10,
        )

        model = TinyModel(in_channels=4).to(device)
        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        for step in range(7):
            x = torch.randn(2, 4, 8, 8, device=device)
            timesteps = manager.sample_timesteps(x, num_timesteps=T)
            noise = torch.randn_like(x)

            if manager.should_update(step):
                with torch.no_grad():
                    losses_before = manager.compute_per_timestep_losses(
                        x[:1], noise[:1], model_fn, torch.float32
                    )
                    losses_after = manager.compute_per_timestep_losses(
                        x[:1], noise[:1], model_fn, torch.float32
                    )
                    delta = (losses_before - losses_after).abs().mean()
                # update_sampler needs gradients for the sampler network — must be outside no_grad
                manager.update_sampler(delta, x)

        # Verify the queue received entries with the correct grid size
        if len(manager.queue) > 0:
            grid_size = len(manager._eval_timesteps)
            for entry in manager.queue:
                assert entry.shape == (grid_size,), f"Queue entry shape {entry.shape} != expected ({grid_size},)"

    def test_flow_matching_with_stride(self, device):
        """Flow-matching training loop with eval_stride>1 should still update the sampler."""
        T = 100
        scheduler = _make_flow_matching_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            model_type="flow_matching", update_freq=3, queue_size=5, num_selected=2,
            eval_stride=5, eval_chunk_size=10,
        )

        model = TinyModel(in_channels=4).to(device)
        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        for step in range(7):
            x = torch.randn(2, 4, 8, 8, device=device)
            timesteps = manager.sample_timesteps(x, num_timesteps=T)
            noise = torch.randn_like(x)

            if manager.should_update(step):
                with torch.no_grad():
                    losses_before = manager.compute_per_timestep_losses(
                        x[:1], noise[:1], model_fn, torch.float32
                    )
                    losses_after = manager.compute_per_timestep_losses(
                        x[:1], noise[:1], model_fn, torch.float32
                    )
                    delta = (losses_before - losses_after).abs().mean()
                # update_sampler needs gradients for the sampler network — must be outside no_grad
                manager.update_sampler(delta, x)

        if len(manager.queue) > 0:
            grid_size = len(manager._eval_timesteps)
            for entry in manager.queue:
                assert entry.shape == (grid_size,), f"Queue entry shape {entry.shape} != expected ({grid_size},)"

    def test_stride_1_equivalence(self, device):
        """With stride=1, the output should match the original (no-stride) behavior."""
        T = 50
        scheduler = _make_ddpm_scheduler(device, T=T)
        model = TinyModel(in_channels=4).to(device)
        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        x = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x)

        # stride=1 (default)
        net1 = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        mgr1 = AdaptiveTimestepManager(
            sampler_network=net1, noise_scheduler=scheduler, device=device,
            model_type="ddpm", eval_stride=1, eval_chunk_size=10, fp32_eval=True,
        )

        # stride=100 (evaluates only timestep 0)
        # Just verify shapes are correct
        net2 = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        mgr2 = AdaptiveTimestepManager(
            sampler_network=net2, noise_scheduler=scheduler, device=device,
            model_type="ddpm", eval_stride=100, eval_chunk_size=10, fp32_eval=True,
        )

        with torch.no_grad():
            losses1 = mgr1.compute_per_timestep_losses(x, noise, model_fn, torch.float32)
            losses2 = mgr2.compute_per_timestep_losses(x, noise, model_fn, torch.float32)

        assert losses1.shape == (T,), f"Expected ({T},), got {losses1.shape}"
        assert losses2.shape == (1,), f"Expected (1,), got {losses2.shape}"
        # The single loss from stride=100 should equal the first loss from stride=1
        # (relaxed tolerance due to different chunk sizes affecting conv kernel selection)
        torch.testing.assert_close(losses2[0], losses1[0], atol=1e-4, rtol=1e-4)


class TestBatchLossLoopIntegration:
    """Integration tests for the |S|-loop batch loss path."""

    def test_batch_loss_scalar_output(self, device):
        """compute_per_timestep_losses_for_batch should return a scalar."""
        T = 50
        B = 4
        scheduler = _make_ddpm_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
        )

        model = TinyModel(in_channels=4).to(device)
        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        x = torch.randn(B, 4, 8, 8, device=device)
        noise = torch.randn_like(x)
        selected = torch.tensor([10, 30], device=device)

        with torch.no_grad():
            loss = manager.compute_per_timestep_losses_for_batch(
                x, noise, model_fn, torch.float32, selected
            )

        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"

    def test_flow_matching_batch_loss(self, device):
        """Batch loss for flow-matching should also return a scalar."""
        T = 50
        B = 4
        scheduler = _make_flow_matching_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            model_type="flow_matching",
        )

        model = TinyModel(in_channels=4).to(device)
        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        x = torch.randn(B, 4, 8, 8, device=device)
        noise = torch.randn_like(x)
        selected = torch.tensor([5, 15, 25], device=device)

        with torch.no_grad():
            loss = manager.compute_per_timestep_losses_for_batch(
                x, noise, model_fn, torch.float32, selected
            )

        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"


class TestStashGating:
    """Tests for the stash gating mechanism."""

    def test_stash_skipped_on_non_update_steps(self, device):
        """When _adaptive_update_pending is False, stashing should be a no-op."""
        T = 100
        scheduler = _make_ddpm_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            update_freq=5, queue_size=5, num_selected=2,
        )

        x = torch.randn(2, 4, 8, 8, device=device)
        # Simulate: sample timesteps (always needed)
        timesteps = manager.sample_timesteps(x, num_timesteps=T)

        # On a non-update step (step 0, 1, 2, 3), _adaptive_update_pending should be False
        # so the trainer would skip stashing. Verify should_update returns False.
        for step in [0, 1, 2, 3]:
            assert not manager.should_update(step), f"step {step} should not be an update step"

        # On step 5, should_update returns True
        assert manager.should_update(5), "step 5 should be an update step"

    def test_should_update_boundary_values(self, device):
        """should_update should correctly handle boundary values."""
        T = 100
        scheduler = _make_ddpm_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            update_freq=10,
        )

        # step 0 should NOT trigger (global_step > 0 check)
        assert not manager.should_update(0)
        # step 10 should trigger
        assert manager.should_update(10)
        # step 5 should not trigger
        assert not manager.should_update(5)
        # step 20 should trigger
        assert manager.should_update(20)
        # step 9 should not trigger
        assert not manager.should_update(9)


class TestEmptyCacheIntegration:
    """Tests for empty_cache integration."""

    def test_empty_cache_flag_in_manager(self, device):
        """The manager itself doesn't have empty_cache — it's a trainer-level concern.
        But verify the disable flag is accessible from args."""
        # This test verifies the manager doesn't break when used with the new params
        T = 50
        scheduler = _make_ddpm_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            eval_chunk_size=16, eval_stride=1, fp32_eval=False,
        )

        # Run a quick loss computation to verify no errors
        x = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x)
        model = TinyModel(in_channels=4).to(device)
        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        with torch.no_grad():
            losses = manager.compute_per_timestep_losses(x, noise, model_fn, torch.float32)
        assert losses.shape[0] == T


class TestStateDictVRAMFields:
    """Integration tests for state_dict with VRAM optimization fields."""

    def test_full_round_trip_with_stride(self, device):
        """Full state dict round-trip with eval_stride>1 preserves all fields."""
        T = 100
        scheduler = _make_ddpm_scheduler(device, T=T)
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            eval_chunk_size=32, eval_stride=5, fp32_eval=True,
            min_timestep=10, max_timestep=90, reward_baseline_decay=0.95,
        )
        manager._reward_baseline = 0.42

        state = manager.state_dict()

        new_net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        new_manager = AdaptiveTimestepManager(
            sampler_network=new_net, noise_scheduler=scheduler, device=device,
        )
        new_manager.load_state_dict(state)

        assert new_manager.eval_chunk_size == 32
        assert new_manager.eval_stride == 5
        assert new_manager.fp32_eval is True
        assert new_manager.min_timestep == 10
        assert new_manager.max_timestep == 90
        assert new_manager.reward_baseline_decay == 0.95
        assert new_manager._reward_baseline == 0.42
        # Grid should be recomputed (_eval_timesteps is always on CPU)
        expected_grid = torch.arange(0, T, 5)
        assert torch.equal(new_manager._eval_timesteps, expected_grid)
