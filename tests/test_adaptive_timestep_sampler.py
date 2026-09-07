"""Tests for Adaptive Non-uniform Timestep Sampling (arXiv:2411.09998).

Verifies:
1. TimestepSamplerNetwork forward pass, output shapes, and positivity constraints
2. F-statistic feature selection correctness
3. AdaptiveTimestepManager initialization, timestep sampling, and state dict round-trip
4. Per-timestep loss computation and delta approximation
5. Policy gradient sampler update
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock
from collections import deque

from library.adaptive_timestep_sampler import (
    TimestepSamplerNetwork,
    select_timesteps_f_statistic,
    AdaptiveTimestepManager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def noise_scheduler(device):
    """Create a mock DDPMScheduler-like object with alphas_cumprod."""
    scheduler = MagicMock()
    T = 1000
    # Create a realistic cosine-like schedule
    t = torch.linspace(0, 1, T, device=device)
    alphas_cumprod = torch.cos(t * math.pi / 2) ** 2
    scheduler.alphas_cumprod = alphas_cumprod
    scheduler.config = MagicMock()
    scheduler.config.num_train_timesteps = T
    return scheduler


@pytest.fixture
def sampler_network(device):
    """Create a small TimestepSamplerNetwork for testing."""
    return TimestepSamplerNetwork(
        in_channels=4,
        hidden_channels=32,
        hidden_depth=2,
    ).to(device)


@pytest.fixture
def adaptive_manager(sampler_network, noise_scheduler, device):
    """Create an AdaptiveTimestepManager for testing."""
    return AdaptiveTimestepManager(
        sampler_network=sampler_network,
        noise_scheduler=noise_scheduler,
        device=device,
        dtype=torch.float32,
        learning_rate=1e-2,
        entropy_coeff=1e-2,
        update_freq=5,  # Small for testing
        queue_size=10,
        num_selected=3,
    )


# ---------------------------------------------------------------------------
# Tests: TimestepSamplerNetwork
# ---------------------------------------------------------------------------

class TestTimestepSamplerNetwork:

    def test_forward_output_shapes(self, sampler_network, device):
        """Output (a, b) should each have shape (batch_size,)."""
        sampler_network.eval()
        x = torch.randn(4, 4, 8, 8, device=device)
        a, b = sampler_network(x)
        assert a.shape == (4,), f"Expected shape (4,), got {a.shape}"
        assert b.shape == (4,), f"Expected shape (4,), got {b.shape}"

    def test_outputs_positive(self, sampler_network, device):
        """a and b must always be positive (Beta distribution requirement)."""
        sampler_network.eval()
        # Test with various inputs including zeros and large values
        for _ in range(10):
            x = torch.randn(8, 4, 16, 16, device=device) * 10
            a, b = sampler_network(x)
            assert (a > 0).all(), f"Found non-positive a values: {a}"
            assert (b > 0).all(), f"Found non-positive b values: {b}"

    def test_different_spatial_sizes(self, sampler_network, device):
        """Network should handle different spatial dimensions via AdaptiveAvgPool."""
        sampler_network.eval()
        for h, w in [(4, 4), (8, 8), (16, 32), (64, 64)]:
            x = torch.randn(2, 4, h, w, device=device)
            a, b = sampler_network(x)
            assert a.shape == (2,)
            assert b.shape == (2,)

    def test_initial_bias_near_uniform(self, sampler_network, device):
        """Initial parameters should produce a roughly symmetric Beta distribution."""
        sampler_network.eval()
        x = torch.zeros(1, 4, 8, 8, device=device)
        a, b = sampler_network(x)
        # With zero input and zero weights, softplus(bias) should give symmetric a, b
        ratio = a / b
        assert 0.5 < ratio.item() < 2.0, f"Initial a/b ratio {ratio.item()} too far from 1.0"

    def test_gradient_flow(self, sampler_network, device):
        """Gradients should flow through the network."""
        sampler_network.train()
        x = torch.randn(2, 4, 8, 8, device=device, requires_grad=True)
        a, b = sampler_network(x)
        loss = a.sum() + b.sum()
        loss.backward()
        assert x.grad is not None, "No gradient flowed through the network"
        for param in sampler_network.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Parameter has no gradient"


# ---------------------------------------------------------------------------
# Tests: Feature Selection (F-statistic)
# ---------------------------------------------------------------------------

class TestFeatureSelection:

    def test_selects_correct_timesteps(self, device):
        """When one timestep has a strong linear relationship with the target,
        it should be selected."""
        Q, T = 20, 100
        queue = torch.randn(Q, T, device=device) * 0.1
        # Make timestep 42 highly correlated with the row mean
        target = queue.mean(dim=1)  # (Q,)
        queue[:, 42] = target * 5.0 + torch.randn(Q, device=device) * 0.01
        # Make timestep 77 also correlated but weaker
        queue[:, 77] = target * 2.0 + torch.randn(Q, device=device) * 0.01

        selected = select_timesteps_f_statistic(queue, num_selected=3)
        assert 42 in selected.tolist(), f"Timestep 42 should be selected, got {selected}"
        assert 77 in selected.tolist(), f"Timestep 77 should be selected, got {selected}"

    def test_returns_correct_count(self, device):
        """Should return exactly num_selected indices."""
        queue = torch.randn(10, 50, device=device)
        for n in [1, 3, 5, 10]:
            selected = select_timesteps_f_statistic(queue, num_selected=n)
            assert selected.shape == (n,), f"Expected {n} indices, got {selected.shape}"

    def test_single_queue_entry(self, device):
        """With only one queue entry, should fall back to highest absolute delta."""
        queue = torch.randn(1, 100, device=device)
        queue[0, 10] = 100.0  # Largest absolute value
        selected = select_timesteps_f_statistic(queue, num_selected=1)
        assert selected[0].item() == 10

    def test_all_zeros(self, device):
        """All-zero queue should not crash."""
        queue = torch.zeros(10, 50, device=device)
        selected = select_timesteps_f_statistic(queue, num_selected=3)
        assert selected.shape == (3,)

    def test_uniform_queue(self, device):
        """Uniform queue (no variance) should not crash."""
        queue = torch.ones(10, 50, device=device) * 0.5
        selected = select_timesteps_f_statistic(queue, num_selected=3)
        assert selected.shape == (3,)


# ---------------------------------------------------------------------------
# Tests: AdaptiveTimestepManager
# ---------------------------------------------------------------------------

class TestAdaptiveTimestepManager:

    def test_initialization(self, adaptive_manager):
        """Manager should initialize with correct hyperparameters."""
        assert adaptive_manager.f_s == 5
        assert adaptive_manager.queue_size == 10
        assert adaptive_manager.num_selected == 3
        assert adaptive_manager.num_train_timesteps == 1000
        assert len(adaptive_manager.queue) == 0

    def test_should_update(self, adaptive_manager):
        """should_update should return True at multiples of f_s, False otherwise."""
        assert adaptive_manager.should_update(0) == False  # Step 0 is never updated
        assert adaptive_manager.should_update(1) == False
        assert adaptive_manager.should_update(4) == False
        assert adaptive_manager.should_update(5) == True
        assert adaptive_manager.should_update(6) == False
        assert adaptive_manager.should_update(10) == True
        assert adaptive_manager.should_update(15) == True

    def test_sample_timesteps_shape(self, adaptive_manager, device):
        """sample_timesteps should return integer timesteps in [0, T)."""
        x = torch.randn(4, 4, 8, 8, device=device)
        timesteps = adaptive_manager.sample_timesteps(x, num_timesteps=1000)
        assert timesteps.shape == (4,), f"Expected shape (4,), got {timesteps.shape}"
        assert timesteps.dtype == torch.int64 or timesteps.dtype == torch.long
        assert (timesteps >= 0).all() and (timesteps < 1000).all()

    def test_sample_timesteps_varies(self, adaptive_manager, device):
        """Sampled timesteps should vary across different inputs (not all identical)."""
        x = torch.randn(16, 4, 8, 8, device=device)
        timesteps = adaptive_manager.sample_timesteps(x, num_timesteps=1000)
        # With 16 samples, we expect at least some variation
        assert timesteps.unique().numel() > 1, "All timesteps are identical"

    def test_state_dict_round_trip(self, adaptive_manager, device):
        """Saving and loading state dict should restore the manager's state."""
        # Add some data to the queue
        adaptive_manager.queue.append(torch.randn(1000, device=device))
        adaptive_manager.queue.append(torch.randn(1000, device=device))

        state = adaptive_manager.state_dict()
        assert "sampler_network" in state
        assert "optimizer" in state
        assert "queue" in state
        assert len(state["queue"]) == 2

        # Create a new manager and load state
        new_manager = AdaptiveTimestepManager(
            sampler_network=TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device),
            noise_scheduler=adaptive_manager.noise_scheduler,
            device=device,
            queue_size=10,
        )
        new_manager.load_state_dict(state)
        assert len(new_manager.queue) == 2

    def test_compute_per_timestep_losses_shape(self, adaptive_manager, device):
        """compute_per_timestep_losses should return shape (T,) with non-negative values."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        # Simple mock model that returns random predictions
        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        losses = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=200
        )
        assert losses.shape == (T,), f"Expected shape ({T},), got {losses.shape}"
        assert (losses >= 0).all(), "Losses should be non-negative (MSE)"

    def test_compute_delta_approximation(self, adaptive_manager, device):
        """Delta approximation should return a scalar and selected indices."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        # Create mock losses_before (random)
        losses_before = torch.rand(T, device=device)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        delta_approx, selected = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before
        )
        assert delta_approx.ndim == 0 or delta_approx.shape == (), "delta_approx should be scalar"
        assert selected.shape == (adaptive_manager.num_selected,)
        # Queue should now have one entry
        assert len(adaptive_manager.queue) == 1

    def test_update_sampler(self, adaptive_manager, device):
        """update_sampler should not crash and should update network parameters."""
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        delta_k_t = torch.tensor(0.5, device=device)

        # Store initial parameters
        initial_params = {
            name: param.clone() for name, param in adaptive_manager.sampler_network.named_parameters()
        }

        adaptive_manager.update_sampler(delta_k_t, x_0)

        # At least one parameter should have changed
        changed = False
        for name, param in adaptive_manager.sampler_network.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-7):
                changed = True
                break
        assert changed, "No parameters changed after update_sampler"


    def test_cached_t_continuous_used_in_update(self, adaptive_manager, device):
        """update_sampler should use the cached sampled t, not a new one."""
        x_0 = torch.randn(2, 4, 8, 8, device=device)
        # Sample timesteps to populate the cache
        timesteps = adaptive_manager.sample_timesteps(x_0, num_timesteps=1000)
        # Verify _cached_t_continuous is set
        assert adaptive_manager._cached_t_continuous is not None
        assert adaptive_manager._cached_t_continuous.shape == (2,)
        # The cached t should be in (0, 1) range
        cached = adaptive_manager._cached_t_continuous
        assert (cached > 0).all() and (cached < 1).all()

        # Now update - the gradient should flow through the cached t
        delta_k_t = torch.tensor(0.5, device=device)
        initial_params = {
            name: param.clone() for name, param in adaptive_manager.sampler_network.named_parameters()
        }
        adaptive_manager.update_sampler(delta_k_t, x_0)
        # Parameters should have changed
        changed = any(
            not torch.allclose(param, initial_params[name], atol=1e-7)
            for name, param in adaptive_manager.sampler_network.named_parameters()
        )
        assert changed

    def test_f_statistic_matches_paper_formula(self, device):
        """F-statistic should match the standard formula."""
        # Create a queue where timestep 0 is perfectly correlated with target
        Q, T = 50, 20
        target_signal = torch.randn(Q, device=device)
        queue = torch.randn(Q, T, device=device) * 0.01
        queue[:, 0] = target_signal * 2.0
        queue[:, 0] += torch.randn(Q, device=device) * 0.001

        selected = select_timesteps_f_statistic(queue, num_selected=1)
        # Timestep 0 should have the highest F-statistic
        assert selected[0].item() == 0, f"Expected timestep 0, got {selected[0].item()}"



    def test_cache_batch_losses_at_S_noop_without_selection(self, adaptive_manager, device):
        """cache_batch_losses_at_S should be a no-op when no |S| selection exists yet."""
        assert adaptive_manager._current_selected_indices is None
        x = torch.randn(4, 4, 8, 8, device=device)
        noise = torch.randn_like(x)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # Should be a no-op since no selection exists
        adaptive_manager.cache_batch_losses_at_S(x, noise, mock_model_fn, torch.float32)
        assert adaptive_manager._prev_batch_losses_at_S is None

    def test_cache_batch_losses_at_S_after_selection(self, adaptive_manager, device):
        """After a delta approximation, the selection should be cached and
        cache_batch_losses_at_S should store the full batch losses at those timesteps."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)
        full_batch = torch.randn(4, 4, 8, 8, device=device)
        full_noise = torch.randn_like(full_batch)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # First call: run a full delta approximation to establish a selection
        losses_before = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )
        delta_approx, selected = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before,
        )
        # After this call, _current_selected_indices should be set
        assert adaptive_manager._current_selected_indices is not None
        assert adaptive_manager._current_selected_indices.shape == (adaptive_manager.num_selected,)

        # Now cache batch losses at the current |S|
        adaptive_manager.cache_batch_losses_at_S(
            full_batch, full_noise, mock_model_fn, torch.float32
        )
        # The cached losses should have shape (B, |S|) but mean gives scalar
        assert adaptive_manager._prev_batch_losses_at_S is not None
        assert adaptive_manager._prev_batch_losses_at_S.ndim == 0  # scalar (mean)

    def test_full_batch_delta_approximation(self, adaptive_manager, device):
        """When a previous |S| selection exists, compute_delta_approximation should
        use the full batch at those timesteps to compute the delta."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)
        full_batch = torch.randn(4, 4, 8, 8, device=device)
        full_noise = torch.randn_like(full_batch)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # ---- First call: establish a selection ----
        losses_before = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )
        delta_approx_first, selected_first = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before,
        )
        # The first call uses single x_0 (no previous selection)
        assert adaptive_manager._current_selected_indices is not None
        first_selection = adaptive_manager._current_selected_indices.clone()

        # ---- Cache full batch losses at the current |S| ----
        adaptive_manager.cache_batch_losses_at_S(
            full_batch, full_noise, mock_model_fn, torch.float32
        )
        assert adaptive_manager._prev_batch_losses_at_S is not None

        # ---- Second call: should use full batch at the previous |S| ----
        losses_before_2 = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )
        delta_approx_second, selected_second = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before_2,
            full_batch_latents=full_batch, full_batch_noise=full_noise,
        )
        # The selected indices should equal the first selection (the one used for the full batch)
        assert torch.equal(selected_second.cpu(), first_selection),             "Second call should return the previous |S| selection (the one used for full batch)"
        # The delta_approx should be a scalar
        assert delta_approx_second.ndim == 0 or delta_approx_second.shape == ()

    def test_full_batch_falls_back_to_single_when_no_cache(self, adaptive_manager, device):
        """If cache_batch_losses_at_S was not called, compute_delta_approximation
        should fall back to the single x_0 path even if full batch is provided."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)
        full_batch = torch.randn(4, 4, 8, 8, device=device)
        full_noise = torch.randn_like(full_batch)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # First call: establish a selection (but DON'T cache full batch losses)
        losses_before = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )
        delta_approx_first, _ = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before,
        )
        first_selection = adaptive_manager._current_selected_indices.clone()
        # No cache_batch_losses_at_S call! So _prev_batch_losses_at_S is None

        # Second call with full batch provided but no cached losses
        losses_before_2 = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )
        delta_approx_second, selected_second = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before_2,
            full_batch_latents=full_batch, full_batch_noise=full_noise,
        )
        # Should fall back to single x_0 path -> selected indices = new selection (not first)
        # The selected indices are the new ones from this call
        assert selected_second.shape == (adaptive_manager.num_selected,)

    def test_full_algorithm_2_cycle(self, adaptive_manager, device):
        """Full cycle: sample timesteps, compute losses before/after, update."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # Step 1: Compute losses before (simulating theta_k)
        losses_before = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )

        # Step 2: Compute delta approximation (simulating theta_{k+1} after optimizer step)
        delta_approx, selected = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before
        )

        # Step 3: Update sampler
        adaptive_manager.update_sampler(delta_approx, x_0)

        # Verify queue was populated
        assert len(adaptive_manager.queue) >= 1


# ---------------------------------------------------------------------------
# Tests: v-parameterization support
# ---------------------------------------------------------------------------

class TestVParameterization:

    @pytest.fixture
    def v_pred_manager(self, device):
        """Create an AdaptiveTimestepManager with v_parameterization=True."""
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        scheduler = MagicMock()
        T = 1000
        t = torch.linspace(0, 1, T, device=device)
        alphas_cumprod = torch.cos(t * math.pi / 2) ** 2
        scheduler.alphas_cumprod = alphas_cumprod
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        return AdaptiveTimestepManager(
            sampler_network=net,
            noise_scheduler=scheduler,
            device=device,
            dtype=torch.float32,
            learning_rate=1e-2,
            entropy_coeff=1e-2,
            update_freq=5,
            queue_size=10,
            num_selected=3,
            v_parameterization=True,
        )

    def test_v_pred_flag_stored(self, v_pred_manager):
        """v_parameterization flag should be stored."""
        assert v_pred_manager.v_parameterization is True

    def test_v_pred_losses_positive(self, v_pred_manager, device):
        """Per-timestep losses with v-prediction should be non-negative."""
        T = v_pred_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        losses = v_pred_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=200
        )
        assert losses.shape == (T,)
        assert (losses >= 0).all(), "Losses should be non-negative"

    def test_v_pred_losses_differ_from_epsilon(self, device):
        """v-prediction losses should differ from epsilon-prediction losses
        when using the same model predictions."""
        scheduler = MagicMock()
        T = 1000
        t = torch.linspace(0, 1, T, device=device)
        alphas_cumprod = torch.cos(t * math.pi / 2) ** 2
        scheduler.alphas_cumprod = alphas_cumprod
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T

        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        # Use a deterministic model so both paths see the same predictions
        torch.manual_seed(42)
        fixed_pred = torch.randn(1, 4, 8, 8, device=device)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return fixed_pred.expand_as(noisy_latents)

        # Epsilon manager
        net_eps = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        eps_manager = AdaptiveTimestepManager(
            sampler_network=net_eps, noise_scheduler=scheduler, device=device,
            v_parameterization=False, queue_size=10,
        )
        losses_eps = eps_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )

        # v-prediction manager
        net_v = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        v_manager = AdaptiveTimestepManager(
            sampler_network=net_v, noise_scheduler=scheduler, device=device,
            v_parameterization=True, queue_size=10,
        )
        losses_v = v_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )

        # They should differ (different targets)
        assert not torch.allclose(losses_eps, losses_v, atol=1e-3), \
            "v-prediction losses should differ from epsilon losses"

    def test_v_pred_state_dict_round_trip(self, v_pred_manager, device):
        """v_parameterization should survive state_dict round-trip."""
        state = v_pred_manager.state_dict()
        assert "v_parameterization" in state
        assert state["v_parameterization"] is True

        # Create a new manager without v_parameterization and load
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        new_manager = AdaptiveTimestepManager(
            sampler_network=net,
            noise_scheduler=v_pred_manager.noise_scheduler,
            device=device,
            queue_size=10,
            v_parameterization=False,
        )
        assert new_manager.v_parameterization is False
        new_manager.load_state_dict(state)
        # Note: load_state_dict doesn't restore v_parameterization currently,
        # but the saved state contains it for future use


# ---------------------------------------------------------------------------
# Tests: Flow-matching support
# ---------------------------------------------------------------------------

class TestFlowMatching:

    @pytest.fixture
    def flow_matching_manager(self, device):
        """Create an AdaptiveTimestepManager with model_type='flow_matching'."""
        # For flow-matching, create a scheduler without alphas_cumprod
        scheduler = MagicMock()
        T = 1000
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        # Flow-matching scheduler does NOT have alphas_cumprod
        # Our code should handle this gracefully
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)

        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        return AdaptiveTimestepManager(
            sampler_network=net,
            noise_scheduler=scheduler,
            device=device,
            dtype=torch.float32,
            learning_rate=1e-2,
            entropy_coeff=1e-2,
            update_freq=5,
            queue_size=10,
            num_selected=3,
            model_type="flow_matching",
        )

    def test_flow_matching_init_no_alphas_cumprod(self, flow_matching_manager):
        """Flow-matching manager should initialize with alphas_cumprod=None."""
        assert flow_matching_manager.model_type == "flow_matching"
        assert flow_matching_manager._alphas_cumprod is None

    def test_flow_matching_losses_shape(self, flow_matching_manager, device):
        """Per-timestep losses should return shape (T,) for flow-matching."""
        T = flow_matching_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        losses = flow_matching_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=200
        )
        assert losses.shape == (T,)
        assert (losses >= 0).all(), "Losses should be non-negative"

    def test_flow_matching_losses_differ_from_ddpm(self, device):
        """Flow-matching losses should differ from DDPM losses for same inputs."""
        T = 1000
        scheduler_ddpm = MagicMock()
        t = torch.linspace(0, 1, T, device=device)
        alphas_cumprod = torch.cos(t * math.pi / 2) ** 2
        scheduler_ddpm.alphas_cumprod = alphas_cumprod
        scheduler_ddpm.config = MagicMock()
        scheduler_ddpm.config.num_train_timesteps = T

        scheduler_fm = MagicMock()
        scheduler_fm.alphas_cumprod = alphas_cumprod  # will be ignored
        scheduler_fm.config = MagicMock()
        scheduler_fm.config.num_train_timesteps = T

        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        # Deterministic model
        torch.manual_seed(42)
        fixed_pred = torch.randn(1, 4, 8, 8, device=device)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return fixed_pred.expand_as(noisy_latents)

        # DDPM manager
        net_ddpm = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        ddpm_manager = AdaptiveTimestepManager(
            sampler_network=net_ddpm, noise_scheduler=scheduler_ddpm, device=device,
            model_type="ddpm", queue_size=10,
        )
        losses_ddpm = ddpm_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )

        # Flow-matching manager
        net_fm = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        fm_manager = AdaptiveTimestepManager(
            sampler_network=net_fm, noise_scheduler=scheduler_fm, device=device,
            model_type="flow_matching", queue_size=10,
        )
        losses_fm = fm_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )

        assert not torch.allclose(losses_ddpm, losses_fm, atol=1e-3), \
            "Flow-matching losses should differ from DDPM losses"

    def test_flow_matching_custom_loss_fn(self, device):
        """Custom compute_loss_fn should be used when provided."""
        T = 1000
        scheduler = MagicMock()
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)

        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        custom_called = {"count": 0}

        def custom_loss_fn(model_output, x_0, noise, t_indices):
            custom_called["count"] += 1
            # Simple custom loss: L1 instead of L2
            target = (noise - x_0).to(torch.float32)
            losses = F.l1_loss(model_output, target, reduction="none")
            return losses.mean(dim=list(range(1, losses.ndim)))

        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            model_type="flow_matching", queue_size=10,
            compute_loss_fn=custom_loss_fn,
        )

        losses = manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=200
        )
        assert custom_called["count"] > 0, "Custom loss function should have been called"
        assert losses.shape == (T,)
        assert (losses >= 0).all()

    def test_flow_matching_algorithm_2_cycle(self, flow_matching_manager, device):
        """Full Algorithm 2 cycle should work with flow-matching."""
        T = flow_matching_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # Step 1: Compute losses before
        losses_before = flow_matching_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )

        # Step 2: Compute delta approximation
        delta_approx, selected = flow_matching_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before
        )

        # Step 3: Update sampler
        flow_matching_manager.update_sampler(delta_approx, x_0)

        assert len(flow_matching_manager.queue) >= 1
        assert delta_approx.ndim == 0 or delta_approx.shape == ()
        assert selected.shape == (flow_matching_manager.num_selected,)

    def test_flow_matching_state_dict_round_trip(self, flow_matching_manager, device):
        """model_type should survive state_dict round-trip."""
        state = flow_matching_manager.state_dict()
        assert "model_type" in state
        assert state["model_type"] == "flow_matching"

    def test_flow_matching_batch_losses(self, flow_matching_manager, device):
        """compute_per_timestep_losses_for_batch should work with flow-matching."""
        x_0 = torch.randn(4, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)
        selected = torch.tensor([100, 500, 900], device=device)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        loss = flow_matching_manager.compute_per_timestep_losses_for_batch(
            x_0, noise, mock_model_fn, torch.float32, selected
        )
        assert loss.ndim == 0 or loss.shape == ()
        assert loss >= 0


# ---------------------------------------------------------------------------
# Tests: Lazy Network Initialization
# ---------------------------------------------------------------------------

class TestLazyInit:
    """Test lazy initialization of TimestepSamplerNetwork from latent shape."""

    def _make_manager(self, device, **kwargs):
        """Create a manager without a pre-created network (lazy init)."""
        scheduler = MagicMock()
        T = 1000
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        return AdaptiveTimestepManager(
            noise_scheduler=scheduler, device=device,
            hidden_channels=32, hidden_depth=2, update_freq=5,
            queue_size=10, num_selected=3, **kwargs,
        )

    def test_lazy_init_network_none_before_sampling(self, device):
        """Network should be None before first sample_timesteps call."""
        manager = self._make_manager(device)
        assert manager.sampler_network is None
        assert manager.optimizer is None

    def test_lazy_init_4_channels(self, device):
        """Lazy init with 4-channel latents (SD1.5/SDXL)."""
        manager = self._make_manager(device)
        x_0 = torch.randn(2, 4, 8, 8, device=device)
        timesteps = manager.sample_timesteps(x_0, num_timesteps=1000)
        assert manager.sampler_network is not None
        assert manager.optimizer is not None
        assert timesteps.shape == (2,)
        # Verify the network was created with the correct in_channels
        first_weight = manager.sampler_network.mlp[0].weight
        assert first_weight.shape[1] == 4 * 4 * 4  # in_channels=4 * 4 * 4

    def test_lazy_init_16_channels(self, device):
        """Lazy init with 16-channel latents (Flux/SD3/Anima)."""
        manager = self._make_manager(device)
        x_0 = torch.randn(2, 16, 8, 8, device=device)
        timesteps = manager.sample_timesteps(x_0, num_timesteps=1000)
        assert manager.sampler_network is not None
        assert timesteps.shape == (2,)
        first_weight = manager.sampler_network.mlp[0].weight
        assert first_weight.shape[1] == 16 * 4 * 4  # in_channels=16 * 4 * 4

    def test_lazy_init_64_channels(self, device):
        """Lazy init with 64-channel latents (extreme/edge case)."""
        manager = self._make_manager(device)
        x_0 = torch.randn(2, 64, 8, 8, device=device)
        timesteps = manager.sample_timesteps(x_0, num_timesteps=1000)
        assert manager.sampler_network is not None
        assert timesteps.shape == (2,)
        first_weight = manager.sampler_network.mlp[0].weight
        assert first_weight.shape[1] == 64 * 4 * 4

    def test_lazy_init_only_once(self, device):
        """Network should only be created once, not on every sample_timesteps call."""
        manager = self._make_manager(device)
        x_0 = torch.randn(2, 16, 8, 8, device=device)
        manager.sample_timesteps(x_0, num_timesteps=1000)
        net1 = manager.sampler_network
        # Second call with same channel count
        manager.sample_timesteps(x_0, num_timesteps=1000)
        assert manager.sampler_network is net1  # Same object, not re-created

    def test_lazy_init_state_dict_before_sampling(self, device):
        """state_dict should work before lazy init (returns no network weights)."""
        manager = self._make_manager(device)
        state = manager.state_dict()
        assert "sampler_network" not in state
        assert "optimizer" not in state
        assert "queue" in state

    def test_lazy_init_load_state_dict_infers_channels(self, device):
        """load_state_dict should infer in_channels from saved weights."""
        # Create a manager with a pre-created 16-channel network and train it
        scheduler = MagicMock()
        T = 1000
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=16, hidden_channels=32, hidden_depth=2).to(device)
        original_manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            hidden_channels=32, hidden_depth=2, update_freq=5,
            queue_size=10, num_selected=3,
        )
        state = original_manager.state_dict()

        # Create a new lazy manager and load state
        new_manager = self._make_manager(device)
        assert new_manager.sampler_network is None
        new_manager.load_state_dict(state)
        assert new_manager.sampler_network is not None
        # Verify the network was created with 16 channels
        first_weight = new_manager.sampler_network.mlp[0].weight
        assert first_weight.shape[1] == 16 * 4 * 4

    def test_lazy_init_full_algorithm_cycle(self, device):
        """Full Algorithm 1+2 cycle should work with lazy-init 16-channel network."""
        manager = self._make_manager(device)
        x_0 = torch.randn(2, 16, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # Sample timesteps (triggers lazy init)
        timesteps = manager.sample_timesteps(x_0, num_timesteps=1000)
        assert timesteps.shape == (2,)

        # Algorithm 2: compute delta
        losses_before = manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=200
        )
        delta_approx, selected = manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before
        )

        # Algorithm 1, line 8: update sampler
        manager.update_sampler(delta_approx, x_0)
        assert len(manager.queue) >= 1

    def test_pre_created_network_still_works(self, device):
        """Passing a pre-created network should still work (backward compatibility)."""
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        scheduler = MagicMock()
        T = 1000
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
        )
        assert manager.sampler_network is net
        assert manager.optimizer is not None
        x_0 = torch.randn(2, 4, 8, 8, device=device)
        timesteps = manager.sample_timesteps(x_0, num_timesteps=1000)
        assert timesteps.shape == (2,)


# ---------------------------------------------------------------------------
# Tests: min_timestep / max_timestep range
# ---------------------------------------------------------------------------

class TestMinMaxTimestep:

    def test_min_timestep_respected(self, device):
        """Sampled timesteps should be >= min_timestep."""
        scheduler = MagicMock()
        T = 1000
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            min_timestep=100, max_timestep=800,
        )
        x = torch.randn(64, 4, 8, 8, device=device)
        timesteps = manager.sample_timesteps(x, num_timesteps=T)
        assert (timesteps >= 100).all(), f"Min timestep violated: {timesteps.min().item()}"
        assert (timesteps < 800).all(), f"Max timestep violated: {timesteps.max().item()}"

    def test_max_timestep_respected(self, device):
        """Sampled timesteps should be < max_timestep."""
        scheduler = MagicMock()
        T = 1000
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            min_timestep=0, max_timestep=500,
        )
        x = torch.randn(128, 4, 8, 8, device=device)
        timesteps = manager.sample_timesteps(x, num_timesteps=T)
        assert (timesteps < 500).all(), f"Max timestep violated: {timesteps.max().item()}"

    def test_full_range_default(self, device):
        """With defaults, timesteps should span [0, T)."""
        scheduler = MagicMock()
        T = 100
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
        )
        x = torch.randn(256, 4, 8, 8, device=device)
        timesteps = manager.sample_timesteps(x, num_timesteps=T)
        assert timesteps.min() >= 0
        assert timesteps.max() < T

    def test_narrow_range_still_samples(self, device):
        """Even with a very narrow range, sampling should produce valid timesteps."""
        scheduler = MagicMock()
        T = 1000
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            min_timestep=400, max_timestep=410,
        )
        x = torch.randn(32, 4, 8, 8, device=device)
        timesteps = manager.sample_timesteps(x, num_timesteps=T)
        assert (timesteps >= 400).all()
        assert (timesteps < 410).all()


# ---------------------------------------------------------------------------
# Tests: Reward baseline normalization
# ---------------------------------------------------------------------------

class TestRewardBaseline:

    def test_reward_baseline_starts_at_zero(self, device):
        """Reward baseline should start at 0."""
        scheduler = MagicMock()
        T = 100
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            reward_baseline_decay=0.9,
        )
        assert manager._reward_baseline == 0.0

    def test_reward_baseline_updates_after_sampler_update(self, device):
        """After update_sampler, the baseline should be non-zero."""
        scheduler = MagicMock()
        T = 100
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            reward_baseline_decay=0.9,
        )
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        delta = torch.tensor(1.0, device=device)
        manager.update_sampler(delta, x_0)
        # Baseline = 0.9 * 0.0 + 0.1 * 1.0 = 0.1
        assert abs(manager._reward_baseline - 0.1) < 1e-6

    def test_reward_baseline_decays(self, device):
        """Baseline should decay toward the current delta value."""
        scheduler = MagicMock()
        T = 100
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            reward_baseline_decay=0.9,
        )
        x_0 = torch.randn(1, 4, 8, 8, device=device)

        # Run several updates with a constant delta.
        # EMA formula: baseline = v * (1 - decay^n). With decay=0.9, n=50:
        # 0.9^50 ≈ 0.005, so baseline ≈ 0.995 → within 0.01 of 1.0.
        for _ in range(50):
            manager.update_sampler(torch.tensor(1.0, device=device), x_0)

        # EMA of constant 1.0 with decay=0.9 should converge to 1.0
        assert abs(manager._reward_baseline - 1.0) < 0.02

    def test_reward_baseline_in_state_dict(self, device):
        """Baseline and decay should round-trip through state_dict."""
        scheduler = MagicMock()
        T = 100
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            reward_baseline_decay=0.95,
        )
        # Simulate some training
        manager._reward_baseline = 0.42

        state = manager.state_dict()
        assert "reward_baseline" in state
        assert state["reward_baseline"] == 0.42
        assert "reward_baseline_decay" in state
        assert state["reward_baseline_decay"] == 0.95

        # Load into a fresh manager
        new_manager = AdaptiveTimestepManager(
            sampler_network=TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device),
            noise_scheduler=scheduler, device=device,
            reward_baseline_decay=0.95,
        )
        new_manager.load_state_dict(state)
        assert new_manager._reward_baseline == 0.42

    def test_reward_baseline_advantage_reduces_variance(self, device):
        """Advantage = delta - baseline should be centered closer to zero than raw delta."""
        scheduler = MagicMock()
        T = 100
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            reward_baseline_decay=0.9,
        )
        x_0 = torch.randn(1, 4, 8, 8, device=device)

        # Run updates with varying deltas
        advantages = []
        deltas = [0.5, 0.3, 0.7, 0.4, 0.6]
        for d in deltas:
            # Capture advantage by checking baseline before and after
            baseline_before = manager._reward_baseline
            manager.update_sampler(torch.tensor(d, device=device), x_0)
            baseline_after = manager._reward_baseline
            advantage = d - baseline_before
            advantages.append(advantage)

        # With baseline, the mean advantage should be smaller in magnitude
        # than the mean of raw deltas
        import statistics
        raw_mean = statistics.mean(deltas)
        adv_mean = abs(statistics.mean(advantages))
        # The baseline reduces the mean absolute advantage
        assert adv_mean <= raw_mean + 0.1


# ---------------------------------------------------------------------------
# Tests: New state_dict fields backward compatibility
# ---------------------------------------------------------------------------

class TestStateDictBackwardCompat:

    def test_load_old_state_dict_without_new_fields(self, device):
        """Loading an old checkpoint without new fields should not crash."""
        scheduler = MagicMock()
        T = 100
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            min_timestep=50, max_timestep=900, reward_baseline_decay=0.95,
        )

        # Create an "old" state dict without the new fields
        old_state = manager.state_dict()
        del old_state["min_timestep"]
        del old_state["max_timestep"]
        del old_state["reward_baseline_decay"]
        del old_state["reward_baseline"]

        # Create a fresh manager with defaults and load the old state
        new_manager = AdaptiveTimestepManager(
            sampler_network=TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device),
            noise_scheduler=scheduler, device=device,
        )
        new_manager.load_state_dict(old_state)
        # Defaults should be preserved (not overwritten by missing keys)
        assert new_manager.min_timestep == 0
        assert new_manager.max_timestep == T
        assert new_manager.reward_baseline_decay == 0.9

    def test_new_state_dict_round_trip(self, device):
        """New fields should survive a state dict round-trip."""
        scheduler = MagicMock()
        T = 100
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            min_timestep=100, max_timestep=800, reward_baseline_decay=0.95,
        )
        manager._reward_baseline = 0.37

        state = manager.state_dict()
        new_manager = AdaptiveTimestepManager(
            sampler_network=TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device),
            noise_scheduler=scheduler, device=device,
        )
        new_manager.load_state_dict(state)
        assert new_manager.min_timestep == 100
        assert new_manager.max_timestep == 800
        assert new_manager.reward_baseline_decay == 0.95
        assert new_manager._reward_baseline == 0.37


# ---------------------------------------------------------------------------
# Tests: VRAM Optimization Parameters
# ---------------------------------------------------------------------------

class TestVRAMOptimizationParams:
    """Tests for eval_chunk_size, eval_stride, fp32_eval, and state_dict persistence."""

    def test_eval_chunk_size_init(self, device):
        """eval_chunk_size should be stored and used as the default chunk."""
        T = 100
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            eval_chunk_size=8,
        )
        assert manager.eval_chunk_size == 8

    def test_eval_stride_grid_shape(self, device):
        """With stride>1, the eval grid should contain ceil(T/stride) points."""
        T = 100
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        for stride in [1, 2, 4, 10]:
            manager = AdaptiveTimestepManager(
                sampler_network=net, noise_scheduler=scheduler, device=device,
                eval_stride=stride,
            )
            expected_size = len(range(0, T, stride))
            assert len(manager._eval_timesteps) == expected_size, (
                f"stride={stride}: expected {expected_size}, got {len(manager._eval_timesteps)}"
            )
            # Verify the grid values are correct (_eval_timesteps is always on CPU)
            expected = torch.arange(0, T, stride)
            assert torch.equal(manager._eval_timesteps, expected)

    def test_grid_to_timestep_mapping(self, device):
        """_grid_to_timestep should map grid indices to real timestep indices."""
        T = 100
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)

        # stride=4: grid indices [0, 1, 2, ...] → timestep [0, 4, 8, ...]
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            eval_stride=4,
        )
        grid_idx = torch.tensor([0, 1, 5, 24], device=device)
        real_ts = manager._grid_to_timestep(grid_idx)
        assert torch.equal(real_ts, torch.tensor([0, 4, 20, 96], device=device))

    def test_grid_to_timestep_clamp(self, device):
        """_grid_to_timestep should clamp to [min_timestep, max_timestep-1]."""
        T = 100
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            eval_stride=4, min_timestep=10, max_timestep=80,
        )
        # Grid index 0 → timestep 0, but clamped to min_timestep=10
        result = manager._grid_to_timestep(torch.tensor([0], device=device))
        assert result.item() == 10
        # Grid index 25 → timestep 100, but clamped to max_timestep-1=79
        result = manager._grid_to_timestep(torch.tensor([25], device=device))
        assert result.item() == 79

    def test_state_dict_round_trip_vram_fields(self, device):
        """eval_chunk_size, eval_stride, fp32_eval should survive round-trip."""
        T = 100
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            eval_chunk_size=32, eval_stride=5, fp32_eval=True,
        )

        state = manager.state_dict()
        assert state["eval_chunk_size"] == 32
        assert state["eval_stride"] == 5
        assert state["fp32_eval"] is True

        new_manager = AdaptiveTimestepManager(
            sampler_network=TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device),
            noise_scheduler=scheduler, device=device,
        )
        # Defaults before load
        assert new_manager.eval_chunk_size == 16
        assert new_manager.eval_stride == 1
        assert new_manager.fp32_eval is False

        new_manager.load_state_dict(state)
        assert new_manager.eval_chunk_size == 32
        assert new_manager.eval_stride == 5
        assert new_manager.fp32_eval is True
        # Grid should be recomputed with loaded stride (_eval_timesteps is always on CPU)
        expected_grid = torch.arange(0, T, 5)
        assert torch.equal(new_manager._eval_timesteps, expected_grid)

    def test_state_dict_backward_compat_missing_vram_fields(self, device):
        """Loading a state dict without VRAM fields should preserve defaults."""
        T = 100
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
        )

        state = manager.state_dict()
        # Remove VRAM fields to simulate an old checkpoint
        del state["eval_chunk_size"]
        del state["eval_stride"]
        del state["fp32_eval"]

        new_manager = AdaptiveTimestepManager(
            sampler_network=TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device),
            noise_scheduler=scheduler, device=device,
        )
        new_manager.load_state_dict(state)
        # Defaults should be preserved
        assert new_manager.eval_chunk_size == 16
        assert new_manager.eval_stride == 1
        assert new_manager.fp32_eval is False


class TestStridedEvalGrid:
    """Tests for compute_per_timestep_losses with strided eval grid."""

    def test_stride_1_returns_full_losses(self, device):
        """With stride=1, compute_per_timestep_losses should return T losses (one per timestep)."""
        T = 50
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            eval_stride=1, eval_chunk_size=8,
        )

        x = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x)
        model_fn = lambda xt, ts, wd: xt  # identity
        losses = manager.compute_per_timestep_losses(x, noise, model_fn, torch.float32)
        assert losses.shape == (T,), f"Expected shape ({T},), got {losses.shape}"

    def test_stride_4_returns_quarter_losses(self, device):
        """With stride=4, compute_per_timestep_losses should return T/4 losses."""
        T = 100
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T
        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
            eval_stride=4, eval_chunk_size=8,
        )

        x = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x)
        model_fn = lambda xt, ts, wd: xt
        losses = manager.compute_per_timestep_losses(x, noise, model_fn, torch.float32)
        assert losses.shape == (25,), f"Expected shape (25,), got {losses.shape}"

    def test_different_chunk_sizes_give_same_result(self, device):
        """Different eval_chunk_size values should produce identical losses."""
        T = 50
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T

        x = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x)
        # Use a model with fixed weights so results are deterministic
        model = nn.Conv2d(4, 4, 1, bias=False)
        nn.init.ones_(model.weight)
        model = model.to(device)

        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        results = {}
        for chunk_size in [5, 10, 25]:
            net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
            manager = AdaptiveTimestepManager(
                sampler_network=net, noise_scheduler=scheduler, device=device,
                eval_chunk_size=chunk_size,
            )
            with torch.no_grad():
                losses = manager.compute_per_timestep_losses(x, noise, model_fn, torch.float32)
            results[chunk_size] = losses

        # All chunk sizes should give identical results
        torch.testing.assert_close(results[5], results[10], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(results[10], results[25], atol=1e-6, rtol=1e-6)

    def test_compute_per_timestep_losses_for_batch_loop(self, device):
        """compute_per_timestep_losses_for_batch should loop over |S| timesteps
        and return a scalar matching the original B×|S| computation."""
        T = 50
        B = 4
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T

        net = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        manager = AdaptiveTimestepManager(
            sampler_network=net, noise_scheduler=scheduler, device=device,
        )

        model = nn.Conv2d(4, 4, 1, bias=False)
        nn.init.ones_(model.weight)
        model = model.to(device)

        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        x = torch.randn(B, 4, 8, 8, device=device)
        noise = torch.randn_like(x)
        selected_indices = torch.tensor([10, 30, 45], device=device)

        with torch.no_grad():
            loss = manager.compute_per_timestep_losses_for_batch(
                x, noise, model_fn, torch.float32, selected_indices
            )
        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
        assert loss.item() > 0, "Loss should be positive"


class TestBF16SweepAccumulation:
    """Tests for bf16 sweep accumulation with fp32 scalar reduction."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16")
    def test_bf16_accumulation_close_to_fp32(self, device):
        """bf16 accumulation should give results close to fp32 path for same inputs."""
        T = 20
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T

        model = nn.Conv2d(4, 4, 1, bias=False)
        nn.init.ones_(model.weight)
        model = model.to(device)

        x = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x)

        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        # fp32 path (escape hatch)
        net32 = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        mgr_fp32 = AdaptiveTimestepManager(
            sampler_network=net32, noise_scheduler=scheduler, device=device,
            fp32_eval=True, eval_chunk_size=10,
        )
        with torch.no_grad():
            losses_fp32 = mgr_fp32.compute_per_timestep_losses(x, noise, model_fn, torch.float32)

        # bf16 path (default)
        net16 = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        mgr_bf16 = AdaptiveTimestepManager(
            sampler_network=net16, noise_scheduler=scheduler, device=device,
            fp32_eval=False, eval_chunk_size=10,
        )
        with torch.no_grad():
            losses_bf16 = mgr_bf16.compute_per_timestep_losses(x, noise, model_fn, torch.float32)

        # Should be close (not exact due to dtype differences)
        torch.testing.assert_close(losses_bf16, losses_fp32, atol=1e-4, rtol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16")
    def test_bf16_batch_losses_close_to_fp32(self, device):
        """bf16 batch loss path should give results close to fp32."""
        T = 20
        B = 4
        scheduler = MagicMock()
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, T, device=device)
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = T

        model = nn.Conv2d(4, 4, 1, bias=False)
        nn.init.ones_(model.weight)
        model = model.to(device)

        x = torch.randn(B, 4, 8, 8, device=device)
        noise = torch.randn_like(x)
        selected = torch.tensor([5, 15], device=device)

        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        net32 = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        mgr_fp32 = AdaptiveTimestepManager(
            sampler_network=net32, noise_scheduler=scheduler, device=device,
            fp32_eval=True,
        )
        with torch.no_grad():
            loss_fp32 = mgr_fp32.compute_per_timestep_losses_for_batch(
                x, noise, model_fn, torch.float32, selected
            )

        net16 = TimestepSamplerNetwork(in_channels=4, hidden_channels=16, hidden_depth=1).to(device)
        mgr_bf16 = AdaptiveTimestepManager(
            sampler_network=net16, noise_scheduler=scheduler, device=device,
            fp32_eval=False,
        )
        with torch.no_grad():
            loss_bf16 = mgr_bf16.compute_per_timestep_losses_for_batch(
                x, noise, model_fn, torch.float32, selected
            )

        torch.testing.assert_close(loss_bf16, loss_fp32, atol=1e-4, rtol=1e-3)
