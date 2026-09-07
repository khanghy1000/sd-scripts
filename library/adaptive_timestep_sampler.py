"""
Adaptive Non-uniform Timestep Sampling for Accelerating Diffusion Model Training.

Implements the method from Kim et al. (arXiv:2411.09998) which adaptively samples
timesteps by tracking the impact of gradient updates on the objective for each timestep,
prioritizing timesteps that are most likely to minimize the objective effectively.

Supports both DDPM-style discrete timesteps (SD1.5, SDXL) and flow-matching
continuous timesteps (Flux, SD3, Anima, Lumina).
"""

import logging
import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TimestepSamplerNetwork(nn.Module):
    """
    Neural network π_φ that parameterizes a Beta distribution for timestep sampling.

    Takes a latent representation x_0 and outputs two positive scalars (a, b)
    which parameterize a Beta(a, b) distribution from which timesteps are sampled.

    Architecture follows the paper's design:
    - Adaptive average pooling to reduce spatial dimensions
    - Flatten
    - MLP with SiLU activations
    - Softplus outputs to ensure a, b > 0
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 128,
        hidden_depth: int = 2,
    ):
        super().__init__()

        # Adaptive average pooling to fixed spatial size (reduces computation)
        self.pool = nn.AdaptiveAvgPool2d(4)

        # Calculate flattened size after pooling
        flat_size = in_channels * 4 * 4

        # Build MLP layers
        layers = []
        in_dim = flat_size
        for i in range(hidden_depth):
            layers.append(nn.Linear(in_dim, hidden_channels))
            layers.append(nn.SiLU())
            in_dim = hidden_channels

        self.mlp = nn.Sequential(*layers)

        # Output heads: each produces a single positive scalar
        self.head_a = nn.Linear(hidden_channels, 1)
        self.head_b = nn.Linear(hidden_channels, 1)

        # Initialize biases for softplus to give reasonable initial Beta parameters
        # Initial Beta(2, 2) is symmetric around 0.5, close to uniform
        nn.init.constant_(self.head_a.bias, 1.2)  # softplus(1.2) ≈ 1.55
        nn.init.constant_(self.head_b.bias, 1.2)
        nn.init.zeros_(self.head_a.weight)
        nn.init.zeros_(self.head_b.weight)

    def forward(self, x_0_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_0_latent: Latent representation of x_0, shape (B, C, H, W)

        Returns:
            a, b: Positive scalars for Beta distribution, each shape (B,)
        """
        # Pool and flatten
        h = self.pool(x_0_latent)
        h = h.reshape(h.shape[0], -1)

        # MLP
        h = self.mlp(h)

        # Output heads with softplus to ensure positivity
        a = F.softplus(self.head_a(h).squeeze(-1)) + 1e-6
        b = F.softplus(self.head_b(h).squeeze(-1)) + 1e-6

        return a, b


def select_timesteps_f_statistic(
    queue_data: torch.Tensor,
    num_selected: int,
) -> torch.Tensor:
    """
    Feature selection using F-statistic from linear regression.

    Given a queue of shape (Q_size, T) containing per-timestep delta values,
    identifies the top |num_selected| timesteps that best explain the overall
    delta (mean across timesteps) using the F-statistic.

    For each timestep tau, the F-statistic measures how well delta_{k,tau}
    predicts the mean delta across all timesteps, based on a simple linear
    regression: F = r^2 / (1 - r^2) * (n - 2), where r is the Pearson
    correlation coefficient.

    Args:
        queue_data: Tensor of shape (Q_size, T) with historical delta values
        num_selected: Number of top timesteps to select

    Returns:
        Tensor of shape (num_selected,) with indices of selected timesteps
    """
    Q_size, T = queue_data.shape

    if Q_size < 2:
        # Not enough data for correlation, select timesteps with highest mean abs delta
        mean_abs = queue_data.mean(dim=0).abs()
        return torch.topk(mean_abs, min(num_selected, T)).indices

    # Compute the target: mean delta across timesteps for each queue entry
    # This represents the overall impact of each gradient update
    target = queue_data.mean(dim=1)  # shape (Q_size,)

    # Compute Pearson correlation between each timestep's delta and the target
    # r = cov(X, Y) / (std(X) * std(Y))
    target_centered = target - target.mean()
    target_std = target_centered.norm()

    if target_std < 1e-10:
        # Target has near-zero variance, fall back to mean absolute delta
        mean_abs = queue_data.mean(dim=0).abs()
        return torch.topk(mean_abs, min(num_selected, T)).indices

    # Center and normalize each timestep column
    data_centered = queue_data - queue_data.mean(dim=0, keepdim=True)  # (Q, T)
    data_std = data_centered.norm(dim=0)  # (T,)

    # Avoid division by zero
    valid_mask = data_std > 1e-10

    # Compute correlation
    correlations = torch.zeros(T, device=queue_data.device)
    if valid_mask.any():
        # cov(X_tau, Y) = X_tau^T @ Y / n
        cov = (data_centered[:, valid_mask].T @ target_centered) / Q_size
        # r = cov / (std_x * std_y)
        correlations[valid_mask] = cov / (data_std[valid_mask] * target_std)

    # F-statistic: F = r^2 / (1 - r^2) * (n - 2)
    r_sq = correlations ** 2
    # Clamp to avoid division by zero
    r_sq = torch.clamp(r_sq, max=1.0 - 1e-6)
    f_stats = r_sq / (1.0 - r_sq) * max(Q_size - 2, 1)

    # Select top |num_selected| timesteps by F-statistic
    actual_selected = min(num_selected, T)
    return torch.topk(f_stats, actual_selected).indices


class AdaptiveTimestepManager:
    """
    Manages adaptive non-uniform timestep sampling for diffusion model training.

    Implements Algorithm 1 (Training DM with Timestep Sampler) and Algorithm 2
    (Approximation of Delta_k^t) from the paper.

    The manager:
    1. Uses a TimestepSamplerNetwork (π_φ) to generate Beta distribution parameters
    2. Samples timesteps from the Beta distribution
    3. Every f_S gradient steps, computes the impact of the gradient update on
       per-timestep losses (Algorithm 2)
    4. Updates the sampler using policy gradient (REINFORCE) with entropy regularization

    VRAM optimization parameters (new, memory-only — results unchanged):
    - eval_chunk_size: batch size for per-timestep loss sweeps (default 16,
      reduced from the original 100 to lower peak activation memory).
    - eval_stride: stride for the evaluation timestep grid (default 1 = all
      timesteps, paper-faithful; stride>1 is an opt-in approximation).
    - fp32_eval: if True, keep model_output/target in fp32 during sweeps
      (original behavior); default False uses weight_dtype accumulation
      with fp32 scalar reduction.
    """

    def __init__(
        self,
        sampler_network: Optional[TimestepSamplerNetwork] = None,
        noise_scheduler=None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        # Hyperparameters from the paper
        learning_rate: float = 1e-2,
        entropy_coeff: float = 1e-2,
        update_freq: int = 40,  # f_S
        queue_size: int = 20,  # |Q|
        num_selected: int = 3,  # |S|
        v_parameterization: bool = False,
        # Flow-matching support
        model_type: str = "ddpm",  # "ddpm" or "flow_matching"
        compute_loss_fn=None,  # Optional: (model_output, x_0, noise, t_indices) -> per_sample_losses
        # Network architecture params for lazy initialization
        hidden_channels: int = 128,
        hidden_depth: int = 2,
        # Timestep range
        min_timestep: int = 0,
        max_timestep: Optional[int] = None,
        # Reward normalization baseline (reduces REINFORCE variance)
        reward_baseline_decay: float = 0.9,
        # VRAM optimization params (memory-only; results unchanged for defaults)
        eval_chunk_size: int = 16,
        eval_stride: int = 1,
        fp32_eval: bool = False,
    ):
        self.device = device
        self.dtype = dtype

        # Timestep range for Beta sampling
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep if max_timestep is not None else self._get_num_train_timesteps(noise_scheduler)

        # Reward normalization baseline (EMA of delta) to reduce REINFORCE variance.
        # A running baseline does not bias the gradient but dramatically reduces variance,
        # making the sampler learn effectively even with small per-step deltas (e.g. LoRA).
        self.reward_baseline_decay = reward_baseline_decay
        self._reward_baseline: float = 0.0

        # Store network architecture params for lazy initialization
        self._hidden_channels = hidden_channels
        self._hidden_depth = hidden_depth

        # If a pre-created network is provided, use it directly; otherwise
        # the network will be lazily initialized on the first sample_timesteps() call
        # by inferring in_channels from the actual latent tensor shape.
        if sampler_network is not None:
            self.sampler_network = sampler_network.to(device=device, dtype=dtype)
            self.optimizer = torch.optim.SGD(
                self.sampler_network.parameters(),
                lr=learning_rate,
            )
        else:
            self.sampler_network = None
            self.optimizer = None

        # Hyperparameters
        self.learning_rate = learning_rate
        self.entropy_coeff = entropy_coeff
        self.f_s = update_freq
        self.queue_size = queue_size
        self.num_selected = num_selected

        # VRAM optimization parameters
        self.eval_chunk_size = eval_chunk_size
        self.eval_stride = max(1, eval_stride)
        self.fp32_eval = fp32_eval

        # Noise scheduler reference for Algorithm 2
        self.noise_scheduler = noise_scheduler
        self.num_train_timesteps = self._get_num_train_timesteps(noise_scheduler)

        # Precompute the evaluation timestep grid (strided subset of [0, T)).
        # With stride=1 (default) this is all timesteps, matching the paper exactly.
        self._eval_timesteps = torch.arange(0, self.num_train_timesteps, self.eval_stride)

        # Queue Q for storing historical delta values (Algorithm 2, line 3)
        self.queue: deque = deque(maxlen=queue_size)

        # Cached sampled t (continuous, in [0,1]) for REINFORCE update
        self._cached_t_continuous: Optional[torch.Tensor] = None
        # Track the currently selected |S| REAL timestep indices (not grid indices)
        # for the next call to Algorithm 2.
        self._current_selected_indices: Optional[torch.Tensor] = None
        # Cache for the previous full-batch losses at |S| timesteps (computed before
        # the optimizer step). Used in Algorithm 2 line 7 to compute the delta for
        # the full batch at the selected timesteps.
        self._prev_batch_losses_at_S: Optional[torch.Tensor] = None
        # Cached previous selection's latents/noise for computing prev losses
        self._prev_batch_latents: Optional[torch.Tensor] = None
        self._prev_batch_noise: Optional[torch.Tensor] = None

        # Model type: "ddpm" or "flow_matching"
        self.model_type = model_type

        # Precompute alphas_cumprod on device for noise addition (DDPM only)
        if model_type == "ddpm":
            self._alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        else:
            self._alphas_cumprod = None

        # Loss target type: epsilon prediction vs v-prediction (DDPM only)
        self.v_parameterization = v_parameterization

        # Custom loss function for flow-matching models
        # Signature: (model_output, x_0, noise, t_indices) -> per_sample_losses
        # When provided, overrides the default DDPM/v-pred loss computation.
        self._compute_loss_fn = compute_loss_fn

        logger.info(
            f"AdaptiveTimestepManager initialized: lr={learning_rate}, "
            f"entropy_coeff={entropy_coeff}, f_S={update_freq}, "
            f"|Q|={queue_size}, |S|={num_selected}, v_pred={v_parameterization}, "
            f"model_type={model_type}, custom_loss_fn={compute_loss_fn is not None}, "
            f"timestep_range=[{min_timestep}, {self.max_timestep}), "
            f"reward_baseline_decay={reward_baseline_decay}, "
            f"eval_chunk_size={eval_chunk_size}, eval_stride={eval_stride} "
            f"(eval_grid_size={len(self._eval_timesteps)}), "
            f"fp32_eval={fp32_eval}, "
            f"network={'pre-created' if sampler_network is not None else 'lazy (will init from first latent shape)'}"
        )

    @staticmethod
    def _get_num_train_timesteps(noise_scheduler) -> int:
        """Extract num_train_timesteps from a scheduler, handling both object and mock configs."""
        if noise_scheduler is None:
            return 1000
        if hasattr(noise_scheduler, "config") and hasattr(noise_scheduler.config, "num_train_timesteps"):
            return noise_scheduler.config.num_train_timesteps
        return getattr(noise_scheduler, "num_train_timesteps", 1000)

    def _init_sampler_network(self, in_channels: int):
        """
        Lazily create the TimestepSamplerNetwork and optimizer if not already done.

        Called on the first sample_timesteps() invocation, using the actual
        channel count from the latent tensor to correctly handle any VAE
        (e.g. 4-channel for SD1.5/SDXL, 16-channel for Flux/SD3/Anima).

        Args:
            in_channels: Number of latent channels (from x_0_latent.shape[1])
        """
        if self.sampler_network is not None:
            return
        logger.info(
            f"Lazily creating TimestepSamplerNetwork: in_channels={in_channels}, "
            f"hidden_channels={self._hidden_channels}, hidden_depth={self._hidden_depth}"
        )
        self.sampler_network = TimestepSamplerNetwork(
            in_channels=in_channels,
            hidden_channels=self._hidden_channels,
            hidden_depth=self._hidden_depth,
        ).to(device=self.device, dtype=self.dtype)
        self.optimizer = torch.optim.SGD(
            self.sampler_network.parameters(),
            lr=self.learning_rate,
        )

    def should_update(self, global_step: int) -> bool:
        """Returns True if the sampler should be updated this step."""
        return global_step > 0 and global_step % self.f_s == 0

    def sample_timesteps(
        self,
        x_0_latent: torch.Tensor,
        num_timesteps: int,
    ) -> torch.Tensor:
        """
        Sample timesteps using the adaptive Beta distribution sampler.

        On the first call, lazily initializes the sampler network using
        x_0_latent.shape[1] to determine in_channels.

        Args:
            x_0_latent: Latent representations, shape (B, C, H, W)
            num_timesteps: Total number of discrete timesteps (T)

        Returns:
            timesteps: Sampled timesteps, shape (B,), dtype long, in range [0, num_timesteps)
        """
        self._init_sampler_network(x_0_latent.shape[1])
        self.sampler_network.eval()
        with torch.no_grad():
            a, b = self.sampler_network(x_0_latent)

        # Sample from Beta distribution
        beta_dist = torch.distributions.Beta(a, b)
        t_continuous = beta_dist.sample()  # shape (B,), values in (0, 1)

        # Cache the sampled t for REINFORCE update (Algorithm 1, line 8)
        # The cached value is the raw Beta sample in [0, 1]; the min/max mapping
        # below is deterministic so log_prob gradients are unaffected.
        self._cached_t_continuous = t_continuous.detach()

        # Map Beta samples [0, 1] to the requested timestep range [min_timestep, max_timestep).
        # The Beta distribution naturally concentrates on high-impact regions while the
        # range constraint ensures we never sample outside the training schedule.
        sigma_min = self.min_timestep / num_timesteps
        sigma_max = self.max_timestep / num_timesteps
        t_scaled = sigma_min + t_continuous * (sigma_max - sigma_min)

        # Convert to discrete timesteps within the constrained range
        timesteps = (t_scaled * num_timesteps).long()
        timesteps = torch.clamp(timesteps, self.min_timestep, self.max_timestep - 1)

        return timesteps

    def _grid_to_timestep(self, grid_indices: torch.Tensor) -> torch.Tensor:
        """Convert evaluation-grid indices to real timestep indices.

        Grid indices are positions in the strided evaluation grid;
        real timestep = grid_index * eval_stride.  Results are clamped
        to [min_timestep, max_timestep - 1].
        """
        real = grid_indices * self.eval_stride
        return torch.clamp(real, self.min_timestep, self.max_timestep - 1)

    def compute_per_timestep_losses(
        self,
        x_0_latent: torch.Tensor,
        noise: torch.Tensor,
        model_fn,
        weight_dtype: torch.dtype,
        chunk_size: Optional[int] = None,
        label: str = "sweep",
    ) -> torch.Tensor:
        """
        Compute the diffusion loss at each timestep in the evaluation grid for a single x_0.

        This is used by Algorithm 2 to evaluate the impact of gradient updates.
        For efficiency, processes timesteps in chunks to avoid OOM, and by default
        accumulates per-sample losses in weight_dtype with fp32 scalar reduction
        (controlled by self.fp32_eval).

        Supports both DDPM-style and flow-matching noise addition.
        For flow-matching models, a custom compute_loss_fn can be provided
        at initialization to handle model-specific loss computation.

        Args:
            x_0_latent: Single latent, shape (1, C, H, W)
            noise: Corresponding noise, shape (1, C, H, W)
            model_fn: Function(noisy_latents, timesteps, weight_dtype) -> noise_pred
            weight_dtype: Data type for model inference
            chunk_size: Number of timesteps to process at once (default: self.eval_chunk_size)
            label: Human-readable tag for log messages (e.g. "theta_k pre-step")

        Returns:
            losses: Per-timestep losses over the evaluation grid, shape (T_eval,)
        """
        T = self.num_train_timesteps
        if chunk_size is None:
            chunk_size = self.eval_chunk_size

        # Expand x_0 and noise to single sample
        x_0 = x_0_latent[:1]  # (1, C, H, W)
        eps = noise[:1]  # (1, C, H, W)

        eval_ts = self._eval_timesteps.to(self.device)  # (T_eval,)
        T_eval = len(eval_ts)
        losses = torch.zeros(T_eval, device=self.device, dtype=torch.float32)

        # Determine whether to use fp32 for model_output/target (original behavior)
        # or weight_dtype with fp32 scalar reduction (optimized path).
        # Always use fp32 when a custom loss function is provided (it may expect fp32).
        use_fp32 = self.fp32_eval or self._compute_loss_fn is not None

        reduce_dims = list(range(1, 4))  # spatial dims for (chunk, C, H, W) tensors

        num_chunks = (T_eval + chunk_size - 1) // chunk_size
        logger.info(
            f"Adaptive sampler sweep start ({label}): {T_eval} eval points in "
            f"{num_chunks} chunks of {chunk_size}, "
            f"precision={'fp32' if use_fp32 else 'weight_dtype+fp32-reduce'}"
        )
        sweep_t0 = time.time()

        for start in range(0, T_eval, chunk_size):
            end = min(start + chunk_size, T_eval)
            t_indices = eval_ts[start:end].long()  # real timestep indices, (chunk_len,)
            chunk_len = end - start

            if self.model_type == "flow_matching":
                # Flow-matching noise addition: x_t = sigma * noise + (1 - sigma) * x_0
                # where sigma = t / T (continuous in [0, 1])
                sigmas = t_indices.float().to(self.device) / T  # (chunk_len,)
                sigmas_view = sigmas.view(-1, 1, 1, 1)  # (chunk_len, 1, 1, 1)

                x_0_expanded = x_0.expand(chunk_len, -1, -1, -1)
                eps_expanded = eps.expand(chunk_len, -1, -1, -1)

                x_t = sigmas_view * eps_expanded + (1.0 - sigmas_view) * x_0_expanded
                x_t = x_t.to(weight_dtype)

                # Forward pass through the model
                with torch.no_grad():
                    model_output = model_fn(x_t, t_indices, weight_dtype)

                if use_fp32:
                    # Original fp32 path (for custom loss fns or fp32_eval escape hatch)
                    model_output = model_output.to(torch.float32)
                    if self._compute_loss_fn is not None:
                        chunk_losses = self._compute_loss_fn(model_output, x_0_expanded.to(torch.float32), eps_expanded.to(torch.float32), t_indices)
                    else:
                        target = (eps_expanded - x_0_expanded).to(torch.float32)
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims)
                else:
                    # Optimized path: accumulate in weight_dtype, reduce to fp32
                    if self._compute_loss_fn is not None:
                        model_output_f32 = model_output.to(torch.float32)
                        chunk_losses = self._compute_loss_fn(model_output_f32, x_0_expanded.to(torch.float32), eps_expanded.to(torch.float32), t_indices)
                    else:
                        target = eps_expanded - x_0_expanded  # stays in weight_dtype
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims, dtype=torch.float32)
            else:
                # DDPM noise addition: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
                alphas_cumprod = self._alphas_cumprod
                alpha_bar_t = alphas_cumprod[t_indices].view(-1, 1, 1, 1)  # (chunk_len, 1, 1, 1)
                sqrt_alpha = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar_t)

                # Expand x_0 and noise to match chunk size
                x_0_expanded = x_0.expand(chunk_len, -1, -1, -1)
                eps_expanded = eps.expand(chunk_len, -1, -1, -1)

                x_t = sqrt_alpha * x_0_expanded + sqrt_one_minus_alpha * eps_expanded
                x_t = x_t.to(weight_dtype)

                # Forward pass through the model
                with torch.no_grad():
                    model_output = model_fn(x_t, t_indices, weight_dtype)

                if use_fp32:
                    # Original fp32 path
                    model_output = model_output.to(torch.float32)
                    if self._compute_loss_fn is not None:
                        chunk_losses = self._compute_loss_fn(model_output, x_0_expanded.to(torch.float32), eps_expanded.to(torch.float32), t_indices)
                    elif self.v_parameterization:
                        target = sqrt_alpha * eps_expanded - sqrt_one_minus_alpha * x_0_expanded
                        target = target.to(torch.float32)
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims)
                    else:
                        target = eps_expanded.to(torch.float32)
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims)
                else:
                    # Optimized path: accumulate in weight_dtype, reduce to fp32
                    if self._compute_loss_fn is not None:
                        model_output_f32 = model_output.to(torch.float32)
                        chunk_losses = self._compute_loss_fn(model_output_f32, x_0_expanded.to(torch.float32), eps_expanded.to(torch.float32), t_indices)
                    elif self.v_parameterization:
                        target = sqrt_alpha * eps_expanded - sqrt_one_minus_alpha * x_0_expanded
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims, dtype=torch.float32)
                    else:
                        target = eps_expanded  # stays in weight_dtype
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims, dtype=torch.float32)

            losses[start:end] = chunk_losses

        # losses.mean().item() also synchronizes CUDA, making elapsed time accurate
        logger.info(
            f"Adaptive sampler sweep done ({label}): "
            f"{time.time() - sweep_t0:.2f}s, mean loss={losses.mean().item():.6f}"
        )

        return losses

    def compute_delta_approximation(
        self,
        model_fn,
        x_0_latent: torch.Tensor,
        noise: torch.Tensor,
        weight_dtype: torch.dtype,
        losses_before: torch.Tensor,
        full_batch_latents: Optional[torch.Tensor] = None,
        full_batch_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Algorithm 2: Approximate Delta_k^t.

        Given per-timestep losses with θ_k (before optimizer step) and θ_{k+1}
        (after optimizer step), computes the approximated delta.

        When full batch data is provided AND we have a previous |S| selection from
        a prior call, the returned delta is computed over the full mini-batch at the
        |S| selected timesteps (per Algorithm 2, line 7: "for x_0s in current mini-batch").
        This requires losses_before_batch_at_S to have been cached via
        `cache_batch_losses_at_S()` BEFORE the optimizer step.

        Otherwise, falls back to computing the delta for a single x_0 at the
        |S| selected timesteps.

        Args:
            model_fn: Function(noisy_latents, timesteps, weight_dtype) -> noise_pred
            x_0_latent: Single latent for the queue, shape (1, C, H, W)
            noise: Corresponding noise for the queue, shape (1, C, H, W)
            weight_dtype: Data type for model inference
            losses_before: Per-timestep losses with θ_k over the eval grid, shape (T_eval,)
            full_batch_latents: Optional full batch latents, shape (B, C, H, W)
            full_batch_noise: Optional full batch noise, shape (B, C, H, W)

        Returns:
            delta_approx: Scalar approximation of Δ̃_k^t
            selected_indices: The REAL timestep indices selected by feature selection
        """
        # Step 2: Compute per-timestep losses with θ_{k+1} (after optimizer step)
        losses_after = self.compute_per_timestep_losses(
            x_0_latent, noise, model_fn, weight_dtype, label="theta_{k+1} post-step"
        )

        # Compute delta for each grid timestep: δ_{k,τ} = L_τ(θ_k) - L_τ(θ_{k+1})
        deltas = losses_before - losses_after  # shape (T_eval,)

        # Step 3: Push into queue Q
        self.queue.append(deltas.detach().cpu())

        # Step 4-5: Feature selection if queue has enough data.
        # Returns grid indices into the evaluation grid.
        if len(self.queue) > 1:
            queue_tensor = torch.stack(list(self.queue), dim=0).to(self.device)  # (Q, T_eval)
            new_selected_grid_indices = select_timesteps_f_statistic(
                queue_tensor, self.num_selected
            )
        else:
            # First iteration: select timesteps with highest absolute delta
            new_selected_grid_indices = torch.topk(deltas.abs(), self.num_selected).indices

        # Convert grid indices to real timestep indices
        new_selected_timestep_indices = self._grid_to_timestep(new_selected_grid_indices)

        # Step 7: Compute approximation for the current mini-batch.
        # Prefer the full batch at the PREVIOUS |S| selection (if available),
        # since we already have losses_before for the full batch at those timesteps.
        # Otherwise, fall back to the single x_0 at the new |S| selection.
        if (
            full_batch_latents is not None
            and full_batch_noise is not None
            and self._current_selected_indices is not None
            and self._prev_batch_losses_at_S is not None
        ):
            prev_S = self._current_selected_indices.to(self.device)
            # Compute losses after the step for the full batch at the PREVIOUS |S| timesteps
            losses_after_batch_at_S = self.compute_per_timestep_losses_for_batch(
                full_batch_latents, full_batch_noise, model_fn, weight_dtype, prev_S
            )
            # Delta for the full batch: mean over batch and |S| timesteps
            delta_approx = self._prev_batch_losses_at_S - losses_after_batch_at_S
            # Use the previous selection as the returned indices (these are the ones
            # for which the delta was actually computed)
            selected_indices = prev_S
        else:
            # Fallback: single x_0 at the new |S| selection (grid indices for indexing deltas)
            delta_approx = deltas[new_selected_grid_indices].mean()
            selected_indices = new_selected_timestep_indices

        # Update the current selection for the NEXT call to Algorithm 2
        # Store REAL timestep indices (not grid indices) so batch loss fns work correctly
        self._current_selected_indices = selected_indices.detach().cpu()
        # Clear the cached prev batch losses (will be set by the next before-step call)
        self._prev_batch_losses_at_S = None
        self._prev_batch_latents = None
        self._prev_batch_noise = None

        return delta_approx, selected_indices

    def cache_batch_losses_at_S(
        self,
        full_batch_latents: torch.Tensor,
        full_batch_noise: torch.Tensor,
        model_fn,
        weight_dtype: torch.dtype,
    ):
        """
        Cache per-timestep losses for the full batch at the current |S| selection.

        This should be called BEFORE the optimizer step, so that after the step
        we can compute the full-batch delta at those |S| timesteps.

        If no |S| selection exists yet (first call), this is a no-op.
        """
        if self._current_selected_indices is None:
            return
        # _current_selected_indices stores REAL timestep indices (not grid indices)
        indices = self._current_selected_indices.to(self.device)
        self._prev_batch_losses_at_S = self.compute_per_timestep_losses_for_batch(
            full_batch_latents, full_batch_noise, model_fn, weight_dtype, indices
        ).detach()
        self._prev_batch_latents = full_batch_latents
        self._prev_batch_noise = full_batch_noise

    def compute_per_timestep_losses_for_batch(
        self,
        x_0_latent: torch.Tensor,
        noise: torch.Tensor,
        model_fn,
        weight_dtype: torch.dtype,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-timestep losses for the full batch at selected real timesteps.

        This is used to compute the final Δ̃_k^t for the full mini-batch
        (Algorithm 2, line 7: "for x_0s in current mini-batch").

        To avoid peak VRAM from a single B×|S| forward, processes one timestep
        at a time with batch-size B (|S| sequential forwards at batch B).

        Args:
            x_0_latent: Latent representations for the batch, shape (B, C, H, W)
            noise: Corresponding noise, shape (B, C, H, W)
            model_fn: Function(noisy_latents, timesteps, weight_dtype) -> noise_pred
            weight_dtype: Data type for model inference
            selected_indices: Real timestep indices to evaluate, shape (|S|,)

        Returns:
            losses: Scalar mean loss across the batch and selected timesteps
        """
        T = self.num_train_timesteps
        B = x_0_latent.shape[0]
        S = selected_indices.shape[0]

        # Determine accumulation precision (same logic as compute_per_timestep_losses)
        use_fp32 = self.fp32_eval or self._compute_loss_fn is not None
        reduce_dims = list(range(1, x_0_latent.ndim))  # spatial dims for per-sample loss

        logger.debug(
            f"Adaptive sampler batch eval start: B={B}, |S|={S} timesteps "
            f"{selected_indices.tolist()}"
        )
        batch_t0 = time.time()

        # Accumulate losses from each timestep separately (one batch-B forward per timestep)
        # to avoid the peak VRAM of a single B×|S| forward.
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        for s_idx in range(S):
            t_idx = selected_indices[s_idx].to(self.device)  # scalar real timestep index
            # Create timestep tensor of shape (B,) for the model
            timesteps_b = t_idx.expand(B)

            if self.model_type == "flow_matching":
                sigma = t_idx.float() / T
                sigma_view = sigma.view(1, 1, 1, 1)

                x_t = sigma_view * noise + (1.0 - sigma_view) * x_0_latent
                x_t = x_t.to(weight_dtype)

                with torch.no_grad():
                    model_output = model_fn(x_t, timesteps_b, weight_dtype)

                if use_fp32:
                    model_output = model_output.to(torch.float32)
                    if self._compute_loss_fn is not None:
                        chunk_losses = self._compute_loss_fn(model_output, x_0_latent.to(torch.float32), noise.to(torch.float32), timesteps_b)
                    else:
                        target = (noise - x_0_latent).to(torch.float32)
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims)
                else:
                    if self._compute_loss_fn is not None:
                        model_output_f32 = model_output.to(torch.float32)
                        chunk_losses = self._compute_loss_fn(model_output_f32, x_0_latent.to(torch.float32), noise.to(torch.float32), timesteps_b)
                    else:
                        target = noise - x_0_latent  # stays in weight_dtype
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims, dtype=torch.float32)
            else:
                # DDPM noise addition
                alphas_cumprod = self._alphas_cumprod
                alpha_bar_t = alphas_cumprod[t_idx].view(1, 1, 1, 1)
                sqrt_alpha = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar_t)

                x_t = sqrt_alpha * x_0_latent + sqrt_one_minus_alpha * noise
                x_t = x_t.to(weight_dtype)

                with torch.no_grad():
                    model_output = model_fn(x_t, timesteps_b, weight_dtype)

                if use_fp32:
                    model_output = model_output.to(torch.float32)
                    if self._compute_loss_fn is not None:
                        chunk_losses = self._compute_loss_fn(model_output, x_0_latent.to(torch.float32), noise.to(torch.float32), timesteps_b)
                    elif self.v_parameterization:
                        target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_0_latent
                        target = target.to(torch.float32)
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims)
                    else:
                        target = noise.to(torch.float32)
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims)
                else:
                    if self._compute_loss_fn is not None:
                        model_output_f32 = model_output.to(torch.float32)
                        chunk_losses = self._compute_loss_fn(model_output_f32, x_0_latent.to(torch.float32), noise.to(torch.float32), timesteps_b)
                    elif self.v_parameterization:
                        target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_0_latent
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims, dtype=torch.float32)
                    else:
                        target = noise  # stays in weight_dtype
                        chunk_losses = F.mse_loss(model_output, target, reduction="none")
                        chunk_losses = chunk_losses.mean(dim=reduce_dims, dtype=torch.float32)

            # Mean over batch for this timestep, accumulate in fp64 for precision
            total_loss += chunk_losses.mean().to(torch.float64)

        logger.debug(
            f"Adaptive sampler batch eval done: {time.time() - batch_t0:.2f}s"
        )

        # Final mean over |S| timesteps
        return (total_loss / S).to(torch.float32)

    def update_sampler(
        self,
        delta_k_t: torch.Tensor,
        x_0_latent: torch.Tensor,
    ):
        """
        Update the timestep sampler π_φ using policy gradient (REINFORCE).

        Implements Algorithm 1, line 8:
        φ_{k+1} = φ_k + γ · Δ̃_k^t · ∇_{φ_k} log π_{φ_k}(a, b | x_0)

        With entropy regularization to prevent premature convergence.

        Args:
            delta_k_t: Approximated Δ̃_k^t, scalar tensor
            x_0_latent: The x_0 latent used for sampling, shape (B, C, H, W)
        """
        self.sampler_network.train()

        # Forward pass to get current (a, b) with gradients
        a, b = self.sampler_network(x_0_latent)

        # Sample from the current Beta distribution (with gradients)
        beta_dist = torch.distributions.Beta(a, b)

        # Use the cached sampled t (the action actually taken in the training step)
        # This is the correct REINFORCE estimator: gradient of log_prob evaluated
        # at the action that was sampled, per Eq. 13 of the paper.
        if self._cached_t_continuous is not None:
            t_continuous = self._cached_t_continuous
        else:
            t_continuous = a / (a + b)  # fallback to mean

        # Compute log probability
        log_prob = beta_dist.log_prob(t_continuous.clamp(1e-6, 1.0 - 1e-6))

        # Reward normalization baseline (EMA of past deltas).
        # Subtracting a baseline from the reward does not bias the REINFORCE gradient
        # but dramatically reduces variance, which is critical for LoRA fine-tuning
        # where per-step model updates (and therefore delta_k_t) are very small.
        with torch.no_grad():
            delta_val = delta_k_t.mean().item()
            self._reward_baseline = (
                self.reward_baseline_decay * self._reward_baseline
                + (1.0 - self.reward_baseline_decay) * delta_val
            )
            advantage = delta_k_t.detach() - self._reward_baseline

        # Policy gradient loss: -advantage * log_prob (we want to maximize delta)
        policy_loss = -(advantage * log_prob).mean()

        # Entropy regularization: H(Beta(a,b)) to prevent premature convergence
        entropy = beta_dist.entropy().mean()
        entropy_loss = -self.entropy_coeff * entropy

        total_loss = policy_loss + entropy_loss

        # Update sampler
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        logger.debug(
            f"Sampler update: delta={delta_val:.6f}, baseline={self._reward_baseline:.6f}, "
            f"advantage={advantage.mean().item():.6f}, "
            f"policy_loss={policy_loss.item():.6f}, "
            f"entropy={entropy.item():.6f}, "
            f"a_mean={a.mean().item():.3f}, b_mean={b.mean().item():.3f}"
        )

    def state_dict(self) -> Dict:
        """Save sampler state for checkpointing.

        If the network has not been lazily initialized yet, only the
        hyperparameters and queue are saved (network/optimizer are omitted).
        """
        state: Dict = {
            "queue": list(self.queue),
            "learning_rate": self.learning_rate,
            "entropy_coeff": self.entropy_coeff,
            "f_s": self.f_s,
            "queue_size": self.queue_size,
            "num_selected": self.num_selected,
            "v_parameterization": self.v_parameterization,
            "model_type": self.model_type,
            "min_timestep": self.min_timestep,
            "max_timestep": self.max_timestep,
            "reward_baseline_decay": self.reward_baseline_decay,
            "reward_baseline": self._reward_baseline,
            "eval_chunk_size": self.eval_chunk_size,
            "eval_stride": self.eval_stride,
            "fp32_eval": self.fp32_eval,
        }
        if self.sampler_network is not None:
            state["sampler_network"] = self.sampler_network.state_dict()
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict):
        """Load sampler state from checkpoint.

        If the network has not been lazily initialized yet, infers
        in_channels from the saved weights' first linear layer.
        """
        # Lazily create the network if needed, inferring in_channels from saved weights
        if self.sampler_network is None and "sampler_network" in state_dict:
            net_sd = state_dict["sampler_network"]
            if net_sd and "mlp.0.weight" in net_sd:
                first_weight = net_sd["mlp.0.weight"]  # shape: (hidden_channels, in_channels * 4 * 4)
                inferred_in_channels = first_weight.shape[1] // 16
                inferred_hidden_channels = first_weight.shape[0]
                # Count MLP linear layers to infer hidden_depth
                inferred_hidden_depth = sum(
                    1 for k in net_sd if k.startswith("mlp.") and k.endswith(".weight")
                )
                self._hidden_channels = inferred_hidden_channels
                self._hidden_depth = inferred_hidden_depth
                self._init_sampler_network(inferred_in_channels)

        if self.sampler_network is not None and "sampler_network" in state_dict:
            self.sampler_network.load_state_dict(state_dict["sampler_network"])
        if self.optimizer is not None and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        self.queue = deque(state_dict["queue"], maxlen=self.queue_size)
        # Restore fields if present (backward-compatible with older checkpoints)
        if "min_timestep" in state_dict:
            self.min_timestep = state_dict["min_timestep"]
        if "max_timestep" in state_dict:
            self.max_timestep = state_dict["max_timestep"]
        if "reward_baseline_decay" in state_dict:
            self.reward_baseline_decay = state_dict["reward_baseline_decay"]
        if "reward_baseline" in state_dict:
            self._reward_baseline = state_dict["reward_baseline"]
        # New VRAM optimization fields (backward-compatible)
        if "eval_chunk_size" in state_dict:
            self.eval_chunk_size = state_dict["eval_chunk_size"]
        if "eval_stride" in state_dict:
            old_stride = self.eval_stride
            self.eval_stride = state_dict["eval_stride"]
            if self.eval_stride != old_stride:
                # Recompute the evaluation grid with the loaded stride
                self._eval_timesteps = torch.arange(0, self.num_train_timesteps, self.eval_stride)
        if "fp32_eval" in state_dict:
            self.fp32_eval = state_dict["fp32_eval"]