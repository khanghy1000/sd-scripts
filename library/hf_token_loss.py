"""
High-Frequency Token Latent Loss — shared module for the sd_scripts train_network family.

Port of `plans/high-frequency-token-loss (1).md` (§4 reference implementation, bit-for-bit).

An opt-in auxiliary loss term for diffusion / flow-matching training that concentrates
training effort on the image tokens that carry fine (high-frequency) detail:

    L_total = L_mse + lambda * L_hf
    L_hf    = mean over batch of mean over tokens of w_token * ||x0_hat_token - x0_token||^2

The per-token weights `w_token` are computed per micro-batch, on-GPU, from the **clean
target only** (never from the prediction / autograd graph). There is no cache, no
extractor network, no RNG draw — the term is deterministic and RNG-neutral, and
`lambda == 0.0` is bit-identical to the loss without the feature.

Invariants (spec §5, do not weaken):
  1. Python-level float gate: `scale > 0.0` decides whether the term runs at all.
  2. Deterministic, RNG-neutral: weights are a pure function of the clean batch.
  3. No host syncs in the per-step path (callers materialize `.item()` at a periodic sync).
  4. Native-dtype arithmetic (shifts/adds only — no FFT).
  5. Weights come from `clean` only, detached from the autograd graph.
  6. `patch` must equal the model's own patchify size; H, W must be divisible by patch.
  7. Supported prediction modes use an HF-only one-sided high-noise SNR gate;
     low-noise samples are unchanged and the main loss weighting is never modified.
  8. Caption-independent: never touches target_v or the dropout mask.
"""

import torch
import torch.nn.functional as F

# Robustness epsilon (bf16-representable); prevents 0/0 on all-flat latents.
HF_EPS = 1e-6

# Default train-time epsilon for x0-residual models whose training target is
# v = (noisy - x0) / (t + eps_train) (e.g. ChromaRadiance `raw` prediction).
HF_DEFAULT_EPS_TRAIN = 5e-2


def hf_sigma_from_timesteps(
    timesteps: torch.Tensor,
    noise_scheduler=None,
    timesteps_in_sigma: bool = False,
) -> torch.Tensor:
    """Convert trainer timesteps to the sigma convention used by HF reconstruction.

    Flow trainers in this repository use either normalized sigma values directly
    (Anima) or discrete timestep values in ``[0, num_train_timesteps]``.
    """
    t = timesteps.detach().to(dtype=torch.float64).reshape(-1)
    if timesteps_in_sigma:
        sigma = t
    else:
        if noise_scheduler is None:
            raise ValueError("noise_scheduler is required for discrete flow timesteps")
        sigma = t / float(noise_scheduler.config.num_train_timesteps)
    return sigma.clamp(min=1e-6, max=1.0)


def hf_one_sided_snr_gate(
    snr: torch.Tensor,
    snr_cut: float,
    min_gate: float = 0.10,
    power: float = 1.0,
) -> torch.Tensor:
    """Leave high-SNR samples unchanged and attenuate only low-SNR samples.

    ``snr_cut`` is the boundary between the untouched low-noise region and the
    attenuated high-noise region. Setting ``min_gate=0`` is useful for epsilon
    prediction, whose x0 reconstruction has an inverse-SNR gradient factor.
    """
    if snr_cut <= 0.0:
        raise ValueError("snr_cut must be > 0")
    if not 0.0 <= min_gate <= 1.0:
        raise ValueError("min_gate must be between 0 and 1")
    if power <= 0.0:
        raise ValueError("power must be > 0")

    snr = snr.detach().to(dtype=torch.float64).reshape(-1).clamp(min=0.0)
    high_noise_gate = (snr / float(snr_cut)).clamp(max=1.0).pow(float(power))
    gate = torch.where(snr >= float(snr_cut), torch.ones_like(snr), high_noise_gate)
    return gate.clamp(min=float(min_gate), max=1.0)


def hf_snr_from_timesteps(
    timesteps: torch.Tensor,
    mode: str,
    noise_scheduler=None,
    timesteps_in_sigma: bool = False,
) -> torch.Tensor:
    """Compute SNR using the schedule appropriate for an HF prediction mode.

    Flow modes and x0-direct models with a flow scheduler use the flow sigma
    convention. DDPM modes and x0-direct models with a DDPM scheduler use the
    scheduler's alpha-bar values.
    """
    if mode in ("flow", "x0_residual_eps") or (
        mode == "x0_direct"
        and (noise_scheduler is None or not hasattr(noise_scheduler, "alphas_cumprod"))
    ):
        sigma = hf_sigma_from_timesteps(
            timesteps,
            noise_scheduler=noise_scheduler,
            timesteps_in_sigma=timesteps_in_sigma,
        )
        return ((1.0 - sigma) / sigma).square()

    if noise_scheduler is None or not hasattr(noise_scheduler, "alphas_cumprod"):
        raise ValueError(f"noise_scheduler with alphas_cumprod is required for mode '{mode}'")

    acp = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float64)
    ts = timesteps.detach().long().clamp(0, acp.numel() - 1)
    alpha_bar = acp[ts].clamp(min=0.0, max=1.0)
    return alpha_bar / (1.0 - alpha_bar).clamp_min(1e-6)


def validate_hf_high_noise_gate_args(
    min_gate: float,
    power: float,
    snr_cut: float,
) -> None:
    """Validate the one-sided high-noise gate configuration."""
    if not 0.0 <= min_gate <= 1.0:
        raise ValueError("hf_high_noise_min_weight must be between 0 and 1")
    if power <= 0.0:
        raise ValueError("hf_high_noise_power must be > 0")
    if snr_cut <= 0.0:
        raise ValueError("hf_high_noise_snr_cut must be > 0")


def laplacian_energy(x: torch.Tensor) -> torch.Tensor:
    """Squared Laplacian response, per spatial position per channel.

    Uses replication padding (load-bearing): a *constant* input gives exactly 0,
    whereas zero padding would produce a spurious non-zero boundary ring.

    Args:
        x: [B, C, H, W]

    Returns:
        [B, C, H, W] of (4x - (up + down + left + right))^2
    """
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    lap = (
        4.0 * x
        - padded[:, :, :-2, 1:-1]
        - padded[:, :, 2:, 1:-1]
        - padded[:, :, 1:-1, :-2]
        - padded[:, :, 1:-1, 2:]
    )
    return lap * lap


def tokenize(x: torch.Tensor, patch: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, N, C*patch*patch] via non-overlapping unfold.

    F.unfold is channel-major per patch; order is irrelevant because only means
    are taken downstream.
    """
    B, C, H, W = x.shape
    assert H % patch == 0 and W % patch == 0, (
        f"patch {patch} must divide latent size {H}x{W}"
    )
    cols = F.unfold(x, kernel_size=patch, stride=patch)  # [B, C*p*p, N]
    return cols.transpose(1, 2).reshape(B, (H // patch) * (W // patch), C * patch * patch)


def hf_token_weights(clean: torch.Tensor, patch: int, exponent: float, eps: float = HF_EPS) -> torch.Tensor:
    """Per-token detail weights derived from the clean target. Detached (constant).

    Args:
        clean: [B, C, H, W] clean target (constant, detached)
        patch: token patch size
        exponent: concentration exponent, must be > 0
        eps: robustness epsilon

    Returns:
        [B, N] per-token weights, per-sample mean exactly 1.
    """
    detail = tokenize(laplacian_energy(clean), patch).mean(dim=-1)  # [B, N]
    mean_d = detail.mean(dim=-1, keepdim=True)
    raw = ((detail + eps) / (mean_d + eps)) ** exponent
    return raw / raw.mean(dim=-1, keepdim=True)  # per-sample mean == 1


def hf_per_sample_loss(x0_pred: torch.Tensor, clean: torch.Tensor, patch: int, exponent: float) -> torch.Tensor:
    """Per-sample high-frequency token loss.

    Args:
        x0_pred: [B, C, H, W] predicted clean estimate (Tweedie), in autograd graph
        clean: [B, C, H, W] clean target (constant, detached)
        patch: token patch size
        exponent: concentration exponent

    Returns:
        [B] per-sample loss.
    """
    w = hf_token_weights(clean, patch, exponent)  # [B, N], clean-derived, detached
    per_token = tokenize(x0_pred - clean, patch).square().mean(dim=-1)  # [B, N]
    return (w * per_token).mean(dim=-1)  # [B]


def validate_hf_args(hf_scale: float, hf_exponent: float, hf_patch: int) -> None:
    """Validate the HF configuration surface (spec §6). Raises ValueError on bad values."""
    if hf_scale < 0.0:
        raise ValueError("hf_scale must be >= 0 (0 = off)")
    if hf_exponent <= 0.0:
        raise ValueError("hf_exponent must be > 0")
    if hf_patch <= 0:
        raise ValueError("hf_patch must be a positive integer")


def hf_x0_hat(
    model_pred: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    mode: str,
    noise_scheduler=None,
    timesteps_in_sigma: bool = False,
    eps_train: float = HF_DEFAULT_EPS_TRAIN,
) -> torch.Tensor:
    """Model-class-specific Tweedie reconstruction of the prediction (spec §2.2).

    Args:
        model_pred: [B, C, H, W] model prediction (post-preconditioning where applicable)
        noisy: [B, C, H, W] forward-diffused input (detached constant)
        timesteps: [B] timesteps; for flow trainers typically in [0, num_train_timesteps],
            or in [0, 1] when `timesteps_in_sigma` is True (e.g. Anima)
        mode: one of
            - "flow":             x0_hat = noisy - sigmas * v
            - "x0_direct":        x0_hat = model_pred (SD3-style preconditioning)
            - "x0_residual_eps":  x0_hat = noisy - v * (sigmas + eps_train)
            - "vpred_ddpm":       x0_hat = sqrt(a_bar)*noisy - sqrt(1-a_bar)*v
            - "eps_ddpm":         x0_hat = (noisy - sqrt(1-a_bar)*eps) / sqrt(a_bar)
        noise_scheduler: scheduler with `config.num_train_timesteps` and, for ddpm modes,
            `alphas_cumprod` (prepared/moved to device)
        timesteps_in_sigma: True when `timesteps` are already in [0, 1] (Anima)
        eps_train: train-time epsilon used by x0-residual models

    Returns:
        [B, C, H, W] predicted clean estimate, float64.
    """
    model_pred = model_pred.to(dtype=torch.float64)

    if mode == "x0_direct":
        # Only mode that does not need noisy/timesteps.
        return model_pred

    if noisy is None or timesteps is None:
        raise ValueError(f"noisy and timesteps are required for mode '{mode}'")
    noisy = noisy.to(dtype=torch.float64)

    if mode in ("flow", "x0_residual_eps"):
        t = timesteps.detach().to(dtype=torch.float64).reshape(-1, 1, 1, 1)
        if timesteps_in_sigma:
            sigmas = t
        else:
            if noise_scheduler is None:
                raise ValueError("noise_scheduler is required for mode 'flow' / 'x0_residual_eps'")
            sigmas = t / float(noise_scheduler.config.num_train_timesteps)
        if mode == "x0_residual_eps":
            return noisy - model_pred * (sigmas + float(eps_train))
        return noisy - sigmas * model_pred

    if mode in ("vpred_ddpm", "eps_ddpm"):
        if noise_scheduler is None or not hasattr(noise_scheduler, "alphas_cumprod"):
            raise ValueError("noise_scheduler with alphas_cumprod is required for ddpm modes")
        acp = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float64)
        ts = timesteps.detach().long().clamp(0, acp.numel() - 1)
        a_bar = acp[ts].reshape(-1, 1, 1, 1)
        sqrt_a = torch.sqrt(a_bar)
        sqrt_1ma = torch.sqrt(1.0 - a_bar)
        if mode == "vpred_ddpm":
            return sqrt_a * noisy - sqrt_1ma * model_pred
        return (noisy - sqrt_1ma * model_pred) / sqrt_a

    raise ValueError(f"Unknown hf_prediction_mode: {mode}")


def hf_apply_term(
    final_loss: torch.Tensor,
    model_pred: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    weighting,
    scale: float,
    exponent: float,
    patch: int,
    mode: str,
    noise_scheduler=None,
    timesteps_in_sigma: bool = False,
    eps_train: float = HF_DEFAULT_EPS_TRAIN,
):
    """Apply the HF term to a scalar loss, honoring the Python-level `scale > 0` gate.

    Args:
        final_loss: scalar loss tensor (L_mse) to add the HF term to
        model_pred: [B, C, H, W] model prediction (in autograd graph)
        clean: [B, C, H, W] clean target (constant; will be detached)
        noisy: [B, C, H, W] forward-diffused input (constant; will be detached)
        timesteps: [B] timesteps
        weighting: optional per-sample timestep importance weights [B] or [B,1,1,1]
            (same convention as the main MSE term); None = no per-sample weighting
        scale: hf_scale (lambda), >= 0; 0 = off (bit-identical no-op)
        exponent: hf_exponent, > 0
        patch: token patch size
        mode: hf prediction mode (see hf_x0_hat)
        noise_scheduler, timesteps_in_sigma, eps_train: forwarded to hf_x0_hat

    Returns:
        (new_final_loss, hf_loss_value) where hf_loss_value is the detached scaled
        contribution `scale * L_hf` (None when off), so loss curves decompose
        L_total == L_mse + hf_scaled.
    """
    if scale <= 0.0:
        return final_loss, None

    clean = clean.detach().to(dtype=torch.float64)
    noisy = noisy.detach().to(dtype=torch.float64)

    # Spec §5.6: p must divide H, W (guaranteed by the model's patchify in practice).
    if patch > 1 and (clean.shape[-2] % patch != 0 or clean.shape[-1] % patch != 0):
        raise ValueError(
            f"hf_patch={patch} must divide latent size {clean.shape[-2]}x{clean.shape[-1]}"
        )

    x0_hat = hf_x0_hat(
        model_pred, noisy, timesteps, mode,
        noise_scheduler=noise_scheduler,
        timesteps_in_sigma=timesteps_in_sigma,
        eps_train=eps_train,
    )
    hf_per_sample = hf_per_sample_loss(x0_hat, clean, patch, exponent)  # [B]
    if weighting is not None:
        w_per_sample = weighting.detach().reshape(-1).to(dtype=hf_per_sample.dtype)
        hf_per_sample = hf_per_sample * w_per_sample
    hf_term = hf_per_sample.mean()
    new_loss = final_loss + scale * hf_term
    return new_loss, (scale * hf_term).detach()
