"""
Multiscale MSE x0-prediction loss (the "anchor" term) — shared module for the
sd_scripts train_network family.

Port of `plans/multiscale-mse-x0-pred-loss.md` (§4 reference implementation,
plus the caching / gating / dtype plumbing of §3).

An opt-in auxiliary data loss stacked on top of the primary denoising objective:

    loss_total = loss_primary + anchor_scale * anchor_term

`anchor_term` decomposes the reconstruction error `delta = x0_pred - x0_clean`
into a Laplacian pyramid (`levels` band-pass octaves plus the coarsest Gaussian
residual), computes the MSE of each band per sample, whitens each band's MSE by
that band operator's filter energy (its mean-square response to a unit-variance
white-noise delta), and averages the whitened band losses uniformly. The result
is a frequency-even reconstruction term: a reconstruction error of a given
energy costs the same in every octave band.

Key properties (spec §1 / §6, do not weaken):
  1. Operates in the x0 domain itself (latents for a latent model). No VAE
     decoder, no extra forward pass.
  2. Deterministic: zero draws from the global RNG stream, zero host syncs,
     zero `.item()` calls in the hot path.
  3. `anchor_scale == 0` (the default) is bit-identical to the plain path
     (Python-level gate at the call site).
  4. The pyramid is built ONCE, on the delta (linearity:
     pyramid(pred) - pyramid(clean) == pyramid(pred - clean)).
  5. Blur and energy calibration use the SAME operator (same sigma, same
     padding, same dtype cast) or the whitening is biased.
  6. Every reduction (band MSEs, energies, weights) is promoted to fp32.
  7. Energy calibration uses a dedicated seeded CPU generator (never the
     global RNG); the energy cache is keyed on (H, W, eff, device, dtype).
  8. Graceful small grids: any grid yields a finite term (levels are clamped
     by the reflect-padding `min(dim) >= 4` ladder); no error path, no NaN.
"""

import math

import torch
import torch.nn.functional as F

from library.hf_token_loss import HF_DEFAULT_EPS_TRAIN, hf_x0_hat

# Burt–Adelson a=0.4 binomial kernel [1,4,6,4,1]/16 ≈ Gaussian sigma 1.0.
PYRAMID_BLUR_SIGMA = 1.0

# Dedicated calibration seed (never the global RNG stream).
BAND_ENERGY_SEED = 0x5A17C0DE

# Number of independent unit-variance white-noise fields used to estimate each
# band operator's filter energy. One field's deep-band estimate is too noisy;
# 32 keeps the denominators tight enough that evenness holds per band.
BAND_ENERGY_FIELDS = 32

# Epsilon for the optional soft SNR gate (guards t = 0 and the degenerate
# all-t=1 batch).
ANCHOR_SNR_EPS = 1e-6

_KERNEL_CACHE = {}  # key: (sigma, device, dtype) -> 1-D kernel tensor
_BAND_ENERGY_CACHE = {}  # key: (H, W, eff, device, dtype) -> list of 0-dim fp32 device tensors


def validate_anchor_args(anchor_scale: float, anchor_levels: int) -> None:
    """Validate the anchor configuration surface (spec §7 / §10). Raises ValueError."""
    if anchor_scale < 0.0:
        raise ValueError("anchor_scale must be >= 0 (0 = off)")
    if anchor_levels < 1:
        raise ValueError("anchor_levels must be >= 1 (a DC-only term, levels = 0, is excluded)")


def _gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    """1-D Gaussian kernel, normalized to sum 1 in fp32 BEFORE the cast (DC-preserving)."""
    r = max(1, math.ceil(3.0 * sigma))
    key = (float(sigma), str(device), dtype)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        c = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (c / sigma) ** 2)
        kernel = (kernel / kernel.sum()).to(dtype)
        _KERNEL_CACHE[key] = kernel
    return kernel


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, reflect-padded, depthwise (groups = C).

    sigma = 0 short-circuits to the exact identity (bit-equal): the kernel path
    cannot represent a 0-width Gaussian (0/0 at the center tap is NaN).

    Reflect keeps border statistics on-manifold; it requires every blurred dim
    to be >= radius + 1 (>= 4 for sigma = 1.0) — see `_effective_levels`.
    """
    if sigma <= 0.0:
        return x
    kernel = _gaussian_kernel(sigma, x.device, x.dtype)
    r = kernel.shape[0] // 2
    vertical = kernel.view(1, 1, -1, 1)
    horizontal = kernel.view(1, 1, 1, -1)
    x = F.conv2d(
        F.pad(x, (0, 0, r, r), mode="reflect"),
        vertical.expand(x.shape[1], 1, -1, 1),
        groups=x.shape[1],
    )
    x = F.conv2d(
        F.pad(x, (r, r, 0, 0), mode="reflect"),
        horizontal.expand(x.shape[1], 1, 1, -1),
        groups=x.shape[1],
    )
    return x


def _pyramid_reduce(x: torch.Tensor) -> torch.Tensor:
    """Anti-alias Gaussian blur (sigma = PYRAMID_BLUR_SIGMA) + even-pixel subsample.

    Odd dims keep the last index (ceil rounding): `x[:, :, ::2, ::2]` on a
    ceil-padded grid is exactly ceil(H/2) x ceil(W/2) because slicing stops at
    the last even index — for odd H the last index H-1 is even and included.
    """
    return _gaussian_blur(x, PYRAMID_BLUR_SIGMA)[:, :, ::2, ::2]


def _pyramid_expand(x: torch.Tensor, size) -> torch.Tensor:
    """Bilinear upsample back to `size`, align_corners=False (half-pixel centers)."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _laplacian_pyramid(delta: torch.Tensor, levels: int) -> list:
    """Laplacian pyramid of `delta`: `levels` band-pass octaves (finest first)
    plus the coarsest Gaussian residual (DC / global-look band). Total bands:
    levels + 1.
    """
    pyramid = []
    g = delta
    for _ in range(levels):
        size = (g.shape[-2], g.shape[-1])
        g_down = _pyramid_reduce(g)
        pyramid.append(g - _pyramid_expand(g_down, size))  # band-pass octave k
        g = g_down
    pyramid.append(g)  # coarsest residual (DC)
    return pyramid


def _effective_levels(h: int, w: int, levels: int) -> int:
    """Clamp the requested level count by the reflect-padding `min(dim) >= 4` ladder.

    Host-side integer math, deterministic per (H, W), zero device work. Small
    buckets simply anchor fewer, coarser bands; the residual band always
    exists, so the term is well-defined for any grid — there is no error path.
    """
    eff = 0
    while eff < levels and min(h, w) >= 4:
        eff += 1
        h, w = (h + 1) // 2, (w + 1) // 2
    return eff


def _band_energies(h: int, w: int, eff: int, device, dtype) -> list:
    """Whitening denominators: per-band filter energies of the pyramid operator.

    e_k = E_white[ mean(L_k^2) ] — the mean-square response of band k to a
    UNIT-VARIANCE white-noise delta — estimated with BAND_ENERGY_FIELDS
    independent fields passed through the SAME pyramid with the SAME dtype cast
    the loss uses (fp32 fields are cast to `dtype` BEFORE the pyramid).

    Fields are drawn from a dedicated CPU generator seeded with a fixed
    constant so the global RNG stream is never touched and results are
    reproducible across runs and devices. Computed under torch.no_grad();
    results are 0-dim fp32 device tensors (never `.item()`-ed). Cached per
    (H, W, eff, device, dtype).
    """
    key = (h, w, eff, str(device), dtype)
    energies = _BAND_ENERGY_CACHE.get(key)
    if energies is not None:
        return energies

    generator = torch.Generator(device="cpu")
    generator.manual_seed(BAND_ENERGY_SEED)
    accumulated = None
    with torch.no_grad():
        for _ in range(BAND_ENERGY_FIELDS):
            white = torch.randn(1, 1, h, w, generator=generator, dtype=torch.float32)
            white = white.to(device=device, dtype=dtype)
            band_ms = [band.float().square().mean() for band in _laplacian_pyramid(white, eff)]
            if accumulated is None:
                accumulated = band_ms
            else:
                accumulated = [acc + ms for acc, ms in zip(accumulated, band_ms)]
    energies = [acc / BAND_ENERGY_FIELDS for acc in accumulated]
    _BAND_ENERGY_CACHE[key] = energies
    return energies


def anchor_per_sample_loss(x0_pred: torch.Tensor, x0_clean: torch.Tensor, levels: int) -> torch.Tensor:
    """Per-sample multiscale MSE anchor loss.

    Args:
        x0_pred: [B, C, H, W] predicted clean estimate (in autograd graph)
        x0_clean: [B, C, H, W] clean target (detached)
        levels: requested Laplacian levels (clamped to what the grid supports)

    Returns:
        [B] fp32 per-sample loss (uniform mean over whitened bands).
    """
    h, w = x0_pred.shape[-2], x0_pred.shape[-1]
    eff = _effective_levels(h, w, levels)
    # ONE pyramid, on the delta (linearity: pyramid(pred) - pyramid(clean) == pyramid(pred - clean)).
    pyramid = _laplacian_pyramid(x0_pred - x0_clean, eff)
    energies = _band_energies(h, w, eff, x0_pred.device, x0_pred.dtype)
    per_band = [
        band.float().square().mean(dim=[1, 2, 3]) / energy  # whitened [B], fp32
        for band, energy in zip(pyramid, energies)
    ]
    return torch.stack(per_band, dim=0).mean(dim=0)  # uniform over bands


def anchor_per_sample_from_prediction(
    model_pred: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    mode: str,
    noise_scheduler=None,
    timesteps_in_sigma: bool = False,
    levels: int = 4,
    eps_train: float = HF_DEFAULT_EPS_TRAIN,
) -> torch.Tensor:
    """x0 extraction (first hop, model-specific) + per-sample anchor loss.

    The pyramid runs in the clean target's dtype (the training working dtype);
    every reduction is fp32. Only the first hop is model-specific — see
    `hf_token_loss.hf_x0_hat` for the supported modes.

    Args:
        model_pred: [B, C, H, W] model prediction (in autograd graph)
        clean: [B, C, H, W] clean target (detached constant)
        noisy: [B, C, H, W] forward-diffused input (detached constant)
        timesteps: [B] timesteps
        mode: prediction mode (see hf_x0_hat): "flow", "x0_direct",
            "x0_residual_eps", "vpred_ddpm", "eps_ddpm"
        noise_scheduler: required for discrete-timestep flow and ddpm modes
        timesteps_in_sigma: True when timesteps are already in [0, 1] (e.g. Anima)
        levels: requested Laplacian levels
        eps_train: train-time epsilon for x0-residual models

    Returns:
        [B] fp32 per-sample loss, differentiable only through x0_pred.
    """
    clean = clean.detach()
    working_dtype = clean.dtype
    x0_pred = hf_x0_hat(
        model_pred,
        noisy.detach(),
        timesteps,
        mode,
        noise_scheduler=noise_scheduler,
        timesteps_in_sigma=timesteps_in_sigma,
        eps_train=eps_train,
    )
    x0_pred = x0_pred.to(dtype=working_dtype)
    return anchor_per_sample_loss(x0_pred, clean.to(dtype=working_dtype), levels)


def anchor_snr_weights(
    timesteps: torch.Tensor,
    mode: str,
    noise_scheduler=None,
    timesteps_in_sigma: bool = False,
) -> torch.Tensor:
    """Optional soft SNR(t) gate for the anchor term (spec §3.7).

    w(t) = signal_energy / (noise_energy + eps), then w <- w / mean(w).clamp_min(1e-8).

    - Flow modes: linear-interpolation SNR `(1-t)^2 / (t^2 + 1e-6)` — early
      (noisy, low-SNR) steps weigh less, late (clean, high-SNR) steps weigh more.
    - DDPM modes: the schedule's SNR `alpha_bar / (1 - alpha_bar)`.

    Batch-mean-1 normalization keeps `anchor_scale`'s magnitude semantics
    unchanged: the gate reshapes the term across t, it does not rescale the
    batch average. The eps guard covers t = 0; the clamp_min on the mean covers
    the degenerate all-t = 1 batch (all-zero weights => term 0, finite, no NaN).

    Returns:
        [B] fp32 detached weights on the timesteps' device.
    """
    t = timesteps.detach().to(dtype=torch.float64).reshape(-1)

    if mode in ("flow", "x0_residual_eps") or (
        mode == "x0_direct"
        and (noise_scheduler is None or not hasattr(noise_scheduler, "alphas_cumprod"))
    ):
        if timesteps_in_sigma:
            sigma = t
        else:
            if noise_scheduler is None:
                raise ValueError("noise_scheduler is required for discrete flow timesteps")
            sigma = t / float(noise_scheduler.config.num_train_timesteps)
        sigma = sigma.clamp(min=0.0, max=1.0)
        snr = (1.0 - sigma).square() / (sigma.square() + ANCHOR_SNR_EPS)
    else:
        if noise_scheduler is None or not hasattr(noise_scheduler, "alphas_cumprod"):
            raise ValueError(f"noise_scheduler with alphas_cumprod is required for mode '{mode}'")
        acp = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float64)
        ts = timesteps.detach().long().clamp(0, acp.numel() - 1)
        alpha_bar = acp[ts].clamp(min=0.0, max=1.0)
        snr = alpha_bar / (1.0 - alpha_bar).clamp_min(ANCHOR_SNR_EPS)

    weights = snr / snr.mean().clamp_min(1e-8)
    return weights.to(dtype=torch.float32)
