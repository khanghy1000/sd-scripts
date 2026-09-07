"""
Standalone Independent Patch Self-Similarity Topology Loss Module (PatchTopologyLoss).

Computes VAE-free spatial topology matching between predicted representations (latents or noise predictions)
and target representations (ground-truth latents or vision backbone patch features like DINOv3).

Uses chunked query processing, fused log_softmax, and mixed precision BMM for memory-efficient
computation of spatial patch affinity distributions.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_4d_spatial(x: torch.Tensor) -> torch.Tensor:
    """Reshapes input to (N, C, H, W) format if passed as sequence (N, L, C)."""
    if x.ndim == 4:
        return x
    elif x.ndim == 3:
        n, l, c = x.shape
        side = int(math.isqrt(l))
        if side * side == l:
            return x.transpose(1, 2).reshape(n, c, side, side)
        else:
            raise ValueError(f"Cannot infer square spatial dimensions (H, W) from sequence length L={l}.")
    else:
        raise ValueError(f"Expected 3D or 4D tensor for PatchTopologyLoss, got shape {x.shape}.")


def extract_spatial_mask(batch: dict, ref_hw, device, dtype) -> Optional[torch.Tensor]:
    """
    Extracts a per-sample spatial mask from a training batch, mirroring the
    semantics of ``library.custom_train_functions.apply_masked_loss``.

    Resolution order:
      1. ``conditioning_images`` (inpainting): R channel remapped from [-1, 1] to [0, 1].
      2. ``alpha_masks``: assumed already in [0, 1].

    Args:
        batch: training batch dict.
        ref_hw: (H, W) spatial size to resize the mask to.
        device, dtype: target device/dtype.
    Returns:
        (B, 1, H, W) float mask tensor, or None if no mask is present.
    """
    mask_image = None
    if "conditioning_images" in batch and batch["conditioning_images"] is not None:
        # conditioning image is -1 to 1. we need to convert it to 0 to 1
        mask_image = batch["conditioning_images"].to(device=device, dtype=dtype)[:, 0].unsqueeze(1)
        mask_image = mask_image / 2 + 0.5
    elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
        # alpha mask is 0 to 1
        mask_image = batch["alpha_masks"].to(device=device, dtype=dtype)
        if mask_image.ndim == 3:
            mask_image = mask_image.unsqueeze(1)  # add channel dimension

    if mask_image is None:
        return None
    if mask_image.ndim != 4:
        return None
    if tuple(mask_image.shape[2:]) != tuple(ref_hw):
        mask_image = F.interpolate(mask_image.float(), size=ref_hw, mode="area").to(dtype)
    return mask_image


class PatchTopologyLoss(nn.Module):
    """
    Standalone VAE-Free Patch Self-Similarity Topology Loss.

    Compares spatial patch attention distributions of predicted representations
    against target representations across spatial pyramid scale octaves.

    Uses chunked query processing (configurable ``chunk_size``) to reduce peak VRAM
    from O(B·N²) to O(B·K·N), fused ``log_softmax``, and native-precision BMM with
    FP32 upcast only for softmax/log_softmax normalization.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        tau_latent: float = 0.1,
        tau_target: float = 0.1,
        scale_levels: int = 2,
        loss_type: str = "kl",
        apply_channel_norm: bool = True,
        apply_timestep_weight: bool = True,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.tau_latent = tau_latent
        self.tau_target = tau_target
        self.scale_levels = max(1, scale_levels)
        self.loss_type = loss_type.lower()
        self.apply_channel_norm = apply_channel_norm
        self.apply_timestep_weight = apply_timestep_weight
        self.chunk_size = chunk_size

    def _compute_octave_loss(
        self,
        curr_p: torch.Tensor,
        curr_t: torch.Tensor,
        query_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes spatial topology distribution loss for a single scale octave using
        chunked query processing, fused log_softmax, and mixed precision BMM.

        Args:
            curr_p: (B, C_p, H, W) tensor
            curr_t: (B, C_t, H, W) tensor
            query_weights: optional (B, N) per-query-patch weights (e.g. from a
                spatial mask). Loss contributions of each query patch are scaled
                by its weight and normalized by the per-sample weight sum.
        Returns:
            Per-sample loss tensor (B,)
        """
        b, c_p, h, w = curr_p.shape
        c_t = curr_t.shape[1]
        n = h * w
        eps = 1e-8

        if self.apply_channel_norm:
            curr_p_norm = F.normalize(curr_p, p=2, dim=1, eps=1e-8)
            curr_t_norm = F.normalize(curr_t, p=2, dim=1, eps=1e-8)
        else:
            curr_p_norm = curr_p
            curr_t_norm = curr_t

        x_p = curr_p_norm.view(b, c_p, n).transpose(1, 2)  # (B, N, C_p)
        x_t = curr_t_norm.view(b, c_t, n).transpose(1, 2)  # (B, N, C_t)

        chunk_size = self.chunk_size if (self.chunk_size is not None and self.chunk_size > 0) else n

        accumulated_loss = torch.zeros(b, device=curr_p.device, dtype=torch.float32)
        if query_weights is not None:
            query_weights = query_weights.to(device=curr_p.device, dtype=torch.float32)
            accumulated_weight = torch.zeros(b, device=curr_p.device, dtype=torch.float32)

        k_p_t = x_p.transpose(1, 2)  # (B, C_p, N)
        k_t_t = x_t.transpose(1, 2)  # (B, C_t, N)

        tau_p = max(self.tau_latent, 1e-6)
        tau_t = max(self.tau_target, 1e-6)

        for start_idx in range(0, n, chunk_size):
            end_idx = min(start_idx + chunk_size, n)

            # Optimization 1: Chunked query slices (B, K, C)
            q_p = x_p[:, start_idx:end_idx, :]
            q_t = x_t[:, start_idx:end_idx, :]

            # Optimization 3: Native precision BMM
            sim_p = torch.bmm(q_p, k_p_t) / float(tau_p)

            with torch.no_grad():
                sim_t = torch.bmm(q_t, k_t_t) / float(tau_t)
                p_target_chunk = F.softmax(sim_t.to(torch.float32), dim=-1)

            # Optimization 2: Fused log_softmax & direct loss computation in FP32
            sim_p_f32 = sim_p.to(torch.float32)

            if self.loss_type == "kl":
                # KL(P_t || P_p) = sum(P_t * (log(P_t) - log(P_p)))
                log_p_pred_chunk = F.log_softmax(sim_p_f32, dim=-1)
                with torch.no_grad():
                    log_p_target_chunk = torch.log(p_target_chunk + eps)
                kl_elem = p_target_chunk * (log_p_target_chunk - log_p_pred_chunk)
                per_query_loss = kl_elem.sum(dim=-1)  # (B, K)
            elif self.loss_type == "ce":
                # Cross-Entropy = -sum(P_t * log(P_p))
                log_p_pred_chunk = F.log_softmax(sim_p_f32, dim=-1)
                ce_elem = -p_target_chunk * log_p_pred_chunk
                per_query_loss = ce_elem.sum(dim=-1)  # (B, K)
            elif self.loss_type == "cosine":
                p_pred_chunk = F.softmax(sim_p_f32, dim=-1)
                cos_sim = F.cosine_similarity(p_pred_chunk, p_target_chunk, dim=-1)  # (B, K)
                per_query_loss = 1.0 - cos_sim  # (B, K)
            else:  # l2
                p_pred_chunk = F.softmax(sim_p_f32, dim=-1)
                diff_sq = (p_pred_chunk - p_target_chunk) ** 2
                per_query_loss = diff_sq.sum(dim=-1)  # (B, K)

            if query_weights is not None:
                w_chunk = query_weights[:, start_idx:end_idx]  # (B, K)
                accumulated_loss += (per_query_loss * w_chunk).sum(dim=-1)
                accumulated_weight += w_chunk.sum(dim=-1)
            else:
                accumulated_loss += per_query_loss.sum(dim=-1)

        if query_weights is not None:
            # Normalize by the per-sample mask coverage; samples with zero mask
            # coverage contribute zero loss (no valid query patches).
            return torch.where(
                accumulated_weight > eps,
                accumulated_loss / accumulated_weight.clamp(min=eps),
                torch.zeros_like(accumulated_loss),
            )
        return accumulated_loss / float(n)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C_p, H, W) or (B, L, C_p)
            target: (B, C_t, H_t, W_t) or (B, L_t, C_t)
            timesteps: Optional diffusion/flow timesteps tensor for time decay weighting
            mask: Optional spatial weight mask (B, 1, H, W) or (B, H, W), aligned
                with pred's spatial resolution. Query patches are weighted by the
                mask (e.g. for masked / inpainting training) and each sample is
                normalized by its mask coverage.
        Returns:
            Per-sample loss tensor (B,) or scalar mean
        """
        pred_4d = _ensure_4d_spatial(pred)
        target_4d = _ensure_4d_spatial(target).detach()

        orig_dtype = pred_4d.dtype
        batch_size = pred_4d.shape[0]

        total_loss = torch.zeros(batch_size, device=pred_4d.device, dtype=torch.float32)

        mask_4d = None
        if mask is not None:
            mask_4d = mask.detach().to(device=pred_4d.device, dtype=torch.float32)
            if mask_4d.ndim == 3:
                mask_4d = mask_4d.unsqueeze(1)
            if mask_4d.shape[0] != batch_size:
                mask_4d = None
            elif tuple(mask_4d.shape[2:]) != tuple(pred_4d.shape[2:]):
                mask_4d = F.interpolate(mask_4d, size=pred_4d.shape[2:], mode="area")

        curr_p = pred_4d
        curr_t = target_4d
        curr_m = mask_4d

        for scale in range(self.scale_levels):
            if scale > 0:
                curr_p = F.avg_pool2d(curr_p, kernel_size=2, stride=2)
                curr_t = F.avg_pool2d(curr_t, kernel_size=2, stride=2)
                if curr_m is not None:
                    curr_m = F.avg_pool2d(curr_m, kernel_size=2, stride=2)

            # Spatial dimension alignment if target and pred spatial resolutions differ
            if curr_t.shape[2:] != curr_p.shape[2:]:
                curr_t = F.interpolate(curr_t, size=curr_p.shape[2:], mode="bilinear", align_corners=False)

            query_weights = None
            if curr_m is not None:
                mb, _, mh, mw = curr_m.shape
                query_weights = curr_m.view(mb, mh * mw)

            level_loss = self._compute_octave_loss(curr_p, curr_t, query_weights=query_weights)
            scale_weight = 1.0 / (2.0 ** scale)
            total_loss += scale_weight * level_loss

        if self.apply_timestep_weight and timesteps is not None:
            # Normalizes timesteps to continuous t in [0, 1] (handles both [0, 1] and [0, 1000] formats)
            t_float = timesteps.to(torch.float32).view(batch_size)
            t_norm = torch.where(t_float > 1.0, t_float / 1000.0, t_float).clamp(min=0.0, max=1.0)
            t_weight = (1.0 - t_norm).clamp(min=0.0, max=1.0)
            total_loss = total_loss * t_weight

        return (total_loss * self.loss_weight).to(orig_dtype)
