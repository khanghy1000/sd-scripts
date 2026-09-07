# Chroma Radiance: pixel-space variant of Chroma with NeRF decoder head.
# Reuses Chroma's backbone (DoubleStreamBlock, SingleStreamBlock, Approximator)
# and adds Conv2d patchify + NeRF decoder for pixel-space operation.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from torch import Tensor

from .chroma_models import (
    Chroma,
    Approximator,
    ModulationOut,
    distribute_modulations,
    modify_mask_to_attend_padding,
    _modulation_shift_scale_fn,
    _modulation_gate_fn,
)
from .flux_models import (
    EmbedND,
    RMSNorm,
    timestep_embedding,
)


PATCH_SIZE = 16


# ---------------------------------------------------------------------------
# NeRF decoder components
# ---------------------------------------------------------------------------


class NerfEmbedder(nn.Module):
    """DCT-like positional encoding + MLP for NeRF pixel embeddings."""

    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int, dtype: torch.dtype | None = None):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input)
        )
        self.nerf_embedder_dtype = dtype

    @lru_cache(maxsize=4)
    def fetch_pos(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")

        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        freqs = torch.linspace(0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]

        coeffs = (1 + freqs_x * freqs_y) ** -1
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs**2)
        return dct

    def forward(self, inputs: Tensor) -> Tensor:
        input_dtype = inputs.dtype
        if self.nerf_embedder_dtype is not None:
            inputs = inputs.to(self.nerf_embedder_dtype)
        with torch.autocast("cuda", enabled=False):
            patch_size = int(inputs.shape[1]**0.5)
            inputs = inputs.float()
            dct = self.fetch_pos(patch_size, inputs.device, torch.float32)
            dct = dct.repeat(inputs.shape[0], 1, 1)
            inputs = torch.cat([inputs, dct], dim=-1)
            inputs = self.embedder.float()(inputs)
        return inputs.to(input_dtype)


class NerfGLUBlock(nn.Module):
    """Hypernetwork GLU block: generates per-pixel MLP weights from backbone conditioning."""

    def __init__(self, hidden_size_s: int, hidden_size_x: int, mlp_ratio: int, use_compiled: bool = False):
        super().__init__()
        total_params = 3 * hidden_size_x**2 * mlp_ratio
        self.param_generator = nn.Linear(hidden_size_s, total_params)
        self.norm = RMSNorm(hidden_size_x)
        self.mlp_ratio = mlp_ratio

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        batch_size, num_x, hidden_size_x = x.shape
        mlp_params = self.param_generator(s)

        fc1_gate_params, fc1_value_params, fc2_params = mlp_params.chunk(3, dim=-1)

        fc1_gate = fc1_gate_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc1_value = fc1_value_params.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc2 = fc2_params.view(batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x)

        fc1_gate = F.normalize(fc1_gate, dim=-2)
        fc1_value = F.normalize(fc1_value, dim=-2)
        fc2 = F.normalize(fc2, dim=-2)

        res_x = x
        x = self.norm(x)
        x = torch.bmm(F.silu(torch.bmm(x, fc1_gate)) * torch.bmm(x, fc1_value), fc2)
        x = x + res_x
        return x


class NerfFinalLayerConv(nn.Module):
    """RMSNorm + 3x3 Conv2d output projection for NeRF decoder."""

    def __init__(self, hidden_size: int, out_channels: int, use_compiled: bool = False):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.conv = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        x_permuted = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_permuted)
        x_norm_permuted = x_norm.permute(0, 3, 1, 2)
        x = self.conv(x_norm_permuted)
        return x


# ---------------------------------------------------------------------------
# Position ID helpers (for patch_size=16)
# ---------------------------------------------------------------------------


def prepare_latent_image_ids(batch_size: int, height: int, width: int, patch_size: int = PATCH_SIZE) -> Tensor:
    h_p = height // patch_size
    w_p = width // patch_size
    ids = torch.zeros(h_p, w_p, 3)
    ids[..., 1] = torch.arange(h_p)[:, None]
    ids[..., 2] = torch.arange(w_p)[None, :]
    ids = ids.reshape(1, h_p * w_p, 3).expand(batch_size, -1, -1)
    return ids.contiguous()


def make_text_position_ids(batch_size: int, seq_len: int) -> Tensor:
    ids = torch.zeros(seq_len, 3)
    ids[:, 0] = torch.arange(seq_len)
    return ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass
class ChromaRadianceParams:
    # Image / text
    in_channels: int = 3
    context_in_dim: int = 4096

    # Backbone
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: list[int] = field(default_factory=lambda: [16, 56, 56])
    theta: int = 10_000
    qkv_bias: bool = True

    # Approximator (modulation distillation)
    approximator_in_dim: int = 64
    approximator_depth: int = 5
    approximator_hidden_size: int = 5120

    # NeRF decoder head
    nerf_hidden_size: int = 64
    nerf_mlp_ratio: int = 4
    nerf_depth: int = 4
    nerf_max_freqs: int = 8
    nerf_tile_size: int = 0  # 0 = disabled (process all patches at once)
    nerf_embedder_dtype: torch.dtype | None = None  # None = use model dtype

    patch_size: int = PATCH_SIZE
    _use_compiled: bool = False


chroma_radiance_params = ChromaRadianceParams(
    in_channels=3,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    approximator_in_dim=64,
    approximator_depth=5,
    approximator_hidden_size=5120,
    nerf_hidden_size=64,
    nerf_mlp_ratio=4,
    nerf_depth=4,
    nerf_max_freqs=8,
    nerf_tile_size=0,
    patch_size=PATCH_SIZE,
    _use_compiled=False,
)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ChromaRadiance(Chroma):
    """Pixel-space Chroma variant with NeRF decoder head.

    Reuses Chroma's backbone (DoubleStreamBlock, SingleStreamBlock, Approximator)
    and replaces img_in (Linear) with Conv2d patchify, final_layer (LastLayer)
    with NeRF decoder.

    forward(x, t, txt, txt_mask) -> predicted x0 in [B, 3, H, W].
    """

    def __init__(self, params: ChromaRadianceParams):
        # Build Chroma with in_channels=3 (pixel space)
        # This creates img_in as Linear(3, hidden_size) which we'll replace
        super().__init__(params)
        self.nerf_params = params
        self.patch_size = params.patch_size

        # Replace img_in (Linear) with Conv2d patchify
        self.img_in_patch = nn.Conv2d(
            params.in_channels,
            params.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        nn.init.zeros_(self.img_in_patch.weight)
        nn.init.zeros_(self.img_in_patch.bias)

        # Remove the Linear img_in (not used)
        del self.img_in

        # Replace final_layer with NeRF decoder
        del self.final_layer

        self.nerf_image_embedder = NerfEmbedder(
            in_channels=params.in_channels,
            hidden_size_input=params.nerf_hidden_size,
            max_freqs=params.nerf_max_freqs,
            dtype=params.nerf_embedder_dtype,
        )
        self.nerf_blocks = nn.ModuleList([
            NerfGLUBlock(
                hidden_size_s=params.hidden_size,
                hidden_size_x=params.nerf_hidden_size,
                mlp_ratio=params.nerf_mlp_ratio,
                use_compiled=params._use_compiled,
            )
            for _ in range(params.nerf_depth)
        ])
        self.nerf_final_layer_conv = NerfFinalLayerConv(
            params.nerf_hidden_size,
            out_channels=params.in_channels,
            use_compiled=params._use_compiled,
        )

    def get_model_type(self) -> str:
        return "chroma_radiance"

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

        self.distilled_guidance_layer.enable_gradient_checkpointing()
        for block in self.double_blocks + self.single_blocks:
            block.enable_gradient_checkpointing()

        print(f"ChromaRadiance: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

        self.distilled_guidance_layer.disable_gradient_checkpointing()
        for block in self.double_blocks + self.single_blocks:
            block.disable_gradient_checkpointing()

        print("ChromaRadiance: Gradient checkpointing disabled.")

    def _forward(
        self,
        img: Tensor,       # [B, C, H, W]  noisy image
        img_ids: Tensor,   # [B, N, 3]
        txt: Tensor,       # [B, L, context_in_dim]
        txt_ids: Tensor,   # [B, L, 3]
        txt_mask: Tensor,  # [B, L]  bool
        timesteps: Tensor, # [B]  in [0, 1]
    ) -> Tensor:
        """Returns predicted x0 [B, C, H, W]."""
        if img.ndim != 4:
            raise ValueError("img must be [B, C, H, W]")
        B, C, H, W = img.shape

        # Extract raw patch pixels for NeRF decoder
        nerf_pixels = F.unfold(img, kernel_size=self.patch_size, stride=self.patch_size)
        nerf_pixels = nerf_pixels.transpose(1, 2)  # [B, N, C*P*P]
        num_patches = nerf_pixels.shape[1]

        # Patchify image for transformer backbone
        img_hidden = self.img_in_patch(img)  # [B, hidden, H/P, W/P]
        img_hidden = img_hidden.flatten(2).transpose(1, 2)  # [B, N, hidden]

        # Text projection
        txt_hidden = self.txt_in(txt)  # [B, L, hidden]

        # Compute txt_seq_len per element (Chroma's per-element attention approach)
        txt_emb_len = txt_hidden.shape[1]
        attn_padding = 1
        txt_seq_len = txt_mask[:, :txt_emb_len].sum(dim=-1).to(torch.int64)
        txt_seq_len = torch.clip(txt_seq_len + attn_padding, 0, txt_emb_len)

        # Trim txt embedding to max text length
        max_txt_len = torch.max(txt_seq_len).item()
        txt_hidden = txt_hidden[:, :max_txt_len, :]

        # Distill all modulation vectors (Approximator in no_grad)
        guidance_val = torch.zeros(B, device=img.device, dtype=timesteps.dtype)
        approx_dtype = next(self.distilled_guidance_layer.parameters()).dtype
        with torch.no_grad():
            self.mod_index = self.mod_index.to(img.device)
            distill_timestep = timestep_embedding(timesteps, self.approximator_in_dim // 4)
            distil_guidance = timestep_embedding(guidance_val, self.approximator_in_dim // 4)
            modulation_index = timestep_embedding(self.mod_index, self.approximator_in_dim // 2)
            modulation_index = modulation_index.unsqueeze(0).expand(B, -1, -1)
            timestep_guidance = (
                torch.cat([distill_timestep, distil_guidance], dim=1)
                .unsqueeze(1)
                .expand(-1, self.mod_index_length, -1)
            )
            input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(approx_dtype)
            mod_vectors = self.distilled_guidance_layer(input_vec.requires_grad_(True))

        mod_vectors_dict = distribute_modulations(
            mod_vectors, self.depth_single_blocks, self.depth_double_blocks
        )

        # RoPE position embeddings (img first, then txt — Chroma convention)
        ids = torch.cat((img_ids, txt_ids[:, :max_txt_len]), dim=1)
        pe = self.pe_embedder(ids)


        # Double-stream blocks (Chroma handles checkpointing internally in each block)
        for i, block in enumerate(self.double_blocks):
            img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            double_mod = [img_mod, txt_mod]
            del img_mod, txt_mod

            img_hidden, txt_hidden = block(
                img=img_hidden, txt=txt_hidden,
                pe=pe, distill_vec=double_mod, txt_seq_len=txt_seq_len,
            )
            del double_mod

        # Merge streams for single-stream blocks
        merged = torch.cat((img_hidden, txt_hidden), dim=1)
        del txt_hidden

        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            merged = block(merged, pe=pe, distill_vec=single_mod, txt_seq_len=txt_seq_len)
            del single_mod

        # Strip text prefix — img_hidden is the first num_patches tokens
        img_hidden = merged[:, :num_patches, :]
        del merged

        # NeRF decoder
        nerf_cond = img_hidden.reshape(B * num_patches, self.params.hidden_size)
        del img_hidden

        nerf_pixels = nerf_pixels.reshape(B * num_patches, C, self.patch_size**2)
        nerf_pixels = nerf_pixels.transpose(1, 2)

        # Tiled or full NeRF processing
        tile_size = self.params.nerf_tile_size
        use_gc = self.gradient_checkpointing

        def run_nerf_blocks(dct, cond):
            for block in self.nerf_blocks:
                if use_gc:
                    dct = ckpt.checkpoint(block, dct, cond, use_reentrant=False)
                else:
                    dct = block(dct, cond)
            return dct

        if tile_size > 0 and num_patches > tile_size:
            output_tiles = []
            for i in range(0, num_patches, tile_size):
                end = min(i + tile_size, num_patches)
                hidden_tile = nerf_cond[i * B:end * B]
                pixels_tile = nerf_pixels[i * B:end * B]
                dct_tile = self.nerf_image_embedder(pixels_tile)
                del pixels_tile
                dct_tile = run_nerf_blocks(dct_tile, hidden_tile)
                del hidden_tile
                output_tiles.append(dct_tile)
            del nerf_cond, nerf_pixels
            img_dct = torch.cat(output_tiles, dim=0)
            del output_tiles
        else:
            img_dct = self.nerf_image_embedder(nerf_pixels)
            del nerf_pixels
            img_dct = run_nerf_blocks(img_dct, nerf_cond)
            del nerf_cond

        # Reconstruct image via fold
        img_dct = self.nerf_final_layer_conv.norm(img_dct)
        img_dct = img_dct.transpose(1, 2)
        img_dct = img_dct.reshape(B, num_patches, -1)
        img_dct = img_dct.transpose(1, 2)
        img_dct = F.fold(
            img_dct,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        output = self.nerf_final_layer_conv.conv(img_dct)
        del img_dct

        return output  # predicted x0

    def forward(
        self,
        x: Tensor,         # [B, 3, H, W]  noisy image
        t: Tensor,         # [B]  timestep in [0, 1]
        txt: Tensor,       # [B, L, context_in_dim]
        txt_mask: Tensor,  # [B, L]  bool
    ) -> Tensor:
        """Returns v-prediction [B, 3, H, W]."""
        t = t.view(-1)
        B, C, H, W = x.shape

        img_ids = prepare_latent_image_ids(B, H, W, self.patch_size).to(x.device)
        txt_ids = make_text_position_ids(B, txt.shape[1]).to(x.device)

        predicted_x0 = self._forward(x, img_ids, txt, txt_ids, txt_mask, t)

        # Convert x0 prediction to v-prediction: v = (noisy - x0) / (t + eps)
        eps = 5e-2 if self.training else 0.0
        return (x - predicted_x0) / (t.view(-1, 1, 1, 1) + eps)

    def prepare_block_swap_before_forward(self):
        """No-op for compatibility with training pipeline."""
        pass
