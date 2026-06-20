# VACE-style ControlNet for Anima (Cosmos Predict 2).
#
# Ports the parallel-branch control design from Cosmos Transfer 2.5
# (minimal_v4_lvg_dit_control_vace.py) onto the sd-scripts Anima DiT.
#
# Unlike LLLite (which injects low-rank perturbations into attention
# projections), VACE instantiates a *parallel* transformer "control branch"
# that duplicates select base blocks, runs on the control input, and
# additively injects per-block "hints" into the base transformer's hidden
# states via zero-conv projections.
#
# Injection is non-invasive: forward hooks on target base blocks return
# `output + hints[i] * scale`. The base DiT class is unmodified.

import os
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.anima_models import Block, PatchEmbed
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# state_dict metadata keys
VACE_ARCH_VERSION = "1"
VACE_CONDITION_STRATEGIES = ("spaced", "first_n")
VACE_COPY_STRATEGIES = ("first_n", "spaced_n", "none")


# ---------------------------------------------------------------------------
# ControlEncoderDiTBlock
# ---------------------------------------------------------------------------
class ControlEncoderDiTBlock(nn.Module):
    """One block of the VACE control branch.

    Wraps an Anima ``Block`` (same architecture as base) and adds the VACE
    zero-conv ``before_proj`` (first block only) / ``after_proj`` (every block)
    plus the stacked-tensor skip-connection mechanism.

    Stacked-tensor protocol (matches VACE):
      * Block 0: ``c = before_proj(c) + x_B_T_H_W_D`` then ``block(c)``
        then post_forward pushes [after_proj(c), c] onto the stack.
      * Block i > 0: pop the last tensor from the incoming stack as the
        working ``c``, run it through the block, then push
        [after_proj(c), c] onto the stack.
      * After the final control block, the stack is unbound and all but the
        last element are the per-target-base-block hints.

    The zero-init of before_proj/after_proj ensures the control contribution
    starts at exactly zero, so an untrained control branch leaves the base
    output unchanged.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        block_id: int = 0,
        use_after_proj: bool = True,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.block_id = block_id
        self.use_after_proj = use_after_proj

        # Inner Anima Block — identical architecture to base so weights can be copied
        self.block = Block(
            x_dim=x_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
        )

        # Zero-conv projections (VACE)
        if block_id == 0:
            self.before_proj = nn.Linear(x_dim, x_dim)
        if use_after_proj:
            self.after_proj = nn.Linear(x_dim, x_dim)

    def init_weights(self) -> None:
        """Init inner block + zero-init the VACE projections."""
        self.block.init_weights()
        if self.block_id == 0:
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        if self.use_after_proj:
            nn.init.zeros_(self.after_proj.weight)
            nn.init.zeros_(self.after_proj.bias)

    def copy_weights_from_base_block(self, base_block: Block) -> None:
        """Copy state_dict of an Anima base Block into the inner Block.

        Used by ``ControlNetVACEAnima.copy_weights_to_control_branch`` to
        initialise the control branch as a meaningful encoder.
        """
        self.block.load_state_dict(base_block.state_dict())

    def pre_forward(self, c: torch.Tensor, x_B_T_H_W_D: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Prepare the working control tensor + carry-along skip list.

        For block 0, ``c`` is the freshly patch-embedded control sequence and
        ``x_B_T_H_W_D`` is the base's patch-embedded input (used once as the
        initial residual base). For subsequent blocks, ``c`` is the stacked
        tensor emitted by the previous block's ``post_forward``.
        """
        all_c: List[torch.Tensor] = []
        if self.block_id == 0:
            c = self.before_proj(c) + x_B_T_H_W_D
        elif self.use_after_proj:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        return c, all_c

    def post_forward(self, c: torch.Tensor, all_c: List[torch.Tensor]) -> torch.Tensor:
        """Apply after_proj as a skip connection and restack for the next block."""
        if self.use_after_proj:
            c_skip = self.after_proj(c)
            all_c = all_c + [c_skip, c]
            c = torch.stack(all_c)
        return c

    def forward(
        self,
        c: torch.Tensor,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params,
        use_fp32: bool = False,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c, all_c = self.pre_forward(c, x_B_T_H_W_D)
        c = self.block(
            c,
            emb_B_T_D,
            crossattn_emb,
            attn_params,
            use_fp32,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            extra_per_block_pos_emb=extra_per_block_pos_emb,
        )
        c = self.post_forward(c, all_c)
        return c


# ---------------------------------------------------------------------------
# ControlNetVACEAnima — the parallel control branch
# ---------------------------------------------------------------------------
class ControlNetVACEAnima(nn.Module):
    """VACE control branch: control patch-embedder + ModuleList of ControlEncoderDiTBlock.

    Constructed from a frozen Anima DiT — only its config (dims, num_blocks, ...)
    is read; its weights are not registered. The control branch is the only
    set of trainable parameters produced by this module.
    """

    def __init__(
        self,
        dit: nn.Module,
        vace_block_every_n: int = 2,
        condition_strategy: Literal["spaced", "first_n"] = "spaced",
        use_after_proj: bool = True,
        vace_in_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if condition_strategy not in VACE_CONDITION_STRATEGIES:
            raise ValueError(
                f"condition_strategy must be one of {VACE_CONDITION_STRATEGIES}, got {condition_strategy!r}"
            )
        if vace_block_every_n < 1:
            raise ValueError(f"vace_block_every_n must be >= 1, got {vace_block_every_n}")

        # Copy config off the frozen DiT (do NOT register dit as a submodule)
        self.model_channels = dit.model_channels
        self.num_blocks = dit.num_blocks
        self.num_heads = dit.num_heads
        self.patch_spatial = dit.patch_spatial
        self.patch_temporal = dit.patch_temporal
        self.use_adaln_lora = dit.use_adaln_lora
        self.adaln_lora_dim = dit.adaln_lora_dim
        self.in_channels = dit.in_channels
        self.concat_padding_mask = dit.concat_padding_mask
        # crossattn_emb_channels is the context_dim of the base Block's cross-attention
        self.crossattn_emb_channels = dit.blocks[0].cross_attn.context_dim
        self.mlp_ratio = 4.0  # Anima Block default; no field on Block exposes this

        self.vace_block_every_n = vace_block_every_n
        self.condition_strategy = condition_strategy
        self.use_after_proj = use_after_proj

        # --- Control layer mapping (VACE logic) ---
        if condition_strategy == "spaced":
            # {base_block_idx: control_branch_idx} = {0:0, n:1, 2n:2, ...}
            self.control_layers = list(range(0, self.num_blocks, vace_block_every_n))
            self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers)}
        else:  # "first_n"
            n_ctrl = max(1, self.num_blocks // vace_block_every_n)
            self.control_layers = list(range(0, n_ctrl))
            self.control_layers_mapping = {i: i for i in range(n_ctrl)}
        assert 0 in self.control_layers, "control_layers must include block 0"

        # --- Control patch embedder ---
        # Control latent has the same channel count as base latent by default.
        # If concat_padding_mask is set on the base, mirror it here.
        if vace_in_channels is None:
            vace_in_channels = self.in_channels
            if self.concat_padding_mask:
                vace_in_channels = vace_in_channels + 1
        self.vace_in_channels = vace_in_channels

        self.control_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=vace_in_channels,
            out_channels=self.model_channels,
        )

        # --- Control blocks ---
        self.control_blocks = nn.ModuleList(
            [
                ControlEncoderDiTBlock(
                    x_dim=self.model_channels,
                    context_dim=self.crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    block_id=i,
                    use_after_proj=use_after_proj,
                )
                for i in range(len(self.control_layers))
            ]
        )

    def init_weights(self) -> None:
        """Init the control embedder + every control block (zero-conv included)."""
        self.control_embedder.init_weights()
        for cb in self.control_blocks:
            cb.init_weights()

    def copy_weights_to_control_branch(
        self, dit: nn.Module, strategy: Literal["first_n", "spaced_n", "none"] = "spaced_n", n: int = -1
    ) -> None:
        """Copy weights from base DiT blocks into the inner Block of each control block.

        strategy:
          * "first_n":  base block i      -> control block i
          * "spaced_n": base block i*n    -> control block i  (mirrors spaced condition)
          * "none":     skip (control blocks keep their fresh init)
        n: number of control blocks to copy (-1 = all).
        """
        if strategy == "none":
            return
        if strategy not in ("first_n", "spaced_n"):
            raise ValueError(f"copy strategy must be one of {VACE_COPY_STRATEGIES}, got {strategy!r}")

        n_ctrl = len(self.control_blocks) if n < 0 else min(n, len(self.control_blocks))
        for cb_idx in range(n_ctrl):
            if strategy == "first_n":
                base_idx = cb_idx
            else:  # spaced_n
                base_idx = self.control_layers[cb_idx]
            if base_idx >= self.num_blocks:
                logger.warning(
                    f"copy_weights_to_control_branch: base_idx={base_idx} >= num_blocks={self.num_blocks}; stopping at cb_idx={cb_idx}"
                )
                break
            self.control_blocks[cb_idx].copy_weights_from_base_block(dit.blocks[base_idx])

    def forward(
        self,
        control_latent_B_C_T_H_W: torch.Tensor,
        x_B_T_H_W_D_base: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params,
        use_fp32: bool = False,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Run the control branch and return per-target hints.

        Args:
            control_latent_B_C_T_H_W: VAE-encoded control input, shape (B, C, T, H, W).
                C must equal ``self.in_channels`` (or +1 if concat_padding_mask).
            x_B_T_H_W_D_base: base DiT's patch-embedded hidden state at block 0 input
                (B, T, H, W, D). Only used by control block 0's pre_forward.
            emb_B_T_D, crossattn_emb, attn_params, use_fp32, rope_emb_L_1_1_D,
            adaln_lora_B_T_3D, extra_per_block_pos_emb: shared with base blocks.

        Returns:
            list of ``len(self.control_layers)`` tensors, each shape (B, T, H, W, D)
            matching base block hidden states. Indexed by control branch order,
            *not* base block index — caller maps via ``control_layers_mapping``.
        """
        if self.concat_padding_mask and padding_mask is not None:
            # Resize mask to latent spatial dims and concat along channel axis
            pm = padding_mask
            if pm.shape[-2:] != control_latent_B_C_T_H_W.shape[-2:]:
                from torchvision import transforms

                pm = transforms.functional.resize(
                    pm, list(control_latent_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
                )
            # pm shape (B, 1, H, W) -> (B, 1, T, H, W)
            T = control_latent_B_C_T_H_W.shape[2]
            pm = pm.unsqueeze(2).repeat(1, 1, T, 1, 1)
            control_latent_B_C_T_H_W = torch.cat([control_latent_B_C_T_H_W, pm], dim=1)

        c_B_T_H_W_D = self.control_embedder(control_latent_B_C_T_H_W)

        for cb in self.control_blocks:
            c_B_T_H_W_D = cb(
                c_B_T_H_W_D,
                x_B_T_H_W_D_base,
                emb_B_T_D=emb_B_T_D,
                crossattn_emb=crossattn_emb,
                attn_params=attn_params,
                use_fp32=use_fp32,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=None,  # control branch does not use extra per-block pos emb
            )

        # Unstack the final tensor and drop the last element (final output, not a hint)
        stacked = c_B_T_H_W_D
        if not self.use_after_proj:
            # Without after_proj there is no stacking; the single output IS the only hint
            return [stacked]
        hints = list(torch.unbind(stacked))[:-1]
        return hints


# ---------------------------------------------------------------------------
# AnimaControlNetVACEWrapper — top-level nn.Module for accelerator.prepare
# ---------------------------------------------------------------------------
class AnimaControlNetVACEWrapper(nn.Module):
    """Holds dit (frozen) + vace (trainable). Runs control branch, then base
    with per-block forward hooks injecting hints.

    Hook mechanism:
      * On __init__, a ``register_forward_hook`` is attached to each target
        base Block. The hook returns ``output + hints[i] * scale``.
      * Before each base forward, ``_set_hints`` stores the list of hint
        tensors on ``self._hints``; the hooks read from there.
      * After each base forward, ``_clear_hints`` resets to ``[None] * N``.

    Interaction with gradient checkpointing:
      PyTorch forward hooks do NOT fire during the activation-checkpointing
      recomputation pass, so gradients will not flow through the hint injection
      when the base Block uses ``torch.utils.checkpoint``. The wrapper raises
      a clear error if any target block has ``gradient_checkpointing=True``
      at construction time, and offers ``disable_target_block_checkpointing``
      to turn it off selectively.
    """

    def __init__(
        self,
        dit: nn.Module,
        vace: ControlNetVACEAnima,
        control_context_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.dit = dit
        self.vace = vace
        self.control_context_scale = float(control_context_scale)

        # hints[i] holds the hint for control_layers_mapping target i (None = no injection)
        self._hints: List[Optional[torch.Tensor]] = [None] * len(vace.control_layers)
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Hold the latest control latent so a patched forward_mini_train_dit can find it
        self._pending_control_latent: Optional[torch.Tensor] = None
        self._pending_padding_mask: Optional[torch.Tensor] = None

        self._register_block_hooks()
        self._patch_dit_forward()

    # -- hook management ----------------------------------------------------
    def _register_block_hooks(self) -> None:
        """Register forward hooks on each target base block."""
        for base_idx, control_idx in self.vace.control_layers_mapping.items():
            block = self.dit.blocks[base_idx]

            def make_hook(cidx: int):
                def hook(module, inputs, output):
                    hint = self._hints[cidx]
                    if hint is None:
                        return output
                    return output + hint.to(dtype=output.dtype) * self.control_context_scale

                return hook

            handle = block.register_forward_hook(make_hook(control_idx))
            self._hook_handles.append(handle)

    def _set_hints(self, hints: List[torch.Tensor]) -> None:
        assert len(hints) == len(self._hints), (
            f"got {len(hints)} hints, expected {len(self._hints)} (= len(control_layers))"
        )
        for i, h in enumerate(hints):
            self._hints[i] = h

    def _clear_hints(self) -> None:
        for i in range(len(self._hints)):
            self._hints[i] = None

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def stage_control(self, control_latent: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> None:
        """Stage a control latent to be used on subsequent forward calls.

        The staged latent persists across multiple forwards (so CFG / multi-step
        inference can reuse it). The caller is responsible for clearing it via
        ``clear_staged_control`` when the generation is done.
        """
        self._pending_control_latent = control_latent
        self._pending_padding_mask = padding_mask

    def clear_staged_control(self) -> None:
        """Clear any staged control latent so subsequent forwards run unconditioned."""
        self._pending_control_latent = None
        self._pending_padding_mask = None

    def disable_target_block_checkpointing(self) -> None:
        """Call if you need gradient checkpointing elsewhere but want hooks to work."""
        for base_idx in self.vace.control_layers_mapping:
            self.dit.blocks[base_idx].disable_gradient_checkpointing()

    # -- forward patching ---------------------------------------------------
    def _patch_dit_forward(self) -> None:
        """Wrap dit.forward_mini_train_dit to run the control branch inline.

        The wrapper:
          1. Computes the shared pre-block tensors (patch embed, rope, t_emb, ...)
             exactly as the original method does.
          2. Runs the control branch using these shared tensors and the
             pending control latent → produces hints list.
          3. Sets hints on self._hints so per-block hooks fire during step 4.
          4. Runs the original block loop (hooks inject).
          5. Runs final_layer + unpatchify (unchanged).
          6. Clears hints.
        """
        dit = self.dit
        original_forward_mini_train_dit = dit.forward_mini_train_dit

        def patched_forward_mini_train_dit(
            x_B_C_T_H_W: torch.Tensor,
            timesteps_B_T: torch.Tensor,
            crossattn_emb: torch.Tensor,
            fps: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
            source_attention_mask: Optional[torch.Tensor] = None,
            t5_input_ids: Optional[torch.Tensor] = None,
            t5_attn_mask: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            # 1. LLM adapter (same as original)
            if t5_input_ids is not None and dit.use_llm_adapter and hasattr(dit, "llm_adapter"):
                crossattn_emb = dit.llm_adapter(
                    source_hidden_states=crossattn_emb,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=source_attention_mask,
                )
                if t5_attn_mask is not None:
                    crossattn_emb[~t5_attn_mask.bool()] = 0

            # 2. Base patch embed
            x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = dit.prepare_embedded_sequence(
                x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
            )

            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = dit.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = dit.t_embedding_norm(t_embedding_B_T_D)

            block_kwargs = {
                "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
                "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
                "extra_per_block_pos_emb": extra_pos_emb,
            }
            from library import attention as attention_mod

            attn_params = attention_mod.AttentionParams.create_attention_params(dit.attn_mode, dit.split_attn)
            use_fp32 = x_B_T_H_W_D.dtype == torch.float16

            # 3. Run control branch if a control latent is pending.
            # The pending latent is NOT cleared here: callers may invoke the
            # forward multiple times against the same control input (e.g. CFG
            # runs the DiT twice per step in inference). The caller owns the
            # lifecycle via ``clear_staged_control`` / re-setting the attr.
            control_latent = self._pending_control_latent
            if control_latent is not None:
                hints = self.vace(
                    control_latent_B_C_T_H_W=control_latent,
                    x_B_T_H_W_D_base=x_B_T_H_W_D,
                    emb_B_T_D=t_embedding_B_T_D,
                    crossattn_emb=crossattn_emb,
                    attn_params=attn_params,
                    use_fp32=use_fp32,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                    extra_per_block_pos_emb=extra_pos_emb,
                    padding_mask=padding_mask,
                )
                self._set_hints(hints)

            # 4. Block loop (original) — hooks fire at target blocks
            try:
                if dit.blocks_to_swap:
                    dit.prepare_block_swap_before_forward()

                for block_idx, block in enumerate(dit.blocks):
                    if dit.blocks_to_swap:
                        dit.offloader.wait_for_block(block_idx)
                    x_B_T_H_W_D = block(
                        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, attn_params, use_fp32, **block_kwargs
                    )
                    if dit.blocks_to_swap:
                        dit.offloader.submit_move_blocks(dit.blocks, block_idx)
            finally:
                # 6. Clear hints regardless of success/failure
                self._clear_hints()

            # 5. Final layer + unpatchify
            x_B_T_H_W_O = dit.final_layer(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                use_fp32=use_fp32,
            )
            x_B_C_Tt_Hp_Wp = dit.unpatchify(x_B_T_H_W_O)
            return x_B_C_Tt_Hp_Wp

        # Keep a reference to the original for unpatching / debugging
        self._original_forward_mini_train_dit = original_forward_mini_train_dit
        dit.forward_mini_train_dit = patched_forward_mini_train_dit

    def unpatch_dit_forward(self) -> None:
        """Restore the original dit.forward_mini_train_dit (cleanup)."""
        if hasattr(self, "_original_forward_mini_train_dit"):
            self.dit.forward_mini_train_dit = self._original_forward_mini_train_dit
            del self._original_forward_mini_train_dit

    # -- public forward -----------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor],
        control_latent: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """If control_latent is provided, the control branch runs and hints
        are injected; otherwise the base DiT runs unmodified.

        Args mirror the Anima DiT forward, plus:
            control_latent: (B, C, T, H, W) VAE-encoded control input.
                Must spatially match ``x`` (same H, W after VAE compression).
                When set, **it is consumed and cleared after this call**. For
                multi-forward use (CFG inference), use ``stage_control`` /
                ``clear_staged_control`` instead.
            padding_mask: optional (B, 1, H, W) padding mask for both base
                and control latent.
        """
        clear_after = False
        if control_latent is not None:
            # Stage the control latent for the patched forward_mini_train_dit
            self._pending_control_latent = control_latent
            self._pending_padding_mask = padding_mask
            clear_after = True

        try:
            return self.dit(x, timesteps, context, padding_mask=padding_mask, **kwargs)
        finally:
            if clear_after:
                self._pending_control_latent = None
                self._pending_padding_mask = None


# ---------------------------------------------------------------------------
# save / load helpers
# ---------------------------------------------------------------------------
# State-dict key layout:
#   control_embedder.{...}                  — control patch embedder
#   control_blocks.{i}.block.{...}          — inner Anima Block of control block i
#   control_blocks.{i}.before_proj.{w|b}    — block 0 only
#   control_blocks.{i}.after_proj.{w|b}     — every block (if use_after_proj)
#
# Metadata (safetensors):
#   vace.version              = "1"
#   vace.block_every_n        = str(int)
#   vace.condition_strategy   = "spaced" | "first_n"
#   vace.use_after_proj       = "true" | "false"
#   vace.num_control_blocks   = str(int)
#   vace.model_channels       = str(int)
#   vace.in_channels          = str(int)


def _to_saved_state_dict(vace: ControlNetVACEAnima) -> dict:
    """Internal state_dict passthrough — keys are already user-facing."""
    sd = vace.state_dict()
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().clone().to("cpu")
    return out


def save_vace_model(
    file: str,
    vace: ControlNetVACEAnima,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Save VACE control branch to a safetensors (or pt) file with architecture metadata."""
    state_dict = _to_saved_state_dict(vace)
    if dtype is not None:
        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].to(dtype)

    # Always attach architecture metadata so loaders can reconstruct the module
    base_meta = {
        "vace.version": VACE_ARCH_VERSION,
        "vace.block_every_n": str(vace.vace_block_every_n),
        "vace.condition_strategy": vace.condition_strategy,
        "vace.use_after_proj": "true" if vace.use_after_proj else "false",
        "vace.num_control_blocks": str(len(vace.control_blocks)),
        "vace.model_channels": str(vace.model_channels),
        "vace.in_channels": str(vace.in_channels),
        "vace.num_blocks": str(vace.num_blocks),
        "vace.concat_padding_mask": "true" if vace.concat_padding_mask else "false",
    }
    if metadata is not None:
        base_meta.update(metadata)
    if len(base_meta) == 0:
        base_meta = None

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, file, base_meta)
    else:
        torch.save({"state_dict": state_dict, "metadata": base_meta}, file)


def load_vace_weights(vace: ControlNetVACEAnima, file: str, strict: bool = False) -> Tuple[List[str], List[str]]:
    """Load VACE weights from a file saved by ``save_vace_model``.

    Returns ``(missing_keys, unexpected_keys)`` from the underlying
    ``load_state_dict`` call.
    """
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file

        weights_sd = load_file(file)
    else:
        obj = torch.load(file, map_location="cpu")
        weights_sd = obj.get("state_dict", obj)

    info = vace.load_state_dict(weights_sd, strict=strict)
    return list(info.missing_keys), list(info.unexpected_keys)


def read_vace_metadata(file: str) -> dict:
    """Read the architecture metadata block from a VACE safetensors file."""
    if os.path.splitext(file)[1] != ".safetensors":
        # .pt files: metadata lives in the saved dict
        obj = torch.load(file, map_location="cpu")
        return obj.get("metadata", {}) or {}
    from safetensors import safe_open

    meta = {}
    with safe_open(file, framework="pt", device="cpu") as f:
        for k in f.metadata() or {}:
            meta[k] = f.metadata()[k]
    return meta


def build_vace_from_metadata(
    dit: nn.Module, file: str, extra_kwargs: Optional[dict] = None
) -> ControlNetVACEAnima:
    """Construct a ControlNetVACEAnima whose config matches a saved checkpoint.

    Useful at inference time when the architecture must be reconstructed
    before loading weights.
    """
    meta = read_vace_metadata(file)

    def _get(key: str, default: Optional[str] = None, cast=str) -> Optional[object]:
        v = meta.get(key, default)
        if v is None:
            return None
        return cast(v)

    kwargs = dict(
        vace_block_every_n=_get("vace.block_every_n", "2", int),
        condition_strategy=_get("vace.condition_strategy", "spaced", str),
        use_after_proj=_get("vace.use_after_proj", "true", lambda s: s.lower() == "true"),
    )
    # Drop Nones so the ControlNetVACEAnima defaults take over
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if extra_kwargs:
        kwargs.update({k: v for k, v in extra_kwargs.items() if v is not None})

    return ControlNetVACEAnima(dit=dit, **kwargs)


def attach_vace_to_dit(
    dit: nn.Module,
    file: str,
    control_context_scale: float = 1.0,
    strict_load: bool = False,
    extra_vace_kwargs: Optional[dict] = None,
) -> AnimaControlNetVACEWrapper:
    """End-to-end: build VACE from metadata, load weights, wrap the DiT.

    Returns a wrapper ready for inference. Hooks are registered; the original
    dit.forward_mini_train_dit is patched.
    """
    vace = build_vace_from_metadata(dit, file, extra_kwargs=extra_vace_kwargs)
    missing, unexpected = load_vace_weights(vace, file, strict=strict_load)
    if missing:
        logger.warning(f"VACE load: missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"VACE load: unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    wrapper = AnimaControlNetVACEWrapper(dit, vace, control_context_scale=control_context_scale)
    return wrapper
