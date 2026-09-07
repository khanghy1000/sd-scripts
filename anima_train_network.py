# Anima LoRA training script

import argparse
import json
import os
import random
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    anima_models,
    anima_train_utils,
    anima_utils,
    flux_train_utils,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
    strategy_anima,
    strategy_base,
    train_util,
)
from library.custom_train_functions import (
    apply_snr_weight_for_flow_matching,
    maybe_apply_antithetic_noise_pairing,
)
import library.compile_utils as compile_utils
import train_network
from library.utils import setup_logging
from library.ramtorch_util import apply_ramtorch_to_module

setup_logging()
import logging

logger = logging.getLogger(__name__)


def build_self_reg_anchor_caption(caption, trigger_word, filler="", shuffle=False, separator=","):
    """Build the self-regularization anchor caption from a training caption.

    The anchor carries the same held tags as the training caption and differs
    only in the trigger slot, so whatever the LoRA changes has to be carried by
    the trigger word alone. The main caption itself is left untouched; only the
    anchor is derived here.

    - `caption`, `trigger_word` and `filler` are split on `separator`, stripped,
      and empties are dropped.
    - Every trigger tag is removed from the caption tags (exact tag match,
      case-insensitive) to form `held_tags`.
    - If `shuffle`, `held_tags` are shuffled once so the LoRA cannot key on tag
      position. `filler` tags keep the vacated trigger slot occupied.
    - If the caption contains no trigger tag, the anchor is filler + full tags
      (still a valid hold target).
    """
    def split_tags(text):
        return [t.strip() for t in str(text or "").split(separator) if t.strip()]

    caption_tags = split_tags(caption)
    trigger_tags = {t.lower() for t in split_tags(trigger_word)}
    filler_tags = split_tags(filler)
    held_tags = [t for t in caption_tags if t.lower() not in trigger_tags]
    if shuffle:
        held_tags = list(held_tags)
        random.shuffle(held_tags)
    return ", ".join(filler_tags + held_tags)


def parse_self_reg_trigger_words(trigger_word):
    """Split a comma-separated trigger word string into a list of non-empty tags."""
    return [t.strip() for t in str(trigger_word or "").split(",") if t.strip()]


class AnimaNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        # Anima is a flow-matching model; timesteps are returned already in [0, 1]
        # (divided by 1000), so the HF Tweedie must treat them as sigmas directly.
        self.hf_prediction_mode = "flow"
        self.hf_timesteps_in_sigma = True
        # Same convention for the multiscale x0-prediction anchor loss (flow Tweedie
        # on sigmas in [0, 1]).
        self.anchor_prediction_mode = "flow"
        self.anchor_timesteps_in_sigma = True
        self._ot_logged = False    # fires a one-time first-batch OT log
        self._cfm_logged = False   # fires a one-time first-batch CFM log
        self.ileco_text_encoder_conds = None
        self.ileco_prompt_pairs = None
        self.addift_pair_settings = None
        # Cache of zero padding masks keyed by (batch, height, width, dtype, device).
        # The Anima DiT consumes padding_mask read-only (resize/expand/concat, never
        # mutated in place), so a single reusable buffer per key avoids a fresh CUDA
        # allocation on every forward pass.
        self._padding_mask_cache = {}
        # Anima-only self-regularization state.
        # `_self_reg_step` counts train steps to alternate main/hold steps;
        # `_self_reg_ema` tracks the held-term EMA; `_self_reg_ctx` carries the
        # per-step anchor context from process_batch to get_noise_pred_and_target.
        self._self_reg_step = 0
        self._self_reg_ema = None
        self._self_reg_loss_value = None
        self._self_reg_ctx = None
        self._self_reg_anchor_stash = None
        self._self_reg_batched_warned = False
        self._self_reg_no_trigger_warned = False

    def get_padding_mask(self, batch_size: int, height: int, width: int, dtype: torch.dtype, device) -> torch.Tensor:
        """Return a cached zero padding mask for the given shape/dtype/device.

        Creates and caches the buffer on first use; subsequent calls with the same
        key return the identical tensor, eliminating per-forward allocation churn.
        """
        key = (batch_size, height, width, dtype, str(device))
        mask = self._padding_mask_cache.get(key)
        if mask is None:
            mask = torch.zeros(batch_size, 1, height, width, dtype=dtype, device=device)
            self._padding_mask_cache[key] = mask
        return mask

    def get_adaptive_model_type(self, args) -> str:
        return "flow_matching"

    def build_adaptive_model_fn(self, unet, accelerator, weight_dtype):
        """Build Anima-compatible model_fn for Algorithm 2 (adaptive timestep sampling).

        Handles:
        - 5D latent expansion ([B,C,H,W] -> [B,C,1,H,W])
        - Timestep scaling to [0, 1] range (divided by 1000)
        - Conditioning expansion to match arbitrary batch sizes from the adaptive sampler
        - Anima model's full conditioning signature (prompt_embeds, padding_mask, t5_ids, masks)
        """
        anima_model: anima_models.Anima = unet
        text_conds = self._adaptive_last_text_conds

        # Unpack the 4 core conditioning tensors
        prompt_embeds = text_conds[0].to(accelerator.device, dtype=weight_dtype)
        attn_mask = text_conds[1].to(accelerator.device) if len(text_conds) > 1 else None
        t5_input_ids = text_conds[2].to(accelerator.device, dtype=torch.long) if len(text_conds) > 2 else None
        t5_attn_mask = text_conds[3].to(accelerator.device) if len(text_conds) > 3 else None

        def model_fn(noisy_latents, timesteps, wdtype):
            """Model forward compatible with AdaptiveTimestepManager's contract.

            Args:
                noisy_latents: (N, C, H, W) 4D latents, N may differ from training batch
                timesteps: (N,) integer timesteps in [0, num_train_timesteps)
                wdtype: weight dtype for model inference

            Returns:
                model_output: (N, C, H, W) 4D prediction
            """
            N = noisy_latents.shape[0]

            # Expand conditioning to match N
            if prompt_embeds.shape[0] != N:
                ep = prompt_embeds[:1].expand(N, -1, -1).contiguous()
            else:
                ep = prompt_embeds

            if attn_mask is not None:
                if attn_mask.shape[0] != N:
                    em = attn_mask[:1].expand(N, -1).contiguous()
                else:
                    em = attn_mask
            else:
                em = None

            if t5_input_ids is not None:
                if t5_input_ids.shape[0] != N:
                    et5 = t5_input_ids[:1].expand(N, -1).contiguous()
                else:
                    et5 = t5_input_ids
            else:
                et5 = None

            if t5_attn_mask is not None:
                if t5_attn_mask.shape[0] != N:
                    et5m = t5_attn_mask[:1].expand(N, -1).contiguous()
                else:
                    et5m = t5_attn_mask
            else:
                et5m = None

            # Scale timesteps to [0, 1] range for Anima (Anima expects timesteps / 1000).
            # IMPORTANT: cast to wdtype (bf16/fp16) to match the training path, where
            # flux_train_utils.get_noisy_model_input_and_timesteps returns timesteps in
            # weight_dtype. The sinusoidal embedder inherits the timestep dtype, and the
            # AdaLN block runs with autocast explicitly disabled (enabled=use_fp32), so a
            # float32 embedding would crash against bf16 AdaLN weights.
            anima_ts = (timesteps.float() / 1000.0).to(wdtype)

            # 4D to 5D: [N, C, H, W] -> [N, C, 1, H, W]
            x_5d = noisy_latents.to(wdtype).unsqueeze(2)

            # Create padding mask matching the latent spatial dimensions
            h_latent = noisy_latents.shape[-2]
            w_latent = noisy_latents.shape[-1]
            padding_mask = self.get_padding_mask(N, h_latent, w_latent, wdtype, noisy_latents.device)

            # Autocast is required: the training forward pass runs under
            # accelerator.autocast(), which casts the float32 timestep embedding to
            # the model's weight dtype (bf16/fp16). Without it, bf16 AdaLN linear
            # layers receive float32 inputs and raise a dtype mismatch error.
            with torch.no_grad(), accelerator.autocast():
                output = anima_model(
                    x_5d,
                    anima_ts,
                    ep,
                    padding_mask=padding_mask,
                    target_input_ids=et5,
                    target_attention_mask=et5m,
                    source_attention_mask=em,
                )

            # 5D to 4D: [N, C, 1, H, W] -> [N, C, H, W]
            return output.squeeze(2)

        return model_fn

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        # --- Flow matching feature ---
        if getattr(args, "flow_use_ot", False):
            logger.info("[Anima] Cosine Optimal Transport (OT): ENABLED -- noise vectors will be batch-reassigned each step.")

        if getattr(args, "contrastive_flow_matching", False):
            logger.info(
                f"[Anima] Contrastive Flow Matching (\u0394FM): ENABLED -- "
                f"cfm_lambda={getattr(args, 'cfm_lambda', 0.05)}. "
                "A negative contrastive loss term will be subtracted from the main loss each step."
            )

        if args.fp8_base or args.fp8_base_unet:
            logger.warning("fp8_base and fp8_base_unet are not supported. / fp8_baseとfp8_base_unetはサポートされていません。")
            args.fp8_base = False
            args.fp8_base_unet = False
        args.fp8_scaled = False  # Anima DiT does not support fp8_scaled

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert train_dataset_group.is_text_encoder_output_cacheable(
                cache_supports_dropout=True
            ), "when caching Text Encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used (use --cache_caption_variants for all but token_warmup_step)"

        if args.ileco:
            assert not args.addift, "--ileco and --addift cannot be enabled at the same time"
            assert (
                args.ileco_prompt_pairs is not None or args.ileco_target_prompt is not None
            ), "--ileco_target_prompt or --ileco_prompt_pairs is required when --ileco is enabled"
            assert args.ileco_loss_weight > 0, "--ileco_loss_weight must be greater than 0"
            assert args.reverse_weight > 0, "--reverse_weight must be greater than 0"
            if args.ileco_min_sigma is not None or args.ileco_max_sigma is not None:
                min_sigma = 0.0 if args.ileco_min_sigma is None else args.ileco_min_sigma
                max_sigma = 1.0 if args.ileco_max_sigma is None else args.ileco_max_sigma
                assert 0.0 <= min_sigma < max_sigma <= 1.0, "--ileco_min_sigma/max_sigma must satisfy 0 <= min < max <= 1"
            if not args.network_train_unet_only:
                logger.warning(
                    "--ileco trains through Anima DiT only. "
                    "Forcing --network_train_unet_only to avoid keeping the text encoder on GPU."
                )
                args.network_train_unet_only = True
            if not args.cache_latents:
                logger.warning(
                    "--ileco without --cache_latents keeps the VAE on GPU during training. "
                    "Enable --cache_latents or --cache_latents_to_disk to reduce VRAM usage."
                )
            if not args.gradient_checkpointing:
                logger.warning(
                    "--ileco without --gradient_checkpointing can use much more VRAM."
                )
        if args.addift:
            for dataset in train_dataset_group.datasets:
                assert hasattr(
                    dataset, "controlnet_subsets"
                ), "ADDifT requires a ControlNet-style dataset with conditioning_data_dir in every subset"
            assert args.addift_loss_weight > 0, "--addift_loss_weight must be greater than 0"
            assert args.reverse_weight > 0, "--reverse_weight must be greater than 0"
            if args.addift_mask_loss:
                assert (
                    args.addift_mask_data_dir is not None
                    or args.addift_alpha_mask is not None
                    or self.has_addift_mask_config(train_dataset_group)
                ), (
                    "--addift_mask_data_dir or --addift_alpha_mask is required in CLI or dataset TOML "
                    "when --addift_mask_loss is enabled"
                )
            if args.addift_pair_settings is not None:
                for dataset in train_dataset_group.datasets:
                    if getattr(dataset, "batch_size", 1) != 1:
                        logger.warning(
                            "--addift_pair_settings is most accurate with batch_size=1 because LoRA multiplier is global per forward pass."
                        )
            if args.addift_min_sigma is not None or args.addift_max_sigma is not None:
                min_sigma = 0.0 if args.addift_min_sigma is None else args.addift_min_sigma
                max_sigma = 1.0 if args.addift_max_sigma is None else args.addift_max_sigma
                assert 0.0 <= min_sigma < max_sigma <= 1.0, "--addift_min_sigma/max_sigma must satisfy 0 <= min < max <= 1"
            if not args.network_train_unet_only:
                logger.warning(
                    "--addift trains through Anima DiT only. "
                    "Forcing --network_train_unet_only to avoid keeping the text encoder on GPU."
                )
                args.network_train_unet_only = True
            if not args.cache_latents:
                logger.warning(
                    "--addift without --cache_latents keeps the VAE on GPU for target image latents. "
                    "Enable --cache_latents or --cache_latents_to_disk to reduce VRAM usage."
                )

        if getattr(args, "self_reg_weight", 0.0):
            self.validate_self_reg_args(args, train_dataset_group)

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
                args.gradient_checkpointing = True
            assert (
                not args.cpu_offload_checkpointing
            ), "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
            assert (
                args.blocks_to_swap is None or args.blocks_to_swap == 0
            ), "blocks_to_swap is not supported with unsloth_offload_checkpointing"

        if args.compile:
            assert not args.torch_compile, (
                "--compile (per-block torch.compile) and --torch_compile (accelerate dynamo) cannot be used together"
                " / --compile（ブロック単位torch.compile）と--torch_compile（accelerate dynamo）は併用できません"
            )
            assert not (args.compile_fullgraph and args.split_attn), (
                "--compile_fullgraph cannot be used with --split_attn (split attention uses dynamic control flow)"
                " / --compile_fullgraphは--split_attnと併用できません（split attentionは動的な制御フローを使用します）"
            )

        train_dataset_group.verify_bucket_reso_steps(16)  # WanVAE spatial downscale = 8 and patch size = 2
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

    @staticmethod
    def is_self_reg_enabled(args) -> bool:
        return float(getattr(args, "self_reg_weight", 0.0) or 0.0) > 0.0

    def validate_self_reg_args(self, args, train_dataset_group=None):
        """Validate Anima-only self-regularization arguments.

        Constraints: LoRA only, explicit trigger word, no cached text encoder
        outputs (the anchor is live-encoded each step).
        """
        triggers = parse_self_reg_trigger_words(getattr(args, "self_reg_trigger_word", ""))
        assert triggers, "--self_reg_trigger_word is required when --self_reg_weight > 0"
        noise_p = float(getattr(args, "self_reg_noise", 0.0) or 0.0)
        assert 0.0 <= noise_p <= 1.0, "--self_reg_noise must be in [0, 1]"
        assert not getattr(args, "ileco", False), "--self_reg_weight cannot be combined with --ileco"
        assert not getattr(args, "addift", False), "--self_reg_weight cannot be combined with --addift"
        if args.cache_text_encoder_outputs:
            raise ValueError(
                "--self_reg_weight requires live text encoder encoding; "
                "--cache_text_encoder_outputs is not supported with self-regularization"
            )
        if not args.network_train_unet_only:
            logger.warning(
                "--self_reg_weight trains through Anima DiT only. "
                "Forcing --network_train_unet_only to avoid keeping the text encoder on GPU."
            )
            args.network_train_unet_only = True

        effective_batch_sizes = []
        if train_dataset_group is not None:
            datasets = getattr(train_dataset_group, "datasets", [train_dataset_group])
            for dataset in datasets:
                subsets = getattr(dataset, "subsets", None)
                if subsets:
                    for subset in subsets:
                        bs = getattr(subset, "batch_size", None)
                        if bs is not None:
                            effective_batch_sizes.append(int(bs))
                bs = getattr(dataset, "batch_size", None)
                if bs is not None:
                    effective_batch_sizes.append(int(bs))
        if not effective_batch_sizes:
            effective_batch_sizes.append(int(getattr(args, "train_batch_size", 1)))

        if getattr(args, "self_reg_batched", False):
            if max(effective_batch_sizes) < 2:
                # Mirror source trainer/train.py: fall back to alternating steps.
                logger.warning(
                    "Self-regularization needs a batch of two or more to hold both halves in one step, alternating instead."
                )
            elif min(effective_batch_sizes) < 2:
                logger.info(
                    "Self-regularization batched mode enabled, but some dataset subsets have batch size < 2; "
                    "those subsets will alternate instead."
                )
        logger.info(
            "[Anima self-reg] ENABLED -- weight=%s, trigger=%s, filler=%s, noise=%s, batched=%s, shuffle_tags=%s. "
            "A starting weight of 1.0 is a good default; alternating mode needs ~2x iterations.",
            getattr(args, "self_reg_weight", 0.0),
            getattr(args, "self_reg_trigger_word", ""),
            getattr(args, "self_reg_filler", ""),
            getattr(args, "self_reg_noise", 0.0),
            getattr(args, "self_reg_batched", False),
            getattr(args, "self_reg_shuffle_tags", False),
        )

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

        # Load Qwen3 text encoder (tokenizers already loaded in get_tokenize_strategy)
        logger.info("Loading Qwen3 text encoder...")
        qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
        qwen3_text_encoder.eval()

        if args.use_ramtorch and not args.cache_text_encoder_outputs:
            qwen3_text_encoder= apply_ramtorch_to_module(qwen3_text_encoder, "qwen3_text_encoder", accelerator.device, weight_dtype)

        # Load VAE
        logger.info("Loading Anima VAE...")
        vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
        vae.to(weight_dtype)
        vae.eval()

        # Return format: (model_type, text_encoders, vae, unet)
        return "anima", [qwen3_text_encoder], vae, None  # unet loaded lazily

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        loading_dtype = None if args.fp8_scaled else weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        # Load DiT
        logger.info(f"Loading Anima DiT model with attn_mode={attn_mode}, split_attn: {args.split_attn}...")
        model = anima_utils.load_anima_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            args.split_attn,
            loading_device,
            loading_dtype,
            args.fp8_scaled,
        )

        # Store unsloth preference so that when the base NetworkTrainer calls
        # dit.enable_gradient_checkpointing(cpu_offload=...), we can override to use unsloth.
        # The base trainer only passes cpu_offload, so we store the flag on the model.
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

        if args.use_ramtorch:
            logger.info("Applying RamTorch to Anima model dit.")
            model = apply_ramtorch_to_module(model, "unet/dit", accelerator.device, model.dtype)

        # Block swap
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        return model, text_encoders

    def get_tokenize_strategy(self, args):
        # Load tokenizers from paths (called before load_target_model, so self.qwen3_tokenizer isn't set yet)
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        return tokenize_strategy

    def get_tokenizers(self, tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check,
            cache_dtype=getattr(args, "cache_latents_dtype", "auto"),
        )

    def get_text_encoding_strategy(self, args):
        return strategy_anima.AnimaTextEncodingStrategy()

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False,
                cache_dtype=getattr(args, "cache_text_encoder_outputs_dtype", "auto"),
            )
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        self.apply_addift_pair_settings_if_needed(args, dataset)
        self.cache_addift_conditioning_latents_if_needed(args, accelerator, vae, dataset, weight_dtype)

        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # We cannot move DiT to CPU because of block swap, so only move VAE
                logger.info("move vae to cpu to save memory")
                org_vae_device = vae.device
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device)

            tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
            text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
            self.cache_ileco_text_encoder_outputs_if_needed(
                args, accelerator, text_encoders, text_encoding_strategy, tokenize_strategy, weight_dtype
            )

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"  cache TE outputs for: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens_and_masks
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # move text encoder back to cpu
            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")

            if not args.lowram:
                logger.info("move vae back to original device")
                vae.to(org_vae_device)

            clean_memory_on_device(accelerator.device)
        else:
            # move text encoder to device for encoding during training/validation
            text_encoders[0].to(accelerator.device)

    def load_addift_pair_settings(self, args):
        if self.addift_pair_settings is not None:
            return self.addift_pair_settings
        if args.addift_pair_settings is None:
            self.addift_pair_settings = {}
            return self.addift_pair_settings

        with open(args.addift_pair_settings, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("pairs", data) if isinstance(data, dict) else data

        settings = {}
        if isinstance(entries, dict):
            iterable = entries.items()
        elif isinstance(entries, list):
            iterable = []
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("ADDifT pair settings list entries must be objects")
                key = entry.get("key", entry.get("stem", entry.get("image", entry.get("filename", None))))
                if key is None:
                    raise ValueError("ADDifT pair settings entry requires key, stem, image, or filename")
                iterable.append((key, entry))
        else:
            raise ValueError("--addift_pair_settings must be an object, a pairs object, or a list")

        for key, entry in iterable:
            if not isinstance(entry, dict):
                raise ValueError(f"ADDifT pair setting for {key} must be an object")
            weight = float(entry.get("weight", 1.0))
            reverse_weight = float(entry.get("reverse_weight", args.reverse_weight))
            if weight <= 0:
                raise ValueError(f"ADDifT pair setting for {key} weight must be greater than 0")
            if reverse_weight <= 0:
                raise ValueError(f"ADDifT pair setting for {key} reverse_weight must be greater than 0")
            settings[os.path.splitext(os.path.basename(str(key)))[0]] = {
                "weight": weight,
                "multiplier": float(entry.get("multiplier", args.addift_multiplier)),
                "reverse_weight": reverse_weight,
                "reverse_multiplier": float(entry.get("reverse_multiplier", args.reverse_multiplier)),
            }

        self.addift_pair_settings = settings
        return self.addift_pair_settings

    def apply_addift_pair_settings_if_needed(self, args, dataset):
        if not args.addift:
            return

        settings = self.load_addift_pair_settings(args)
        for ds in dataset.datasets:
            if not hasattr(ds, "dreambooth_dataset_delegate"):
                continue
            for info in ds.dreambooth_dataset_delegate.image_data.values():
                stem = os.path.splitext(os.path.basename(info.absolute_path))[0]
                pair_setting = settings.get(stem, {})
                info.addift_pair_weight = float(pair_setting.get("weight", 1.0))
                info.addift_pair_multiplier = float(pair_setting.get("multiplier", args.addift_multiplier))
                info.addift_pair_reverse_weight = float(pair_setting.get("reverse_weight", args.reverse_weight))
                info.addift_pair_reverse_multiplier = float(pair_setting.get("reverse_multiplier", args.reverse_multiplier))

                if args.addift_mask_loss:
                    subset = self.get_addift_subset_for_image(ds, info.image_key)
                    mask_data_dir = getattr(subset, "addift_mask_data_dir", None) or args.addift_mask_data_dir
                    alpha_mask_mode = getattr(subset, "addift_alpha_mask", None) or args.addift_alpha_mask
                    if mask_data_dir is not None:
                        mask_path = self.find_addift_mask_path(mask_data_dir, stem)
                        if mask_path is None:
                            raise ValueError(f"ADDifT mask not found for {stem} in {mask_data_dir}")
                        info.addift_mask_path = mask_path
                    elif alpha_mask_mode is not None:
                        info.addift_alpha_mask = alpha_mask_mode
                    else:
                        raise ValueError(f"ADDifT mask directory or alpha mask mode is not configured for {stem}")

    def get_addift_subset_for_image(self, dataset, image_key):
        db_subset = dataset.dreambooth_dataset_delegate.image_to_subset[image_key]
        for subset in getattr(dataset, "controlnet_subsets", []):
            if subset.image_dir == db_subset.image_dir:
                return subset
        return db_subset

    def has_addift_mask_config(self, dataset):
        for ds in dataset.datasets:
            for subset in getattr(ds, "controlnet_subsets", getattr(ds, "subsets", [])):
                if getattr(subset, "addift_mask_data_dir", None) is not None or getattr(subset, "addift_alpha_mask", None) is not None:
                    return True
        return False

    def find_addift_mask_path(self, mask_dir, stem):
        for ext in train_util.IMAGE_EXTENSIONS:
            path = os.path.join(mask_dir, stem + ext)
            if os.path.exists(path):
                return path
        return None

    def cache_addift_conditioning_latents_if_needed(self, args, accelerator, vae, dataset, weight_dtype):
        if not args.addift or not args.addift_cache_conditioning_latents:
            return

        logger.info("cache ADDifT conditioning latents")
        org_vae_device = vae.device
        org_vae_dtype = vae.dtype
        if accelerator.is_main_process:
            vae.to(accelerator.device, dtype=weight_dtype)
            vae.requires_grad_(False)
            vae.eval()

        try:
            for ds in dataset.datasets:
                if not hasattr(ds, "dreambooth_dataset_delegate") or not hasattr(ds, "get_conditioning_image_tensor"):
                    continue

                image_infos = list(ds.dreambooth_dataset_delegate.image_data.values())
                uncached = []
                for info in image_infos:
                    bucket_w, bucket_h = info.bucket_reso
                    cache_path = os.path.splitext(info.cond_img_path)[0] + f"_{bucket_w:04d}x{bucket_h:04d}_addift_anima.npz"
                    info.addift_conditioning_latents_npz = cache_path
                    if not accelerator.is_main_process:
                        continue
                    if args.skip_cache_check and os.path.exists(cache_path):
                        continue
                    if os.path.exists(cache_path):
                        try:
                            data = train_util.load_npz(cache_path)
                            if "latents" in data and "latents_flipped" in data:
                                continue
                        except Exception:
                            pass
                    uncached.append(info)

                if not uncached:
                    continue

                batch_size = args.vae_batch_size or 1
                for i in range(0, len(uncached), batch_size):
                    batch_infos = uncached[i : i + batch_size]
                    images = []
                    flipped_images = []
                    for info in batch_infos:
                        target_size_hw = (info.bucket_reso[1], info.bucket_reso[0])
                        original_size_hw = (info.image_size[1], info.image_size[0])
                        images.append(ds.get_conditioning_image_tensor(info, target_size_hw, original_size_hw, False))
                        flipped_images.append(ds.get_conditioning_image_tensor(info, target_size_hw, original_size_hw, True))

                    image_tensor = torch.stack(images).to(accelerator.device, dtype=weight_dtype)
                    flipped_image_tensor = torch.stack(flipped_images).to(accelerator.device, dtype=weight_dtype)
                    with torch.no_grad(), accelerator.autocast():
                        latents = self.encode_images_to_latents(args, vae, image_tensor).to("cpu")
                        flipped_latents = self.encode_images_to_latents(args, vae, flipped_image_tensor).to("cpu")

                    for info, latent, flipped_latent in zip(batch_infos, latents, flipped_latents):
                        train_util.save_npz(
                            info.addift_conditioning_latents_npz,
                            {"latents": latent, "latents_flipped": flipped_latent},
                            cache_dtype=getattr(args, "cache_latents_dtype", "auto"),
                        )
        finally:
            if accelerator.is_main_process:
                vae.to(org_vae_device, dtype=org_vae_dtype)
                clean_memory_on_device(accelerator.device)
            accelerator.wait_for_everyone()

    def load_ileco_prompt_pairs(self, args):
        if self.ileco_prompt_pairs is not None:
            return self.ileco_prompt_pairs

        if args.ileco_prompt_pairs is None:
            pairs = [
                {
                    "original": args.ileco_original_prompt or "",
                    "target": args.ileco_target_prompt,
                    "weight": 1.0,
                    "multiplier": 1.0,
                }
            ]
        else:
            with open(args.ileco_prompt_pairs, "r", encoding="utf-8") as f:
                data = json.load(f)
            pairs = data.get("pairs", data) if isinstance(data, dict) else data

        if not isinstance(pairs, list) or len(pairs) == 0:
            raise ValueError("--ileco_prompt_pairs must contain at least one prompt pair")

        normalized_pairs = []
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                raise ValueError(f"iLECO prompt pair #{i} must be an object")
            original = pair.get("original", pair.get("original_prompt", pair.get("source", pair.get("source_prompt", ""))))
            target = pair.get("target", pair.get("target_prompt", None))
            if target is None:
                raise ValueError(f"iLECO prompt pair #{i} does not have target or target_prompt")
            weight = float(pair.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(f"iLECO prompt pair #{i} weight must be greater than 0")
            normalized_pairs.append(
                {
                    "original": str(original or ""),
                    "target": str(target),
                    "weight": weight,
                    "multiplier": float(pair.get("multiplier", 1.0)),
                }
            )

        if args.add_reverse_pairs:
            reverse_pairs = []
            for pair in normalized_pairs:
                reverse_pairs.append(
                    {
                        "original": pair["target"],
                        "target": pair["original"],
                        "weight": args.reverse_weight,
                        "multiplier": args.reverse_multiplier,
                    }
                )
            normalized_pairs.extend(reverse_pairs)

        self.ileco_prompt_pairs = normalized_pairs
        return self.ileco_prompt_pairs

    def cache_ileco_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator,
        text_encoders,
        text_encoding_strategy,
        tokenize_strategy,
        weight_dtype,
    ):
        if not args.ileco or self.ileco_text_encoder_conds is not None:
            return

        models = text_encoders if args.cache_text_encoder_outputs else self.get_models_for_text_encoding(args, accelerator, text_encoders)
        if models is None:
            raise ValueError("Text encoder models are not available for iLECO prompt encoding")

        pairs = self.load_ileco_prompt_pairs(args)
        prompts = []
        for pair in pairs:
            prompts.extend([pair["original"], pair["target"]])
        logger.info(f"cache iLECO Text Encoder outputs: {len(pairs)} prompt pair(s)")

        tokens_and_masks = tokenize_strategy.tokenize(prompts)
        tokens_and_masks = [t.to(accelerator.device) for t in tokens_and_masks]
        with torch.no_grad(), accelerator.autocast():
            encoded = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens_and_masks)

        if args.full_fp16:
            encoded = [t.to(weight_dtype) if t is not None and t.dtype.is_floating_point else t for t in encoded]

        self.ileco_text_encoder_conds = []
        for i, pair in enumerate(pairs):
            original_index = i * 2
            target_index = original_index + 1
            self.ileco_text_encoder_conds.append(
                {
                    "original": [t[original_index : original_index + 1].detach() if t is not None else None for t in encoded],
                    "target": [t[target_index : target_index + 1].detach() if t is not None else None for t in encoded],
                    "weight": pair["weight"],
                    "multiplier": pair["multiplier"],
                }
            )

    def repeat_ileco_text_encoder_conds(self, conds, batch_size, device, weight_dtype):
        repeated = []
        for t in conds:
            if t is None:
                repeated.append(None)
                continue
            dtype = weight_dtype if t.dtype.is_floating_point else t.dtype
            t = t.to(device=device, dtype=dtype)
            if t.shape[0] == 1 and batch_size > 1:
                t = t.repeat(batch_size, *([1] * (t.ndim - 1)))
            elif t.shape[0] != batch_size:
                raise ValueError(f"Unexpected iLECO Text Encoder batch size: {t.shape[0]} != {batch_size}")
            repeated.append(t)
        return repeated

    def apply_limited_sigma_range(self, args, noise_scheduler, latents, noise, sigmas, min_sigma_arg, max_sigma_arg, dtype):
        if min_sigma_arg is None and max_sigma_arg is None:
            return None, None, None

        min_sigma = 0.0 if min_sigma_arg is None else min_sigma_arg
        max_sigma = 1.0 if max_sigma_arg is None else max_sigma_arg
        sigmas = min_sigma + sigmas * (max_sigma - min_sigma)
        timesteps = sigmas.flatten() * noise_scheduler.config.num_train_timesteps

        if args.ip_noise_gamma:
            xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
            if args.ip_noise_gamma_random_strength:
                ip_noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma
            else:
                ip_noise_gamma = args.ip_noise_gamma
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * (noise + ip_noise_gamma * xi)
        else:
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

        return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas

    def get_ileco_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        unet,
        network,
        weight_dtype,
        is_train=True,
    ):
        # iLECO is a teacher-distillation path: the training target is the teacher's
        # velocity, so there is no analytic clean x0 for the HF token term — skip it.
        self._hf_noisy_latents = None
        # Same for the multiscale x0-prediction anchor loss (no analytic clean x0).
        self._anchor_noisy_latents = None
        anima: anima_models.Anima = unet

        if self.ileco_text_encoder_conds is None:
            raise ValueError("iLECO Text Encoder outputs are not cached")
        if network is None or not hasattr(network, "set_multiplier"):
            raise ValueError("iLECO training requires a network with set_multiplier() support")

        if latents.ndim == 5:
            latents = latents.squeeze(2)
        noise = torch.randn_like(latents)
        noise = maybe_apply_antithetic_noise_pairing(args, noise)
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )
        ranged_noisy_model_input, ranged_timesteps, ranged_sigmas = self.apply_limited_sigma_range(
            args,
            noise_scheduler,
            latents,
            noise,
            sigmas,
            args.ileco_min_sigma,
            args.ileco_max_sigma,
            weight_dtype,
        )
        if ranged_noisy_model_input is not None:
            noisy_model_input = ranged_noisy_model_input
            timesteps = ranged_timesteps
            sigmas = ranged_sigmas
        timesteps = timesteps / 1000.0

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)

        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask = self.get_padding_mask(bs, h_latent, w_latent, weight_dtype, accelerator.device)
        noisy_model_input = noisy_model_input.unsqueeze(2)

        pair_index = torch.randint(len(self.ileco_text_encoder_conds), (1,)).item()
        pair_conds = self.ileco_text_encoder_conds[pair_index]
        original_conds = self.repeat_ileco_text_encoder_conds(
            pair_conds["original"], bs, accelerator.device, weight_dtype
        )
        target_conds = self.repeat_ileco_text_encoder_conds(pair_conds["target"], bs, accelerator.device, weight_dtype)
        original_prompt_embeds, original_attn_mask, original_t5_input_ids, original_t5_attn_mask = original_conds[:4]
        target_prompt_embeds, target_attn_mask, target_t5_input_ids, target_t5_attn_mask = target_conds[:4]

        old_multiplier = getattr(network, "multiplier", 1.0)
        try:
            network.set_multiplier(0.0)
            with torch.no_grad(), accelerator.autocast():
                target = anima(
                    noisy_model_input,
                    timesteps,
                    target_prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=target_t5_input_ids,
                    target_attention_mask=target_t5_attn_mask,
                    source_attention_mask=target_attn_mask,
                )

            network.set_multiplier(pair_conds["multiplier"])
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                model_pred = anima(
                    noisy_model_input,
                    timesteps,
                    original_prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=original_t5_input_ids,
                    target_attention_mask=original_t5_attn_mask,
                    source_attention_mask=original_attn_mask,
                )
        finally:
            network.set_multiplier(old_multiplier)

        model_pred = model_pred.squeeze(2)
        target = target.detach().squeeze(2)

        weighting = anima_train_utils.compute_loss_weighting_for_anima(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        weighting = weighting * args.ileco_loss_weight * pair_conds["weight"]

        return model_pred, target, timesteps, weighting, noise

    def get_addift_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        is_train=True,
    ):
        # ADDifT is a teacher-distillation path: the student regresses the teacher's
        # velocity (x0_hat converges to a sigma-dependent blend), so there is no analytic
        # clean x0 for the HF token term — skip it.
        self._hf_noisy_latents = None
        # Same for the multiscale x0-prediction anchor loss (no analytic clean x0).
        self._anchor_noisy_latents = None
        anima: anima_models.Anima = unet

        if network is None or not hasattr(network, "set_multiplier"):
            raise ValueError("ADDifT training requires a network with set_multiplier() support")
        if "addift_conditioning_latents" not in batch or batch["addift_conditioning_latents"] is None:
            raise ValueError("ADDifT training requires paired conditioning latents. Use a dataset with conditioning_data_dir.")

        if latents.ndim == 5:
            latents = latents.squeeze(2)
        target_latents = latents
        conditioning_latents = batch["addift_conditioning_latents"].to(accelerator.device, dtype=latents.dtype)
        if conditioning_latents.ndim == 5:
            conditioning_latents = conditioning_latents.squeeze(2)
        if conditioning_latents.shape != target_latents.shape:
            raise ValueError(f"ADDifT conditioning latent shape mismatch: {conditioning_latents.shape} != {target_latents.shape}")

        is_reverse_pair = args.add_reverse_pairs and torch.randint(2, (1,)).item() == 1
        pair_weights = batch.get("addift_pair_weights", None)
        pair_multipliers = batch.get("addift_pair_multipliers", None)
        pair_reverse_weights = batch.get("addift_pair_reverse_weights", None)
        pair_reverse_multipliers = batch.get("addift_pair_reverse_multipliers", None)
        if is_reverse_pair:
            source_latents = target_latents
            teacher_latents = conditioning_latents
            if pair_reverse_multipliers is not None:
                pair_reverse_multipliers = pair_reverse_multipliers.to(accelerator.device)
                student_multiplier = pair_reverse_multipliers.mean().item()
            else:
                student_multiplier = args.reverse_multiplier
            if pair_reverse_weights is not None:
                addift_pair_weight = pair_reverse_weights.to(accelerator.device, dtype=weight_dtype).view(-1, 1, 1, 1)
            else:
                addift_pair_weight = args.reverse_weight
        else:
            source_latents = conditioning_latents
            teacher_latents = target_latents
            if pair_multipliers is not None:
                pair_multipliers = pair_multipliers.to(accelerator.device)
                student_multiplier = pair_multipliers.mean().item()
            else:
                student_multiplier = args.addift_multiplier
            if pair_weights is not None:
                addift_pair_weight = pair_weights.to(accelerator.device, dtype=weight_dtype).view(-1, 1, 1, 1)
            else:
                addift_pair_weight = 1.0

        noise = torch.randn_like(source_latents)
        noise = maybe_apply_antithetic_noise_pairing(args, noise)
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, source_latents, noise, accelerator.device, weight_dtype
        )
        ranged_noisy_model_input, ranged_timesteps, ranged_sigmas = self.apply_limited_sigma_range(
            args,
            noise_scheduler,
            source_latents,
            noise,
            sigmas,
            args.addift_min_sigma,
            args.addift_max_sigma,
            weight_dtype,
        )
        if ranged_noisy_model_input is not None:
            noisy_model_input = ranged_noisy_model_input
            timesteps = ranged_timesteps
            sigmas = ranged_sigmas
        target_noisy_model_input = ((1.0 - sigmas) * teacher_latents + sigmas * noise).to(weight_dtype)
        timesteps = timesteps / 1000.0

        addift_mask = None
        if args.addift_mask_loss:
            addift_mask = batch.get("addift_masks", None)
            if addift_mask is None:
                raise ValueError("ADDifT mask loss requires addift_masks in batch")
            addift_mask = addift_mask.to(accelerator.device, dtype=weight_dtype)
            addift_mask = torch.nn.functional.interpolate(addift_mask, size=latents.shape[-2:], mode="area")

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)

        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds[:4]
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)
        t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask = self.get_padding_mask(bs, h_latent, w_latent, weight_dtype, accelerator.device)
        noisy_model_input = noisy_model_input.unsqueeze(2)
        target_noisy_model_input = target_noisy_model_input.unsqueeze(2)

        old_multiplier = getattr(network, "multiplier", 1.0)
        try:
            network.set_multiplier(0.0)
            with torch.no_grad(), accelerator.autocast():
                target = anima(
                    target_noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=attn_mask,
                )

            network.set_multiplier(student_multiplier)
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                model_pred = anima(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=attn_mask,
                )
        finally:
            network.set_multiplier(old_multiplier)

        model_pred = model_pred.squeeze(2)
        target = target.detach().squeeze(2)

        weighting = anima_train_utils.compute_loss_weighting_for_anima(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        weighting = weighting * args.addift_loss_weight * addift_pair_weight
        if addift_mask is not None:
            weighting = weighting * addift_mask

        return model_pred, target, timesteps, weighting, noise

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]  # compatibility
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            qwen3_te,
            tokenize_strategy,
            text_encoding_strategy,
            self.sample_prompts_te_outputs,
        )

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        return vae.encode_pixels_to_latents(images)  # Keep 4D for input/output

    def shift_scale_latents(self, args, latents):
        # Latents already normalized by vae.encode with scale
        return latents

    def anima_predict(self, anima, noisy_5d, timesteps_scaled, conds, padding_mask):
        """Single Anima DiT forward returning a 4D prediction.

        Callers are responsible for the surrounding autocast / grad context.
        `conds` is the 4-tensor Anima conditioning
        (prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask).
        """
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = conds[:4]
        return anima(
            noisy_5d,
            timesteps_scaled,
            prompt_embeds,
            padding_mask=padding_mask,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        ).squeeze(2)

    def encode_self_reg_conds(
        self, args, accelerator, text_encoders, anchor_captions, tokenize_strategy, text_encoding_strategy, weight_dtype
    ):
        """Live-encode self-reg anchor captions through the frozen text encoder.

        Mirrors the non-cached main path (tokenize -> move ids to device ->
        encode_tokens under no_grad/autocast). The Anima TE is always frozen, so
        no grad context is needed here.
        """
        tokens_and_masks = tokenize_strategy.tokenize(anchor_captions)
        tokens_and_masks = [t.to(accelerator.device) for t in tokens_and_masks]
        models = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        if models is None:
            raise ValueError("Text encoder models are not available for self-regularization anchor encoding")
        with torch.no_grad(), accelerator.autocast():
            encoded = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens_and_masks)
        if getattr(args, "full_fp16", False):
            encoded = [c.to(weight_dtype) for c in encoded]
        return encoded

    def move_self_reg_conds(self, conds, accelerator, weight_dtype):
        """Move anchor conds to device/dtype exactly like the main conds."""
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = conds[:4]
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)
        t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(accelerator.device)
        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask] + list(conds[4:])

    def get_self_reg_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        is_train=True,
    ):
        """Self-regularization forward.

        At the same noisy point in the latent, with the trigger word taken out
        of the caption, the LoRA's anchor prediction has to match what the
        frozen model predicted there: `loss = main + weight * MSE(LoRA(anchor),
        frozen(anchor))`. The loss is exactly zero while the LoRA is zero, so
        there is no floor to drift on.

        Mode comes from `self._self_reg_ctx` (built in `process_batch`):
        - "hold": odd alternating steps train on the hold term only. Returns
          (anchor_pred, teacher_pred, timesteps, weighting * weight, noise) so
          the base loss pipeline computes the hold loss directly.
        - "together": same-step half-batch. Returns the main triple on the
          first half and stashes the held-term scalar in
          `self._self_reg_anchor_stash` for `process_batch` to add.
        """
        anima: anima_models.Anima = unet
        ctx = self._self_reg_ctx or {}
        mode = ctx.get("mode", "hold")
        half = ctx.get("half", None)
        anchor_conds = ctx.get("anchor_conds", None)
        if anchor_conds is None:
            raise ValueError("Self-regularization anchor conds are not prepared")
        if network is None or not hasattr(network, "set_multiplier"):
            raise ValueError("Self-regularization training requires a network with set_multiplier() support")

        # Teacher-distillation hold term: no analytic clean x0 exists for the
        # auxiliary x0-based losses on this branch — skip them like iLECO.
        self._hf_noisy_latents = None
        self._anchor_noisy_latents = None
        self._noisy_latents = None

        if latents.ndim == 5:
            latents = latents.squeeze(2)

        if mode == "hold" and float(getattr(args, "self_reg_noise", 0.0) or 0.0) > 0:
            if random.random() < float(args.self_reg_noise):
                # Somewhere the training images never reach (pure-noise anchor).
                latents = torch.randn_like(latents)

        # Shared noise path (keeps flow_use_ot, antithetic pairing, adaptive
        # sampler and grad checkpointing consistent for both sides).
        noise = torch.randn_like(latents)

        if getattr(args, "flow_use_ot", False) and latents.size(0) > 1:
            with torch.no_grad():
                b_size = latents.size(0)
                lat_flat = latents.view(b_size, -1)
                noise_flat = noise.view(b_size, -1)
                _, (_, col_indices) = train_util.cosine_optimal_transport(lat_flat, noise_flat)
                noise = noise[col_indices.squeeze(0)]

        noise = maybe_apply_antithetic_noise_pairing(args, noise, is_train=is_train)

        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args,
            noise_scheduler,
            latents,
            noise,
            accelerator.device,
            weight_dtype,
            fixed_timesteps=None,
            is_train=is_train,
        )

        # Set T-LoRA timestep mask after any together-mode truncation below (the
        # mask expects [0, max_timestep]-range timesteps matching the forwards).
        timesteps_scaled = timesteps / 1000.0  # scale to [0, 1] range

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)

        main_conds = self.move_self_reg_conds(text_encoder_conds, accelerator, weight_dtype)
        anchor_conds = self.move_self_reg_conds(anchor_conds, accelerator, weight_dtype)
        if args.gradient_checkpointing:
            for t in list(main_conds) + list(anchor_conds):
                if t is not None and torch.is_tensor(t) and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        if mode == "together":
            if half is None or half < 1:
                raise ValueError("Self-regularization together mode requires half >= 1")
            latents = latents[:half]
            noise = noise[:half]
            noisy_model_input = noisy_model_input[:half]
            timesteps_scaled = timesteps_scaled[:half]
            timesteps = timesteps[:half]
            sigmas = sigmas[:half]
            main_conds = [
                t[:half] if torch.is_tensor(t) and t.shape[0] >= half else t for t in main_conds
            ]
            anchor_conds = [
                t[:half] if torch.is_tensor(t) and t.shape[0] >= half else t for t in anchor_conds
            ]

        self.apply_tlora_mask(timesteps)

        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        bs = latents.shape[0]
        padding_mask = self.get_padding_mask(bs, h_latent, w_latent, weight_dtype, accelerator.device)
        noisy_5d = noisy_model_input.unsqueeze(2)  # 4D to 5D

        if mode == "together":
            # The main-half stores keep the base auxiliaries (HF / wavelet /
            # multiscale anchor / CFM) working on the main term exactly as
            # without self-reg.
            if is_train and getattr(self, "wavelet_masking_enabled", False):
                self._noisy_latents = noisy_model_input.detach()
            if is_train and self.hf_scale > 0.0:
                self._hf_noisy_latents = noisy_model_input.detach()
            if is_train and self.anchor_scale > 0.0:
                self._anchor_noisy_latents = noisy_model_input.detach()

        old_multiplier = getattr(network, "multiplier", 1.0)
        try:
            network.set_multiplier(0.0)
            with torch.no_grad(), accelerator.autocast():
                base_pred = self.anima_predict(anima, noisy_5d, timesteps_scaled, anchor_conds, padding_mask)
                base_pred = base_pred.detach()

            network.set_multiplier(old_multiplier)
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                if mode != "hold":
                    model_pred = self.anima_predict(anima, noisy_5d, timesteps_scaled, main_conds, padding_mask)
                anchor_pred = self.anima_predict(anima, noisy_5d, timesteps_scaled, anchor_conds, padding_mask)
        finally:
            network.set_multiplier(old_multiplier)

        self.clear_tlora_mask_if_needed()

        base_weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=args.weighting_scheme, sigmas=sigmas
        )
        reg_weight = float(getattr(args, "self_reg_weight", 0.0) or 0.0)

        if mode == "hold":
            weighting = base_weighting * reg_weight
            return anchor_pred, base_pred, timesteps_scaled, weighting, noise

        # together: main triple flows through the base loss pipeline; the held
        # term is stashed for process_batch to add.
        latents_f64 = latents.to(torch.float64)
        noise_f64 = noise.to(torch.float64)
        target = noise_f64 - latents_f64  # rectified flow target

        huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
        anchor_loss_elem = train_util.conditional_loss(
            anchor_pred, base_pred, args.loss_type, "none", huber_c, scale=float(args.loss_scale)
        )
        anchor_loss_elem = anchor_loss_elem * (base_weighting * reg_weight)
        anchor_per_sample = anchor_loss_elem.mean(dim=list(range(1, anchor_loss_elem.ndim)))
        anchor_per_sample = anchor_per_sample * batch["loss_weights"]
        anchor_per_sample = self.post_process_loss(anchor_per_sample, args, timesteps, noise_scheduler)
        self._self_reg_anchor_stash = anchor_per_sample.mean()

        return model_pred, target, timesteps_scaled, base_weighting, noise

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        text_encoder_masks,
        unet,
        network,
        weight_dtype,
        train_unet,
        fixed_timesteps=None,
        is_train=True,
    ):
        anima: anima_models.Anima = unet

        if args.ileco:
            return self.get_ileco_noise_pred_and_target(
                args,
                accelerator,
                noise_scheduler,
                latents,
                unet,
                network,
                weight_dtype,
                is_train=is_train,
            )

        if args.addift:
            return self.get_addift_noise_pred_and_target(
                args,
                accelerator,
                noise_scheduler,
                latents,
                batch,
                text_encoder_conds,
                unet,
                network,
                weight_dtype,
                is_train=is_train,
            )

        if is_train and self._self_reg_ctx is not None:
            return self.get_self_reg_noise_pred_and_target(
                args,
                accelerator,
                noise_scheduler,
                latents,
                batch,
                text_encoder_conds,
                unet,
                network,
                weight_dtype,
                is_train=is_train,
            )

        # Sample noise
        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]

        # Adaptive timestep sampling: use Beta distribution sampler if enabled.
        # Sample timesteps BEFORE noise so we can store all data for Algorithm 2 after noise is computed.
        adaptive_fixed_timesteps = fixed_timesteps
        if is_train and self.adaptive_manager is not None and fixed_timesteps is None:
            adaptive_fixed_timesteps = self.adaptive_manager.sample_timesteps(
                latents, noise_scheduler.config.num_train_timesteps
            )
            # Store latents and args for Algorithm 2. Noise will be stored after it is computed.
            # Only pin tensors when this is an update step to avoid wasting VRAM.
            if self._adaptive_update_pending:
                self._adaptive_last_latents = latents.detach()
                self._adaptive_last_args = args

        noise = torch.randn_like(latents)

        if getattr(args, "flow_use_ot", False) and latents.size(0) > 1:
            with torch.no_grad():
                b_size = latents.size(0)
                lat_flat = latents.view(b_size, -1)
                noise_flat = noise.view(b_size, -1)
                _, (_, col_indices) = train_util.cosine_optimal_transport(lat_flat, noise_flat)
                noise = noise[col_indices.squeeze(0)]
            if not self._ot_logged:
                logger.info(
                    f"[Anima OT] First batch: noise reordered by cosine OT. "
                    f"New noise assignment indices: {col_indices.squeeze(0).tolist()}"
                )
                self._ot_logged = True

        # Antithetic noise pairing (after OT so pair structure matches sigmas)
        noise = maybe_apply_antithetic_noise_pairing(args, noise, is_train=is_train)

        # Now that noise is computed, store it for Algorithm 2
        if is_train and self.adaptive_manager is not None and fixed_timesteps is None and self._adaptive_update_pending:
            self._adaptive_last_noise = noise.detach()

        # Get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args,
            noise_scheduler,
            latents,
            noise,
            accelerator.device,
            weight_dtype,
            fixed_timesteps=adaptive_fixed_timesteps,
            is_train=is_train,
        )
        # Store noisy latents for LWD wavelet masking (used in process_batch via base class)
        # Must be stored BEFORE unsqueeze to 5D (wavelet DWT expects 4D)
        if is_train and getattr(self, "wavelet_masking_enabled", False):
            self._noisy_latents = noisy_model_input.detach()

        # Store noisy latents for High-Frequency Token loss (4D, before 5D unsqueeze)
        if is_train and self.hf_scale > 0.0:
            self._hf_noisy_latents = noisy_model_input.detach()

        # Store noisy latents for the multiscale x0-prediction anchor loss (4D, before 5D unsqueeze)
        if is_train and self.anchor_scale > 0.0:
            self._anchor_noisy_latents = noisy_model_input.detach()

        # Set T-LoRA timestep mask before timestep scaling (mask expects [0, max_timestep] range)
        self.apply_tlora_mask(timesteps)

        timesteps = timesteps / 1000.0  # scale to [0, 1] range. timesteps is float32

        # Gradient checkpointing support
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Unpack text encoder conditions
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds[
            :4
        ]  # ignore caption_dropout_rate which is not needed for training step

        # Move to device
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)
        t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

        # Create padding mask
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask = self.get_padding_mask(bs, h_latent, w_latent, weight_dtype, accelerator.device)

        # Call model
        noisy_model_input = noisy_model_input.unsqueeze(2)  # 4D to 5D, [B, C, H, W] -> [B, C, 1, H, W]
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = anima(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )
        model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, 1, H, W] -> [B, C, H, W]

        # Clear T-LoRA mask after the forward pass
        self.clear_tlora_mask_if_needed()

        # Upcast for grokking
        latents = latents.to(torch.float64)
        noise = noise.to(torch.float64)

        # Rectified flow target: noise - latents
        target = noise - latents

        # Loss weighting
        weighting = anima_train_utils.compute_loss_weighting_for_anima(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        return model_pred, target, timesteps, weighting, noise

    def process_batch(
        self, 
        batch, 
        text_encoders, 
        unet, 
        network, 
        vae, 
        noise_scheduler,
        vae_dtype, 
        weight_dtype, 
        accelerator, 
        args,
        text_encoding_strategy, 
        tokenize_strategy,
        is_train=True, 
        train_text_encoder=True, 
        train_unet=True,
        edm2_model=None,
    ) -> torch.Tensor:
        """Override base process_batch for caption dropout with cached text encoder outputs."""

        self.cache_ileco_text_encoder_outputs_if_needed(
            args, accelerator, text_encoders, text_encoding_strategy, tokenize_strategy, weight_dtype
        )

        if args.addift:
            if batch.get("addift_conditioning_latents") is not None:
                batch["addift_conditioning_latents"] = batch["addift_conditioning_latents"].to(accelerator.device)
            elif "conditioning_images" not in batch or batch["conditioning_images"] is None:
                raise ValueError("ADDifT requires a dataset with conditioning_data_dir")
            else:
                if vae.device != accelerator.device:
                    vae.to(accelerator.device, dtype=vae_dtype)
                    vae.requires_grad_(False)
                    vae.eval()

                with torch.no_grad():
                    conditioning_images = batch["conditioning_images"]
                    if args.vae_batch_size is None or len(conditioning_images) <= args.vae_batch_size:
                        target_latents = self.encode_images_to_latents(
                            args, vae, conditioning_images.to(accelerator.device, dtype=vae_dtype)
                        )
                    else:
                        chunks = [
                            conditioning_images[i : i + args.vae_batch_size]
                            for i in range(0, len(conditioning_images), args.vae_batch_size)
                        ]
                        target_latents_list = []
                        for chunk in chunks:
                            target_latents_list.append(
                                self.encode_images_to_latents(args, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                            )
                        target_latents = torch.cat(target_latents_list, dim=0)

                    if torch.any(torch.isnan(target_latents)):
                        accelerator.print("NaN found in ADDifT conditioning latents, replacing with zeros")
                        target_latents = torch.nan_to_num(target_latents, 0, out=target_latents)

                    batch["addift_conditioning_latents"] = self.shift_scale_latents(args, target_latents)

        # Text encoder conditions
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        anima_text_encoding_strategy: strategy_anima.AnimaTextEncodingStrategy = text_encoding_strategy
        if text_encoder_outputs_list is not None:
            caption_dropout_rates = text_encoder_outputs_list[-1]
            text_encoder_outputs_list = text_encoder_outputs_list[:-1]

            # Apply caption dropout to cached outputs
            text_encoder_outputs_list = anima_text_encoding_strategy.drop_cached_text_encoder_outputs(
                *text_encoder_outputs_list, caption_dropout_rates=caption_dropout_rates
            )
            # Add the caption dropout rates back to the list for validation dataset (which is re-used batch items)
            batch["text_encoder_outputs_list"] = text_encoder_outputs_list + [caption_dropout_rates]

        if is_train and not self._cfm_logged and getattr(args, "contrastive_flow_matching", False):
            logger.info(
                f"[Anima CFM] First batch: Contrastive Flow Matching is active. "
                f"Negative rolled targets will be computed and subtracted with lambda={getattr(args, 'cfm_lambda', 0.05)}."
            )
            self._cfm_logged = True

        self_reg_ctx = self.prepare_self_reg_step(
            args, accelerator, text_encoders, batch, text_encoding_strategy, tokenize_strategy, weight_dtype,
            is_train=is_train,
        )
        if self_reg_ctx is None:
            return super().process_batch(
                batch,
                text_encoders,
                unet,
                network,
                vae,
                noise_scheduler,
                vae_dtype,
                weight_dtype,
                accelerator,
                args,
                text_encoding_strategy,
                tokenize_strategy,
                is_train,
                train_text_encoder,
                train_unet,
                edm2_model,
            )

        saved = self.apply_self_reg_batch_mutations(args, batch, self_reg_ctx)
        try:
            final_loss, pre_scaling_loss, loss_scaled = super().process_batch(
                batch,
                text_encoders,
                unet,
                network,
                vae,
                noise_scheduler,
                vae_dtype,
                weight_dtype,
                accelerator,
                args,
                text_encoding_strategy,
                tokenize_strategy,
                is_train,
                train_text_encoder,
                train_unet,
                edm2_model,
            )
            held = None
            if self_reg_ctx["mode"] == "together":
                if self._self_reg_anchor_stash is None:
                    raise ValueError("Self-regularization together step did not produce a held-term loss")
                final_loss = final_loss + self._self_reg_anchor_stash
                pre_scaling_loss = pre_scaling_loss + self._self_reg_anchor_stash
                held = self._self_reg_anchor_stash.detach()
            else:  # hold step: the base pre-scaling loss IS the held term
                held = pre_scaling_loss.detach()
            if held is not None:
                held_value = float(held.float().mean().item())
                if self._self_reg_ema is None:
                    self._self_reg_ema = held_value
                else:
                    self._self_reg_ema = self._self_reg_ema * 0.9 + held_value * 0.1
                self._self_reg_loss_value = held_value
            return final_loss, pre_scaling_loss, loss_scaled
        finally:
            self.restore_self_reg_batch_mutations(args, batch, saved)
            self._self_reg_ctx = None
            self._self_reg_anchor_stash = None

    def prepare_self_reg_step(
        self, args, accelerator, text_encoders, batch, text_encoding_strategy, tokenize_strategy, weight_dtype,
        is_train=True,
    ):
        """Decide the self-reg mode for this step and live-encode anchors.

        Returns a context dict consumed by `get_self_reg_noise_pred_and_target`,
        or None when this step runs the plain base path (self-reg disabled,
        validation, or an even alternating "main" step).
        """
        if not is_train or not self.is_self_reg_enabled(args):
            return None
        if getattr(args, "cache_text_encoder_outputs", False):
            raise ValueError(
                "--self_reg_weight requires live text encoder encoding; "
                "--cache_text_encoder_outputs is not supported with self-regularization"
            )
        captions = batch.get("captions", None)
        if not captions:
            return None

        self._self_reg_step += 1
        batch_size = len(captions)
        batched = bool(getattr(args, "self_reg_batched", False))
        if batched and batch_size < 2:
            if not self._self_reg_batched_warned:
                logger.warning(
                    "Self-regularization needs a batch of two or more to hold both halves in one step, alternating instead"
                )
                self._self_reg_batched_warned = True
            batched = False

        if batched:
            mode = "together"
            half = batch_size // 2
        elif self._self_reg_step % 2 == 1:
            mode = "hold"
            half = None
        else:
            return None  # even alternating step: plain main path, no anchor overhead

        trigger_word = str(getattr(args, "self_reg_trigger_word", "") or "")
        filler = str(getattr(args, "self_reg_filler", "") or "")
        shuffle_tags = bool(getattr(args, "self_reg_shuffle_tags", False))
        anchor_captions = [
            build_self_reg_anchor_caption(caption, trigger_word, filler, shuffle_tags) for caption in captions
        ]
        if not self._self_reg_no_trigger_warned and parse_self_reg_trigger_words(trigger_word):
            lowered = [t.lower() for t in parse_self_reg_trigger_words(trigger_word)]
            for caption in captions:
                tags = [t.strip().lower() for t in str(caption or "").split(",")]
                if not any(t in lowered for t in tags):
                    logger.debug(
                        "Self-regularization: caption contains no trigger tag; "
                        "the anchor still holds filler + full tags."
                    )
                    break
            self._self_reg_no_trigger_warned = True

        anchor_conds = self.encode_self_reg_conds(
            args, accelerator, text_encoders, anchor_captions, tokenize_strategy, text_encoding_strategy, weight_dtype
        )
        self._self_reg_ctx = {"mode": mode, "half": half, "anchor_conds": anchor_conds}
        return self._self_reg_ctx

    def apply_self_reg_batch_mutations(self, args, batch, ctx):
        """Align per-sample batch entries with the truncated together halves.

        Both halves derive from samples [:half], so loss_weights / alpha_masks
        are sliced to match. The adaptive timestep sampler is gated off for the
        step to keep its (latents, conds) bookkeeping consistent. On hold steps
        the contrastive flow-matching term is disabled since the whole step is
        the teacher branch.
        """
        saved = {}
        if ctx["mode"] == "together":
            half = ctx["half"]
            if "loss_weights" in batch and batch["loss_weights"] is not None:
                saved["loss_weights"] = batch["loss_weights"]
                batch["loss_weights"] = batch["loss_weights"][:half]
            if batch.get("alpha_masks") is not None:
                saved["alpha_masks"] = batch["alpha_masks"]
                batch["alpha_masks"] = batch["alpha_masks"][:half]
            if getattr(self, "_adaptive_update_pending", False):
                saved["_adaptive_update_pending"] = True
                self._adaptive_update_pending = False
        else:  # hold
            if getattr(args, "contrastive_flow_matching", False):
                saved["contrastive_flow_matching"] = True
                args.contrastive_flow_matching = False
        return saved

    def restore_self_reg_batch_mutations(self, args, batch, saved):
        if "loss_weights" in saved:
            batch["loss_weights"] = saved["loss_weights"]
        if "alpha_masks" in saved:
            batch["alpha_masks"] = saved["alpha_masks"]
        if saved.get("_adaptive_update_pending", False):
            self._adaptive_update_pending = True
        if saved.get("contrastive_flow_matching", False):
            args.contrastive_flow_matching = True

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        if args.min_snr_gamma:
            # Anima timesteps are already in [0, 1] range (sigmas) — they are divided by 1000
            # in get_noise_pred_and_target before being returned
            loss = apply_snr_weight_for_flow_matching(loss, timesteps, args.min_snr_gamma, soft=args.min_snr_gamma_soft)
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(None, args, False, True, False, anima="preview").to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        metadata["ss_ileco"] = args.ileco
        if args.ileco:
            metadata["ss_ileco_original_prompt"] = args.ileco_original_prompt
            metadata["ss_ileco_target_prompt"] = args.ileco_target_prompt
            metadata["ss_ileco_prompt_pairs"] = args.ileco_prompt_pairs
            metadata["ss_ileco_loss_weight"] = args.ileco_loss_weight
            metadata["ss_add_reverse_pairs"] = args.add_reverse_pairs
            metadata["ss_reverse_multiplier"] = args.reverse_multiplier
            metadata["ss_reverse_weight"] = args.reverse_weight
            metadata["ss_ileco_min_sigma"] = args.ileco_min_sigma
            metadata["ss_ileco_max_sigma"] = args.ileco_max_sigma
        metadata["ss_addift"] = args.addift
        if args.addift:
            metadata["ss_addift_loss_weight"] = args.addift_loss_weight
            metadata["ss_addift_multiplier"] = args.addift_multiplier
            metadata["ss_addift_pair_settings"] = args.addift_pair_settings
            metadata["ss_addift_mask_loss"] = args.addift_mask_loss
            metadata["ss_addift_mask_data_dir"] = args.addift_mask_data_dir
            metadata["ss_addift_alpha_mask"] = args.addift_alpha_mask
            metadata["ss_add_reverse_pairs"] = args.add_reverse_pairs
            metadata["ss_reverse_multiplier"] = args.reverse_multiplier
            metadata["ss_reverse_weight"] = args.reverse_weight
            metadata["ss_addift_min_sigma"] = args.addift_min_sigma
            metadata["ss_addift_max_sigma"] = args.addift_max_sigma

        # Anima-only self-regularization config
        metadata["ss_self_reg_weight"] = getattr(args, "self_reg_weight", 0.0)
        metadata["ss_self_reg_trigger_word"] = getattr(args, "self_reg_trigger_word", "")
        metadata["ss_self_reg_filler"] = getattr(args, "self_reg_filler", "")
        metadata["ss_self_reg_noise"] = getattr(args, "self_reg_noise", 0.0)
        metadata["ss_self_reg_batched"] = getattr(args, "self_reg_batched", False)
        metadata["ss_self_reg_shuffle_tags"] = getattr(args, "self_reg_shuffle_tags", False)

        # Patch Topology Loss config (runs through inherited NetworkTrainer.process_batch)
        metadata["ss_patch_topology_loss"] = bool(getattr(args, "patch_topology_loss", False))
        metadata["ss_patch_topology_weight"] = getattr(args, "patch_topology_weight", 1.0)
        metadata["ss_patch_topology_tau"] = getattr(args, "patch_topology_tau", 0.1)
        metadata["ss_patch_topology_scale_levels"] = getattr(args, "patch_topology_scale_levels", 2)
        metadata["ss_patch_topology_loss_type"] = getattr(args, "patch_topology_loss_type", "kl")
        metadata["ss_patch_topology_disable_timestep_weight"] = bool(
            getattr(args, "patch_topology_disable_timestep_weight", False)
        )
        metadata["ss_patch_topology_chunk_size"] = getattr(args, "patch_topology_chunk_size", 512)
        metadata["ss_patch_topology_start_step"] = getattr(args, "patch_topology_start_step", 0)
        metadata["ss_patch_topology_warmup_steps"] = getattr(args, "patch_topology_warmup_steps", 0)
        metadata["ss_patch_topology_dynamic_weighting"] = getattr(args, "patch_topology_dynamic_weighting", "none")
        metadata["ss_patch_topology_dwa_temperature"] = getattr(args, "patch_topology_dwa_temperature", 2.0)
        metadata["ss_patch_topology_gradnorm_alpha"] = getattr(args, "patch_topology_gradnorm_alpha", 1.5)
        metadata["ss_patch_topology_dynamic_max_weight"] = getattr(args, "patch_topology_dynamic_max_weight", 10.0)

    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
        edm2_lr_scheduler=None,
        current_loss_scaled=None,
        average_loss_scaled=None,
        current_loss_edm2=None,
        average_loss_edm2=None,
        current_val_loss=None,
        average_val_loss=None,
        current_ffl_loss=None,
        current_patch_topology_loss=None,
        current_patch_topology_weight=None,
        current_wav_mask_ratio=None,
        current_weight_noise_norm=None,
        current_hf_loss=None,
        current_anchor_loss=None,
        it_s: float = 0.0,
    ):
        logs = super().generate_step_logs(
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            mean_grad_norm,
            mean_combined_norm,
            edm2_lr_scheduler,
            current_loss_scaled,
            average_loss_scaled,
            current_loss_edm2,
            average_loss_edm2,
            current_val_loss=current_val_loss,
            average_val_loss=average_val_loss,
            current_ffl_loss=current_ffl_loss,
            current_patch_topology_loss=current_patch_topology_loss,
            current_patch_topology_weight=current_patch_topology_weight,
            current_wav_mask_ratio=current_wav_mask_ratio,
            current_weight_noise_norm=current_weight_noise_norm,
            current_hf_loss=current_hf_loss,
            current_anchor_loss=current_anchor_loss,
            it_s=it_s,
        )
        if self._self_reg_loss_value is not None:
            logs["loss/current_self_reg"] = self._self_reg_loss_value
        return logs

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Set first parameter's requires_grad to True to workaround Accelerate gradient checkpointing bug
        first_param = next(text_encoder.parameters())
        first_param.requires_grad_(True)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        # The base NetworkTrainer only calls enable_gradient_checkpointing(cpu_offload=True/False),
        # so we re-apply with unsloth_offload if needed (after base has already enabled it).
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            model = super().prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            model = unet
            model = accelerator.prepare(model, device_placement=[not self.is_swapping_blocks])
            accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
            accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        # CUDA perf switches are independent of torch.compile; apply whenever requested.
        compile_utils.apply_cuda_optimizations(args)

        if args.compile:
            # Apply per-block torch.compile to the DiT blocks. Reach the real Anima via
            # unwrap_model so we mutate the underlying ModuleList regardless of any DDP wrapper.
            dit = accelerator.unwrap_model(model)
            compile_utils.compile_transformer(args, dit, [dit.blocks], disable_linear=self.is_swapping_blocks)

        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    # parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    # Anima-specific default: lower cfm_lambda than the SDXL default of 0.05
    parser.set_defaults(cfm_lambda=0.02)
    parser.add_argument("--ileco", action="store_true", help="enable dataset-backed iLECO prompt-to-prompt training")
    parser.add_argument("--ileco_original_prompt", type=str, default="", help="original prompt for single-pair iLECO")
    parser.add_argument("--ileco_target_prompt", type=str, default=None, help="target prompt for single-pair iLECO")
    parser.add_argument("--ileco_prompt_pairs", type=str, default=None, help="UTF-8 JSON file with iLECO prompt pairs")
    parser.add_argument("--ileco_loss_weight", type=float, default=1.0, help="loss multiplier for iLECO")
    parser.add_argument("--ileco_min_sigma", type=float, default=None, help="minimum sigma for iLECO timestep sampling")
    parser.add_argument("--ileco_max_sigma", type=float, default=None, help="maximum sigma for iLECO timestep sampling")
    parser.add_argument("--add_reverse_pairs", action="store_true", help="add reverse iLECO prompt pairs")
    parser.add_argument("--reverse_multiplier", type=float, default=-1.0, help="LoRA multiplier for reverse iLECO pairs")
    parser.add_argument("--reverse_weight", type=float, default=1.0, help="loss weight for reverse iLECO pairs")
    parser.add_argument("--addift", action="store_true", help="enable ADDifT paired-image training")
    parser.add_argument("--addift_cache_conditioning_latents", action="store_true", help="cache ADDifT source image latents")
    parser.add_argument("--addift_pair_settings", type=str, default=None, help="UTF-8 JSON file with ADDifT per-pair settings")
    parser.add_argument("--addift_mask_data_dir", type=str, default=None, help="directory containing ADDifT loss masks")
    parser.add_argument("--addift_mask_loss", action="store_true", help="apply ADDifT spatial loss masks")
    parser.add_argument(
        "--addift_alpha_mask",
        type=str,
        default=None,
        choices=["target", "source", "union", "intersection", "difference"],
        help="derive ADDifT loss masks from image alpha channels",
    )
    parser.add_argument("--addift_loss_weight", type=float, default=1.0, help="loss multiplier for ADDifT")
    parser.add_argument("--addift_multiplier", type=float, default=1.0, help="LoRA multiplier for ADDifT student prediction")
    parser.add_argument("--addift_min_sigma", type=float, default=None, help="minimum sigma for ADDifT timestep sampling")
    parser.add_argument("--addift_max_sigma", type=float, default=None, help="maximum sigma for ADDifT timestep sampling")
    parser.add_argument(
        "--self_reg_weight",
        type=float,
        default=0.0,
        help="Self-regularization weight: hold the LoRA to the frozen model's own behaviour on "
        "everything but the trigger word. 0 = off. 1.0 is a good starting value.",
    )
    parser.add_argument(
        "--self_reg_trigger_word",
        type=str,
        default="",
        help="Trigger word(s) for self-regularization, comma-separated, matched case-insensitively "
        "against comma-separated caption tags. Required when --self_reg_weight > 0.",
    )
    parser.add_argument(
        "--self_reg_filler",
        type=str,
        default="",
        help="Word(s) placed in the vacated trigger slot of the anchor caption so tag positions stay stable.",
    )
    parser.add_argument(
        "--self_reg_noise",
        type=float,
        default=0.0,
        help="Probability (0..1) of using a pure-noise anchor on hold steps, covering latents the images never reach.",
    )
    parser.add_argument(
        "--self_reg_batched",
        action="store_true",
        help="Hold both halves in one step (same-step half-batch) instead of alternating steps. "
        "Needs batch size >= 2; otherwise falls back to alternating steps.",
    )
    parser.add_argument(
        "--self_reg_shuffle_tags",
        action="store_true",
        help="Shuffle the held tags once when building the anchor caption so tag positions are unreliable.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    # Automatically switch to Anima-specific LoRA module if generic one is provided
    if args.network_module == "networks.lora":
        print("Override network module: networks.lora -> networks.lora_anima")
        args.network_module = "networks.lora_anima"

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    # Anima is a Rectified Flow model. Use a private flag to bypass the CFM guard
    # in train_network.py WITHOUT triggering the generic "Using Rectified Flow" log block.
    args._anima_model = True

    trainer = AnimaNetworkTrainer()
    trainer.train(args)
