# Anima DMAD (Distribution Matching as Adversarial Distillation) training script.
#
# Based on ``DMAD/train/image`` (SDXL: ``main/train_sd.py`` + ``main/sd_guidance.py``,
# paper arXiv:2610.02188) ported to the Anima rectified-flow DiT, and integrated
# with this repo's dataset / accelerator / caching / sampling infrastructure
# (like ``anima_train.py``) plus LoRA support (like ``anima_train_network.py``).
#
# What it does per iteration (same as DMAD):
#   generator turn: student DiT renders latents G (1-step or multi-step Euler,
#     optionally with backward simulation), critic is frozen, generator minimizes
#     ``-d_teacher(G) * teacher_loss_weight - d_real(G) * real_loss_weight``.
#   critic turn:    critic (DiT copy reading final-block tokens + real/teacher
#     MLP heads) minimizes ``BCE(T=1, G=0) + BCE(Q=1, G=0)`` on real latents T,
#     generated G (detached) and teacher samples Q.
#
# Teacher samples Q come from a pre-generated LMDB cache (--teacher_latent_path,
# built by anima_dmad_generate_teacher.py; no online teacher evaluation, as in
# the paper) or, as a fallback, from on-the-fly multi-step Euler sampling with
# the frozen teacher DiT (--teacher_model_path, --teacher_steps, --teacher_cfg).
# The generator trains full-finetune by default or LoRA (--network_module
# networks.lora_anima ...); the critic trains fully or with LoRA on its backbone
# (--dmad_critic_lora_dim) plus full heads.
# Memory (full, on-the-fly): generator + teacher + critic + VAE + Qwen3.
# Cached-Q mode drops the teacher DiT; LoRA modes shrink the trainable state.
# Use --gradient_checkpointing / --blocks_to_swap / smaller batch sizes on small GPUs.

import argparse
import copy
import gc
import importlib
import math
import os
import sys
from multiprocessing import Value

import toml
from tqdm import tqdm

import torch
import torch.nn.functional as F

from library import flux_train_utils
from library.device_utils import init_ipex, clean_memory_on_device
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

init_ipex()

from accelerate.utils import set_seed
from library import anima_models, anima_train_utils, anima_utils, strategy_base, strategy_anima, sai_model_spec
from library.anima_dmad import AnimaDMADCritic, AnimaTeacherLatentDataset, apply_scaled_spectral_norm, euler_sample_x0, velocity_to_x0

import library.accelerator_setup as accelerator_setup
import library.args as args_util
import library.checkpoint_io as checkpoint_io
import library.dataset as dataset_util
import library.deepspeed_utils as deepspeed_utils
import library.logging_util as logging_util
import library.loss as loss_util  # noqa: F401 (kept for parity with anima_train.py)
import library.optimizer as optimizer_util
import library.sampling as sampling

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util
from library.config_util import ConfigSanitizer, BlueprintGenerator
from library.custom_train_functions import add_custom_train_arguments


def add_dmad_arguments(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("DMAD arguments")
    g.add_argument("--teacher_model_path", type=str, default=None,
                   help="Frozen teacher Anima checkpoint for on-the-fly Q samples. Default: same as --pretrained_model_name_or_path")
    g.add_argument("--teacher_latent_path", type=str, default=None,
                   help="LMDB of pre-generated teacher latents (Q) built by anima_dmad_generate_teacher.py. "
                   "When set, Q is read from cache (no online teacher evaluation, as in the paper) and no teacher DiT is loaded")
    g.add_argument("--dmad_teacher_batch_size", type=int, default=8,
                   help="Batch size for cached teacher latents (exact match with the train batch is not required)")
    g.add_argument("--conditioning_sigma", type=float, default=1.0,
                   help="Noise level of the one-step generator (1.0 = pure noise)")
    g.add_argument("--num_denoising_steps", type=int, default=1,
                   help="Generator steps at inference. 1 = one-step; >1 = multi-step Euler")
    g.add_argument("--dmad_flow_shift", type=float, default=3.0,
                   help="Flow shift of the multi-step generator sigma grid")
    g.add_argument("--backward_simulation", action="store_true",
                   help="Multi-step generator: run the student's own sampler no-grad, then one grad forward (like DMAD-SDXL)")
    g.add_argument("--teacher_steps", type=int, default=20,
                   help="Euler steps for on-the-fly teacher (Q) sampling (ignored with --teacher_latent_path)")
    g.add_argument("--teacher_flow_shift", type=float, default=3.0,
                   help="Flow shift for on-the-fly teacher sampling")
    g.add_argument("--teacher_cfg", type=float, default=1.0,
                   help="CFG scale for on-the-fly teacher sampling (bakes guidance into Q; ignored with --teacher_latent_path)")
    g.add_argument("--teacher_negative_prompt", type=str, default="",
                   help="Negative prompt for on-the-fly teacher CFG")
    g.add_argument("--critic_num_blocks", type=int, default=None,
                   help="Critic backbone depth (DiT blocks). Default: full DiT depth (paper's DiT discriminators read final-block tokens)")
    g.add_argument("--critic_max_sigma", type=float, default=1.0,
                   help="Critic noise levels are uniform in [0, this)")
    g.add_argument("--critic_spectral_norm", action="store_true",
                   help="Frozen-gain spectral norm on the critic")
    g.add_argument("--generator_spectral_norm", action="store_true",
                   help="Frozen-gain spectral norm on the generator encoder (patch embed + first-half blocks)")
    g.add_argument("--critic_freeze_backbone", action="store_true",
                   help="Freeze the critic backbone (heads train)")
    g.add_argument("--teacher_gap_route", action="store_true",
                   help="Weight the teacher loss per noise band by the online gap E[d_real(T)] - E[d_real(Q)]")
    g.add_argument("--gap_tau", type=float, default=2.0,
                   help="Gap-routing temperature (paper: 0.5 for ImageNet/SDXL, 2 for Wan/MiniMax-H3 DiTs)")
    g.add_argument("--gap_num_bands", type=int, default=10,
                   help="Number of equal-width gap-routing bands over [0, 1). Default: 10 (paper H3/Wan); "
                   "ImageNet-style setups use fewer (e.g. 6)")
    g.add_argument("--teacher_loss_weight", type=float, default=5e-3)
    g.add_argument("--real_loss_weight", type=float, default=5e-3)
    g.add_argument("--critic_loss_weight", type=float, default=1e-2)
    g.add_argument("--guidance_lr", type=float, default=5e-7, help="Learning rate for the critic")
    g.add_argument("--dmad_adam_beta1", type=float, default=0.0)
    g.add_argument("--dmad_adam_beta2", type=float, default=0.99)
    g.add_argument("--dmad_warmup_steps", type=int, default=500)
    g.add_argument("--generator_ema_kimg", type=float, default=1000, help="Generator EMA half-life in thousand images")
    g.add_argument("--generator_ema_rampup", type=float, default=0.05)
    # LoRA (generator). Mirrors train_network.py; LoRA trains the student, base DiT stays frozen.
    g.add_argument("--network_module", type=str, default=None,
                   help="LoRA network module for the generator (e.g. networks.lora_anima). Unset = full finetune")
    g.add_argument("--network_dim", type=int, default=None, help="LoRA rank (paper H3: 128)")
    g.add_argument("--network_alpha", type=float, default=1, help="LoRA alpha (paper H3: 128)")
    g.add_argument("--network_dropout", type=float, default=None, help="LoRA neuron dropout")
    g.add_argument("--network_args", type=str, default=None, nargs="*",
                   help="additional network arguments (key=value), e.g. train_llm_adapter=true")
    g.add_argument("--network_weights", type=str, default=None, help="resume generator LoRA from this file")
    g.add_argument("--dim_from_weights", action="store_true",
                   help="determine LoRA rank from --network_weights")
    # LoRA (critic backbone). Heads always train fully.
    g.add_argument("--dmad_critic_lora_dim", type=int, default=None,
                   help="LoRA rank on the critic backbone (paper H3: 128). Unset = train critic fully")
    g.add_argument("--dmad_critic_lora_alpha", type=float, default=None,
                   help="LoRA alpha for the critic backbone (default: same as rank)")


def constant_with_warmup(optimizer, warmup_steps):
    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_lora_network(args, target, dim, alpha, log_name):
    """Import the network module and build a LoRA network on ``target``.

    ``target`` is the generator DiT or the critic's ``blocks`` ModuleList
    (module selection is by class name, so both work). Text-encoder LoRAs are
    skipped: the Qwen3 encoder stays frozen in DMAD training.
    """
    sys.path.append(os.path.dirname(__file__))
    accelerator_print = logger.info
    accelerator_print(f"import network module: {args.network_module}")
    network_module = importlib.import_module(args.network_module)

    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=", 1)
            net_kwargs[key] = value
    if "dropout" not in net_kwargs:
        net_kwargs["dropout"] = args.network_dropout

    if args.dim_from_weights:
        assert args.network_weights, "--network_weights is required with --dim_from_weights"
        network, _ = network_module.create_network_from_weights(1, args.network_weights, None, [], target, **net_kwargs)
    else:
        network = network_module.create_network(
            1.0, dim, alpha, None, [], target,
            neuron_dropout=args.network_dropout, **net_kwargs)
    logger.info(f"created LoRA network for {log_name}: dim={dim}, alpha={alpha}")
    return network_module, network


@torch.no_grad()
def sample_backward_simulation(dit, noise4d, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
                               num_steps, flow_shift, dtype):
    """Run the student's own sampler no-grad for a random number of steps.

    Returns (clean_estimate_4D, sigma_k) mirroring SDUniModel.sample_backward.
    """
    device = noise4d.device
    b = noise4d.shape[0]
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    if float(flow_shift) != 1.0:
        fs = float(flow_shift)
        sigmas = (sigmas * fs) / (1 + (fs - 1) * sigmas)
    k = int(torch.randint(0, num_steps, (1,), device=device).item())
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        kk = torch.tensor([k], device=device, dtype=torch.long)
        torch.distributed.broadcast(kk, src=0)
        k = int(kk.item())
    x = noise4d.unsqueeze(2).float()
    _, _, _, h, w = x.shape
    padding_mask = torch.zeros(b, 1, h, w, dtype=torch.float32, device=device)
    clean = noise4d.float()
    for j in range(k):
        s = float(sigmas[j])
        t = torch.full((b,), s, device=device, dtype=noise4d.dtype)
        v = dit(x.to(dtype), t, prompt_embeds.to(dtype) if torch.is_tensor(prompt_embeds) else prompt_embeds,
                padding_mask=padding_mask.to(dtype), source_attention_mask=attn_mask,
                target_input_ids=t5_input_ids, target_attention_mask=t5_attn_mask).float()
        clean = (x.squeeze(2) - s * v.squeeze(2))
        s_next = float(sigmas[j + 1])
        clean_full = clean.unsqueeze(2)
        clean_full = (1.0 - s_next) * clean_full + s_next * torch.randn_like(clean_full)
        x = clean_full
        clean = clean_full.squeeze(2)
    return clean.to(dtype), float(sigmas[k])


def student_forward(dit, noise4d, cond, args, real_latents=None, dtype=None):
    """Render G latents (4D) with the student. Returns (G, sigma_info)."""
    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = cond
    b = noise4d.shape[0]
    device = noise4d.device
    steps = max(1, int(args.num_denoising_steps))
    if steps <= 1:
        s = float(args.conditioning_sigma)
        # timesteps must match the DiT dtype (float32 t vs bf16 weights errors)
        t = torch.full((b,), s, device=device, dtype=noise4d.dtype)
        x = noise4d.unsqueeze(2)
        _, _, _, h, w = x.shape
        padding_mask = torch.zeros(b, 1, h, w, dtype=dtype, device=device)
        v = dit(x, t, prompt_embeds, padding_mask=padding_mask, source_attention_mask=attn_mask,
                target_input_ids=t5_input_ids, target_attention_mask=t5_attn_mask)
        g = velocity_to_x0(noise4d, v.squeeze(2), torch.full((b,), s, device=device, dtype=v.dtype))
        return g, t
    if args.backward_simulation:
        with torch.no_grad():
            clean, s_k = sample_backward_simulation(dit, noise4d, prompt_embeds, attn_mask,
                                                    t5_input_ids, t5_attn_mask, steps, args.dmad_flow_shift, dtype)
        s = torch.full((b,), s_k, device=device, dtype=noise4d.dtype)
        noised = (1.0 - s.reshape(-1, 1, 1, 1)) * clean + s.reshape(-1, 1, 1, 1) * noise4d
        x = noised.unsqueeze(2)
        _, _, _, h, w = x.shape
        padding_mask = torch.zeros(b, 1, h, w, dtype=dtype, device=device)
        t = s.to(noise4d.dtype)
        v = dit(x, t, prompt_embeds, padding_mask=padding_mask, source_attention_mask=attn_mask,
                target_input_ids=t5_input_ids, target_attention_mask=t5_attn_mask)
        g = velocity_to_x0(noised, v.squeeze(2), s.to(v.dtype))
        return g, t
    # no backward simulation: noise real latents to a random grid step, one forward
    assert real_latents is not None, "multi-step without backward simulation needs real latents"
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    if float(args.dmad_flow_shift) != 1.0:
        fs = float(args.dmad_flow_shift)
        sigmas = (sigmas * fs) / (1 + (fs - 1) * sigmas)
    idx = torch.randint(0, steps, (b,), device=device)
    s = sigmas[idx].to(noise4d.dtype)
    noised = (1.0 - s.reshape(-1, 1, 1, 1)) * real_latents + s.reshape(-1, 1, 1, 1) * noise4d
    x = noised.unsqueeze(2)
    _, _, _, h, w = x.shape
    padding_mask = torch.zeros(b, 1, h, w, dtype=dtype, device=device)
    t = s.to(noise4d.dtype)
    v = dit(x, t, prompt_embeds, padding_mask=padding_mask, source_attention_mask=attn_mask,
            target_input_ids=t5_input_ids, target_attention_mask=t5_attn_mask)
    g = velocity_to_x0(noised, v.squeeze(2), s.to(v.dtype))
    return g, t


def train(args):
    args_util.verify_training_args(args)
    accelerator_setup.prepare_dataset_args(args, True)
    setup_logging(args, reset=True)
    flux_train_utils.log_timestep_sampling_info(args)

    assert not getattr(args, "fused_backward_pass", False), "fused_backward_pass is not supported with the DMAD script"
    assert not getattr(args, "deepspeed", False), "deepspeed is not supported yet with the DMAD script"

    use_teacher_cache = args.teacher_latent_path is not None
    use_gen_lora = args.network_module is not None
    use_critic_lora = args.dmad_critic_lora_dim is not None and args.dmad_critic_lora_dim > 0
    if use_gen_lora and args.generator_spectral_norm:
        raise ValueError("generator_spectral_norm is not supported with generator LoRA")
    if use_critic_lora and args.critic_spectral_norm:
        logger.warning("critic_spectral_norm with critic LoRA applies to heads only")
    # Qwen3 must stay resident when teacher prompts are encoded live: cached-Q
    # mode (Q prompts) or on-the-fly CFG (negative prompt).
    need_qwen3_live = use_teacher_cache or (not use_teacher_cache and float(args.teacher_cfg) > 1.0)

    if not args.skip_cache_check:
        args.skip_cache_check = args.skip_latents_validity_check
    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
        args.cache_text_encoder_outputs = True
    if args.cpu_offload_checkpointing and not args.gradient_checkpointing:
        logger.warning("cpu_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
        args.gradient_checkpointing = True
    if args.unsloth_offload_checkpointing:
        if not args.gradient_checkpointing:
            logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
            args.gradient_checkpointing = True
        assert not args.cpu_offload_checkpointing, "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
    assert (args.blocks_to_swap is None or args.blocks_to_swap == 0) or not args.cpu_offload_checkpointing, \
        "blocks_to_swap is not supported with cpu_offload_checkpointing"

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None
    if args.seed is not None:
        set_seed(args.seed)

    if args.cache_latents:
        latents_caching_strategy = strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
        else:
            if use_dreambooth_method:
                user_config = {"datasets": [{"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                    args.train_data_dir, args.reg_data_dir)}]}
            else:
                user_config = {"datasets": [{"subsets": [{"image_dir": args.train_data_dir,
                                                          "metadata_file": args.in_json}]}]}
        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, _ = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = dataset_util.load_arbitrary_dataset(args)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = dataset_util.collator_class(current_epoch, current_step, ds_for_collator)
    train_dataset_group.verify_bucket_reso_steps(16)

    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify the metadata file and train_data_dir option.")
        return
    if cache_latents:
        assert train_dataset_group.is_latent_cacheable(), "when caching latents, either color_aug or random_crop cannot be used"
    if args.cache_text_encoder_outputs:
        assert train_dataset_group.is_text_encoder_output_cacheable(cache_supports_dropout=True), \
            "when caching text encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

    logger.info("prepare accelerator")
    accelerator = accelerator_setup.prepare_accelerator(args)
    weight_dtype, save_dtype = accelerator_setup.prepare_dtype(args)

    logger.info("Loading tokenizers...")
    qwen3_text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
    t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer,
        qwen3_max_length=args.qwen3_max_token_length, t5_max_length=args.t5_max_token_length)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)
    qwen3_text_encoder.to(weight_dtype)
    qwen3_text_encoder.requires_grad_(False)

    sample_prompts_te_outputs = None
    if args.cache_text_encoder_outputs:
        qwen3_text_encoder.to(accelerator.device)
        qwen3_text_encoder.eval()
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
            strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, is_partial=False))
        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([qwen3_text_encoder], accelerator)
        if args.sample_prompts is not None:
            prompts = sampling.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                        if p not in sample_prompts_te_outputs:
                            tokens_and_masks = tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy, [qwen3_text_encoder], tokens_and_masks)
        accelerator.wait_for_everyone()
        if need_qwen3_live:
            logger.info("keeping Qwen3 resident for live teacher-prompt encoding")
        else:
            qwen3_text_encoder = None
            gc.collect()
            clean_memory_on_device(accelerator.device)

    logger.info("Loading Anima VAE...")
    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
    if cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()
        train_dataset_group.new_cache_latents(vae, accelerator)
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    logger.info("Loading Anima DiT (%s)...", "generator base for LoRA" if use_gen_lora else "generator")
    dit = anima_utils.load_anima_model("cpu", args.pretrained_model_name_or_path, args.attn_mode,
                                       args.split_attn, "cpu", dit_weight_dtype=None)
    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing(cpu_offload=args.cpu_offload_checkpointing,
                                           unsloth_offload=args.unsloth_offload_checkpointing)
    if use_gen_lora:
        dit.requires_grad_(False)  # base frozen; LoRA trains
    else:
        dit.requires_grad_(True)

    teacher_dit = None
    if use_teacher_cache:
        logger.info(f"Using cached teacher latents from {args.teacher_latent_path}; no teacher DiT loaded")
    else:
        teacher_path = args.teacher_model_path or args.pretrained_model_name_or_path
        logger.info(f"Loading frozen teacher DiT from {teacher_path}...")
        teacher_dit = anima_utils.load_anima_model("cpu", teacher_path, args.attn_mode, args.split_attn,
                                                   "cpu", dit_weight_dtype=None)
        teacher_dit.requires_grad_(False)
        teacher_dit.eval()

    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    if is_swapping_blocks:
        logger.info(f"Enable block swap on generator: blocks_to_swap={args.blocks_to_swap}")
        dit.enable_block_swap(args.blocks_to_swap, accelerator.device)
        logger.warning("block swap applies to the generator only; teacher and critic stay resident")

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    if args.generator_spectral_norm:
        n = apply_scaled_spectral_norm(dit.x_embedder) + apply_scaled_spectral_norm(
            torch.nn.ModuleList(list(dit.blocks[:len(dit.blocks) // 2])))
        logger.info(f"generator spectral norm: {n} layers")

    logger.info("Building DMAD critic (from base weights, pre-LoRA)...")
    critic = AnimaDMADCritic(
        dit,
        critic_num_blocks=args.critic_num_blocks,
        freeze_backbone=args.critic_freeze_backbone or use_critic_lora,
        spectral_norm=args.critic_spectral_norm and not use_critic_lora,
        teacher_gap_route=args.teacher_gap_route,
        gap_tau=args.gap_tau,
        gap_num_bands=args.gap_num_bands,
        critic_max_sigma=args.critic_max_sigma,
    )
    if args.critic_spectral_norm and use_critic_lora:
        # heads only; backbone trains via LoRA
        n = apply_scaled_spectral_norm(critic.head_real) + apply_scaled_spectral_norm(critic.head_teacher)
        logger.info(f"critic head spectral norm: {n} layers")

    # LoRA networks (created after critic init so the critic starts from base weights)
    network = None
    if use_gen_lora:
        _, network = create_lora_network(args, dit, args.network_dim, args.network_alpha, "generator")
        network.apply_to([], dit, False, True)
        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load generator LoRA weights from {args.network_weights}: {info}")

    critic_network = None
    if use_critic_lora:
        _, critic_network = create_lora_network(
            args, critic.blocks,
            args.dmad_critic_lora_dim,
            args.dmad_critic_lora_alpha if args.dmad_critic_lora_alpha is not None else args.dmad_critic_lora_dim,
            "critic backbone",
        )
        critic_network.apply_to([], critic.blocks, False, True)

    # requires_grad mask of the critic backbone/heads (frozen backbone stays
    # frozen when the turn flips it back on; LoRA params live on critic_network)
    critic_grad_mask = {n: p.requires_grad for n, p in critic.named_parameters()}

    def set_critic_requires_grad(mod, flag: bool):
        if not flag:
            mod.requires_grad_(False)
        else:
            for n, p in mod.named_parameters():
                p.requires_grad_(critic_grad_mask.get(n, True))

    if network is None:
        param_groups = anima_train_utils.get_anima_param_groups(
            dit, base_lr=args.learning_rate, self_attn_lr=args.self_attn_lr, cross_attn_lr=args.cross_attn_lr,
            mlp_lr=args.mlp_lr, mod_lr=args.mod_lr, llm_adapter_lr=args.llm_adapter_lr)
        n_gen_trainable = sum(p.numel() for group in param_groups for p in group["params"])
    else:
        param_groups, lr_descriptions = network.prepare_optimizer_params_with_multiple_te_lrs(
            None, args.learning_rate, args.learning_rate)
        accelerator.print(f"generator LoRA groups: {lr_descriptions}")
        n_gen_trainable = sum(p.numel() for group in param_groups for p in group["params"])
    accelerator.print(f"number of trainable generator parameters: {n_gen_trainable:,}")
    n_critic_trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    if critic_network is not None:
        n_critic_trainable += sum(p.numel() for p in critic_network.parameters() if p.requires_grad)
    accelerator.print(f"number of trainable critic parameters: {n_critic_trainable:,}")

    _, _, optimizer_gen = optimizer_util.get_optimizer(args, trainable_params=param_groups)
    optimizer_train_fn, optimizer_eval_fn = optimizer_util.get_optimizer_train_eval_fn(optimizer_gen, args)
    critic_train_params = [p for p in critic.parameters() if p.requires_grad]
    if critic_network is not None:
        critic_train_params += [p for p in critic_network.parameters() if p.requires_grad]
    optimizer_critic = torch.optim.AdamW(
        critic_train_params,
        lr=args.guidance_lr, betas=(args.dmad_adam_beta1, args.dmad_adam_beta2), weight_decay=0.01)
    scheduler_critic = constant_with_warmup(optimizer_critic, args.dmad_warmup_steps)

    train_dataset_group.set_current_strategies()
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group, batch_size=1, shuffle=True, collate_fn=collator,
        num_workers=n_workers, persistent_workers=args.persistent_data_loader_workers)
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
    train_dataset_group.set_max_train_steps(args.max_train_steps)
    lr_scheduler = optimizer_util.get_scheduler_fix(args, optimizer_gen, accelerator.num_processes)

    dit_weight_dtype = weight_dtype
    if args.full_fp16:
        assert args.mixed_precision == "fp16"
    elif args.full_bf16:
        assert args.mixed_precision == "bf16"
    else:
        dit_weight_dtype = torch.float32
    dit.to(dit_weight_dtype)
    if teacher_dit is not None:
        teacher_dit.to(accelerator.device, dtype=weight_dtype)
    critic.to(dit_weight_dtype)

    if not args.cache_text_encoder_outputs and qwen3_text_encoder is not None:
        qwen3_text_encoder.to(accelerator.device)
    clean_memory_on_device(accelerator.device)

    dit = accelerator.prepare(dit, device_placement=[not is_swapping_blocks])
    if is_swapping_blocks:
        accelerator.unwrap_model(dit).move_to_device_except_swap_blocks(accelerator.device)
    critic = accelerator.prepare(critic)
    if network is not None:
        network = accelerator.prepare(network)
    if critic_network is not None:
        critic_network = accelerator.prepare(critic_network)
    optimizer_gen, optimizer_critic, train_dataloader, lr_scheduler, scheduler_critic = accelerator.prepare(
        optimizer_gen, optimizer_critic, train_dataloader, lr_scheduler, scheduler_critic)

    if args.full_fp16:
        accelerator_setup.patch_accelerator_for_fp16_training(accelerator)
    args_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # Trainable generator params live on the LoRA network (LoRA mode) or the DiT.
    gen_holder = network if network is not None else dit
    gen_ema_params = [p.detach().clone() for p in accelerator.unwrap_model(gen_holder).parameters()
                      if p.requires_grad]

    def gen_clip_params():
        if network is not None:
            return [p for p in accelerator.unwrap_model(network).parameters() if p.requires_grad]
        return accelerator.unwrap_model(dit).parameters()

    def critic_clip_params():
        params = [p for p in accelerator.unwrap_model(critic).parameters() if p.requires_grad]
        if critic_network is not None:
            params += [p for p in accelerator.unwrap_model(critic_network).parameters() if p.requires_grad]
        return params

    # Cached teacher latents (Q): independent cycling loader.
    teacher_iter = None
    if use_teacher_cache:
        teacher_ds = AnimaTeacherLatentDataset(args.teacher_latent_path)
        teacher_loader = torch.utils.data.DataLoader(
            teacher_ds, batch_size=args.dmad_teacher_batch_size, shuffle=True, drop_last=True, num_workers=0)

        def _cycle(dl):
            while True:
                for b in dl:
                    yield b

        teacher_iter = _cycle(teacher_loader)

    # Negative conditioning for on-the-fly teacher CFG (prompt-independent: encode once).
    neg_cond = None
    if not use_teacher_cache and float(args.teacher_cfg) > 1.0:
        assert qwen3_text_encoder is not None, "Qwen3 is required for on-the-fly teacher CFG"
        with torch.no_grad():
            neg_tokens = tokenize_strategy.tokenize([args.teacher_negative_prompt])
            neg_enc = text_encoding_strategy.encode_tokens(tokenize_strategy, [qwen3_text_encoder], neg_tokens)
        neg_cond = (neg_enc[0].to(accelerator.device, dtype=dit_weight_dtype),
                    neg_enc[1].to(accelerator.device),
                    neg_enc[2].to(accelerator.device, dtype=torch.long),
                    neg_enc[3].to(accelerator.device))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    accelerator.print("running DMAD training / 学習開始")
    os.makedirs(args.output_dir, exist_ok=True)
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0,
                        disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers("anima-dmad" if args.log_tracker_name is None else args.log_tracker_name,
                                  config=args_util.get_sanitized_config_or_none(args), init_kwargs=init_kwargs)

    if is_swapping_blocks:
        accelerator.unwrap_model(dit).prepare_block_swap_before_forward()

    optimizer_eval_fn()
    anima_train_utils.sample_images(accelerator, args, 0, global_step, dit, vae, qwen3_text_encoder,
                                    tokenize_strategy, text_encoding_strategy, sample_prompts_te_outputs)
    optimizer_train_fn()

    unwrapped_dit = accelerator.unwrap_model(dit)
    logger.info(f"dit device: {unwrapped_dit.device}, dtype: {unwrapped_dit.dtype}")

    loss_recorder = logging_util.LossRecorder()

    def lora_metadata(step, epoch_no):
        from library import model_io

        meta = {
            "ss_network_module": args.network_module,
            "ss_network_dim": str(args.network_dim),
            "ss_network_alpha": str(args.network_alpha),
            "ss_steps": str(step),
            "ss_epoch": str(epoch_no),
        }
        try:
            sai = model_io.get_sai_model_spec_dataclass(None, args, False, True, False, anima="preview").to_metadata_dict()
            meta.update(sai)
        except Exception as e:
            logger.warning(f"sai model spec metadata failed: {e}")
        return meta

    def save_ema_lora(nw, ema_params, ckpt_file, metadata):
        from safetensors.torch import save_file as safetensors_save

        live_sd = nw.state_dict()
        ema_sd = dict(live_sd)
        for (name, _), s in zip([(n, p) for n, p in nw.named_parameters() if p.requires_grad], ema_params):
            if name in ema_sd:
                ema_sd[name] = s.cpu()
        for k, v in list(ema_sd.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                ema_sd[k] = v.detach().clone().to("cpu").to(save_dtype).contiguous()
        safetensors_save(ema_sd, ckpt_file, metadata=dict(metadata))

    def save_dmad(step, epoch, is_epoch_end):
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        os.makedirs(args.output_dir, exist_ok=True)
        base = args.output_name or "anima-dmad"
        if network is None:
            anima_train_utils.save_anima_model_on_epoch_end_or_stepwise(
                args, is_epoch_end, accelerator, save_dtype, epoch, num_train_epochs, step,
                accelerator.unwrap_model(dit))
            # extra: EMA generator + critic weights next to the main checkpoint
            try:
                from safetensors.torch import save_file as safetensors_save
                live = {k: v.detach().cpu() for k, v in accelerator.unwrap_model(dit).state_dict().items()}
                ema_sd = dict(live)
                for (name, _), s in zip([(n, p) for n, p in accelerator.unwrap_model(dit).named_parameters() if p.requires_grad],
                                        gen_ema_params):
                    if name in ema_sd:
                        ema_sd[name] = s.cpu()
                meta = {"format": "pt"}
                safetensors_save({("net." + k): v.contiguous().to(save_dtype) for k, v in ema_sd.items()},
                                 os.path.join(args.output_dir, f"{base}_ema_{step:06d}.safetensors"), metadata=meta)
            except Exception as e:
                logger.warning(f"extra DMAD EMA checkpoint save failed: {e}")
        else:
            metadata = lora_metadata(step, epoch)
            accelerator.unwrap_model(network).save_weights(
                os.path.join(args.output_dir, f"{base}_{step:06d}.safetensors"), save_dtype, dict(metadata))
            save_ema_lora(accelerator.unwrap_model(network), gen_ema_params,
                          os.path.join(args.output_dir, f"{base}_ema_{step:06d}.safetensors"), metadata)
        # critic weights next to the main checkpoint
        try:
            from safetensors.torch import save_file as safetensors_save
            safetensors_save({k: v.detach().cpu().contiguous() for k, v in
                              accelerator.unwrap_model(critic).state_dict().items()},
                             os.path.join(args.output_dir, f"{base}_critic_{step:06d}.safetensors"),
                             metadata={"format": "pt"})
            if critic_network is not None:
                accelerator.unwrap_model(critic_network).save_weights(
                    os.path.join(args.output_dir, f"{base}_critic_lora_{step:06d}.safetensors"),
                    save_dtype, {"format": "pt", "ss_network_module": args.network_module,
                                 "ss_network_dim": str(args.dmad_critic_lora_dim),
                                 "ss_network_alpha": str(args.dmad_critic_lora_alpha
                                                         if args.dmad_critic_lora_alpha is not None
                                                         else args.dmad_critic_lora_dim)})
        except Exception as e:
            logger.warning(f"extra DMAD critic checkpoint save failed: {e}")

    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1
        accelerator.unwrap_model(dit).train()
        accelerator.unwrap_model(critic).train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(dit):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=dit_weight_dtype)
                    if latents.ndim == 5:
                        latents = latents.squeeze(2)
                else:
                    with torch.no_grad():
                        images = batch["images"].to(accelerator.device, dtype=weight_dtype)
                        latents = vae.encode_pixels_to_latents(images).to(accelerator.device, dtype=dit_weight_dtype)
                    if torch.any(torch.isnan(latents)):
                        latents = torch.nan_to_num(latents, 0, out=latents)

                te_list = batch.get("text_encoder_outputs_list", None)
                if te_list is not None:
                    dropout_rates = te_list[-1]
                    te_list = te_list[:-1]
                    te_list = text_encoding_strategy.drop_cached_text_encoder_outputs(*te_list, caption_dropout_rates=dropout_rates)
                    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = te_list
                else:
                    input_ids_list = batch["input_ids_list"]
                    with torch.no_grad():
                        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, [qwen3_text_encoder], input_ids_list)

                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=dit_weight_dtype)
                attn_mask = attn_mask.to(accelerator.device)
                t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
                t5_attn_mask = t5_attn_mask.to(accelerator.device)
                cond = (prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask)

                noise = torch.randn_like(latents)
                critic_mod = accelerator.unwrap_model(critic)

                # ---- generator turn (critic frozen) ----
                set_critic_requires_grad(critic_mod, False)
                with accelerator.autocast():
                    gen_latents, _ = student_forward(accelerator.unwrap_model(dit), noise, cond, args,
                                                     real_latents=latents, dtype=dit_weight_dtype)
                    gen_losses = critic_mod.compute_generator_loss(gen_latents, *cond)
                    gen_loss = (gen_losses["gen_teacher_loss"] * args.teacher_loss_weight
                                + gen_losses["gen_real_loss"] * args.real_loss_weight)
                accelerator.backward(gen_loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    accelerator.clip_grad_norm_(gen_clip_params(), args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler.step()
                optimizer_gen.zero_grad(set_to_none=True)
                optimizer_critic.zero_grad(set_to_none=True)

                # EMA update
                if accelerator.sync_gradients:
                    total_batch = gen_latents.shape[0] * accelerator.num_processes
                    cur_nimg = (global_step + 1) * total_batch
                    ema_nimg = args.generator_ema_kimg * 1000
                    if args.generator_ema_rampup > 0:
                        ema_nimg = min(ema_nimg, cur_nimg * args.generator_ema_rampup)
                    decay = 0.5 ** (total_batch / max(ema_nimg, 1e-8))
                    with torch.no_grad():
                        live_params = [p for p in accelerator.unwrap_model(gen_holder).parameters() if p.requires_grad]
                        for s, p in zip(gen_ema_params, live_params):
                            s.lerp_(p.detach(), 1.0 - decay)

                # ---- critic turn ----
                set_critic_requires_grad(critic_mod, True)
                if use_teacher_cache:
                    t_batch = next(teacher_iter)
                    teacher_q = t_batch["latents"].to(accelerator.device, dtype=dit_weight_dtype)
                    with torch.no_grad():
                        t_tokens = tokenize_strategy.tokenize(t_batch["prompt"])
                        t_enc = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, [qwen3_text_encoder], t_tokens)
                    teacher_cond = (t_enc[0].to(accelerator.device, dtype=dit_weight_dtype),
                                    t_enc[1].to(accelerator.device),
                                    t_enc[2].to(accelerator.device, dtype=torch.long),
                                    t_enc[3].to(accelerator.device))
                else:
                    with torch.no_grad():
                        teacher_noise = torch.randn_like(latents)
                        teacher_q = euler_sample_x0(
                            teacher_dit, teacher_noise, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
                            steps=max(1, int(args.teacher_steps)), flow_shift=float(args.teacher_flow_shift),
                            guidance_scale=float(args.teacher_cfg), neg_cond=neg_cond)
                        teacher_q = teacher_q.to(dtype=dit_weight_dtype)
                    teacher_cond = cond
                with accelerator.autocast():
                    critic_losses, critic_logs = critic_mod.compute_critic_loss(
                        gen_latents.detach(), prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
                        latents, cond, teacher_q, teacher_cond)
                    critic_loss = critic_losses["critic_loss"] * args.critic_loss_weight
                accelerator.backward(critic_loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    accelerator.clip_grad_norm_(critic_clip_params(), args.max_grad_norm)
                optimizer_critic.step()
                scheduler_critic.step()
                optimizer_critic.zero_grad(set_to_none=True)
                optimizer_gen.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                optimizer_eval_fn()
                anima_train_utils.sample_images(accelerator, args, None, global_step, dit, vae, qwen3_text_encoder,
                                                tokenize_strategy, text_encoding_strategy, sample_prompts_te_outputs)
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    save_dmad(global_step, epoch, False)
                optimizer_train_fn()

            current_loss = gen_loss.detach().item() + critic_loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss,
                        "gen_teacher_loss": gen_losses["gen_teacher_loss"].detach().item(),
                        "gen_real_loss": gen_losses["gen_real_loss"].detach().item(),
                        "critic_loss": critic_losses["critic_loss"].detach().item()}
                for k, v in critic_logs.items():
                    logs[k] = v.item() if torch.is_tensor(v) else v
                accelerator.log(logs, step=global_step)
            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            progress_bar.set_postfix(**{"avr_loss": loss_recorder.moving_average})
            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            accelerator.log({"loss/epoch": loss_recorder.moving_average, "epoch": epoch + 1}, step=global_step)
        accelerator.wait_for_everyone()
        optimizer_eval_fn()
        if args.save_every_n_epochs is not None:
            save_dmad(global_step, epoch, True)
        anima_train_utils.sample_images(accelerator, args, epoch + 1, global_step, dit, vae, qwen3_text_encoder,
                                        tokenize_strategy, text_encoding_strategy, sample_prompts_te_outputs)

    is_main_process = accelerator.is_main_process
    dit = accelerator.unwrap_model(dit)
    if network is not None:
        network = accelerator.unwrap_model(network)
    accelerator.end_training()
    optimizer_eval_fn()
    if args.save_state or args.save_state_on_train_end:
        checkpoint_io.save_state_on_train_end(args, accelerator)
    del accelerator
    if is_main_process:
        if network is None:
            anima_train_utils.save_anima_model_on_train_end(args, save_dtype, epoch, global_step, dit)
        else:
            base = args.output_name or "anima-dmad"
            metadata = lora_metadata(global_step, epoch)
            network.save_weights(os.path.join(args.output_dir, f"{base}.safetensors"), save_dtype, dict(metadata))
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    args_util.add_sd_models_arguments(parser)
    args_util.add_dataset_arguments(parser, True, True, True)
    args_util.add_training_arguments(parser, False)
    args_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    args_util.add_sd_saving_arguments(parser)
    args_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    args_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    add_dmad_arguments(parser)
    parser.add_argument("--cpu_offload_checkpointing", action="store_true")
    parser.add_argument("--unsloth_offload_checkpointing", action="store_true")
    parser.add_argument("--skip_latents_validity_check", action="store_true", help="[Deprecated] use 'skip_cache_check' instead")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    if args.show_timesteps:
        anima_train_utils.show_timesteps(args)
    else:
        train(args)
