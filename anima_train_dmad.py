# Anima DMAD (Distribution Matching as Adversarial Distillation) training script.
#
# This is a port of the MOSS implementation (``RamTorchRun/moss/train_dmad.py``,
# paper arXiv 2610.02188) to the Anima rectified-flow DiT, integrated with this
# repo's dataset / accelerator / caching / sampling infrastructure (like
# ``anima_train.py``), with teacher-sample caching support and kohya LoRA exports.
#
# ONE frozen DiT backbone carries the three LoRA roles of the moss trainer
# (``library.anima_dmad``):
#
#   None     teacher : frozen base (+ optional frozen always-on teacher LoRA,
#             ``--teacher_lora_path`` -- e.g. a DMD LoRA, applied unmerged in
#             fp32); ``teacher_steps`` Euler on the shift-``dmad_flow_shift``
#             grid, no CFG, sampled online for every caption (except with
#             ``--teacher_latent_path``, which supplies Q from cache)
#   default  student: same backbone + new LoRA (zero-init B: starts AT the
#             teacher); ``student_steps``-step re-noise sampler (predict
#             x0 = x - t v, renew the noise at the next t); k ~ U{0..steps-1}
#             no-grad re-noise steps, then one step with grad -> x0_G; noise to
#             t = shift(u); L_G = -mean(lambda_T w(u) h_T + lambda_R h_R), the
#             disc frozen (gradients flow through its input).
#   disc     discriminator: same backbone + disc LoRA; the final block's image
#             tokens go through a non-affine RMSNorm into two heads (teacher,
#             real), each Linear-SiLU-Linear to a per-token logit, token mean.
#
# discriminator turn: balanced BCE per head on noisy (target, x0_G.detach())
# pairs; the teacher head scores teacher samples, the real head scores the
# real images; t = shift(u), u ~ U(disc_u_range). Gap reweighting: per u band an
# EMA of mean h_R(real) - mean h_R(teacher), gathered over ranks;
# w = sigmoid((median - gap) / tau), normalized over the measured bands.
#
# Exports (on every save / at the end, like moss): ``*_student_*.safetensors``
# (student LoRA, kohya anima format, applied ON TOP of the teacher LoRA),
# ``*_data_lora_<steps>step_*.safetensors`` (one exact-rank kohya LoRA =
# strength * teacher + student, used at strength 1 on the stock model; only
# with --teacher_lora_path), ``*_disc_*.safetensors`` + ``*_disc_heads_*.safetensors``.
#
# Memory: backbone (frozen) + student/disc role LoRAs + heads + VAE + Qwen3.
# The old separate-critic copy, full-finetune mode, spectral norm and the
# per-band routing args are gone; the moss algorithm needs LoRA roles.
# Use --gradient_checkpointing / --blocks_to_swap / smaller batch sizes on
# small GPUs.

import argparse
import gc
import math
import os
import time
from multiprocessing import Value

import toml
import torch
import torch.distributed as dist
from tqdm import tqdm

from library import flux_train_utils
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from library import anima_train_utils, anima_utils, strategy_base, strategy_anima, sai_model_spec
from library.anima_dmad import (
    AnimaDmad,
    AnimaTeacherLatentDataset,
    PowerEMA,
    broadcast_params,
    clip_step,
    enable_dmad_block_checkpointing,
    set_lora_role,
)

import library.accelerator_setup as accelerator_setup
import library.args as args_util
import library.dataset as dataset_util
import library.deepspeed_utils as deepspeed_utils
import library.logging_util as logging_util
import library.sampling as sampling

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util
from library.config_util import ConfigSanitizer, BlueprintGenerator
from library.custom_train_functions import add_custom_train_arguments


def add_dmad_arguments(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("DMAD arguments (method of moss/train_dmad.py)")
    # teacher
    g.add_argument("--teacher_model_path", type=str, default=None,
                   help="Frozen teacher Anima checkpoint; its weights also INITIALIZE the shared backbone "
                        "(the student LoRA starts at the teacher). Default: --pretrained_model_name_or_path")
    g.add_argument("--teacher_lora_path", type=str, default=None,
                   help="Frozen, always-on teacher LoRA (kohya anima format) on the block projections; the "
                        "teacher = backbone + strength * this LoRA (moss teacher-style). Student and disc "
                        "roles start at the teacher automatically (zero-init LoRA B)")
    g.add_argument("--teacher_lora_strength", type=float, default=1.0,
                   help="Strength of the frozen teacher LoRA")
    g.add_argument("--teacher_lora_alpha", type=float, default=None,
                   help="Alpha of the frozen teacher LoRA (default: the alpha stored in the file, else the rank)")
    g.add_argument("--teacher_latent_path", type=str, default=None,
                   help="LMDB of pre-generated teacher latents (Q) built by anima_dmad_generate_teacher.py. "
                        "When set, Q is read from cache (no online teacher evaluation) and the disc's teacher "
                        "chunk uses the cached sample's own prompt")
    g.add_argument("--dmad_teacher_batch_size", type=int, default=8,
                   help="Batch size for cached teacher latents (exact match with the train batch is not required)")
    g.add_argument("--teacher_steps", type=int, default=8,
                   help="Euler steps for the online teacher samples (moss default: 8; ignored with --teacher_latent_path)")
    g.add_argument("--teacher_flow_shift", type=float, default=None,
                   help="Flow shift of the teacher Euler grid (default: --dmad_flow_shift)")
    g.add_argument("--teacher_cfg", type=float, default=1.0,
                   help="CFG scale for online teacher sampling, baked into Q (> 1 needs --teacher_negative_prompt; "
                        "the moss teacher runs without CFG; ignored with --teacher_latent_path)")
    g.add_argument("--teacher_negative_prompt", type=str, default="", help="Negative prompt for on-the-fly teacher CFG")
    # student / student grid
    g.add_argument("--student_steps", "--num_denoising_steps", dest="student_steps", type=int, default=1,
                   help="Generator steps at inference (re-noise sampler). 1 = one-step (t = 1 pure noise); "
                        ">1 = multi-step (moss: 4)")
    g.add_argument("--dmad_flow_shift", type=float, default=3.0,
                   help="Flow shift of the student / teacher sigma grids (moss uses its model's shift)")
    g.add_argument("--dmad_lora_rank", type=int, default=128,
                   help="Rank of BOTH the student and the disc role LoRAs (moss default)")
    g.add_argument("--dmad_lora_alpha", type=float, default=128.0,
                   help="Alpha of both role LoRAs (moss default)")
    g.add_argument("--dmad_head_hidden", type=int, default=1024,
                   help="Hidden width of the two discriminator heads (Linear-SiLU-Linear; moss default)")
    g.add_argument("--disc_u_range", type=float, nargs=2, default=[0.001, 0.999],
                   help="Uniform base-time u range for the generator and disc noising (0 < lo < hi < 1)")
    # losses / gap routing
    g.add_argument("--lambda_teacher", type=float, default=1.0, help="Teacher-head weight in the generator loss")
    g.add_argument("--lambda_real", type=float, default=1.0, help="Real-head weight in the generator loss")
    g.add_argument("--gap_bands", type=int, default=10, help="Number of equal-width gap-routing bands over [0, 1)")
    g.add_argument("--gap_beta", type=float, default=0.99, help="Gap EMA coefficient")
    g.add_argument("--gap_tau", type=float, default=2.0, help="Gap-routing temperature (paper: 0.5 ImageNet/SDXL, 2 for Wan/H3 DiTs)")
    g.add_argument("--gap_floor", type=float, default=1e-4, help="Floor for the band-weight normalization")
    # optimizers (moss-style plain AdamW + warmup ramps for both roles)
    g.add_argument("--disc_lr", type=float, default=4e-5,
                   help="Learning rate for the disc role + heads (moss default; the generator uses --learning_rate)")
    g.add_argument("--dmad_adam_beta1", type=float, default=0.0)
    g.add_argument("--dmad_adam_beta2", type=float, default=0.99)
    g.add_argument("--dmad_weight_decay", type=float, default=0.0, help="Weight decay for both optimizers (moss default: 0)")
    g.add_argument("--dmad_warmup_steps", type=int, default=0, help="LR ramp warmup for both optimizers (moss warmup)")
    g.add_argument("--disc_warmup_steps", type=int, default=0, help="Updates where only the disc trains (moss disc_warmup)")
    g.add_argument("--dmad_ema_sigma_rel", type=float, default=0.1,
                   help="Power-EMA relative std for the student role (moss ema_sigma_rel)")


def ramp(n, warmup):
    """Linear warmup factor (moss): min(1, (n + 1) / warmup) when warmup > 0."""
    return min(1.0, (n + 1) / warmup) if warmup else 1.0


def lora_metadata(args, step, epoch_no):
    """SAI-style metadata for the exported kohya LoRA files."""
    from library import model_io

    meta = {
        "ss_network_module": "dmad_patch_lora (student/disc role LoRA, moss/train_dmad.py)",
        "ss_network_dim": str(args.dmad_lora_rank),
        "ss_network_alpha": str(args.dmad_lora_alpha),
        "ss_steps": str(step),
        "ss_epoch": str(epoch_no),
    }
    try:
        sai = model_io.get_sai_model_spec_dataclass(None, args, False, True, False, anima="preview").to_metadata_dict()
        meta.update(sai)
    except Exception as e:
        logger.warning(f"sai model spec metadata failed: {e}")
    return meta


def save_dmad(args, accelerator, dmad, step, epoch_no):
    """Write the moss-style exports: student LoRA (+ EMA), combined teacher+student
    LoRA, disc LoRA and heads (kohya anima format, ``lora_unet_`` keys)."""
    from safetensors.torch import save_file as safetensors_save

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    base = args.output_name or "anima-dmad"
    metadata = lora_metadata(args, step, epoch_no)

    # moss naming -> per-step output files
    stem_map = {
        "student_lora.safetensors": f"{base}_student_{step:06d}.safetensors",
        "student_ema.safetensors": f"{base}_student_ema_{step:06d}.safetensors",
        f"data_lora_{dmad.steps}step.safetensors": f"{base}_data_lora_{dmad.steps}step_{step:06d}.safetensors",
        "disc_lora.safetensors": f"{base}_disc_{step:06d}.safetensors",
        "disc_heads.safetensors": f"{base}_disc_heads_{step:06d}.safetensors",
    }
    lora_stems = {stag for stag in stem_map if "lora" in stag and "heads" not in stag}
    try:
        rows = dmad.exports(dmad.ema.shadow)  # keyed by file stem: {stem: {key: tensor}}
        for stem, tensors in rows.items():
            payload = {k: v.detach().float().contiguous().cpu() if torch.is_tensor(v) else v for k, v in tensors.items()}
            meta = dict(metadata) if stem in lora_stems else {"format": "pt"}
            safetensors_save(payload, os.path.join(args.output_dir, stem_map[stem]), metadata=meta)
        logger.info(f"DMAD exports written at step {step}")
    except Exception as e:
        logger.warning(f"DMAD export save failed: {e}")


def train(args):
    args_util.verify_training_args(args)
    accelerator_setup.prepare_dataset_args(args, True)
    setup_logging(args, reset=True)
    flux_train_utils.log_timestep_sampling_info(args)

    assert not getattr(args, "fused_backward_pass", False), "fused_backward_pass is not supported with the DMAD script"
    assert not getattr(args, "deepspeed", False), "deepspeed is not supported yet with the DMAD script"
    assert 0 < args.disc_u_range[0] < args.disc_u_range[1] < 1, "--disc_u_range must satisfy 0 < lo < hi < 1"
    assert args.dmad_lora_rank >= 1 and args.student_steps >= 1 and args.teacher_steps >= 1
    if args.optimizer_type is not None and args.optimizer_type.lower() not in ("adamw", "adamw_with_warmup"):
        logger.warning(f"--optimizer_type {args.optimizer_type} is ignored: DMAD always uses plain AdamW "
                       f"(betas {args.dmad_adam_beta1}, {args.dmad_adam_beta2}) for the generator and disc roles")
    if args.teacher_flow_shift is None:
        args.teacher_flow_shift = args.dmad_flow_shift
    if getattr(args, "resume", None) and str(args.resume) not in ("none", ""):
        logger.warning("--resume is not supported by the DMAD script (moss-style role LoRA state); ignoring it")

    use_teacher_cache = args.teacher_latent_path is not None
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

    # Shared frozen backbone: the teacher checkpoint when given (the student LoRA
    # then starts AT the teacher), otherwise the base checkpoint.
    backbone_path = args.teacher_model_path or args.pretrained_model_name_or_path
    logger.info(f"Loading frozen DiT backbone from {backbone_path} "
                f"({'teacher checkpoint (student starts AT the teacher)' if args.teacher_model_path else 'base checkpoint'})")
    dit = anima_utils.load_anima_model("cpu", backbone_path, args.attn_mode, args.split_attn,
                                       "cpu", dit_weight_dtype=None)
    dit.requires_grad_(False)

    # cast the backbone to its master dtype (fp32 unless --full_fp16/--full_bf16)
    dit_weight_dtype = weight_dtype
    if args.full_fp16:
        assert args.mixed_precision == "fp16"
        dit_weight_dtype = torch.float16
    elif args.full_bf16:
        assert args.mixed_precision == "bf16"
        dit_weight_dtype = torch.bfloat16
    else:
        dit_weight_dtype = torch.float32
    dit.to(device=accelerator.device, dtype=dit_weight_dtype)

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        enable_dmad_block_checkpointing(dit, cpu_offload=args.cpu_offload_checkpointing)

    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    if is_swapping_blocks:
        logger.info(f"Enable block swap on the shared backbone: blocks_to_swap={args.blocks_to_swap}")
        dit.enable_block_swap(args.blocks_to_swap, accelerator.device)
        dit.move_to_device_except_swap_blocks(accelerator.device)

    # Role LoRAs are injected on the device-resident backbone, AFTER all casts.
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    dmad = AnimaDmad(args, dit, accelerator.device, rank=rank, world=world, autocast_ctx=accelerator.autocast)
    logger.info(f"DMAD roles: student {sum(p.numel() for p in dmad.gen_named.values()):,} params, "
                f"disc {sum(p.numel() for p in dmad.disc_named.values()):,} params; student sigmas "
                f"{[round(s, 4) for s in dmad.student_sigmas]}; teacher sigmas "
                f"{[round(s, 4) for s in dmad.teacher_sigmas]}")
    accelerator.print(f"teacher: {'cached LMDB' if use_teacher_cache else 'backbone (role None)'} "
                      f"+ {'frozen LoRA ' + str(args.teacher_lora_path) if args.teacher_lora_path else 'no teacher LoRA'}; "
                      f"teacher steps {args.teacher_steps}, shift {args.dmad_flow_shift}")

    # identical replicas across ranks (moss does the same after inject)
    broadcast_params(list(dmad.gen_named.values()) + list(dmad.disc_named.values()))
    dmad.ema = PowerEMA(dmad.gen_named, args.dmad_ema_sigma_rel)

    gen_params = list(dmad.gen_named.values())
    disc_params = list(dmad.disc_named.values())
    betas = (args.dmad_adam_beta1, args.dmad_adam_beta2)
    gen_opt = torch.optim.AdamW([{"params": gen_params, "base_lr": args.learning_rate}],
                                lr=args.learning_rate, betas=betas, weight_decay=args.dmad_weight_decay)
    disc_opt = torch.optim.AdamW([{"params": disc_params, "base_lr": args.disc_lr}],
                                 lr=args.disc_lr, betas=betas, weight_decay=args.dmad_weight_decay)

    train_dataset_group.set_current_strategies()
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group, batch_size=1, shuffle=True, collate_fn=collator,
        num_workers=n_workers, persistent_workers=args.persistent_data_loader_workers)
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
    train_dataset_group.set_max_train_steps(args.max_train_steps)
    accelerator.print(f"number of trainable generator (student role) parameters: {sum(p.numel() for p in gen_params):,}")
    accelerator.print(f"number of trainable discriminator (disc role + heads) parameters: {sum(p.numel() for p in disc_params):,}")
    train_dataloader = accelerator.prepare(train_dataloader)

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

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers("anima-dmad" if args.log_tracker_name is None else args.log_tracker_name,
                                  config=args_util.get_sanitized_config_or_none(args), init_kwargs=init_kwargs)

    if is_swapping_blocks:
        dit.prepare_block_swap_before_forward()

    loss_recorder = logging_util.LossRecorder()

    def render_samples(epoch_no, steps_no):
        """Sample previews with the EMA student (moss-style swap), live at step 0."""
        set_lora_role(dit, "default")
        dmad.set_model_mode(train=False)
        swapped = False
        if dmad.ema.count > 0:
            dmad.swap_ema()
            swapped = True
        anima_train_utils.sample_images(accelerator, args, epoch_no, steps_no, dit, vae, qwen3_text_encoder,
                                        tokenize_strategy, text_encoding_strategy, sample_prompts_te_outputs)
        if swapped:
            dmad.swap_ema()
        dmad.set_model_mode(train=True)

    render_samples(0, global_step)

    # micro-batch iteration with epoch bookkeeping
    it = iter(train_dataloader)
    epoch_no = 0
    epoch_end_pending = False

    def next_batch():
        nonlocal it, epoch_no, epoch_end_pending
        try:
            return next(it)
        except StopIteration:
            epoch_end_pending = True
            epoch_no += 1
            it = iter(train_dataloader)
            return next(it)

    def encode_cond(batch):
        """latents (4D), cond tuple."""
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
        return latents, (prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask)

    accum = args.gradient_accumulation_steps
    warmup_eval = lambda n: ramp(n, args.dmad_warmup_steps)

    dit.train()

    for _step in range(args.max_train_steps):
        step_started = time.monotonic()
        train_gen = global_step >= args.disc_warmup_steps
        scale = warmup_eval(dmad.ema.count) if train_gen else 0.0
        for group in gen_opt.param_groups:
            group["lr"] = group["base_lr"] * scale
        for group in disc_opt.param_groups:
            group["lr"] = group["base_lr"] * warmup_eval(global_step)

        stats, items, teacher_seconds = [], [], 0.0
        gen_opt.zero_grad(set_to_none=True)

        # ---- generator turn (micro-batches accumulated) ----
        for _ in range(accum):
            batch = next_batch()
            latents, cond = encode_cond(batch)
            noise = torch.randn_like(latents)
            if use_teacher_cache:
                t_batch = next(teacher_iter)
                teacher_q = t_batch["latents"].to(accelerator.device, dtype=dit_weight_dtype)
                with torch.no_grad():
                    t_tokens = tokenize_strategy.tokenize(t_batch["prompt"])
                    tm, attn_mask, t5_input_ids, t5_attn_mask = text_encoding_strategy.encode_tokens(
                        tokenize_strategy, [qwen3_text_encoder], t_tokens)
                tm = tm.to(accelerator.device, dtype=dit_weight_dtype)
                attn_mask = attn_mask.to(accelerator.device)
                t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
                t5_attn_mask = t5_attn_mask.to(accelerator.device)
                teacher_cond = (tm, attn_mask, t5_input_ids, t5_attn_mask)
            else:
                mark = time.monotonic()
                teacher_q = dmad.teacher_sample(cond, torch.randn_like(latents), neg_cond=neg_cond)
                teacher_seconds += time.monotonic() - mark
                teacher_cond = cond

            item, record = dmad.generator_step(cond, teacher_cond, noise, latents, teacher_q, accum, train_gen)
            items.append(item)
            stats.append(record)

        gen_norm = torch.zeros((), device=accelerator.device)
        if train_gen:
            gen_norm = clip_step(gen_params, gen_opt, args.max_grad_norm, "generator")
            dmad.ema.update(dmad.gen_named)

        # ---- discriminator turn ----
        disc_opt.zero_grad(set_to_none=True)
        gap_stats = []
        disc_norm = torch.zeros((), device=accelerator.device)
        for item in items:
            record, gap_part = dmad.disc_step(item, accum)
            stats.append(record)
            gap_stats.append(gap_part)
        disc_norm = clip_step(disc_params, disc_opt, args.max_grad_norm, "discriminator")
        dmad.set_trainable(gen=False, disc=False)
        dmad.update_gap(torch.cat(gap_stats, 1))
        del items

        # ---- stats / logging ----
        keys = sorted({k for s in stats for k in s})
        values = [sum(s.get(k, 0.0) for s in stats) / len(stats) for k in keys]
        if world > 1:
            reduced = torch.tensor(values, device=accelerator.device)
            dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
            values = reduced.tolist()
        metrics = dict(zip(keys, values))
        current_loss = metrics.get("loss_g", 0.0) + metrics.get("loss_d", 0.0)
        loss_recorder.add(epoch=epoch_no, step=global_step, loss=current_loss)

        peak_gpu = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        seconds = time.monotonic() - step_started

        progress_bar.update(1)
        global_step += 1
        if len(accelerator.trackers) > 0:
            logged = {"loss": current_loss, "gen_grad_norm": gen_norm.item(), "disc_grad_norm": disc_norm.item(),
                      "lr_scale": scale, "train_gen": train_gen, "peak_gpu_bytes": peak_gpu,
                      "seconds": seconds, "teacher_seconds": teacher_seconds}
            logged.update(metrics)
            logs_gap = dmad.gap.gap.tolist()
            for b, v in enumerate(logs_gap):
                if math.isnan(v):
                    continue
                logged[f"gap_b{b}"] = v
            accelerator.log(logged, step=global_step)
        progress_bar.set_postfix(**{"avr_loss": loss_recorder.moving_average})
        if accelerator.is_main_process:
            gen_part = (f"G={metrics.get('loss_g', 0.0):.3f} hT={metrics.get('g_h_teacher', 0.0):.2f} "
                        f"hR={metrics.get('g_h_real', 0.0):.2f} w={metrics.get('teacher_weight', 0.0):.2f} "
                        f"k={metrics.get('k', 0)} |g|={gen_norm.item():.3g} " if train_gen else "")
            accelerator.print(f"[update {global_step}/{args.max_train_steps}] "
                              f"D={metrics.get('loss_d', 0.0):.3f} (T {metrics.get('loss_d_teacher', 0.0):.3f} "
                              f"R {metrics.get('loss_d_real', 0.0):.3f}) {gen_part}|d|={disc_norm.item():.3g} "
                              f"teacher {teacher_seconds:.1f}s {seconds:.1f}s peak={peak_gpu / 2**30:.0f}GiB")

        if args.sample_every_n_steps is not None and global_step % args.sample_every_n_steps == 0:
            render_samples(None, global_step)
        if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
            save_dmad(args, accelerator, dmad, global_step, epoch_no + 1)

        if epoch_end_pending:
            epoch_end_pending = False
            accelerator.print(f"\nepoch {epoch_no} complete")
            accelerator.wait_for_everyone()
            render_samples(epoch_no, global_step)
            if args.save_every_n_epochs is not None:
                save_dmad(args, accelerator, dmad, global_step, epoch_no)
            if len(accelerator.trackers) > 0:
                accelerator.log({"loss/epoch": loss_recorder.moving_average, "epoch": epoch_no}, step=global_step)

    # ---- final save ----
    accelerator.wait_for_everyone()
    set_lora_role(dit, "default")
    save_dmad(args, accelerator, dmad, global_step, epoch_no + 1)
    if accelerator.is_main_process:
        logger.info("model saved.")

    accelerator.end_training()


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
