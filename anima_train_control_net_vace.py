# Anima ControlNet-VACE training script
# (derived from anima_train.py; DiT frozen, only the VACE control branch is trained)
#
# VACE = parallel transformer "control branch" duplicating select base blocks,
# with zero-conv additive hint injection. See
# networks/control_net_vace_anima.py for the architecture.

import argparse
import copy
import gc
import math
import os
from pathlib import Path
from multiprocessing import Value
from typing import Optional

# bucket 切替で発生しうる稀な断片化 OOM 対策
# torch import より前に環境変数を設定する必要があるため、ここで setdefault しておく.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import toml
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from library import flux_train_utils, qwen_image_autoencoder_kl
from library.device_utils import init_ipex, clean_memory_on_device
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

init_ipex()

from accelerate.utils import set_seed
from library import (
    deepspeed_utils,
    anima_train_utils,
    anima_utils,
    strategy_base,
    strategy_anima,
    sai_model_spec,
)
import library.accelerator_setup as accelerator_setup
import library.args as args_util
import library.dataset as dataset_util
import library.model_io as model_io
import library.optimizer as optimizer_util
import library.logging_util as logging_util
import library.loss as loss_util
import library.checkpoint_io as checkpoint_io
import library.sampling as sampling
import library.config_util as config_util
from library.config_util import ConfigSanitizer, BlueprintGenerator
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments
from library.utils import setup_logging, add_logging_arguments

import networks.control_net_vace_anima as vace_module
from networks.control_net_vace_anima import (
    ControlNetVACEAnima,
    AnimaControlNetVACEWrapper,
    save_vace_model,
    load_vace_weights,
    VACE_ARCH_VERSION,
    VACE_CONDITION_STRATEGIES,
    VACE_COPY_STRATEGIES,
)

setup_logging()
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample-image hook builder: stages a VAE-encoded control latent before each prompt
# ---------------------------------------------------------------------------
def _load_control_image(path: str, width: int, height: int, device, dtype) -> torch.Tensor:
    """Load a control image and return (1, 3, H, W) in [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # HWC, [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 3, H, W)
    return tensor.to(device=device, dtype=dtype)


def _make_vace_sample_hooks(args, wrapper: AnimaControlNetVACEWrapper, vae, dit_dtype):
    """Build (on_prompt_start, on_prompt_end) callbacks that VAE-encode the
    control image and stage it on the wrapper before each sample prompt is
    rendered. The wrapper's patched forward picks up the staged latent."""

    unwrapped = wrapper  # caller passes the already-unwrapped wrapper

    def on_prompt_start(prompt_dict: dict, accelerator):
        ci_path = prompt_dict.get("controlnet_image")
        if ci_path is None:
            logger.warning(
                "no control image for sample prompt (use '--cn <path>'); running base DiT without VACE cond"
            )
            unwrapped._pending_control_latent = None
            return

        if not os.path.isfile(ci_path):
            logger.warning(f"control image not found: {ci_path}; running base DiT without VACE cond")
            unwrapped._pending_control_latent = None
            return

        # Match the dimensions used by _sample_image_inference (rounded to multiple of 16)
        w = prompt_dict.get("width", 512)
        h = prompt_dict.get("height", 512)
        h = max(64, h - h % 16)
        w = max(64, w - w % 16)
        cond_image = _load_control_image(ci_path, w, h, accelerator.device, dit_dtype)

        # VAE-encode the control image to a 16-ch latent
        with torch.no_grad(), accelerator.autocast():
            control_latent = vae.encode_pixels_to_latents(cond_image)  # (1, C, H/8, W/8) or (1, C, 1, H/8, W/8)
            if control_latent.dim() == 4:
                control_latent = control_latent.unsqueeze(2)  # -> (1, C, 1, H/8, W/8)
            control_latent = control_latent.to(dtype=dit_dtype)

        # Stage the latent; the patched forward_mini_train_dit will consume it
        unwrapped._pending_control_latent = control_latent

    def on_prompt_end(prompt_dict: dict):
        unwrapped._pending_control_latent = None

    return on_prompt_start, on_prompt_end


def add_anima_vace_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--vace_block_every_n",
        type=int,
        default=2,
        help=(
            "create one control block every n base model blocks. "
            "E.g. 2 with --vace_condition_strategy=spaced conditions base blocks {0,2,4,...}. "
            "default: 2"
        ),
    )
    parser.add_argument(
        "--vace_condition_strategy",
        type=str,
        default="spaced",
        choices=list(VACE_CONDITION_STRATEGIES),
        help=(
            "how control blocks correspond to base blocks. "
            "'spaced' (default): every n-th base block, where n=--vace_block_every_n. "
            "'first_n': the first num_blocks//n base blocks."
        ),
    )
    parser.add_argument(
        "--vace_copy_strategy",
        type=str,
        default="spaced_n",
        choices=list(VACE_COPY_STRATEGIES),
        help=(
            "how base weights are copied into the control branch at init. "
            "'spaced_n' (default): control block i gets base block i*n's weights (mirrors spaced). "
            "'first_n': control block i gets base block i's weights. "
            "'none': random init (no copy)."
        ),
    )
    parser.add_argument(
        "--vace_copy_n",
        type=int,
        default=-1,
        help="number of control blocks to copy weights into (-1 = all control blocks).",
    )
    parser.add_argument(
        "--vace_control_context_scale",
        type=float,
        default=1.0,
        help="global multiplier on the injected hint at every base block (default: 1.0).",
    )
    parser.add_argument(
        "--vace_after_proj",
        action="store_true",
        help="enable the per-block after_proj (zero-conv skip connection). On by default; pass --no_vace_after_proj to disable.",
    )
    parser.add_argument(
        "--no_vace_after_proj",
        action="store_true",
        help="disable the per-block after_proj; the control branch becomes a plain sequential chain.",
    )
    parser.add_argument(
        "--vace_weights",
        type=str,
        default=None,
        help="pretrained VACE weights to resume from / 学習を再開する VACE の初期重み",
    )


def train(args):
    args_util.verify_training_args(args)
    accelerator_setup.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    # Dual-output logging: console (via setup_logging) + append-mode file so
    # progress can be tailed from another SSH session without losing the
    # interactive console output.
    _log_file = getattr(args, "log_file", None)
    if _log_file is None:
        _log_file = Path(args.output_dir) / "train.log" if args.output_dir else None
    if _log_file:
        _log_path = Path(_log_file)
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(str(_log_path), mode="a", encoding="utf-8")
        _fh.setLevel(logging.INFO)
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(_fh)
        logger.info("logging to %s", _log_path)

    if not args.skip_cache_check:
        args.skip_cache_check = args.skip_latents_validity_check

    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
        args.cache_text_encoder_outputs = True

    # MVP では未対応の機能を明示的に弾く (LLLite と同じ制約 + gradient_checkpointing)
    assert args.blocks_to_swap is None or args.blocks_to_swap == 0, (
        "blocks_to_swap is not supported in Anima ControlNet-VACE training (MVP)"
    )
    assert not args.cpu_offload_checkpointing, (
        "cpu_offload_checkpointing is not supported in Anima ControlNet-VACE training (hooks + checkpointing incompatible)"
    )
    assert not args.unsloth_offload_checkpointing, (
        "unsloth_offload_checkpointing is not supported in Anima ControlNet-VACE training"
    )
    assert not args.deepspeed, "deepspeed is not supported in Anima ControlNet-VACE training (MVP)"
    assert not args.fused_backward_pass, "fused_backward_pass is not supported in Anima ControlNet-VACE training (MVP)"

    cache_latents = args.cache_latents

    if args.seed is not None:
        set_seed(args.seed)

    # latents caching strategy
    if cache_latents:
        latents_caching_strategy = strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # dataset (ControlNet 形式) — conditioning_data_dir に control 画像を置く
    if args.dataset_class is not None:
        train_dataset_group = dataset_util.load_arbitrary_dataset(args)
        val_dataset_group = None
    else:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(False, False, True, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "conditioning_data_dir"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning("ignore following options because config file is found: {0}".format(", ".join(ignored)))
        else:
            user_config = {
                "datasets": [
                    {
                        "subsets": config_util.generate_controlnet_subsets_config_by_subdirs(
                            args.train_data_dir,
                            args.conditioning_data_dir,
                            args.caption_extension,
                        )
                    }
                ]
            }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = dataset_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(16)

    if args.debug_dataset:
        if args.cache_text_encoder_outputs:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
                strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                    args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
                )
            )
        logger.info("Loading tokenizers...")
        weight_dtype, save_dtype = accelerator_setup.prepare_dtype(args)
        qwen3_text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
        t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_tokenizer=qwen3_tokenizer,
            t5_tokenizer=t5_tokenizer,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        train_dataset_group.set_current_strategies()
        dataset_util.debug_dataset(train_dataset_group, True)
        return

    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify train_data_dir / conditioning_data_dir / dataset_config.")
        return

    if cache_latents:
        assert train_dataset_group.is_latent_cacheable(), "when caching latents, color_aug/random_crop cannot be used"
    if args.cache_text_encoder_outputs:
        assert train_dataset_group.is_text_encoder_output_cacheable(
            cache_supports_dropout=True
        ), "when caching text encoder output, shuffle_caption / token_warmup_step / caption_tag_dropout_rate cannot be used"

    # accelerator
    logger.info("prepare accelerator")
    accelerator = accelerator_setup.prepare_accelerator(args)
    weight_dtype, save_dtype = accelerator_setup.prepare_dtype(args)

    # tokenizers and strategies
    logger.info("Loading tokenizers...")
    qwen3_text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
    t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)

    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    qwen3_text_encoder.to(weight_dtype)
    qwen3_text_encoder.requires_grad_(False)

    sample_prompts_te_outputs = None
    if args.cache_text_encoder_outputs:
        qwen3_text_encoder.to(accelerator.device)
        qwen3_text_encoder.eval()

        text_encoder_caching_strategy = strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, is_partial=False
        )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_caching_strategy)

        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([qwen3_text_encoder], accelerator)

        if args.sample_prompts is not None:
            if not os.path.isfile(args.sample_prompts):
                logger.warning(
                    "sample_prompts file not found: %s — sampling disabled for this run. "
                    "Create the file to enable training progress samples.",
                    args.sample_prompts,
                )
                args.sample_prompts = None
            else:
                logger.info(f"Cache Text Encoder outputs for sample prompts: {args.sample_prompts}")
                prompts = sampling.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"  cache TE outputs for: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, [qwen3_text_encoder], tokens_and_masks
                                )

        accelerator.wait_for_everyone()
        qwen3_text_encoder = None
        gc.collect()
        clean_memory_on_device(accelerator.device)

    # VAE (also used to VAE-encode control images)
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

    # DiT (frozen)
    logger.info("Loading Anima DiT...")
    dit = anima_utils.load_anima_model(
        "cpu", args.pretrained_model_name_or_path, args.attn_mode, args.split_attn, "cpu", dit_weight_dtype=None
    )

    # NOTE: VACE injects via forward hooks, which are incompatible with
    # torch.utils.checkpoint on the target base blocks (hooks do not fire
    # during recomputation in backward). For correctness we forbid
    # gradient_checkpointing for now; revisit if a per-block override is added.
    assert not args.gradient_checkpointing, (
        "gradient_checkpointing is not supported with VACE hook injection. "
        "Disable --gradient_checkpointing or use a larger GPU. "
        "(Hooks do not fire during the checkpoint recomputation pass.)"
    )

    dit.requires_grad_(False)

    # Build VACE
    use_after_proj = True
    if args.no_vace_after_proj:
        use_after_proj = False

    logger.info(
        f"Building ControlNet-VACE (Anima): block_every_n={args.vace_block_every_n}, "
        f"strategy={args.vace_condition_strategy}, after_proj={use_after_proj}"
    )
    vace = ControlNetVACEAnima(
        dit,
        vace_block_every_n=args.vace_block_every_n,
        condition_strategy=args.vace_condition_strategy,
        use_after_proj=use_after_proj,
    )

    if args.vace_weights is not None:
        logger.info(f"Loading VACE weights from {args.vace_weights}")
        missing, unexpected = load_vace_weights(vace, args.vace_weights, strict=False)
        if missing:
            logger.warning(f"missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.warning(f"unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    else:
        # Init first (random + zeroed before/after projs), then copy base
        # weights into the inner control blocks. Copy-then-init would
        # re-randomize the just-copied blocks via Block.init_weights().
        vace.init_weights()
        vace.copy_weights_to_control_branch(dit, strategy=args.vace_copy_strategy, n=args.vace_copy_n)

    wrapper = AnimaControlNetVACEWrapper(
        dit, vace, control_context_scale=args.vace_control_context_scale
    )

    # Optimizer — only VACE params are trainable
    trainable_params = list(vace.parameters())
    n_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
    accelerator.print(f"number of VACE control blocks: {len(vace.control_blocks)}")
    accelerator.print(f"control_layers (base block indices): {vace.control_layers}")
    accelerator.print(f"number of trainable parameters: {n_trainable:,}")

    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = optimizer_util.get_optimizer(args, trainable_params=trainable_params)
    optimizer_train_fn, optimizer_eval_fn = optimizer_util.get_optimizer_train_eval_fn(optimizer, args)

    # dataloader
    train_dataset_group.set_current_strategies()
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs: {args.max_train_steps}")

    train_dataset_group.set_max_train_steps(args.max_train_steps)
    lr_scheduler = optimizer_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # dtype: DiT frozen but forward goes through autocast; align to weight_dtype
    dit_weight_dtype = weight_dtype
    if args.full_fp16:
        assert args.mixed_precision == "fp16", "full_fp16 requires mixed_precision='fp16'"
        accelerator.print("enable full fp16 training.")
    elif args.full_bf16:
        assert args.mixed_precision == "bf16", "full_bf16 requires mixed_precision='bf16'"
        accelerator.print("enable full bf16 training.")
    dit.to(dit_weight_dtype)
    dit.to(accelerator.device)

    # VACE in fp32 by default (mixed-precision training), or weight_dtype under full_*16
    vace_dtype = torch.float32
    if args.full_fp16 or args.full_bf16:
        vace_dtype = weight_dtype
    vace.to(vace_dtype)
    vace.to(accelerator.device)

    if not args.cache_text_encoder_outputs and qwen3_text_encoder is not None:
        qwen3_text_encoder.to(accelerator.device)
    # VACE VAE-encodes control images on-the-fly every step, so the VAE must
    # stay on device even when target latents are cached (the LLLite trainer
    # can drop the VAE after caching, but VACE cannot).
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device, dtype=weight_dtype)

    clean_memory_on_device(accelerator.device)

    # accelerator.prepare — wrapper を渡す (DiT + VACE を一括で prepare)
    wrapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        wrapper, optimizer, train_dataloader, lr_scheduler
    )

    if args.full_fp16:
        accelerator_setup.patch_accelerator_for_fp16_training(accelerator)

    args_util.resume_from_local_or_hf_if_specified(accelerator, args)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    accelerator.print("running training (Anima ControlNet-VACE)")
    accelerator.print(f"  num train images x repeats: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
    accelerator.print(f"  num epochs: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    accelerator.print(f"  gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps: {args.max_train_steps}")

    # Restore global_step from the checkpoint so the training loop continues
    # toward the original max_train_steps target (not max_train_steps MORE).
    # Without this, global_step resets to 0 on resume → the loop runs
    # max_train_steps additional steps, and the LR scheduler overflows past
    # its configured cosine end.
    global_step = 0
    if args.resume:
        import re as _re
        m = _re.search(r"(\d+)", os.path.basename(args.resume.rstrip("/")))
        if m:
            global_step = int(m.group(1))
            accelerator.print(f"  resumed global_step = {global_step} (from {args.resume})")
    # total=max_train_steps (not range length) so the bar shows X/20000 on
    # resume; initial=global_step sets the start position WITHOUT feeding
    # tqdm fake updates (which would inflate the it/s rate).
    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "anima_controlnet_vace" if args.log_tracker_name is None else args.log_tracker_name,
            config=args_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    # sample-image hooks
    unwrapped_wrapper_for_hooks = accelerator.unwrap_model(wrapper)
    on_prompt_start, on_prompt_end = _make_vace_sample_hooks(
        args, unwrapped_wrapper_for_hooks, vae, dit_weight_dtype
    )

    def _sample_images(epoch_arg, step_arg):
        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch_arg,
            step_arg,
            unwrapped_wrapper_for_hooks.dit,
            vae,
            qwen3_text_encoder,
            tokenize_strategy,
            text_encoding_strategy,
            sample_prompts_te_outputs,
            on_prompt_start=on_prompt_start,
            on_prompt_end=on_prompt_end,
        )

    # --sample_at_first
    optimizer_eval_fn()
    _sample_images(0, global_step)
    optimizer_train_fn()

    # save helper (VACE のみ)
    def _save_vace(ckpt_file: str):
        sai_metadata = model_io.get_sai_model_spec_dataclass(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, anima="preview"
        ).to_metadata_dict()
        sai_metadata["modelspec.architecture"] = "anima-preview/control-net-vace"
        sai_metadata["vace.version"] = VACE_ARCH_VERSION
        sai_metadata["vace.block_every_n"] = str(args.vace_block_every_n)
        sai_metadata["vace.condition_strategy"] = args.vace_condition_strategy
        sai_metadata["vace.use_after_proj"] = "true" if not args.no_vace_after_proj else "false"
        sai_metadata["vace.copy_strategy"] = args.vace_copy_strategy
        sai_metadata["vace.copy_n"] = str(args.vace_copy_n)
        sai_metadata["vace.control_context_scale"] = str(args.vace_control_context_scale)
        unwrapped = accelerator.unwrap_model(wrapper).vace
        save_vace_model(ckpt_file, unwrapped, dtype=save_dtype, metadata=sai_metadata)

    def _save_step(global_step_: int, epoch_: int):
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        ckpt_name = checkpoint_io.get_step_ckpt_name(args, "." + args.save_model_as, global_step_)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        _save_vace(ckpt_file)
        if args.save_state:
            checkpoint_io.save_and_remove_state_stepwise(args, accelerator, global_step_)
        remove_step_no = checkpoint_io.get_remove_step_no(args, global_step_)
        if remove_step_no is not None:
            old_ckpt = os.path.join(
                args.output_dir, checkpoint_io.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
            )
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

    def _save_epoch(epoch_no: int):
        if not accelerator.is_main_process:
            return
        ckpt_name = checkpoint_io.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch_no)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        _save_vace(ckpt_file)
        if args.save_state:
            checkpoint_io.save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)
        remove_epoch_no = checkpoint_io.get_remove_epoch_no(args, epoch_no)
        if remove_epoch_no is not None:
            old_ckpt = os.path.join(
                args.output_dir, checkpoint_io.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
            )
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

    loss_recorder = logging_util.LossRecorder()
    epoch = 0
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        wrapper.train()
        # DiT is frozen; keep eval mode except where train mode is required
        accelerator.unwrap_model(wrapper).dit.eval()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            with accelerator.accumulate(wrapper):
                # --- target latents ---
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=dit_weight_dtype)
                    if latents.ndim == 5:
                        latents = latents.squeeze(2)
                else:
                    with torch.no_grad():
                        images = batch["images"].to(accelerator.device, dtype=weight_dtype)
                        latents = vae.encode_pixels_to_latents(images).to(accelerator.device, dtype=dit_weight_dtype)
                    if torch.any(torch.isnan(latents)):
                        accelerator.print("NaN found in latents, replacing with zeros")
                        latents = torch.nan_to_num(latents, 0, out=latents)

                # --- control latents (VAE-encode the conditioning image) ---
                with torch.no_grad():
                    cond_image = batch["conditioning_images"].to(accelerator.device, dtype=weight_dtype)
                    control_latent = vae.encode_pixels_to_latents(cond_image).to(
                        accelerator.device, dtype=dit_weight_dtype
                    )
                    if torch.any(torch.isnan(control_latent)):
                        accelerator.print("NaN found in control_latent, replacing with zeros")
                        control_latent = torch.nan_to_num(control_latent, 0, out=control_latent)

                # --- text encoder outputs ---
                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    caption_dropout_rates = text_encoder_outputs_list[-1]
                    text_encoder_outputs_list = text_encoder_outputs_list[:-1]
                    text_encoder_outputs_list = text_encoding_strategy.drop_cached_text_encoder_outputs(
                        *text_encoder_outputs_list, caption_dropout_rates=caption_dropout_rates
                    )
                    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_outputs_list
                else:
                    input_ids_list = batch["input_ids_list"]
                    with torch.no_grad():
                        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, [qwen3_text_encoder], input_ids_list
                        )

                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=dit_weight_dtype)
                attn_mask = attn_mask.to(accelerator.device)
                t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
                t5_attn_mask = t5_attn_mask.to(accelerator.device)

                # --- noise + timesteps ---
                noise = torch.randn_like(latents)
                noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
                    args, noise_scheduler_copy, latents, noise, accelerator.device, dit_weight_dtype
                )
                timesteps = timesteps / 1000.0
                if torch.any(torch.isnan(noisy_model_input)):
                    accelerator.print("NaN found in noisy_model_input, replacing with zeros")
                    noisy_model_input = torch.nan_to_num(noisy_model_input, 0, out=noisy_model_input)

                # --- padding mask (zeros; Anima expects (B, 1, H_lat, W_lat)) ---
                bs = latents.shape[0]
                h_latent, w_latent = latents.shape[-2], latents.shape[-1]
                padding_mask = torch.zeros(
                    bs, 1, h_latent, w_latent, dtype=dit_weight_dtype, device=accelerator.device
                )

                # --- 5D-ize for Anima (B, C, T=1, H, W) ---
                noisy_model_input = noisy_model_input.unsqueeze(2)
                if control_latent.dim() == 4:
                    control_latent = control_latent.unsqueeze(2)

                with accelerator.autocast():
                    model_pred = wrapper(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        control_latent=control_latent,
                        padding_mask=padding_mask,
                        source_attention_mask=attn_mask,
                        t5_input_ids=t5_input_ids,
                        t5_attn_mask=t5_attn_mask,
                    )
                model_pred = model_pred.squeeze(2)

                target = noise - latents

                weighting = anima_train_utils.compute_loss_weighting_for_anima(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                huber_c = loss_util.get_huber_threshold_if_needed(args, timesteps, None)
                loss = loss_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean([1, 2, 3])

                if weighting is not None:
                    loss = loss * weighting

                loss_weights = batch["loss_weights"]
                loss = loss * loss_weights
                loss = loss.mean()

                try:
                    accelerator.backward(loss)
                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        f"OOM at step={global_step} epoch={epoch} "
                        f"latents={tuple(latents.shape)} "
                        f"control_latent={tuple(control_latent.shape)} "
                        f"prompt_embeds={tuple(prompt_embeds.shape)}"
                    )
                    try:
                        logger.error(torch.cuda.memory_summary(abbreviated=False))
                    except Exception as e:
                        logger.error(f"failed to dump memory_summary: {e}")
                    raise

                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = list(accelerator.unwrap_model(wrapper).vace.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # clear staged control latent to prevent leakage
            unwrapped = accelerator.unwrap_model(wrapper)
            unwrapped._pending_control_latent = None
            unwrapped._pending_padding_mask = None

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                optimizer_eval_fn()
                _sample_images(None, global_step)
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    _save_step(global_step, epoch)
                optimizer_train_fn()

            current_loss = loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss, "lr": lr_scheduler.get_last_lr()[0]}
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            progress_bar.set_postfix(**{"avr_loss": avr_loss})

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average, "epoch": epoch + 1}
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()

        optimizer_eval_fn()
        if (
            args.save_every_n_epochs is not None
            and (epoch + 1) % args.save_every_n_epochs == 0
            and (epoch + 1) < num_train_epochs
        ):
            _save_epoch(epoch + 1)
        _sample_images(epoch + 1, global_step)
        optimizer_train_fn()

    is_main_process = accelerator.is_main_process

    accelerator.end_training()
    optimizer_eval_fn()

    if args.save_state or args.save_state_on_train_end:
        checkpoint_io.save_state_on_train_end(args, accelerator)

    if is_main_process:
        ckpt_name = checkpoint_io.get_last_ckpt_name(args, "." + args.save_model_as)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving final checkpoint: {ckpt_file}")
        _save_vace(ckpt_file)
        logger.info("model saved.")

    # Restore original dit.forward_mini_train_dit so the dit object can be reused cleanly
    unwrapped = accelerator.unwrap_model(wrapper)
    unwrapped.unpatch_dit_forward()
    unwrapped.remove_hooks()

    del accelerator


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

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="(unsupported in MVP) offload gradient checkpointing to CPU",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="(unsupported in MVP) offload activations to CPU async",
    )
    parser.add_argument(
        "--skip_latents_validity_check",
        action="store_true",
        help="[Deprecated] use 'skip_cache_check' instead",
    )

    add_anima_vace_arguments(parser)

    parser.add_argument("--log-file", type=Path, default=None,
                        help="append-mode log file for cross-session monitoring (default: <output_dir>/train.log)")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    train(args)
