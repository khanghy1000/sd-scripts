"""Anima TDM few-step distillation (data-free, LoRA).

Trajectory Distribution Matching (arXiv:2503.06674) distills a pretrained
rectified-flow teacher into a K-step deterministic student without any images
(captions/prompts only). Ported to Anima from the RamTorch Krea-2 example
(``RamTorchRun/krea2/train_tdm.py``, itself a port of the official TDM demo
from DDPM/eps to rectified flow).

Anima convention (same as the regular trainer): ``x_t = (1-t) x0 + t eps``,
the DiT predicts ``v = eps - x0`` with ``t`` in [0, 1], Euler steps run
``t: 1 -> 0`` as ``x += v * (t_prev - t)``.

The classic recipe needs THREE model copies (frozen teacher, trainable fake
score, trainable student). Here all three share ONE frozen DiT buffer with
TWO LoRA adapters nested on the same Linears (see ``networks/lora_anima.py``:
``LoRAModule`` has no enable flag, so roles are selected with multipliers --
the nested forwards add, i.e. ``out = base + ms*student + mf*fake``)::

    teacher : (ms, mf) = (0, 0)   base weights alone
    student : (ms, mf) = (1, 0)
    fake    : (ms, mf) = (0, 1)

Per iteration (K = ``--tdm_steps``):

  1. Rollout (no grad, student): K Euler steps from noise on the shifted
     schedule, CFG-free, saving each step input ``x_{t_m}`` and its ``x0`` /
     ``eps`` predictions (``x0 = x - t*v``, ``eps = x + (1-t)*v``).
  2. Fake update: per sample draw a segment ``m`` and the ODE point at the
     segment end ``x_mid = (1-t_mid) x0hat + t_mid epshat``, renoise to a
     random ``tau`` under the flow marginal and train the fake adapter to
     denoise ``x_tau`` back toward ``sg(x0hat)``, weighted by the paper's
     importance-sampling ratio and a min-SNR(5) cap.
  3. Student update: fresh segment/tau draw; teacher cond, teacher uncond and
     fake scores at ``(x_tau, tau)`` (no grad); revised target
     ``coop = x0hat + (x0_real_cfg - x0_fake)``; re-run the ONE student step
     WITH grad and pull its ``x0`` prediction toward ``sg(coop)`` with the
     element-wise Pseudo-Huber surrogate (``c = 1e-3``) and the
     DMD normalizer ``1 / mean|x0hat - x0_real_cfg|``.

``tau`` is sampled inside the segment (non-overlapping intervals, paper
Sec. 3.1) unless ``--tdm_no_separate_intervals`` is given.

Run (single GPU, data-free -- prompts file only, one caption per line)::

    python anima_train_tdm_network.py \
      --pretrained_model_name_or_path anima-base.safetensors \
      --qwen3 /path/to/qwen3-0.6b.safetensors --vae /path/to/vae.safetensors \
      --tdm_prompts_file prompts.txt --resolution 512,512 \
      --train_batch_size 8 --tdm_steps 4 --output_dir output/anima-tdm-4step

Optimizer/LR/scheduler flags are shared with ``anima_train_network.py``:
``--learning_rate`` (student), ``--tdm_fake_lr`` (fake score),
``--optimizer_type``/``--optimizer_args``, ``--lr_scheduler`` /
``--lr_warmup_steps`` / ``--lr_decay_steps`` / ..., ``--max_train_steps``,
``--train_batch_size``, ``--max_grad_norm``, ``--save_every_n_steps``,
``--seed``.

Outputs: ``tdm_student_step_<n>.safetensors`` (student LoRA only, same key
layout as ``anima_train_network.py`` -- directly usable for inference /
ComfyUI conversion) and ``tdm_state_step_<n>.safetensors`` (both adapters,
``student.``/``fake.`` prefixes -- resume with ``--tdm_resume_state``).
With EMA enabled (default, ``--tdm_ema_decay``), an extra
``<output_name>_ema_step_<n>.safetensors`` holds the EMA student LoRA (the
recommended inference weight), the state file also carries ``ema.`` entries,
and previews render with the EMA weights.
"""

import argparse
import csv
import os
import random
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    anima_train_utils,
    anima_utils,
    flux_train_utils,
    strategy_anima,
)
import library.accelerator_setup as accelerator_setup
import library.args as args_util
import library.logging_util as logging_util
import library.model_io as model_io
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TDM CLI
# ---------------------------------------------------------------------------

def add_tdm_arguments(parser: argparse.ArgumentParser):
    g = parser.add_argument_group(
        "TDM distillation",
        "Trajectory Distribution Matching few-step distillation (arXiv:2503.06674). Data-free: training needs only --tdm_prompts_file. "
        "Optimizer/LR/scheduler/batch/steps/clipping flags are shared with anima_train_network.py "
        "(--learning_rate, --optimizer_type/--optimizer_args, --lr_scheduler/--lr_warmup_steps, "
        "--train_batch_size, --max_train_steps, --max_grad_norm, --save_every_n_steps, --seed).",
    )
    g.add_argument("--tdm_prompts_file", type=str, default=None,
                   help="Text file with one caption per line. REQUIRED (data-free distillation uses captions only).")
    g.add_argument("--tdm_steps", type=int, default=4,
                   help="K: number of deterministic student steps to distill (default: 4).")
    g.add_argument("--tdm_cfg", type=float, default=3.5,
                   help="Teacher CFG scale for the revised target (paper default: 3.5).")
    g.add_argument("--tdm_neg_prompt", type=str, default="",
                   help="Teacher CFG negative prompt (uncond anchor). Default: empty string (standard). Baked into the student permanently since the student is CFG-free.")
    g.add_argument("--tdm_neg_file", type=str, default=None,
                   help="Optional text file with one negative prompt per line, sampled per batch. Overrides --tdm_neg_prompt when given.")
    g.add_argument("--tdm_shift", type=float, default=3.0,
                   help="Flow shift for the K-step student schedule (Anima sampling default: 3.0).")
    g.add_argument("--tdm_fake_lr", type=float, default=1e-5,
                   help="Fake-score optimizer LR. Student uses shared --learning_rate (default: 2e-6). "
                   "Paper default for PixArt was 2e-5; 1e-5 here (default: 1e-5).")
    g.add_argument("--tdm_no_huber", action="store_true",
                   help="Use plain MSE for the student loss instead of the Pseudo-Huber surrogate.")
    g.add_argument("--tdm_huber_c", type=float, default=1e-3,
                   help="Element-wise Pseudo-Huber c for the student loss (default: 1e-3). Ignored with --tdm_no_huber.")
    g.add_argument("--tdm_no_separate_intervals", action="store_true",
                   help="Sample tau up to the schedule top (demo default) instead of inside the segment (paper default).")
    g.add_argument("--tdm_randmid", action="store_true",
                   help="Land the ODE point at a random point inside the segment (demo --use_randmid regularizer).")
    g.add_argument("--tdm_tau_min", type=float, default=0.025)
    g.add_argument("--tdm_tau_gap", type=float, default=0.0125)
    g.add_argument("--tdm_tau_max", type=float, default=0.9875)
    g.add_argument("--tdm_fake_weights", type=str, default=None,
                   help="Init fake adapter from a LoRA file (student init uses --network_weights).")
    g.add_argument("--tdm_resume_state", type=str, default=None,
                   help="Resume both adapters from a tdm_state_*.safetensors file.")
    g.add_argument("--tdm_resume_optim", type=str, default=None,
                   help="Resume both optimizer (+scheduler) states from a tdm_optim_*.pt file.")
    g.add_argument("--tdm_eval_every", type=int, default=0, help="Render K-step student previews every N steps (0 = disabled, default).")
    g.add_argument("--tdm_preview_n", type=int, default=4, help="Preview images per eval.")
    g.add_argument("--tdm_no_ema", action="store_true",
                   help="Disable the student EMA shadow (by default an EMA of the student LoRA is kept for previews/saving).")
    g.add_argument("--tdm_ema_decay", type=float, default=0.999,
                   help="EMA decay for the student LoRA shadow (default: 0.999). Updated once per optimizer step.")
    g.add_argument("--tdm_ema_warmup", type=int, default=0,
                   help="Optimizer steps before EMA starts tracking. At this step the EMA is re-synced from the student, then decayed (default: 0).")
    g.add_argument("--tdm_ema_update_every", type=int, default=1,
                   help="Update the EMA every N optimizer steps (default: 1).")


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    args_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    add_tdm_arguments(parser)
    # --gradient_checkpointing / --cpu_offload_checkpointing are inherited
    # from train_network.setup_parser(); add the unsloth variant for parity
    # with anima_train.py / anima_train_network.py.
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing. Implies --gradient_checkpointing.",
    )
    parser.set_defaults(
        train_batch_size=8,
        max_train_steps=1000,
        save_every_n_steps=50,
        lr_warmup_steps=100,
        lr_scheduler="constant_with_warmup",
    )
    return parser


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def parse_resolution(resolution: str | None) -> tuple[int, int]:
    if not resolution:
        return (512, 512)
    if "," in resolution:
        w, h = resolution.split(",")[:2]
        return (int(h), int(w))
    s = int(resolution)
    return (s, s)


def make_schedule(steps: int, shift: float, device) -> torch.Tensor:
    """K+1 shifted sigmas from 1.0 to 0.0 (same shift as sampling/do_sample)."""
    sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    if shift != 1.0:
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    return sigmas.to(device)


@torch.no_grad()
def encode_cond(
    tokenize_strategy: strategy_anima.AnimaTokenizeStrategy,
    text_encoding_strategy: strategy_anima.AnimaTextEncodingStrategy,
    qwen3,
    captions: list[str],
    weight_dtype: torch.dtype,
    uncond: bool = False,
):
    """Qwen3 (+T5 ids) encoding. Uncond patches T5 to the </s>+mask convention
    used by cached caption dropout, so teacher CFG has a stable anchor."""
    tokens = tokenize_strategy.tokenize(captions)
    out = text_encoding_strategy.encode_tokens(tokenize_strategy, [qwen3], tokens)
    prompt_embeds, attn_mask, t5_ids, t5_am = out
    device = prompt_embeds.device
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
    attn_mask = attn_mask.to(device)
    t5_ids = t5_ids.to(device, dtype=torch.long)
    t5_am = t5_am.to(device)
    if uncond:
        t5_ids = t5_ids.clone()
        t5_am = t5_am.clone()
        t5_ids[:, 0] = 1  # </s>
        t5_am[:, 0] = 1
    return prompt_embeds, attn_mask, t5_ids, t5_am


def dit_predict_v(dit, x_4d: torch.Tensor, t_vec: torch.Tensor, cond, weight_dtype,
                  ckpt_input_grad: bool = False) -> torch.Tensor:
    """One DiT forward. x_4d: [B, C, H, W] latent; t_vec: [B] in [0, 1].
    Returns v-prediction [B, C, H, W] (caller's dtype).

    ckpt_input_grad: when gradient checkpointing is enabled, mark the latent
    input as requiring grad (mirrors the anima_train_network.py workaround so
    torch checkpointing keeps a grad path on the frozen-base + LoRA graph).
    Only takes effect when grad is enabled (no-op under torch.no_grad()).
    """
    if ckpt_input_grad and torch.is_grad_enabled() and x_4d.dtype.is_floating_point:
        x_4d.requires_grad_(True)
    B = x_4d.shape[0]
    device = x_4d.device
    x_5d = x_4d.unsqueeze(2)  # [B, C, 1, H, W]
    prompt_embeds, attn_mask, t5_ids, t5_am = cond
    h, w = x_4d.shape[-2], x_4d.shape[-1]
    padding_mask = torch.zeros(B, 1, h, w, dtype=weight_dtype, device=device)
    out_5d = dit(
        x_5d, t_vec,
        prompt_embeds,
        padding_mask=padding_mask,
        target_input_ids=t5_ids,
        target_attention_mask=t5_am,
        source_attention_mask=attn_mask,
    )
    return out_5d.squeeze(2)


def set_role(student_net, fake_net, role: str):
    """Flip which LoRA delta the shared base applies (nested adds)."""
    if role == "teacher":
        student_net.set_multiplier(0.0)
        fake_net.set_multiplier(0.0)
    elif role == "student":
        student_net.set_multiplier(1.0)
        fake_net.set_multiplier(0.0)
    elif role == "fake":
        student_net.set_multiplier(0.0)
        fake_net.set_multiplier(1.0)
    else:
        raise ValueError(f"unknown role: {role}")


def zero_all(*nets):
    for net in nets:
        for p in net.parameters():
            if p.grad is not None:
                p.grad = None


# ---------------------------------------------------------------------------
# Student EMA (fp32 shadow of the student LoRA state_dict)
# ---------------------------------------------------------------------------

def init_ema_state(student_net, device) -> dict:
    """Snapshot the student LoRA state_dict in fp32 on ``device``."""
    ema = {}
    for k, v in student_net.state_dict().items():
        if torch.is_tensor(v) and v.is_floating_point():
            ema[k] = v.detach().to(device, dtype=torch.float32).clone()
        else:
            ema[k] = v.detach().clone() if torch.is_tensor(v) else v
    return ema


@torch.no_grad()
def update_ema_state(ema: dict, student_net, decay: float):
    """ema <- decay * ema + (1 - decay) * student (fp32, in place)."""
    src_sd = student_net.state_dict()
    for k, e in ema.items():
        s = src_sd[k]
        if torch.is_tensor(s) and s.is_floating_point():
            e.mul_(decay).add_(s.detach().to(e.device, dtype=torch.float32), alpha=1.0 - decay)


def ema_casted_state_dict(ema: dict, student_net) -> dict:
    """EMA entries cast back to the student's dtype/device (for load/swap)."""
    dst_sd = student_net.state_dict()
    out = {}
    for k, v in dst_sd.items():
        e = ema.get(k, v)
        if torch.is_tensor(e) and torch.is_tensor(v) and e.is_floating_point():
            out[k] = e.to(device=v.device, dtype=v.dtype)
        else:
            out[k] = e
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if not args.tdm_prompts_file or not os.path.isfile(args.tdm_prompts_file):
        raise ValueError("--tdm_prompts_file is required (data-free distillation uses captions only).")
    with open(args.tdm_prompts_file, encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    if not prompts:
        raise ValueError(f"No captions found in {args.tdm_prompts_file}.")
    logger.info(f"Loaded {len(prompts)} captions from {args.tdm_prompts_file} (images are not used).")
    if getattr(args, "tdm_neg_file", None):
        if not os.path.isfile(args.tdm_neg_file):
            raise ValueError(f"--tdm_neg_file not found: {args.tdm_neg_file}.")
        with open(args.tdm_neg_file, encoding="utf-8") as f:
            neg_prompts = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        if not neg_prompts:
            raise ValueError(f"No negative prompts found in {args.tdm_neg_file}.")
        logger.info(f"Loaded {len(neg_prompts)} negative prompts from {args.tdm_neg_file}.")
    else:
        neg_prompts = None
    neg_single = getattr(args, "tdm_neg_prompt", "") or ""
    if neg_prompts is None and neg_single:
        logger.info(f"Using fixed teacher negative prompt ({len(neg_single)} chars). Baked into the student; student stays CFG-free.")

    flux_train_utils.log_timestep_sampling_info(args)
    if args.fp8_base or args.fp8_base_unet:
        logger.warning("fp8_base/fp8_base_unet are not supported for Anima TDM; disabled.")
        args.fp8_base = False
        args.fp8_base_unet = False

    # Gradient checkpointing flags (--gradient_checkpointing /
    # --cpu_offload_checkpointing come from train_network.setup_parser()).
    if getattr(args, "cpu_offload_checkpointing", False) and not getattr(args, "gradient_checkpointing", False):
        logger.warning("cpu_offload_checkpointing is enabled, so gradient_checkpointing is also enabled.")
        args.gradient_checkpointing = True
    if getattr(args, "unsloth_offload_checkpointing", False):
        if not getattr(args, "gradient_checkpointing", False):
            logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled.")
            args.gradient_checkpointing = True
        assert not getattr(args, "cpu_offload_checkpointing", False), \
            "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing."
    use_ckpt = bool(getattr(args, "gradient_checkpointing", False))

    seed = args.seed if args.seed is not None else 42
    set_seed(seed)
    random.seed(seed)

    accelerator = accelerator_setup.prepare_accelerator(args)
    logging_util.init_trackers(accelerator, args, "anima-tdm")
    device = accelerator.device
    weight_dtype, save_dtype = accelerator_setup.prepare_dtype(args)
    logger.info(f"device={device}, weight_dtype={weight_dtype}, save_dtype={save_dtype}")

    # ---- models: frozen base + Qwen3 TE + VAE (previews only) ----
    logger.info("Loading Qwen3 text encoder...")
    qwen3, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
    qwen3.eval()
    qwen3.requires_grad_(False)
    qwen3.to(device)

    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.qwen3,
        t5_tokenizer_path=args.t5_tokenizer_path,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length,
    )
    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()

    logger.info("Loading Anima VAE (preview decode only)...")
    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
    vae.to(weight_dtype)
    vae.eval()
    vae.requires_grad_(False)

    attn_mode = args.attn_mode or ("xformers" if args.xformers else "torch")
    logger.info(f"Loading Anima DiT teacher from {args.pretrained_model_name_or_path} (attn_mode={attn_mode})...")
    dit = anima_utils.load_anima_model(
        device, args.pretrained_model_name_or_path, attn_mode, args.split_attn, device, weight_dtype,
    )
    # Buffers not stored in the checkpoint (RoPE seq/dim ranges, adapter
    # inv_freq) are created on CPU; move the whole module so they land on the
    # training device (params are already there). Mirrors anima_train.py.
    dit.to(device)
    dit.eval()

    if use_ckpt:
        dit.enable_gradient_checkpointing(
            cpu_offload=bool(getattr(args, "cpu_offload_checkpointing", False)),
            unsloth_offload=bool(getattr(args, "unsloth_offload_checkpointing", False)),
        )
        logger.info(
            "gradient checkpointing ENABLED "
            f"(cpu_offload={bool(getattr(args, 'cpu_offload_checkpointing', False))}, "
            f"unsloth_offload={bool(getattr(args, 'unsloth_offload_checkpointing', False))}). "
            "DiT blocks are recomputed on backward; DiT runs in train() mode for the "
            "two WITH-grad forwards and eval() everywhere else."
        )
    else:
        logger.info("gradient checkpointing disabled.")

    # ---- two LoRA adapters nested on the same frozen base ----
    from networks import lora_anima

    rank = args.network_dim or 32
    # Repo convention defaults --network_alpha to 1; for distillation the
    # sane default is alpha == rank (same as the Krea-2 TDM recipe), so an
    # untouched default follows rank. Pass --network_alpha explicitly to override.
    if args.network_alpha in (None, 1) and rank != 1:
        alpha = float(rank)
        logger.info(f"--network_alpha left at default; using alpha=rank={rank} for TDM.")
    else:
        alpha = args.network_alpha if args.network_alpha is not None else float(rank)
    lora_kw = dict(
        verbose="true" if getattr(args, "verbose", False) else "false",
    )
    if getattr(args, "network_args", None):
        for kv in args.network_args:
            if "=" in kv:
                k, v = kv.split("=", 1)
                lora_kw[k.strip()] = v.strip()
    llm_adapter_on = str(lora_kw.get("train_llm_adapter", "false")).lower() == "true"
    logger.info(f"Creating student/fake LoRA adapters (rank={rank}, alpha={alpha}, llm_adapter={llm_adapter_on})...")
    student_net = lora_anima.create_network(1.0, rank, alpha, None, None, dit, neuron_dropout=args.network_dropout, **lora_kw)
    fake_net = lora_anima.create_network(1.0, rank, alpha, None, None, dit, neuron_dropout=args.network_dropout, **lora_kw)
    # Student first so the fake wraps it: out = base + ms*student + mf*fake.
    student_net.apply_to([], dit, apply_text_encoder=False, apply_unet=True)
    fake_net.apply_to([], dit, apply_text_encoder=False, apply_unet=True)
    student_net.to(device, dtype=weight_dtype)
    fake_net.to(device, dtype=weight_dtype)

    # The base (incl. shared norms/modulation/embedders) stays frozen forever;
    # roles must differ ONLY through their own adapters.
    dit.requires_grad_(False)
    dit.eval()
    student_net.train()
    fake_net.train()
    n_s = sum(p.numel() for p in student_net.parameters())
    n_f = sum(p.numel() for p in fake_net.parameters())
    n_base = sum(p.numel() for p in dit.parameters())
    logger.info(f"Params: base {n_base/1e9:.2f}B frozen | student {n_s/1e6:.1f}M | fake {n_f/1e6:.1f}M trainable.")

    if args.network_weights:
        logger.info(f"Init student adapter from {args.network_weights}")
        student_net.load_weights(args.network_weights)
    if args.tdm_fake_weights:
        logger.info(f"Init fake adapter from {args.tdm_fake_weights}")
        fake_net.load_weights(args.tdm_fake_weights)
    if args.tdm_resume_state:
        logger.info(f"Resume both adapters from {args.tdm_resume_state}")
        sd = load_file(args.tdm_resume_state, device="cpu")
        s_sd = {k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")}
        f_sd = {k[len("fake."):]: v for k, v in sd.items() if k.startswith("fake.")}
        student_net.load_state_dict(s_sd, strict=True)
        fake_net.load_state_dict(f_sd, strict=True)

    # ---- student EMA shadow (fp32, student LoRA only) ----
    use_ema = not bool(getattr(args, "tdm_no_ema", False))
    ema_decay = float(getattr(args, "tdm_ema_decay", 0.999) or 0.0)
    ema_warmup = max(0, int(getattr(args, "tdm_ema_warmup", 0) or 0))
    ema_update_every = max(1, int(getattr(args, "tdm_ema_update_every", 1) or 1))
    if use_ema and not (0.0 < ema_decay < 1.0):
        logger.warning(f"--tdm_ema_decay={ema_decay} out of (0, 1); disabling EMA.")
        use_ema = False
    ema_state = None
    if use_ema:
        ema_state = init_ema_state(student_net, device)
        if args.tdm_resume_state:
            e_sd = {k[len("ema."):]: v for k, v in sd.items() if k.startswith("ema.")}
            if e_sd:
                missing = [k for k in student_net.state_dict().keys() if k not in e_sd]
                if missing:
                    logger.warning(f"EMA state in resume file misses {len(missing)} keys; init those from student.")
                    for k in missing:
                        e_sd[k] = student_net.state_dict()[k].detach().cpu()
                with torch.no_grad():
                    for k, e in ema_state.items():
                        v = e_sd[k]
                        e.copy_(v.to(e.device, dtype=torch.float32) if torch.is_tensor(v) and v.is_floating_point()
                                else (v if torch.is_tensor(v) else e))
                logger.info(f"Resumed student EMA from {args.tdm_resume_state} ({len(e_sd)} tensors).")
            else:
                logger.info("Resume file has no ema.* entries; EMA re-initialized from student.")
        logger.info(f"Student EMA ENABLED (decay={ema_decay:g}, warmup={ema_warmup}, update_every={ema_update_every}). "
                    f"Previews and *_ema_*.safetensors use the EMA weights.")
    else:
        logger.info("Student EMA disabled (--tdm_no_ema).")

    from library import optimizer as optimizer_util

    # Optimizer factory shared with anima_train_network.py (same
    # --optimizer_type/--optimizer_args handling via get_optimizer).
    # Two adapters need two LRs: the student uses --learning_rate directly,
    # while the fake score carries its own --tdm_fake_lr as an explicit param
    # group -- the same pattern train_network.py uses for --unet_lr /
    # --text_encoder_lr (per-group "lr" overrides the constructor default).
    student_lr = float(args.learning_rate)
    fake_lr = float(args.tdm_fake_lr)
    _, _, opt_s = optimizer_util.get_optimizer(args, student_net.parameters())
    _, _, opt_f = optimizer_util.get_optimizer(
        args, [{"params": list(fake_net.parameters()), "lr": fake_lr}]
    )
    logger.info(f"Optimizers: student lr={student_lr:g}, fake lr={fake_lr:g}, "
                f"type={args.optimizer_type or 'AdamW'}, optimizer_args={args.optimizer_args}.")
    opt_s_train_fn, opt_s_eval_fn = optimizer_util.get_optimizer_train_eval_fn(opt_s, args)
    opt_f_train_fn, opt_f_eval_fn = optimizer_util.get_optimizer_train_eval_fn(opt_f, args)

    # Shared LR schedulers (same --lr_scheduler/--lr_warmup_steps/--lr_decay_steps/
    # --lr_scheduler_num_cycles/--lr_scheduler_power/... as the regular trainer).
    # One scheduler per optimizer; both stepped once per optimizer step (not per
    # grad-accumulation micro-step). get_scheduler_fix derives total steps from
    # --max_train_steps.
    sched_s = optimizer_util.get_scheduler_fix(args, opt_s, accelerator.num_processes)
    sched_f = optimizer_util.get_scheduler_fix(args, opt_f, accelerator.num_processes)
    logger.info(f"LR schedulers: {args.lr_scheduler} (warmup={args.lr_warmup_steps}, "
                f"max_train_steps={args.max_train_steps}).")
    if args.tdm_resume_optim:
        logger.info(f"Resume optimizers (+schedulers) from {args.tdm_resume_optim}")
        o = torch.load(args.tdm_resume_optim, map_location="cpu", weights_only=False)
        opt_s.load_state_dict(o["student"])
        opt_f.load_state_dict(o["fake"])
        for sched, key in ((sched_s, "sched_student"), (sched_f, "sched_fake")):
            if key in o and hasattr(sched, "load_state_dict"):
                try:
                    sched.load_state_dict(o[key])
                except Exception as e:
                    logger.warning(f"Could not load {key} state (old checkpoint?): {e}")
        global_step = int(o.get("step", 0))
    else:
        global_step = 0

    # ---- geometry / schedule ----
    img_h, img_w = parse_resolution(args.resolution)
    lat_h, lat_w = img_h // 8, img_w // 8
    latent_ch = 16
    logger.info(f"Distilling at {img_w}x{img_h} (latent {lat_w}x{lat_h}x{latent_ch}), K={args.tdm_steps}.")
    K = args.tdm_steps
    ts = make_schedule(K, args.tdm_shift, device)  # [K+1], 1 -> 0
    if args.tdm_no_huber:
        huber_c = None
        logger.info("Student loss: MSE + DMD normalizer.")
    else:
        huber_c = args.tdm_huber_c
        logger.info(f"Student loss: Pseudo-Huber(c={huber_c:.3g}) + DMD normalizer.")
    logger.info(f"tau sampled in {'segment (non-overlapping)' if not args.tdm_no_separate_intervals else 'full range'}.")

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = args.output_name or "anima-tdm"
    csv_path = os.path.join(args.output_dir, "tdm_loss_log.csv")
    new_csv = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if new_csv:
        csv_writer.writerow(["step", "loss_g", "loss_d", "lr_student", "lr_fake", "time"])

    autocast_kw = dict(device_type=device.type if isinstance(device, torch.device) else str(device),
                       dtype=weight_dtype)
    if autocast_kw["device_type"] not in ("cuda", "cpu", "xpu"):
        autocast_kw["device_type"] = "cuda"
    t0 = time.time()

    def save_all(step: int):
        # Schedule-free optimizers must be in eval mode for checkpointing
        # (mirrors train_network.py); no-ops for all other optimizer types.
        opt_s_eval_fn()
        opt_f_eval_fn()
        try:
            _save_all(step)
        finally:
            opt_s_train_fn()
            opt_f_train_fn()

    def _save_all(step: int):
        s_sd = {f"student.{k}": v.detach().cpu().contiguous() for k, v in student_net.state_dict().items()}
        f_sd = {f"fake.{k}": v.detach().cpu().contiguous() for k, v in fake_net.state_dict().items()}
        state_dict = {**s_sd, **f_sd}
        if use_ema and ema_state is not None:
            for k, e in ema_state.items():
                state_dict[f"ema.{k}"] = e.detach().cpu().contiguous()
        state_path = os.path.join(args.output_dir, f"tdm_state_step_{step}.safetensors")
        save_file(state_dict, state_path)
        logger.info(f"[ckpt] both adapters{' + EMA' if use_ema else ''} -> {state_path}")
        student_path = os.path.join(args.output_dir, f"{out_name}_step_{step}.safetensors")
        meta = model_io.get_sai_model_spec_dataclass(None, args, False, True, False, anima="preview").to_metadata_dict()
        meta["ss_tdm_step"] = str(step)
        student_net.save_weights(student_path, save_dtype, meta)
        logger.info(f"[ckpt] student LoRA -> {student_path}")
        if use_ema and ema_state is not None:
            # Save the EMA student via a temporary swap so hashing/metadata
            # reuse the LoRA save path. Raw student is restored afterwards.
            backup_sd = {k: v.detach().clone() for k, v in student_net.state_dict().items()}
            try:
                student_net.load_state_dict(ema_casted_state_dict(ema_state, student_net), strict=True)
                ema_meta = dict(meta)
                ema_meta["ss_tdm_ema"] = f"decay={ema_decay:g}"
                ema_path = os.path.join(args.output_dir, f"{out_name}_ema_step_{step}.safetensors")
                student_net.save_weights(ema_path, save_dtype, ema_meta)
                logger.info(f"[ckpt] student EMA LoRA -> {ema_path}")
            finally:
                student_net.load_state_dict(backup_sd, strict=True)
        optim_state = {"student": opt_s.state_dict(), "fake": opt_f.state_dict(), "step": step + 1}
        for sched, key in ((sched_s, "sched_student"), (sched_f, "sched_fake")):
            try:
                if hasattr(sched, "state_dict"):
                    optim_state[key] = sched.state_dict()
            except Exception as e:
                logger.warning(f"Could not save {key} state: {e}")
        torch.save(optim_state,
                   os.path.join(args.output_dir, f"tdm_optim_step_{step}.pt"))

    @torch.no_grad()
    def preview(step: int):
        dit.eval()
        opt_s_eval_fn()
        opt_f_eval_fn()
        # Previews render with EMA weights when available (restored afterwards);
        # the training rollout always uses the raw student.
        backup_sd = None
        if use_ema and ema_state is not None:
            backup_sd = {k: v.detach().clone() for k, v in student_net.state_dict().items()}
            student_net.load_state_dict(ema_casted_state_dict(ema_state, student_net), strict=True)
        try:
            _preview_rollout(step)
        finally:
            if backup_sd is not None:
                student_net.load_state_dict(backup_sd, strict=True)
            opt_s_train_fn()
            opt_f_train_fn()

    @torch.no_grad()
    def _preview_rollout(step: int):
        dit.eval()
        set_role(student_net, fake_net, "student")
        vae_device = vae.device if hasattr(vae, "device") else device
        try:
            vae.to(device)
        except Exception:
            pass
        shots = prompts[: min(args.tdm_preview_n, len(prompts))]
        imgs = []
        preview_base_seed = seed  # shared --seed drives training RNG and preview base seed
        for i, cap in enumerate(shots):
            with torch.autocast(**autocast_kw):
                cond = encode_cond(tokenize_strategy, text_encoding_strategy, qwen3, [cap], weight_dtype)
                pe, am, t5i, t5a = cond
                if dit.use_llm_adapter:
                    cross = dit.llm_adapter(source_hidden_states=pe, target_input_ids=t5i,
                                            target_attention_mask=t5a, source_attention_mask=am)
                    cross[~t5a.bool()] = 0
                else:
                    cross = pe
                lat = anima_train_utils.do_sample(
                    img_h, img_w, preview_base_seed + i, dit, cross, K, weight_dtype, device,
                    guidance_scale=1.0, flow_shift=args.tdm_shift,
                )
            decoded = vae.decode_to_pixels(lat).float()
            img = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)[0]
            if img.ndim == 4:
                img = img[:, 0]
            img = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
            imgs.append(Image.fromarray(img))
        w0, h0 = imgs[0].size
        grid = Image.new("RGB", (w0 * len(imgs), h0))
        for j, im in enumerate(imgs):
            grid.paste(im, (j * w0, 0))
        path = os.path.join(args.output_dir, f"sample/tdm_step_{step}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        grid.save(path)
        logger.info(f"[preview] K={K}-step student samples -> {path}")
        # Log preview grid to wandb (mirrors library/anima_train_utils.py sample logging).
        if "wandb" in [tracker.name for tracker in accelerator.trackers]:
            wandb_tracker = accelerator.get_tracker("wandb")
            import wandb

            wandb_tracker.log({f"preview_step_{step}": wandb.Image(grid)}, commit=False)
            # Commit the pending preview image (same pattern as train_network.py).
            accelerator.log({}, step=step)
        try:
            vae.to(vae_device)
        except Exception:
            pass
        clean_memory_on_device(device)

    # ---- data-free loop: random caption batches; images never used ----
    # Shared flags: --train_batch_size, --max_train_steps, --max_grad_norm,
    # --save_every_n_steps (same semantics as anima_train_network.py).
    B = args.train_batch_size
    max_steps = args.max_train_steps
    max_grad_norm = float(args.max_grad_norm)
    save_every = int(args.save_every_n_steps) if args.save_every_n_steps else 0
    accum_steps = max(1, int(getattr(args, "gradient_accumulation_steps", 1) or 1))
    if accum_steps > 1:
        logger.info(
            f"gradient accumulation ENABLED: {accum_steps} micro-steps x batch {B} "
            f"= effective batch {B * accum_steps} per optimizer step. "
            f"--max_train_steps={max_steps} counts optimizer steps."
        )
    else:
        logger.info("gradient accumulation disabled (gradient_accumulation_steps=1).")
    pbar = tqdm(range(global_step, max_steps), initial=global_step, total=max_steps, desc="TDM")
    for opt_step in pbar:
        # Accumulation window: zero once, run `accum_steps` micro-batches,
        # step once. Losses are scaled by 1/accum so the accumulated grad
        # equals the mean over the effective (B*accum) batch.
        zero_all(student_net, fake_net)
        loss_g_sum, loss_d_sum = 0.0, 0.0
        for _micro in range(accum_steps):
            batch_caps = random.choices(prompts, k=B)
            if neg_prompts is not None:
                neg_caps = random.choices(neg_prompts, k=B)
            else:
                neg_caps = [neg_single] * B

            with torch.no_grad():
                with torch.autocast(**autocast_kw):
                    cond = encode_cond(tokenize_strategy, text_encoding_strategy, qwen3, batch_caps, weight_dtype)
                    # The </s>+mask patch inside encode_cond(uncond=True) is the cached-dropout
                    # empty-anchor convention: keep it only for blank negatives so a real
                    # negative prompt's first T5 token is not clobbered.
                    use_blank_anchor = all(not s.strip() for s in neg_caps)
                    uncond = encode_cond(tokenize_strategy, text_encoding_strategy, qwen3, neg_caps, weight_dtype, uncond=use_blank_anchor)

            x = torch.randn(B, latent_ch, lat_h, lat_w, device=device, dtype=weight_dtype)

            # 1. student rollout (no grad): K Euler steps, save inputs + x0/eps preds
            xs, x0s, epss = [], [], []
            with torch.no_grad():
                dit.eval()  # ensure eval (grad-ckpt train mode is only for WITH-grad forwards)
                set_role(student_net, fake_net, "student")
                for k in range(K):
                    t_vec = torch.full((B,), ts[k].item(), dtype=torch.float32, device=device)
                    with torch.autocast(**autocast_kw):
                        v = dit_predict_v(dit, x, t_vec, cond, weight_dtype).float()
                    xf = x.float()
                    xs.append(x)
                    x0s.append(xf - ts[k].item() * v)
                    epss.append(xf + (1.0 - ts[k].item()) * v)
                    x = (xf + (ts[k + 1].item() - ts[k].item()) * v).to(weight_dtype)

            def draw_segment():
                """Random segment per sample + flow renoise of its ODE point to tau."""
                m = torch.randint(0, K, (B,), device=device)
                t_m = ts[m]
                t_next = ts[m + 1]
                idx = m.view(1, B, 1, 1, 1).expand(1, B, latent_ch, lat_h, lat_w)
                x_tm = torch.stack(xs).gather(0, idx)[0]
                x0_hat = torch.stack(x0s).gather(0, idx)[0].float()
                eps_hat = torch.stack(epss).gather(0, idx)[0].float()
                t_mid = t_next.clone()
                if args.tdm_randmid:
                    u = torch.rand(B, device=device)
                    t_mid = t_next + u * (0.975 * t_m - t_next).clamp(min=0.0)
                tm4 = t_mid.view(B, 1, 1, 1)
                x_mid = (1.0 - tm4) * x0_hat + tm4 * eps_hat
                lo = torch.maximum(t_mid, torch.full_like(t_mid, args.tdm_tau_min))
                if args.tdm_no_separate_intervals:
                    hi = torch.full_like(t_m, args.tdm_tau_max)
                else:
                    hi = t_m - args.tdm_tau_gap
                hi = torch.maximum(hi, lo + 1e-4)
                tau = lo + torch.rand(B, device=device) * (hi - lo)
                ta4 = tau.view(B, 1, 1, 1)
                c = (1.0 - ta4) / (1.0 - tm4).clamp(min=1e-6)
                beta = (ta4.square() - (c * tm4).square()).clamp(min=0.0).sqrt()
                xi = torch.randn_like(x_mid)
                x_tau = c * x_mid + beta * xi
                eps_mix = (c * tm4 * eps_hat + beta * xi) / ta4.clamp(min=1e-6)
                return t_m, x_tm, x0_hat, tau, x_tau, eps_mix, xi

            # 2. fake-score forward: denoise x_tau -> sg(x0hat). Grad accumulates;
            #    optimizer steps once per window after the micro loop.
            with torch.no_grad():
                _, _, x0_hat_f, tau_f, x_tau_f, eps_mix, xi = draw_segment()
                is_w = torch.exp(-0.5 * eps_mix.square().mean(dim=(1, 2, 3)) + 0.5 * xi.square().mean(dim=(1, 2, 3)))
                snr = ((1.0 - tau_f) / tau_f.clamp(min=1e-6)).square()
                w_fake = is_w * torch.minimum(snr, torch.full_like(snr, 5.0))
            set_role(student_net, fake_net, "fake")
            if use_ckpt:
                dit.train()  # AnimaBlock gates checkpointing on self.training
            with torch.autocast(**autocast_kw):
                v_fake_train = dit_predict_v(dit, x_tau_f.to(weight_dtype), tau_f, cond, weight_dtype,
                                             ckpt_input_grad=use_ckpt).float()
            ta4 = tau_f.view(B, 1, 1, 1)
            x0_fake_pred = x_tau_f - ta4 * v_fake_train
            loss_d = (((x0_fake_pred - x0_hat_f).square().mean(dim=(1, 2, 3))) * w_fake).mean()
            (loss_d / accum_steps).backward()
            if use_ckpt:
                dit.eval()  # back to deterministic eval for no-grad scores
            loss_d_sum += loss_d.item()

            # 3a. scores (no grad): teacher cond/uncond + fake at fresh (x_tau, tau)
            with torch.no_grad():
                dit.eval()
                t_m, x_tm, x0_hat, tau, x_tau, _, _ = draw_segment()
                x_tau_b = x_tau.to(weight_dtype)
                set_role(student_net, fake_net, "teacher")
                with torch.autocast(**autocast_kw):
                    v_real = dit_predict_v(dit, x_tau_b, tau, cond, weight_dtype).float()
                    v_unc = dit_predict_v(dit, x_tau_b, tau, uncond, weight_dtype).float()
                set_role(student_net, fake_net, "fake")
                with torch.autocast(**autocast_kw):
                    v_fake = dit_predict_v(dit, x_tau_b, tau, cond, weight_dtype).float()
                ta4 = tau.view(B, 1, 1, 1)
                x0_real = x_tau - ta4 * v_real
                x0_unc = x_tau - ta4 * v_unc
                x0_fake = x_tau - ta4 * v_fake
                x0_real_cfg = x0_real + (args.tdm_cfg - 1.0) * (x0_real - x0_unc)
                coop = (x0_hat + (x0_real_cfg - x0_fake)).detach()
                norm = (x0_hat - x0_real_cfg).abs().mean(dim=(1, 2, 3)).clamp(min=1e-6)
                w_gen = 1.0 / norm

            # 3b. one student step WITH grad toward sg(coop)
            set_role(student_net, fake_net, "student")
            if use_ckpt:
                dit.train()  # AnimaBlock gates checkpointing on self.training
            t_m_vec = t_m.to(dtype=torch.float32)
            with torch.autocast(**autocast_kw):
                v_stu = dit_predict_v(dit, x_tm, t_m_vec, cond, weight_dtype, ckpt_input_grad=use_ckpt).float()
            tm4 = t_m.view(B, 1, 1, 1)
            x0_stu = x_tm.float() - tm4 * v_stu
            diff = x0_stu - coop
            if huber_c is None:
                el = diff.square()
            else:
                el = torch.sqrt(diff.square() + huber_c * huber_c) - huber_c
            loss_g = (el.mean(dim=(1, 2, 3)) * w_gen).mean()
            (loss_g / accum_steps).backward()
            if use_ckpt:
                dit.eval()  # back to deterministic eval for next rollout
            loss_g_sum += loss_g.item()

        # End of accumulation window: single clip + step per optimizer, then
        # step both shared LR schedulers once per optimizer step.
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(fake_net.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), max_grad_norm)
        opt_f.step()
        opt_s.step()
        sched_f.step()
        sched_s.step()
        lr_s_now = opt_s.param_groups[0]["lr"]
        lr_f_now = opt_f.param_groups[0]["lr"]
        if use_ema and ema_state is not None:
            if opt_step < ema_warmup:
                pass  # keep init until warmup elapses
            elif opt_step == ema_warmup and ema_warmup > 0:
                # Fresh start from the warmed-up student, then decay afterwards.
                # In-place re-sync (closures hold a reference to this dict).
                with torch.no_grad():
                    for k, e in ema_state.items():
                        s = student_net.state_dict()[k]
                        if torch.is_tensor(s) and s.is_floating_point():
                            e.copy_(s.detach().to(e.device, dtype=torch.float32))
            elif (opt_step - ema_warmup) % ema_update_every == 0:
                update_ema_state(ema_state, student_net, ema_decay)
        loss_d_v = loss_d_sum / accum_steps
        loss_g_v = loss_g_sum / accum_steps

        csv_writer.writerow([opt_step, f"{loss_g_v:.6f}", f"{loss_d_v:.6f}", f"{lr_s_now:.2e}", f"{lr_f_now:.2e}", f"{time.time() - t0:.1f}"])
        pbar.set_postfix(loss_g=f"{loss_g_v:.4f}", loss_d=f"{loss_d_v:.4f}", lr=f"{lr_s_now:.2e}/{lr_f_now:.2e}")
        if len(accelerator.trackers) > 0:
            accelerator.log({"loss_g": loss_g_v, "loss_d": loss_d_v, "lr_student": lr_s_now, "lr_fake": lr_f_now}, step=opt_step)
        if opt_step % 10 == 0:
            csv_file.flush()

        if save_every > 0 and opt_step > 0 and opt_step % save_every == 0:
            save_all(opt_step)
        if args.tdm_eval_every > 0 and opt_step % args.tdm_eval_every == 0:
            preview(opt_step)

    logger.info(f"Reached max_train_steps={max_steps}. Saving final checkpoint.")
    save_all(max_steps)
    if args.tdm_eval_every > 0:
        preview(max_steps)
    csv_file.close()
    accelerator.end_training()


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility
    if getattr(args, "verbose", None) is None:
        args.verbose = False

    main(args)
