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
     Pseudo-Huber surrogate (paper Sec. 3.3, ``c = 0.00054*sqrt(d)``) and the
     DMD normalizer ``1 / mean|x0hat - x0_real_cfg|``.

``tau`` is sampled inside the segment (non-overlapping intervals, paper
Sec. 3.1) unless ``--tdm_no_separate_intervals`` is given.

Run (single GPU, data-free -- prompts file only, one caption per line)::

    python anima_train_tdm_network.py \
      --pretrained_model_name_or_path anima-base.safetensors \
      --qwen3 /path/to/qwen3-0.6b.safetensors --vae /path/to/vae.safetensors \
      --tdm_prompts_file prompts.txt --resolution 512,512 \
      --tdm_batch_size 8 --tdm_steps 4 --output_dir output/anima-tdm-4step

Outputs: ``tdm_student_step_<n>.safetensors`` (student LoRA only, same key
layout as ``anima_train_network.py`` -- directly usable for inference /
ComfyUI conversion) and ``tdm_state_step_<n>.safetensors`` (both adapters,
``student.``/``fake.`` prefixes -- resume with ``--tdm_resume_state``).
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
        "Trajectory Distribution Matching few-step distillation (arXiv:2503.06674). Data-free: training needs only --tdm_prompts_file.",
    )
    g.add_argument("--tdm_prompts_file", type=str, default=None,
                   help="Text file with one caption per line. REQUIRED (data-free distillation uses captions only).")
    g.add_argument("--tdm_steps", type=int, default=4,
                   help="K: number of deterministic student steps to distill (default: 4).")
    g.add_argument("--tdm_batch_size", type=int, default=8,
                   help="Prompts per iteration (default: 8).")
    g.add_argument("--tdm_cfg", type=float, default=3.5,
                   help="Teacher CFG scale for the revised target (paper default: 3.5).")
    g.add_argument("--tdm_neg_prompt", type=str, default="",
                   help="Teacher CFG negative prompt (uncond anchor). Default: empty string (standard). Baked into the student permanently since the student is CFG-free.")
    g.add_argument("--tdm_neg_file", type=str, default=None,
                   help="Optional text file with one negative prompt per line, sampled per batch. Overrides --tdm_neg_prompt when given.")
    g.add_argument("--tdm_shift", type=float, default=3.0,
                   help="Flow shift for the K-step student schedule (Anima sampling default: 3.0).")
    g.add_argument("--tdm_lr", type=float, default=2e-6, help="Student AdamW LR (paper default: 2e-6).")
    g.add_argument("--tdm_fake_lr", type=float, default=1e-5, help="Fake-score AdamW LR (paper default: 2e-5 for PixArt; 1e-5 here).")
    g.add_argument("--tdm_betas", type=float, nargs=2, default=[0.0, 0.999],
                   help="Adam betas for both optimizers (paper default: 0 0.999).")
    g.add_argument("--tdm_weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    g.add_argument("--tdm_max_grad_norm", type=float, default=1.0, help="Gradient clip norm for both optimizers.")
    g.add_argument("--tdm_warmup_steps", type=int, default=100, help="Linear LR warmup steps (then constant).")
    g.add_argument("--tdm_max_steps", type=int, default=2000, help="Total distillation iterations.")
    g.add_argument("--tdm_no_huber", action="store_true",
                   help="Use plain MSE for the student loss instead of the Pseudo-Huber surrogate.")
    g.add_argument("--tdm_huber_c", type=float, default=None,
                   help="Explicit Pseudo-Huber c (overrides both --tdm_no_huber and the auto 0.00054*sqrt(d)).")
    g.add_argument("--tdm_no_separate_intervals", action="store_true",
                   help="Sample tau up to the schedule top (demo default) instead of inside the segment (paper default).")
    g.add_argument("--tdm_randmid", action="store_true",
                   help="Land the ODE point at a random point inside the segment (demo --use_randmid regularizer).")
    g.add_argument("--tdm_tau_min", type=float, default=0.025)
    g.add_argument("--tdm_tau_gap", type=float, default=0.0125)
    g.add_argument("--tdm_tau_max", type=float, default=0.9875)
    g.add_argument("--tdm_train_llm_adapter", action="store_true",
                   help="Include LLM-adapter Linears in both LoRA adapters (default: DiT blocks only).")
    g.add_argument("--tdm_fake_weights", type=str, default=None,
                   help="Init fake adapter from a LoRA file (student init uses --network_weights).")
    g.add_argument("--tdm_resume_state", type=str, default=None,
                   help="Resume both adapters from a tdm_state_*.safetensors file.")
    g.add_argument("--tdm_resume_optim", type=str, default=None,
                   help="Resume both optimizer states from a tdm_optim_*.pt file.")
    g.add_argument("--tdm_save_every", type=int, default=500, help="Save adapters + optim state every N steps.")
    g.add_argument("--tdm_eval_every", type=int, default=0, help="Render K-step student previews every N steps (0 = disabled, default).")
    g.add_argument("--tdm_preview_n", type=int, default=4, help="Preview images per eval.")
    g.add_argument("--tdm_seed", type=int, default=None, help="Preview noise seed (default: fixed 42 for comparability).")


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
        logger.info("gradient checkpointing disabled (pass --gradient_checkpointing to reduce activation VRAM).")

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
        train_llm_adapter="true" if args.tdm_train_llm_adapter else "false",
        verbose="true" if getattr(args, "verbose", False) else "false",
    )
    if getattr(args, "network_args", None):
        for kv in args.network_args:
            if "=" in kv:
                k, v = kv.split("=", 1)
                lora_kw[k.strip()] = v.strip()
    logger.info(f"Creating student/fake LoRA adapters (rank={rank}, alpha={alpha}, llm_adapter={args.tdm_train_llm_adapter})...")
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

    betas = tuple(args.tdm_betas)
    opt_s = torch.optim.AdamW(student_net.parameters(), lr=args.tdm_lr, betas=betas, weight_decay=args.tdm_weight_decay)
    opt_f = torch.optim.AdamW(fake_net.parameters(), lr=args.tdm_fake_lr, betas=betas, weight_decay=args.tdm_weight_decay)
    logger.info(f"Optimizers: student lr={args.tdm_lr:g}, fake lr={args.tdm_fake_lr:g}, betas={betas}.")
    if args.tdm_resume_optim:
        logger.info(f"Resume optimizers from {args.tdm_resume_optim}")
        o = torch.load(args.tdm_resume_optim, map_location="cpu", weights_only=True)
        opt_s.load_state_dict(o["student"])
        opt_f.load_state_dict(o["fake"])
        global_step = int(o.get("step", 0))
    else:
        global_step = 0

    warmup = max(1, args.tdm_warmup_steps)

    def apply_warmup(step: int):
        mult = min(1.0, (step + 1) / warmup)
        for pg in opt_s.param_groups:
            pg["lr"] = args.tdm_lr * mult
        for pg in opt_f.param_groups:
            pg["lr"] = args.tdm_fake_lr * mult
        return args.tdm_lr * mult

    # ---- geometry / schedule ----
    img_h, img_w = parse_resolution(args.resolution)
    lat_h, lat_w = img_h // 8, img_w // 8
    latent_ch = 16
    logger.info(f"Distilling at {img_w}x{img_h} (latent {lat_w}x{lat_h}x{latent_ch}), K={args.tdm_steps}.")
    K = args.tdm_steps
    ts = make_schedule(K, args.tdm_shift, device)  # [K+1], 1 -> 0
    if args.tdm_huber_c is not None:
        huber_c = args.tdm_huber_c
        logger.info(f"Student loss: Pseudo-Huber(c={huber_c:.3g} explicit) + DMD normalizer.")
    elif args.tdm_no_huber:
        huber_c = None
        logger.info("Student loss: MSE + DMD normalizer.")
    else:
        huber_c = 0.00054 * (latent_ch * lat_h * lat_w) ** 0.5
        logger.info(f"Student loss: Pseudo-Huber(c={huber_c:.3g} auto = 0.00054*sqrt(d)) + DMD normalizer.")
    logger.info(f"tau sampled in {'segment (non-overlapping)' if not args.tdm_no_separate_intervals else 'full range'}.")

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = args.output_name or "anima-tdm"
    csv_path = os.path.join(args.output_dir, "tdm_loss_log.csv")
    new_csv = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if new_csv:
        csv_writer.writerow(["step", "loss_g", "loss_d", "lr", "time"])

    autocast_kw = dict(device_type=device.type if isinstance(device, torch.device) else str(device),
                       dtype=weight_dtype)
    if autocast_kw["device_type"] not in ("cuda", "cpu", "xpu"):
        autocast_kw["device_type"] = "cuda"
    t0 = time.time()

    def save_all(step: int):
        s_sd = {f"student.{k}": v.detach().cpu().contiguous() for k, v in student_net.state_dict().items()}
        f_sd = {f"fake.{k}": v.detach().cpu().contiguous() for k, v in fake_net.state_dict().items()}
        state_path = os.path.join(args.output_dir, f"tdm_state_step_{step}.safetensors")
        save_file({**s_sd, **f_sd}, state_path)
        logger.info(f"[ckpt] both adapters -> {state_path}")
        student_path = os.path.join(args.output_dir, f"{out_name}_step_{step}.safetensors")
        meta = model_io.get_sai_model_spec_dataclass(None, args, False, True, False, anima="preview").to_metadata_dict()
        meta["ss_tdm_step"] = str(step)
        student_net.save_weights(student_path, save_dtype, meta)
        logger.info(f"[ckpt] student LoRA -> {student_path}")
        torch.save({"student": opt_s.state_dict(), "fake": opt_f.state_dict(), "step": step + 1},
                   os.path.join(args.output_dir, f"tdm_optim_step_{step}.pt"))

    @torch.no_grad()
    def preview(step: int):
        dit.eval()
        set_role(student_net, fake_net, "student")
        vae_device = vae.device if hasattr(vae, "device") else device
        try:
            vae.to(device)
        except Exception:
            pass
        shots = prompts[: min(args.tdm_preview_n, len(prompts))]
        imgs = []
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
                    img_h, img_w, 42 + i, dit, cross, K, weight_dtype, device,
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
        try:
            vae.to(vae_device)
        except Exception:
            pass
        clean_memory_on_device(device)

    # ---- data-free loop: random caption batches; images never used ----
    B = args.tdm_batch_size
    max_steps = args.tdm_max_steps
    pbar = tqdm(range(global_step, max_steps), initial=global_step, total=max_steps, desc="TDM")
    for step in pbar:
        batch_caps = random.choices(prompts, k=B)
        if neg_prompts is not None:
            neg_caps = random.choices(neg_prompts, k=B)
        else:
            neg_caps = [neg_single] * B
        lr_now = apply_warmup(step)

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

        # 2. fake-score update: denoise x_tau -> sg(x0hat)
        with torch.no_grad():
            _, _, x0_hat_f, tau_f, x_tau_f, eps_mix, xi = draw_segment()
            is_w = torch.exp(-0.5 * eps_mix.square().mean(dim=(1, 2, 3)) + 0.5 * xi.square().mean(dim=(1, 2, 3)))
            snr = ((1.0 - tau_f) / tau_f.clamp(min=1e-6)).square()
            w_fake = is_w * torch.minimum(snr, torch.full_like(snr, 5.0))
        set_role(student_net, fake_net, "fake")
        zero_all(student_net, fake_net)
        if use_ckpt:
            dit.train()  # AnimaBlock gates checkpointing on self.training
        with torch.autocast(**autocast_kw):
            v_fake_train = dit_predict_v(dit, x_tau_f.to(weight_dtype), tau_f, cond, weight_dtype,
                                         ckpt_input_grad=use_ckpt).float()
        ta4 = tau_f.view(B, 1, 1, 1)
        x0_fake_pred = x_tau_f - ta4 * v_fake_train
        loss_d = (((x0_fake_pred - x0_hat_f).square().mean(dim=(1, 2, 3))) * w_fake).mean()
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(fake_net.parameters(), args.tdm_max_grad_norm)
        opt_f.step()
        if use_ckpt:
            dit.eval()  # back to deterministic eval for no-grad scores
        loss_d_v = loss_d.item()

        # 3a. scores (no grad): teacher cond/uncond + fake at fresh (x_tau, tau)
        with torch.no_grad():
            dit.eval()
            _, t_m, x_tm, x0_hat, tau, x_tau, _, _ = draw_segment()
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
        zero_all(student_net, fake_net)
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
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(student_net.parameters(), args.tdm_max_grad_norm)
        opt_s.step()
        if use_ckpt:
            dit.eval()  # back to deterministic eval for next rollout
        loss_g_v = loss_g.item()

        csv_writer.writerow([step, f"{loss_g_v:.6f}", f"{loss_d_v:.6f}", f"{lr_now:.2e}", f"{time.time() - t0:.1f}"])
        pbar.set_postfix(loss_g=f"{loss_g_v:.4f}", loss_d=f"{loss_d_v:.4f}", lr=f"{lr_now:.2e}")
        if step % 10 == 0:
            csv_file.flush()

        if args.tdm_save_every > 0 and step > 0 and step % args.tdm_save_every == 0:
            save_all(step)
        if args.tdm_eval_every > 0 and step % args.tdm_eval_every == 0:
            preview(step)

    logger.info(f"Reached max_steps={args.tdm_max_steps}. Saving final checkpoint.")
    save_all(max_steps)
    if args.tdm_eval_every > 0:
        preview(max_steps)
    csv_file.close()


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility
    if getattr(args, "verbose", None) is None:
        args.verbose = False

    main(args)
