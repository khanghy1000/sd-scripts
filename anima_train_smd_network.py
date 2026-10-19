"""Anima SMD few-step distillation (data-free, LoRA + training-only Pivot).

Simplified Matching Distillation (arXiv:2610.18472, Microsoft Research)
distills a pretrained rectified-flow teacher into a K-step deterministic
student without any images (captions/prompts only). Notably simpler than
trajectory/DMD-based recipes: entirely regression-based, no fake score
model, no adversarial loss, no architecture change at inference.

Anima convention (same as the regular trainer / TDM script):
``x_t = (1-t) x0 + t eps``, the DiT predicts ``v = eps - x0`` with ``t`` in
[0, 1], Euler steps run ``t: 1 -> 0`` as ``x += v * (t_prev - t)``.

Parts of the method (paper Sec. 3):

  1. Simple on-policy baseline (Sec. 3.1, Algorithm 1). Roll out the current
     student K steps from noise (no grad across transitions). For EVERY
     coarse transition ``t_i -> t_{i+1}`` (``dt = t_{i+1} - t_i < 0``) make
     ONE student prediction with grad, ``v_th = v(x_t, t_i)``, then query the
     frozen teacher at Q intermediate states on the student's own transition
     and regress the SAME student prediction against each teacher velocity::

         for alpha ~ U(0, 1) (paper symbol: tau; Q query points, per transition):
             s   = t_i + alpha * dt                  # query time
             x_s = x_t + alpha * dt * v_th           # query state
             L  += || v_th - v_T(x_s, s) ||_2

  2. Pivot proxy (Sec. 3.2, Algorithm 2). Within a transition the teacher
     velocity can curve a lot, so a single student prediction cannot fit all
     local targets without jitter. Pivot is a TRAINING-ONLY residual proxy
     predicting P zero-integral basis coefficients ``a_m`` (same shape as the
     velocity) once per transition from the features of the K-th block from
     the end of the student (``--smd_pivot_hook_layer``, paper default K=4).
     It processes those features with M Transformer blocks copied from the
     final M student blocks (``--smd_pivot_blocks``, paper default M=3) and a
     zero-initialized output projection, so training starts exactly at the
     simple baseline. Inside the transition it bends the path::

         v(alpha) = v_th + sum_m a_m b_m(alpha)                    # velocity
         x_s    = x_t + alpha*dt*v_th + dt * sum_m a_m B_m(alpha)  # state
         B_m(u) = u(1-u)(2u-1)^(m-1),  b_m = dB_m/du               # poly basis

     Since B_m(0) = B_m(1) = 0 the transitions endpoints (what the deployed
     student step is!) are untouched; Pivot only shapes the interior query
     states/velocities and is never evaluated at inference.

  3. Repulsion from a reference prediction (Sec. 3.3). At each query point,
     besides the guided teacher target ``v`` (CFG ``--smd_cfg``), the teacher
     ALSO predicts a weaker reference ``ev`` at the same state (CFG
     ``--smd_ref_cfg``, 1.0 = the plain conditional branch). The student is
     pulled toward the target and pushed away from the reference::

         L = || v(alpha) - vT ||_2 - lambda * || v(alpha) - evT ||_2  (lambda<1)

Like the TDM script, the teacher is the same frozen DiT with the student LoRA
multiplier zeroed; the student is one trainable LoRA adapter nested on the frozen
base (``networks/lora_anima.py``). Unlike TDM there is NO fake-score adapter --
the only extra trainable module is the Pivot.

Default hyper-parameters follow the paper's image-generation recipe
(App. B Table 15, Qwen-Image row) -- the closest analog to Anima:

    Student LoRA (rank 256, alpha 256)  -- pass --network_dim 256 --network_alpha 256
    Pivot full-rank, M=3 blocks, hook K=4, Polynomial basis P=4
    Queries / transition 2 | target CFG 5.0 / reference CFG 1.0 | lambda 0.2
    AdamW (zero weight decay), LR 1e-4 student and pivot, 100-iter warm-up,
    constant schedule thereafter, 1000 iterations, 4-NFE student.

NOTE on --smd_final_step_size_scale: its default is 1.0 (uniform raw grid), NOT
the paper's 0.5. With 1.0 the trained K-step grid is identical to what the
standard anima sampler builds (``do_sample``: linspace(1,0,K+1) shifted,
CFG-free at guidance_scale=1.0), so a trained SMD student runs in the EXISTING
inference code unmodified. Set 0.5 only if you control the deployment sampler.

Deployment sampler (must match training; used for previews here):
K Euler steps on the schedule ``raw knots 1 -> 0 with the final step scaled
by --smd_final_step_size_scale (default 1.0 = uniform; 0.5 = paper recipe,
requires a matching non-uniform sampler)``, shifted with Anima's rational shift
``S(ts) = shift*ts / (1 + (shift-1)*ts)`` (--smd_shift), CFG-free.

Run (single GPU, data-free -- prompts file only, one caption per line)::

    python anima_train_smd_network.py \
      --pretrained_model_name_or_path anima-base.safetensors \
      --qwen3 /path/to/qwen3-0.6b.safetensors --vae /path/to/vae.safetensors \
      --smd_prompts_file prompts.txt --resolution 512,512 \
      --network_dim 256 --network_alpha 256 \
      --train_batch_size 8 --max_train_steps 1000 --output_dir output/anima-smd-4step

Optimizer/LR/scheduler flags are shared with ``anima_train_network.py``:
``--learning_rate`` (student; recommended 1e-4, AdamW zero weight decay e.g.
``--optimizer_args weight_decay``), ``--smd_pivot_lr`` (default 1e-4),
``--optimizer_type``/``--optimizer_args``, ``--lr_scheduler`` /
``--lr_warmup_steps`` / ``--lr_decay_steps`` / ..., ``--max_train_steps``,
``--train_batch_size``, ``--max_grad_norm``, ``--save_every_n_steps``,
``--seed``.

Per-step losses (``loss`` = imitation - lambda * repulsion, SMD normalizer
``0.5 * --smd_w_rescale / K``) are logged to ``smd_loss_log.csv``.

Outputs: ``{output_name}_step_<n>.safetensors`` (student LoRA only, same key
layout as ``anima_train_network.py`` -- directly usable for inference /
ComfyUI conversion) and ``smd_state_step_<n>.safetensors`` (student LoRA +
Pivot, ``student.``/``pivot.`` prefixes -- resume with ``--smd_resume_state``).
With EMA enabled (default, ``--smd_ema_decay``), an extra
``{output_name}_ema_step_<n>.safetensors`` holds the EMA student LoRA (the
recommended inference weight), the state file also carries ``ema.`` entries,
and previews render with the EMA weights.
"""

import argparse
import contextlib
import copy
import csv
import json
import os
import random
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from tqdm import tqdm

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    anima_train_utils,
    anima_utils,
    attention,
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
# SMD CLI
# ---------------------------------------------------------------------------

def add_smd_arguments(parser: argparse.ArgumentParser):
    g = parser.add_argument_group(
        "SMD distillation",
        "Simplified Matching Distillation few-step distillation (arXiv:2610.18472). Data-free: training needs only "
        "--smd_prompts_file. Optimizer/LR/scheduler/batch/steps/clipping flags are shared with anima_train_network.py "
        "(--learning_rate student, --smd_pivot_lr pivot, --optimizer_type/--optimizer_args, --lr_scheduler/--lr_warmup_steps, "
        "--train_batch_size, --max_train_steps, --max_grad_norm, --save_every_n_steps, --seed).",
    )
    g.add_argument("--smd_prompts_file", type=str, default=None,
                   help="Text file with one caption per line. REQUIRED (data-free distillation uses captions only).")
    g.add_argument("--smd_steps", type=int, default=4,
                   help="K: number of deterministic student steps (NFE at inference) to distill (default: 4).")
    g.add_argument("--smd_shift", type=float, default=3.0,
                   help="Rational flow shift for the K-step student schedule (Anima sampling default: 3.0).")
    g.add_argument("--smd_final_step_size_scale", type=float, default=1.0,
                   help="Scale of the LAST Euler step relative to the others on the student schedule. "
                        "1.0 (default) = uniform raw grid, byte-for-byte the schedule that "
                        "library/anima_train_utils.do_sample (and standard uniform+shift samplers) build, "
                        "so NO inference-code changes are needed. The paper's image recipe uses 0.5, but "
                        "that requires the deployment sampler to reproduce the non-uniform last segment; "
                        "whichever value is chosen MUST be mirrored at inference.")
    g.add_argument("--smd_cfg", type=float, default=5.0,
                   help="Teacher CFG scale for the imitation target (paper image recipe: 5.0).")
    g.add_argument("--smd_neg_prompt", type=str, default="",
                   help="Teacher CFG negative prompt (uncond branch). Default: empty string (standard). "
                        "Baked into the student permanently since the student is CFG-free.")
    g.add_argument("--smd_neg_file", type=str, default=None,
                   help="Optional text file with one negative prompt per line, sampled per batch. Overrides --smd_neg_prompt when given.")
    g.add_argument("--smd_queries", type=int, default=2,
                   help="Q: teacher query points per transition, alpha ~ U(0,1) (paper image recipe: 2).")
    g.add_argument("--smd_imitation_weight", type=float, default=1.0,
                   help="Weight of the imitation term (paper: 1.0).")
    g.add_argument("--smd_repulsion_lambda", type=float, default=0.2,
                   help="Repulsion weight lambda in (0, 1). 0 disables the repulsion term (paper image recipe: 0.2).")
    g.add_argument("--smd_ref_cfg", type=float, default=1.0,
                   help="CFG scale of the negative reference prediction ev at each query state (1.0 = plain conditional "
                        "branch, paper image recipe; 0.0 = unconditional branch).")
    g.add_argument("--smd_w_rescale", type=float, default=30.0,
                   help="SMD loss normalizer multiplier: total loss = 0.5 * this / K * (imitation - lambda * repulsion).")
    g.add_argument("--smd_beta", type=float, default=1.0,
                   help="Per-sample velocity distance beta: distance = per-sample MSE ** (beta / 2). beta=2 is plain MSE; "
                        "beta=1 (paper default) is a Moment-style smoothed L2.")
    g.add_argument("--smd_eps", type=float, default=1e-6,
                   help="Epsilon smoothing inside the velocity distance (default: 1e-6).")
    g.add_argument("--smd_no_pivot", action="store_true",
                   help="Disable the Pivot proxy and train the plain on-policy baseline (paper Sec. 3.1 ablation).")
    g.add_argument("--smd_no_repulsion", action="store_true",
                   help="Disable the repulsion term entirely (same as --smd_repulsion_lambda 0).")
    # Pivot
    g.add_argument("--smd_pivot_hook_layer", type=int, default=-4,
                   help="Index (negative) of the student block whose output features feed the Pivot (paper default: -4, "
                        "i.e. the 4th block from the end; K=1 would be the final block). Must leave at least the M copied "
                        "blocks after it.")
    g.add_argument("--smd_pivot_blocks", type=int, default=None,
                   help="M Pivot transformer blocks, copied from the final M student blocks (paper image recipe: 3). "
                        "Default: derive from --smd_pivot_hook_layer (all blocks after the hook).")
    g.add_argument("--smd_p", type=int, default=4,
                   help="P polynomial basis terms for the Pivot residual (paper image recipe: 4).")
    g.add_argument("--smd_pivot_lr", type=float, default=1e-4,
                   help="Pivot optimizer LR. Student uses shared --learning_rate (paper image recipe: 1e-4 for both).")
    g.add_argument("--smd_pivot_lora_rank", type=int, default=None,
                   help="If set, train the Pivot blocks with LoRA of this rank instead of full-rank "
                        "(H3-style recipe). The input projection and the zero-init output projection always "
                        "stay fully trainable.")
    # Resume / eval / EMA (same conventions as the TDM script)
    g.add_argument("--smd_resume_state", type=str, default=None,
                   help="Resume student LoRA (+Pivot) from a smd_state_*.safetensors file.")
    g.add_argument("--smd_resume_optim", type=str, default=None,
                   help="Resume both optimizer (+scheduler) states from a smd_optim_*.pt file.")
    g.add_argument("--smd_eval_every", type=int, default=0, help="Render K-step student previews every N steps (0 = disabled, default).")
    g.add_argument("--smd_preview_n", type=int, default=4, help="Preview images per eval.")
    g.add_argument("--smd_no_ema", action="store_true",
                   help="Disable the student EMA shadow (by default an EMA of the student LoRA is kept for previews/saving).")
    g.add_argument("--smd_ema_decay", type=float, default=0.999,
                   help="EMA decay for the student LoRA shadow (default: 0.999). Updated once per optimizer step.")
    g.add_argument("--smd_ema_warmup", type=int, default=0,
                   help="Optimizer steps before EMA starts tracking. At this step the EMA is re-synced from the student, then decayed (default: 0).")
    g.add_argument("--smd_ema_update_every", type=int, default=1,
                   help="Update the EMA every N optimizer steps (default: 1).")


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    args_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    add_smd_arguments(parser)
    # --gradient_checkpointing / --cpu_offload_checkpointing are inherited
    # from train_network.setup_parser(); add the unsloth variant for parity
    # with anima_train.py / anima_train_network.py / the TDM script.
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing. Implies --gradient_checkpointing.",
    )
    parser.set_defaults(
        learning_rate=1e-4,  # paper image recipe
        train_batch_size=8,  # paper uses global batch 16; lower this for single-GPU VRAM
        max_train_steps=1000,
        save_every_n_steps=100,
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


def make_smd_schedule(steps: int, shift: float, final_step_size_scale: float, device) -> torch.Tensor:
    """K+1 shifted sigmas from 1.0 to 0.0 on the SMD grid (paper Sec. 2/3).

    Knots are uniform except the final Euler segment, which is scaled by
    ``final_step_size_scale`` (SMD ``compute_raw_grid_knots``), then Anima's
    rational shift ``S(t) = shift*t / (1 + (shift-1)*t)`` is applied (same as
    sampling/do_sample). The resulting values are what both the student and
    the teacher pass as the timestep input to the DiT.
    """
    fss = max(float(final_step_size_scale), 1e-4)
    base_seg = 1.0 / (steps - 1 + fss)
    raw = [1.0]
    for i in range(steps):
        seg = base_seg * (fss if i == steps - 1 else 1.0)
        raw.append(raw[-1] - seg)
    raw[-1] = 0.0
    raw = torch.tensor(raw, dtype=torch.float32)
    if shift != 1.0:
        raw = (raw * shift) / (1.0 + (shift - 1.0) * raw)
    return raw.to(device)


def poly_basis_eval(coeffs: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Polynomial endpoint-preserving Pivot basis (paper App. A.1 Eq. 22).

    B_m(t) = t(1-t)(2t-1)**(m-1);  b_m = dB_m/dt;  q = sum_m a_m B_m;
    r = sum_m a_m b_m. B_m(0) = B_m(1) = 0 keeps the transition endpoints
    fixed for a given coarse student prediction.

    coeffs: [B, P, C, H, W]; alpha: [B] in [0, 1).
    Returns (q, r), each [B, C, H, W].
    """
    B, P = coeffs.shape[:2]
    # a_m = coeffs[:, m-1] is [B, C, H, W], so alpha broadcasts as [B, 1, 1, 1].
    # This is the within-transition interpolation parameter the paper writes
    # as tau in Eqs. 9-16 (query time s = t_i + alpha*dt) and the upstream SMD
    # code calls alpha (query_alphas / "diagnostic/query/alpha"); naming
    # follows the code, not the paper symbol, since --network_alpha already
    # owns "alpha" at the CLI level but not in this loss-local scope.
    alpha = alpha.to(coeffs.dtype).view(B, 1, 1, 1)
    two_a_m1 = 2.0 * alpha - 1.0
    factor = alpha * (1.0 - alpha)
    q = torch.zeros_like(coeffs[:, 0])
    r = torch.zeros_like(coeffs[:, 0])
    for m in range(1, P + 1):
        a_m = coeffs[:, m - 1]
        pow_ = two_a_m1 ** (m - 1)
        q = q + a_m * (factor * pow_)
        b_m = (1.0 - 2.0 * alpha) * pow_
        if m >= 2:
            b_m = b_m + 2.0 * (m - 1) * factor * two_a_m1 ** (m - 2)
        r = r + a_m * b_m
    return q, r


def per_sample_velocity_distance(lhs: torch.Tensor, rhs: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    """SMD velocity distance (paper Eq. 10/SMD ``per_sample_velocity_distance``).

    Per-sample mean squared error over all non-batch dimensions, raised to
    ``beta / 2`` (beta=2: exact MSE; beta=1: sqrt(MSE + eps), the default).
    Returns [B].
    """
    mse = (lhs.float() - rhs.float()).reshape(lhs.shape[0], -1).square().mean(dim=1)
    if abs(beta - 2.0) < 1e-12:
        return mse
    return (mse + eps).pow(beta / 2.0)


def cfg_combine(out_cond: torch.Tensor, out_uncond: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    """Teacher CFG in velocity form (paper Eq. 4). 1.0 = conditional, 0.0 = unconditional."""
    if cfg_scale == 1.0:
        return out_cond
    if cfg_scale == 0.0:
        return out_uncond
    return out_uncond + float(cfg_scale) * (out_cond - out_uncond)


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
    input as requiring grad (mirrors anima_train_network.py's workaround so
    torch checkpointing keeps a grad path on the frozen-base + LoRA graph).
    Only takes effect when grad is enabled (no-op under torch.no_grad()).
    """
    if ckpt_input_grad and torch.is_grad_enabled() and x_4d.dtype.is_floating_point:
        x_4d = x_4d.detach().requires_grad_(True)
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


def set_role(student_net, role: str):
    """Flip which LoRA delta the shared frozen base applies (nested adds).

    teacher : multiplier 0 -> base weights alone
    student : multiplier 1 -> base + student adapter
    """
    if role == "teacher":
        student_net.set_multiplier(0.0)
    elif role == "student":
        student_net.set_multiplier(1.0)
    else:
        raise ValueError(f"unknown role: {role}")


def zero_all(*nets):
    for net in nets:
        for p in net.parameters():
            if p.grad is not None:
                p.grad = None


# ---------------------------------------------------------------------------
# Pivot: training-only residual proxy (paper Sec. 3.2 / App. A.3)
# ---------------------------------------------------------------------------

class SMDPivot(nn.Module):
    """Pivot proxy for the Anima DiT.

    Architecture per the paper (Fig. 36): features captured from the student
    block at ``hook_pos`` are processed by an input Linear, M Transformer
    blocks copied from the final M student blocks (pretrained init), and a
    ZERO-INITIALIZED output projection producing P basis coefficient vectors
    (same shape as the student velocity). With the zero-init output the
    network starts exactly at the no-Pivot simple baseline.

    The copied blocks receive the SAME conditioning inputs the student block
    saw at the transition start (t embedding, adaln_lora, rope, cross-attn),
    captured together with the features in one hook.
    """

    def __init__(self, dit, hook_layer: int, basis_p: int, attn_mode: str, split_attn: bool):
        super().__init__()
        n_blocks = dit.num_blocks
        pos = n_blocks + int(hook_layer)
        if not 0 <= pos < n_blocks:
            raise ValueError(
                f"--smd_pivot_hook_layer={hook_layer} is out of range for {n_blocks} student blocks."
            )
        self.hook_pos = pos
        self.basis_p = basis_p
        self.patch_spatial = dit.patch_spatial
        self.patch_temporal = dit.patch_temporal
        self.out_channels = dit.out_channels
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        tail = list(dit.blocks)[pos + 1:]
        if len(tail) == 0:
            raise ValueError(
                f"--smd_pivot_hook_layer={hook_layer} leaves no student blocks after it to copy into the Pivot; "
                "use a more negative index (paper: -4)."
            )
        # Pre-trained init: exact copies of the natural student tail.
        self.blocks = nn.ModuleList([copy.deepcopy(blk) for blk in tail])
        model_channels = dit.model_channels
        self.proj = nn.Linear(model_channels, model_channels, bias=False)
        per_token_modes = (
            self.patch_spatial * self.patch_spatial * self.patch_temporal * (basis_p * self.out_channels)
        )
        self.out_proj = nn.Linear(model_channels, per_token_modes, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def trainables(self) -> list:
        """Trainable Pivot parameters (all in full mode; proj/out_proj only in LoRA mode)."""
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        feat_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params,
        use_fp32: bool,
        block_kwargs: dict,
    ) -> torch.Tensor:
        """Return coefficient tensor a_m with the shape of the student
        velocity: [B, P, C, H, W] (T == 1 for images)."""
        h = self.proj(feat_B_T_H_W_D)
        for blk in self.blocks:
            h = blk(h, emb_B_T_D, crossattn_emb, attn_params, use_fp32, **block_kwargs)
        tokens = self.out_proj(h)  # [B, T, H', W', P * p*p*t * C]
        # Token channel layout mirrors the base DiT's unpatchify order
        # "(p1 p2 t (P C))" (einops forbids nested brackets, so split the
        # channel into (p1*p2*t, P, C) first and merge the patch scales in a
        # second one-level rearrange).
        a = rearrange(tokens, "B T Hp Wp (z P C) -> B T Hp Wp z P C",
                      P=self.basis_p, C=self.out_channels)
        v_5d = rearrange(
            a,
            "B T Hp Wp (p1 p2 t) P C -> B P C (T t) (Hp p1) (Wp p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return v_5d.contiguous().squeeze(3)  # [B, P, C, T=1, H, W] -> [B, P, C, H, W]


@contextlib.contextmanager
def capture_dit_block_features(hook_block):
    """Capture the output + conditioning args of one Anima DiT block.

    Fires when ``hook_block`` runs (i.e. during the student's with-grad
    forward). Keys written per call: ``feat`` (block output [B, T, H', W', D]),
    ``args`` (x, emb, cross, attn_params, use_fp32) and ``kwargs`` (rope /
    adaln_lora / extra pos emb). Downstream recompute-side hook firings
    (gradient checkpointing) overwrite the cache harmlessly.
    """
    cache: dict = {}
    if hook_block is None:
        raise ValueError("hook_block is None")

    def hook(module, args, kwargs, output):
        cache["feat"] = output
        cache["args"] = args
        cache["kwargs"] = kwargs

    handle = hook_block.register_forward_hook(hook, with_kwargs=True)
    try:
        yield cache
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Student EMA (fp32 shadow of the student LoRA state_dict) -- TDM parity
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


def build_smd_metadata(
    args,
    *,
    session_id: int,
    training_started_at: float,
    rank: int,
    alpha: float,
    lora_kw: dict,
    optimizer_name: str,
    optimizer_args_str: str,
    student_lr: float,
    pivot_lr: float,
    num_prompts: int,
    batch_size_per_device: int,
    total_batch_size: int,
    accum_steps: int,
    seed: int,
    use_ckpt: bool,
    use_pivot: bool,
    pivot_mode: str,
    use_ema: bool,
    ema_decay: float,
    ema_warmup: int,
    ema_update_every: int,
    neg_single: str = "",
):
    """Build the ``ss_*`` metadata dict for SMD LoRA outputs.

    Same standard keys as ``anima_train_network.py`` (via
    ``train_network._build_metadata``: session, LR, optimizer, batch size,
    network rank/alpha/args, scheduler, seed, model hashes, ...) plus the
    Anima trainer keys (``ss_weighting_scheme`` etc.) and ``ss_smd_*`` keys
    for every SMD-specific argument.

    Returns ``(metadata, minimum_metadata)`` with all values stringified,
    like the regular trainer.
    """
    g = lambda k, d=None: getattr(args, k, d)  # noqa: E731

    metadata = {
        "ss_session_id": session_id,
        "ss_training_started_at": training_started_at,
        "ss_output_name": g("output_name"),
        "ss_learning_rate": student_lr,
        "ss_smd_pivot_lr": pivot_lr,
        "ss_text_encoder_lr": g("text_encoder_lr"),
        "ss_unet_lr": g("unet_lr"),
        "ss_gradient_checkpointing": use_ckpt,
        "ss_gradient_accumulation_steps": accum_steps,
        "ss_max_train_steps": g("max_train_steps"),
        "ss_lr_warmup_steps": g("lr_warmup_steps"),
        "ss_lr_scheduler": g("lr_scheduler"),
        "ss_network_module": "networks.lora_anima",
        "ss_network_dim": rank,
        "ss_network_alpha": alpha,
        "ss_network_dropout": g("network_dropout"),
        "ss_mixed_precision": g("mixed_precision"),
        "ss_full_fp16": bool(g("full_fp16", False)),
        "ss_v2": bool(g("v2", False)),
        "ss_base_model_version": "anima",
        "ss_clip_skip": g("clip_skip"),
        "ss_max_token_length": g("qwen3_max_token_length", g("max_token_length")),
        "ss_cache_latents": bool(g("cache_latents", False)),
        "ss_seed": seed,
        "ss_lowram": g("lowram"),
        "ss_noise_offset": g("noise_offset"),
        "ss_multires_noise_iterations": g("multires_noise_iterations"),
        "ss_multires_noise_discount": g("multires_noise_discount"),
        "ss_adaptive_noise_scale": g("adaptive_noise_scale"),
        "ss_zero_terminal_snr": g("zero_terminal_snr"),
        "ss_training_comment": g("training_comment"),
        "ss_sd_scripts_commit_hash": model_io.get_git_revision_hash(),
        "ss_optimizer": optimizer_name + (f"({optimizer_args_str})" if optimizer_args_str else ""),
        "ss_max_grad_norm": g("max_grad_norm"),
        "ss_min_snr_gamma": g("min_snr_gamma"),
        "ss_scale_weight_norms": g("scale_weight_norms"),
        "ss_ip_noise_gamma": g("ip_noise_gamma"),
        "ss_debiased_estimation": bool(g("debiased_estimation_loss", False)),
        "ss_noise_offset_random_strength": g("noise_offset_random_strength"),
        "ss_ip_noise_gamma_random_strength": g("ip_noise_gamma_random_strength"),
        "ss_loss_type": "smd_velocity_regression",
        "ss_huber_schedule": g("huber_schedule"),
        "ss_huber_scale": g("huber_scale"),
        "ss_huber_c": g("huber_c"),
        "ss_fp8_base": bool(g("fp8_base", False)),
        "ss_fp8_base_unet": bool(g("fp8_base_unet", False)),
        # Anima trainer keys (AnimaNetworkTrainer.update_metadata)
        "ss_weighting_scheme": g("weighting_scheme"),
        "ss_logit_mean": g("logit_mean"),
        "ss_logit_std": g("logit_std"),
        "ss_mode_scale": g("mode_scale"),
        "ss_timestep_sampling": g("timestep_sampling"),
        "ss_sigmoid_scale": g("sigmoid_scale"),
        "ss_discrete_flow_shift": g("discrete_flow_shift"),
        # Batch / data-free dataset keys (regular-trainer non-user-config branch)
        "ss_batch_size_per_device": batch_size_per_device,
        "ss_total_batch_size": total_batch_size,
        "ss_resolution": g("resolution"),
    }

    if lora_kw:
        metadata["ss_network_args"] = json.dumps(lora_kw)

    # Base model / VAE / text-encoder names + hashes (regular-trainer style)
    sd_model_name = g("pretrained_model_name_or_path")
    if sd_model_name is not None:
        if os.path.exists(sd_model_name):
            metadata["ss_sd_model_hash"] = model_io.model_hash(sd_model_name)
            metadata["ss_new_sd_model_hash"] = model_io.calculate_sha256(sd_model_name)
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name
    vae_name = g("vae")
    if vae_name is not None:
        if os.path.exists(vae_name):
            metadata["ss_vae_hash"] = model_io.model_hash(vae_name)
            metadata["ss_new_vae_hash"] = model_io.calculate_sha256(vae_name)
            vae_name = os.path.basename(vae_name)
        metadata["ss_vae_name"] = vae_name
    qwen3_name = g("qwen3")
    if qwen3_name is not None:
        metadata["ss_smd_qwen3_name"] = (
            os.path.basename(qwen3_name) if os.path.exists(qwen3_name) else qwen3_name
        )

    # SMD-specific args
    neg_file = g("smd_neg_file")
    metadata.update(
        {
            "ss_smd_prompts_file": os.path.basename(g("smd_prompts_file") or ""),
            "ss_smd_num_prompts": num_prompts,
            "ss_smd_steps": g("smd_steps"),
            "ss_smd_shift": g("smd_shift"),
            "ss_smd_final_step_size_scale": g("smd_final_step_size_scale"),
            "ss_smd_cfg": g("smd_cfg"),
            "ss_smd_neg_prompt": neg_single,
            "ss_smd_neg_file": os.path.basename(neg_file) if neg_file else None,
            "ss_smd_queries": g("smd_queries"),
            "ss_smd_imitation_weight": g("smd_imitation_weight"),
            "ss_smd_repulsion_lambda": g("smd_repulsion_lambda"),
            "ss_smd_ref_cfg": g("smd_ref_cfg"),
            "ss_smd_w_rescale": g("smd_w_rescale"),
            "ss_smd_beta": g("smd_beta"),
            "ss_smd_eps": g("smd_eps"),
            "ss_smd_pivot": bool(use_pivot),
            "ss_smd_pivot_mode": pivot_mode,
            "ss_smd_pivot_hook_layer": g("smd_pivot_hook_layer"),
            "ss_smd_pivot_blocks": g("smd_pivot_blocks"),
            "ss_smd_p": g("smd_p"),
            "ss_smd_pivot_lora_rank": g("smd_pivot_lora_rank"),
            "ss_smd_ema": bool(use_ema),
            "ss_smd_ema_decay": ema_decay if use_ema else None,
            "ss_smd_ema_warmup": ema_warmup if use_ema else None,
            "ss_smd_ema_update_every": ema_update_every if use_ema else None,
            "ss_smd_eval_every": g("smd_eval_every"),
            "ss_smd_preview_n": g("smd_preview_n"),
        }
    )

    metadata = {k: str(v) for k, v in metadata.items()}

    minimum_metadata = {}
    for key in model_io.SS_METADATA_MINIMUM_KEYS:
        if key in metadata:
            minimum_metadata[key] = metadata[key]

    return metadata, minimum_metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    session_id = random.randint(0, 2**32)
    training_started_at = time.time()
    if not args.smd_prompts_file or not os.path.isfile(args.smd_prompts_file):
        raise ValueError("--smd_prompts_file is required (data-free distillation uses captions only).")
    with open(args.smd_prompts_file, encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    if not prompts:
        raise ValueError(f"No captions found in {args.smd_prompts_file}.")
    logger.info(f"Loaded {len(prompts)} captions from {args.smd_prompts_file} (images are not used).")
    if getattr(args, "smd_neg_file", None):
        if not os.path.isfile(args.smd_neg_file):
            raise ValueError(f"--smd_neg_file not found: {args.smd_neg_file}.")
        with open(args.smd_neg_file, encoding="utf-8") as f:
            neg_prompts = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        if not neg_prompts:
            raise ValueError(f"No negative prompts found in {args.smd_neg_file}.")
        logger.info(f"Loaded {len(neg_prompts)} negative prompts from {args.smd_neg_file}.")
    else:
        neg_prompts = None
    neg_single = getattr(args, "smd_neg_prompt", "") or ""
    if neg_prompts is None and neg_single:
        logger.info(f"Using fixed teacher negative prompt ({len(neg_single)} chars). Baked into the student; student stays CFG-free.")

    flux_train_utils.log_timestep_sampling_info(args)
    if args.fp8_base or args.fp8_base_unet:
        logger.warning("fp8_base/fp8_base_unet are not supported for Anima SMD; disabled.")
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

    # SMD validation (mirrors the upstream method's settings check).
    K = int(args.smd_steps)
    Q = int(args.smd_queries)
    if K < 1 or Q < 1:
        raise ValueError("At least one --smd_steps step and one --smd_queries query are required.")
    beta = float(args.smd_beta)
    if not 0.0 < beta <= 2.0:
        raise ValueError(f"--smd_beta must be in (0, 2], got {beta}")
    lam = 0.0 if args.smd_no_repulsion else float(args.smd_repulsion_lambda)
    if not 0.0 <= lam < 1.0:
        raise ValueError(f"--smd_repulsion_lambda must be in [0, 1), got {lam}")
    w_imitation = float(args.smd_imitation_weight)
    eps = float(args.smd_eps)
    w_rescale = float(args.smd_w_rescale)
    if not (w_imitation > 0 or lam > 0):
        raise ValueError("At least one of --smd_imitation_weight / --smd_repulsion_lambda must be positive.")
    teacher_cfg = float(args.smd_cfg)
    ref_cfg = float(args.smd_ref_cfg)
    use_pivot = not bool(args.smd_no_pivot)
    # Teacher branches actually needed per query state.
    need_cond = any(abs(sc) > 1e-9 for sc in (teacher_cfg, ref_cfg))
    need_uncond = any(abs(sc - 1.0) > 1e-9 for sc in (teacher_cfg, ref_cfg))

    seed = args.seed if args.seed is not None else 42
    set_seed(seed)
    random.seed(seed)

    accelerator = accelerator_setup.prepare_accelerator(args)
    logging_util.init_trackers(accelerator, args, "anima-smd")
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
            "DiT blocks are recomputed on backward; the DiT runs in train() mode for the "
            "WITH-grad student forwards and eval() everywhere else."
        )
    else:
        logger.info("gradient checkpointing disabled.")

    # ---- Pivot FIRST (pure pretrained copies, before any LoRA attach) ----
    pivot = None
    pivot_lora_net = None
    pivot_params: list = []
    if use_pivot:
        from networks import lora_anima as _lora_anima_mod  # noqa: F401 (used for pivot LoRA below)
        pivot = SMDPivot(
            dit,
            hook_layer=int(args.smd_pivot_hook_layer),
            basis_p=int(args.smd_p),
            attn_mode=dit.attn_mode,
            split_attn=bool(dit.split_attn),
        )
        M = len(pivot.blocks)
        if args.smd_pivot_blocks is not None and int(args.smd_pivot_blocks) != M:
            raise ValueError(
                f"--smd_pivot_blocks={args.smd_pivot_blocks} does not match the number of blocks after "
                f"--smd_pivot_hook_layer={args.smd_pivot_hook_layer} ({M}). Pretrained Pivot init requires the "
                "Pivot blocks to be the natural student tail; move the hook layer or drop the flag to derive M."
            )
        logger.info(
            f"Pivot: hook student block {int(args.smd_pivot_hook_layer)} ({pivot.hook_pos}), "
            f"M={M} copied blocks, P={args.smd_p} polynomial basis, zero-init output projection."
        )
        if args.smd_pivot_lora_rank:
            rank_p = int(args.smd_pivot_lora_rank)
            logger.info(f"Pivot blocks train via LoRA rank={rank_p} (bases frozen); proj/out_proj full-rank.")
            pivot_lora_net = _lora_anima_mod.create_network(
                1.0, rank_p, float(rank_p), None, None, pivot, neuron_dropout=args.network_dropout,
                verbose="false",
            )
            pivot_lora_net.apply_to([], pivot, apply_text_encoder=False, apply_unet=True)
            # Freeze the copied block bases; proj + out_proj (+ LoRA) train.
            pivot.requires_grad_(False)
            for p in pivot.proj.parameters():
                p.requires_grad_(True)
            for p in pivot.out_proj.parameters():
                p.requires_grad_(True)
            pivot_mode = "lora"
            pivot_params = list(pivot_lora_net.parameters()) + pivot.trainables()
        else:
            logger.info("Pivot trains full-rank (paper image recipe).")
            pivot.requires_grad_(True)
            pivot_mode = "full"
            pivot_params = pivot.trainables()
        if use_ckpt:
            for blk in pivot.blocks:
                blk.enable_gradient_checkpointing(
                    cpu_offload=bool(getattr(args, "cpu_offload_checkpointing", False)),
                    unsloth_offload=bool(getattr(args, "unsloth_offload_checkpointing", False)),
                )
        pivot.to(device, dtype=weight_dtype)
        if pivot_lora_net is not None:
            pivot_lora_net.to(device, dtype=weight_dtype)
        n_pivot_tr = sum(p.numel() for p in pivot_params)
        logger.info(f"Pivot trainable params: {n_pivot_tr/1e6:.1f}M ({pivot_mode}).")

    # ---- student LoRA on the frozen base (single adapter; teacher = mult 0) ----
    from networks import lora_anima

    rank = args.network_dim or 32
    # Repo convention defaults --network_alpha to 1; for distillation the
    # sane default is alpha == rank (same as the TDM/Krea recipe), so an
    # untouched default follows rank. Pass --network_alpha explicitly to override.
    if args.network_alpha in (None, 1) and rank != 1:
        alpha = float(rank)
        logger.info(f"--network_alpha left at default; using alpha=rank={rank} for SMD.")
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
    logger.info(f"Creating student LoRA (rank={rank}, alpha={alpha}, llm_adapter={llm_adapter_on})...")
    student_net = lora_anima.create_network(1.0, rank, alpha, None, None, dit, neuron_dropout=args.network_dropout, **lora_kw)
    student_net.apply_to([], dit, apply_text_encoder=False, apply_unet=True)
    student_net.to(device, dtype=weight_dtype)

    # The base (incl. shared norms/modulation/embedders) stays frozen forever;
    # roles differ ONLY through the student adapter.
    dit.requires_grad_(False)
    dit.eval()
    student_net.train()
    if use_pivot:
        pivot.train()
    n_s = sum(p.numel() for p in student_net.parameters())
    n_base = sum(p.numel() for p in dit.parameters())
    logger.info(f"Params: base {n_base/1e9:.2f}B frozen | student {n_s/1e6:.1f}M"
                + (f" | pivot {n_pivot_tr/1e6:.1f}M" if use_pivot else "") + " trainable.")

    if args.network_weights:
        logger.info(f"Init student adapter from {args.network_weights}")
        student_net.load_weights(args.network_weights)
    if args.smd_resume_state:
        logger.info(f"Resume student LoRA (+Pivot) from {args.smd_resume_state}")
        sd = load_file(args.smd_resume_state, device="cpu")
        s_sd = {k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")}
        student_net.load_state_dict(s_sd, strict=True)
        if use_pivot:
            p_sd = {k[len("pivot."):]: v for k, v in sd.items() if k.startswith("pivot.")}
            if p_sd:
                pivot.load_state_dict(p_sd, strict=True)
            else:
                logger.warning("Resume file has no pivot.* entries; Pivot re-initialized from pretrained copies.")
            if pivot_lora_net is not None:
                pl_sd = {k[len("pivot_lora."):]: v for k, v in sd.items() if k.startswith("pivot_lora.")}
                if pl_sd:
                    pivot_lora_net.load_state_dict(pl_sd, strict=True)
                else:
                    logger.warning("Resume file has no pivot_lora.* entries; pivot LoRA re-initialized.")

    # ---- student EMA shadow (fp32, student LoRA only) ----
    use_ema = not bool(getattr(args, "smd_no_ema", False))
    ema_decay = float(getattr(args, "smd_ema_decay", 0.999) or 0.0)
    ema_warmup = max(0, int(getattr(args, "smd_ema_warmup", 0) or 0))
    ema_update_every = max(1, int(getattr(args, "smd_ema_update_every", 1) or 1))
    if use_ema and not (0.0 < ema_decay < 1.0):
        logger.warning(f"--smd_ema_decay={ema_decay} out of (0, 1); disabling EMA.")
        use_ema = False
    ema_state = None
    if use_ema:
        ema_state = init_ema_state(student_net, device)
        if args.smd_resume_state:
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
                logger.info(f"Resumed student EMA from {args.smd_resume_state} ({len(e_sd)} tensors).")
            else:
                logger.info("Resume file has no ema.* entries; EMA re-initialized from student.")
        logger.info(f"Student EMA ENABLED (decay={ema_decay:g}, warmup={ema_warmup}, update_every={ema_update_every}). "
                    f"Previews and *_ema_*.safetensors use the EMA weights.")
    else:
        logger.info("Student EMA disabled (--smd_no_ema).")

    from library import optimizer as optimizer_util

    # Optimizer factory shared with anima_train_network.py (same
    # --optimizer_type/--optimizer_args handling via get_optimizer).
    # Two trainables need two LRs: the student uses --learning_rate directly,
    # while Pivot carries its own --smd_pivot_lr as an explicit param group --
    # the same pattern train_network.py uses for --unet_lr / --text_encoder_lr.
    student_lr = float(args.learning_rate)
    pivot_lr = float(args.smd_pivot_lr)
    optimizer_name, optimizer_args_str, opt_s = optimizer_util.get_optimizer(args, student_net.parameters())
    if use_pivot:
        _, _, opt_p = optimizer_util.get_optimizer(
            args, [{"params": pivot_params, "lr": pivot_lr}]
        )
    logger.info(f"Optimizers: student lr={student_lr:g}, "
                + (f"pivot lr={pivot_lr:g}, " if use_pivot else "")
                + f"type={args.optimizer_type or 'AdamW'}, optimizer_args={args.optimizer_args}.")
    opt_s_train_fn, opt_s_eval_fn = optimizer_util.get_optimizer_train_eval_fn(opt_s, args)
    if use_pivot:
        opt_p_train_fn, opt_p_eval_fn = optimizer_util.get_optimizer_train_eval_fn(opt_p, args)
    else:
        opt_p_train_fn, opt_p_eval_fn = (lambda: None), (lambda: None)

    # Shared LR schedulers (same --lr_scheduler/--lr_warmup_steps/--lr_decay_steps/
    # --lr_scheduler_num_cycles/--lr_scheduler_power/... as the regular trainer).
    # One scheduler per optimizer; both stepped once per optimizer step (not per
    # grad-accumulation micro-step). get_scheduler_fix derives total steps from
    # --max_train_steps.
    sched_s = optimizer_util.get_scheduler_fix(args, opt_s, accelerator.num_processes)
    sched_p = optimizer_util.get_scheduler_fix(args, opt_p, accelerator.num_processes) if use_pivot else None
    logger.info(f"LR schedulers: {args.lr_scheduler} (warmup={args.lr_warmup_steps}, "
                f"max_train_steps={args.max_train_steps}).")
    if args.smd_resume_optim:
        logger.info(f"Resume optimizers (+schedulers) from {args.smd_resume_optim}")
        o = torch.load(args.smd_resume_optim, map_location="cpu", weights_only=False)
        opt_s.load_state_dict(o["student"])
        if use_pivot:
            opt_p.load_state_dict(o["pivot"])
        for sched, key in ((sched_s, "sched_student"), (sched_p, "sched_pivot")):
            if sched is not None and key in o and hasattr(sched, "load_state_dict"):
                try:
                    sched.load_state_dict(o[key])
                except Exception as e:
                    logger.warning(f"Could not load {key} state (old checkpoint?): {e}")
        global_step = int(o.get("step", 0))
    else:
        global_step = 0

    # ---- geometry / schedule / loss weights ----
    img_h, img_w = parse_resolution(args.resolution)
    lat_h, lat_w = img_h // 8, img_w // 8
    latent_ch = 16
    logger.info(
        f"Distilling at {img_w}x{img_h} (latent {lat_w}x{lat_h}x{latent_ch}), K={K} steps, Q={Q} queries/transition."
    )
    ts = make_smd_schedule(K, args.smd_shift, args.smd_final_step_size_scale, device)  # [K+1], 1 -> 0
    logger.info(f"Student schedule (shift={args.smd_shift}, final_step_size_scale={args.smd_final_step_size_scale}): "
                f"{[round(float(v), 4) for v in ts.cpu().tolist()]}")
    normalization = 0.5 * w_rescale / K  # SMD LossWeights normalization
    logger.info(
        f"Loss: imitation(w={w_imitation:g}, beta={beta:g}) - lambda*repulsion(lambda={lam:g}, ref CFG={ref_cfg:g}) "
        f"at teacher CFG={teacher_cfg:g}; normalization={normalization:.3g} (0.5*{w_rescale:g}/K)."
    )
    logger.info("Pivot ENABLED (training only; zero inference overhead)." if use_pivot else
                "Pivot DISABLED -- plain on-policy baseline (paper Sec. 3.1).")

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = args.output_name or "anima-smd"
    csv_path = os.path.join(args.output_dir, "smd_loss_log.csv")
    new_csv = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if new_csv:
        csv_writer.writerow(["step", "loss", "loss_imitation", "loss_repulsion", "lr_student", "lr_pivot", "time"])

    autocast_kw = dict(device_type=device.type if isinstance(device, torch.device) else str(device),
                       dtype=weight_dtype)
    if autocast_kw["device_type"] not in ("cuda", "cpu", "xpu"):
        autocast_kw["device_type"] = "cuda"
    t0 = time.time()

    attn_params = attention.AttentionParams.create_attention_params(dit.attn_mode, bool(dit.split_attn))

    def save_all(step: int):
        # Schedule-free optimizers must be in eval mode for checkpointing
        # (mirrors train_network.py); no-ops for all other optimizer types.
        opt_s_eval_fn()
        opt_p_eval_fn()
        try:
            _save_all(step)
        finally:
            opt_s_train_fn()
            opt_p_train_fn()

    def _save_all(step: int):
        # Full ss_* metadata like anima_train_network.py (standard keys +
        # Anima keys + ss_smd_*), merged with the SAI ModelSpec block. The
        # regular trainer lets the ModelSpec block win on conflicts; same here.
        # Honors --no_metadata (minimum subset only).
        metadata, minimum_metadata = build_smd_metadata(
            args,
            session_id=session_id,
            training_started_at=training_started_at,
            rank=rank,
            alpha=alpha,
            lora_kw=lora_kw,
            optimizer_name=optimizer_name,
            optimizer_args_str=optimizer_args_str,
            student_lr=student_lr,
            pivot_lr=pivot_lr,
            num_prompts=len(prompts),
            batch_size_per_device=B,
            total_batch_size=B * accelerator.num_processes * accum_steps,
            accum_steps=accum_steps,
            seed=seed,
            use_ckpt=use_ckpt,
            use_pivot=use_pivot,
            pivot_mode=pivot_mode if use_pivot else "none",
            use_ema=bool(use_ema and ema_state is not None),
            ema_decay=ema_decay,
            ema_warmup=ema_warmup,
            ema_update_every=ema_update_every,
            neg_single=neg_single,
        )
        metadata["ss_training_finished_at"] = str(time.time())
        metadata["ss_steps"] = str(step)
        metadata["ss_epoch"] = "1"  # data-free streaming: one virtual epoch spans the run
        metadata["ss_smd_step"] = str(step)
        minimum_metadata = dict(minimum_metadata)
        minimum_metadata["ss_smd_step"] = str(step)
        metadata_to_save = dict(minimum_metadata) if getattr(args, "no_metadata", False) else dict(metadata)
        sai_metadata = model_io.get_sai_model_spec_dataclass(None, args, False, True, False, anima="preview").to_metadata_dict()
        metadata_to_save.update(sai_metadata)
        metadata_to_save = {str(k): str(v) for k, v in metadata_to_save.items()}
        s_sd = {f"student.{k}": v.detach().cpu().contiguous() for k, v in student_net.state_dict().items()}
        state_dict = {**s_sd}
        if use_ema and ema_state is not None:
            for k, e in ema_state.items():
                state_dict[f"ema.{k}"] = e.detach().cpu().contiguous()
        if use_pivot:
            state_dict.update({f"pivot.{k}": v.detach().cpu().contiguous() for k, v in pivot.state_dict().items()})
            if pivot_lora_net is not None:
                state_dict.update({f"pivot_lora.{k}": v.detach().cpu().contiguous()
                                   for k, v in pivot_lora_net.state_dict().items()})
        state_path = os.path.join(args.output_dir, f"smd_state_step_{step}.safetensors")
        save_file(state_dict, state_path, metadata=metadata_to_save)
        logger.info(f"[ckpt] student LoRA + pivot (+EMA) -> {state_path}")
        student_path = os.path.join(args.output_dir, f"{out_name}_step_{step}.safetensors")
        student_net.save_weights(student_path, save_dtype, dict(metadata_to_save))
        logger.info(f"[ckpt] student LoRA -> {student_path}")
        if use_ema and ema_state is not None:
            # Save the EMA student via a temporary swap so hashing/metadata
            # reuse the LoRA save path. Raw student is restored afterwards.
            backup_sd = {k: v.detach().clone() for k, v in student_net.state_dict().items()}
            try:
                student_net.load_state_dict(ema_casted_state_dict(ema_state, student_net), strict=True)
                ema_meta = dict(metadata_to_save)
                ema_meta["ss_smd_ema"] = f"decay={ema_decay:g}"
                ema_path = os.path.join(args.output_dir, f"{out_name}_ema_step_{step}.safetensors")
                student_net.save_weights(ema_path, save_dtype, ema_meta)
                logger.info(f"[ckpt] student EMA LoRA -> {ema_path}")
            finally:
                student_net.load_state_dict(backup_sd, strict=True)
        optim_state = {"student": opt_s.state_dict(), "step": step + 1}
        if use_pivot:
            optim_state["pivot"] = opt_p.state_dict()
        for sched, key in ((sched_s, "sched_student"), (sched_p, "sched_pivot")):
            if sched is None:
                continue
            try:
                if hasattr(sched, "state_dict"):
                    optim_state[key] = sched.state_dict()
            except Exception as e:
                logger.warning(f"Could not save {key} state: {e}")
        torch.save(optim_state,
                   os.path.join(args.output_dir, f"smd_optim_step_{step}.pt"))

    @torch.no_grad()
    def smd_student_rollout(x, t_schedule, cond_tenant_list, weight_dtype):
        """CFG-free K-step student sampling with the EXACT training schedule."""
        out = x.clone()
        n = len(t_schedule) - 1
        for k in range(n):
            t_vec = torch.full((out.shape[0],), float(t_schedule[k].item()), dtype=torch.float32, device=out.device)
            v = dit_predict_v(dit, out, t_vec, cond_tenant_list, weight_dtype).float()
            out = (out.float() + (float(t_schedule[k + 1].item()) - float(t_schedule[k].item())) * v).to(weight_dtype)
        return out

    @torch.no_grad()
    def preview(step: int):
        dit.eval()
        opt_s_eval_fn()
        opt_p_eval_fn()
        # Previews render with EMA weights when available (restored afterwards);
        # training rollouts always use the raw student.
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
            opt_p_train_fn()

    @torch.no_grad()
    def _preview_rollout(step: int):
        dit.eval()
        set_role(student_net, "student")
        vae_device = vae.device if hasattr(vae, "device") else device
        try:
            vae.to(device)
        except Exception:
            pass
        shots = prompts[: min(args.smd_preview_n, len(prompts))]
        imgs = []
        preview_base_seed = seed  # shared --seed drives training RNG and preview base seed
        for i, cap in enumerate(shots):
            gen = torch.Generator(device="cpu").manual_seed(preview_base_seed + i)
            noise = torch.randn(1, latent_ch, lat_h, lat_w, generator=gen, dtype=torch.float32).to(weight_dtype).to(device)
            with torch.autocast(**autocast_kw):
                cond = encode_cond(tokenize_strategy, text_encoding_strategy, qwen3, [cap], weight_dtype)
                lat = smd_student_rollout(noise, ts, cond, weight_dtype)
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
        path = os.path.join(args.output_dir, f"sample/smd_step_{step}.png")
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
    pbar = tqdm(range(global_step, max_steps), initial=global_step, total=max_steps, desc="SMD")
    for opt_step in pbar:
        # Accumulation window: zero once, run `accum_steps` micro-batches,
        # step once. Losses are scaled by 1/accum so the accumulated grad
        # equals the mean over the effective (B*accum) batch.
        zero_all(student_net, *( [pivot_lora_net, pivot] if use_pivot else [] ))
        loss_sum, loss_imit_sum, loss_rep_sum = 0.0, 0.0, 0.0
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
                    uncond = encode_cond(tokenize_strategy, text_encoding_strategy, qwen3, neg_caps, weight_dtype, uncond=use_blank_anchor) if need_uncond else None

            x = torch.randn(B, latent_ch, lat_h, lat_w, device=device, dtype=weight_dtype)

            # ---- K-transition on-policy rollout (paper Algorithm 2) ----
            # stopgrad across transitions; per-transition backward keeps graph
            # memory tight (SMD per_segment_backward=True).
            for i in range(K):
                t_start = float(ts[i].item())
                dt = float(ts[i + 1].item()) - t_start  # < 0
                x = x.detach()
                if dt >= 0:
                    raise RuntimeError(f"Non-decreasing schedule segment at {i}: dt={dt}")
                t_vec = torch.full((B,), t_start, dtype=torch.float32, device=device)
                alphas = torch.rand(B, Q, device=device, dtype=torch.float32).sort(dim=1).values

                # (a) one student prediction WITH grad at the transition start.
                # With Pivot, the same forward captures the K-th-block-from-end
                # features + conditioning args needed by the Pivot branch.
                set_role(student_net, "student")
                hook_block = dit.blocks[pivot.hook_pos] if use_pivot else None
                if use_ckpt:
                    dit.train()  # AnimaBlock gates checkpointing on self.training
                if use_pivot:
                    with capture_dit_block_features(hook_block) as cap:
                        with torch.autocast(**autocast_kw):
                            v_theta = dit_predict_v(dit, x, t_vec, cond, weight_dtype, ckpt_input_grad=use_ckpt).float()
                else:
                    with torch.autocast(**autocast_kw):
                        v_theta = dit_predict_v(dit, x, t_vec, cond, weight_dtype, ckpt_input_grad=use_ckpt).float()
                if use_ckpt:
                    dit.eval()  # back to deterministic eval for the no-grad scores

                # (b) Pivot coefficients from the captured features
                # (gradients reach BOTH the Pivot and the student via the features).
                if use_pivot:
                    use_fp32 = cap["args"][4]
                    bkw = cap["kwargs"]
                    with torch.autocast(**autocast_kw):
                        coeffs = pivot(
                            cap["feat"].to(x.dtype), cap["args"][1], cap["args"][2], attn_params, use_fp32, bkw
                        ).float()  # [B, P, C, H, W]

                # (c) proxy query states + velocities (both carry grad).
                # Paper Sec. 3.2 (tau there == alpha here):
                # v(alpha) = v_th + r(alpha);
                # x_s = x_t + alpha*dt*v_th + dt*q(alpha) with B_m endpoints fixed.
                v_q_list, x_q_list, s_q_list = [], [], []
                for k in range(Q):
                    alpha_k = alphas[:, k]
                    if use_pivot:
                        q_k, r_k = poly_basis_eval(coeffs, alpha_k)  # [B, C, H, W] each
                    else:
                        q_k, r_k = torch.zeros(1), torch.zeros(1)
                    v_q_k = v_theta + r_k
                    ta4 = alpha_k.view(B, 1, 1, 1)
                    s_k = torch.full((B,), t_start, device=device) + alpha_k * dt
                    x_q_k = x.float() + ta4 * dt * v_theta + dt * q_k
                    v_q_list.append(v_q_k)
                    x_q_list.append(x_q_k)
                    s_q_list.append(s_k)
                v_q_all = torch.stack(v_q_list)          # [Q, B, C, H, W] grad
                x_all = torch.cat(x_q_list, dim=0)       # [B*Q, C, H, W] grad
                s_all = torch.cat(s_q_list, dim=0)       # [B*Q] no grad

                # (d) frozen teacher queries at the SAME states (targets; teacher
                # outputs are detached per the paper). Separate cond/uncond
                # branches are combined with the CFG scales. The student adapter
                # is zeroed so the teacher is the plain base weights.
                set_role(student_net, "teacher")
                with torch.autocast(**autocast_kw), torch.no_grad():
                    v_real_all, v_unc_all = None, None
                    if need_cond:
                        v_real_all = dit_predict_v(dit, x_all.detach(), s_all, cond, weight_dtype).float()
                    if need_uncond:
                        v_unc_all = dit_predict_v(dit, x_all.detach(), s_all, uncond, weight_dtype).float()
                v_target_all = cfg_combine(v_real_all, v_unc_all, teacher_cfg).view(Q, B, *v_theta.shape[1:])
                v_ref_all = cfg_combine(v_real_all, v_unc_all, ref_cfg).view(Q, B, *v_theta.shape[1:])

                # (e) imitation + repulsion over the query points
                imitation, repulsion = None, None
                for k in range(Q):
                    if w_imitation > 0:
                        d_t = per_sample_velocity_distance(v_q_all[k], v_target_all[k], beta, eps).mean()
                        imitation = d_t if imitation is None else imitation + d_t
                    if lam > 0:
                        d_r = per_sample_velocity_distance(v_q_all[k], v_ref_all[k], beta, eps).mean()
                        repulsion = d_r if repulsion is None else repulsion + d_r
                imitation = imitation / Q if imitation is not None else torch.zeros((), device=device)
                repulsion = repulsion / Q if repulsion is not None else torch.zeros((), device=device)
                seg_loss = normalization * (w_imitation * imitation - lam * repulsion)

                # (f) per-transition backward (equivalent to the paper's full
                # rollout loss since graphs only span one transition each).
                (seg_loss / accum_steps).backward()
                loss_sum += seg_loss.item()
                loss_imit_sum += float(imitation.item())
                loss_rep_sum += float(repulsion.item())

                # (g) next rollout state follows the COARSE student update (no pivot,
                # no grad) -- paper Algorithm 2 line 17 / Algorithm 1 line 13.
                x = (x.float() + dt * v_theta.detach()).to(weight_dtype).detach()

        # End of accumulation window: single clip + step per optimizer, then
        # step both LR schedulers once per optimizer step.
        if max_grad_norm > 0:
            if use_pivot:
                torch.nn.utils.clip_grad_norm_(pivot_params, max_grad_norm)
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), max_grad_norm)
        if use_pivot:
            opt_p.step()
        opt_s.step()
        if use_pivot:
            sched_p.step()
        sched_s.step()
        lr_s_now = opt_s.param_groups[0]["lr"]
        lr_p_now = opt_p.param_groups[0]["lr"] if use_pivot else 0.0
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
        loss_v = loss_sum / accum_steps
        imit_v = loss_imit_sum / accum_steps
        rep_v = loss_rep_sum / accum_steps

        csv_writer.writerow(
            [opt_step, f"{loss_v:.6f}", f"{imit_v:.6f}", f"{rep_v:.6f}",
             f"{lr_s_now:.2e}", f"{lr_p_now:.2e}", f"{time.time() - t0:.1f}"]
        )
        pbar.set_postfix(loss=f"{loss_v:.4f}", imit=f"{imit_v:.4f}", rep=f"{rep_v:.4f}",
                         lr=f"{lr_s_now:.2e}/{lr_p_now:.2e}")
        if len(accelerator.trackers) > 0:
            log_dict = {"loss": loss_v, "loss_imitation": imit_v, "loss_repulsion": rep_v, "lr_student": lr_s_now}
            if use_pivot:
                log_dict["lr_pivot"] = lr_p_now
            accelerator.log(log_dict, step=opt_step)
        if opt_step % 10 == 0:
            csv_file.flush()

        if save_every > 0 and opt_step > 0 and opt_step % save_every == 0:
            save_all(opt_step)
        if args.smd_eval_every > 0 and opt_step % args.smd_eval_every == 0:
            preview(opt_step)

    logger.info(f"Reached max_train_steps={max_steps}. Saving final checkpoint.")
    save_all(max_steps)
    if args.smd_eval_every > 0:
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
