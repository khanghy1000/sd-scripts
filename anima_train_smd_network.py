"""Anima SMD few-step distillation (data-free, LoRA + training-only Pivot).

Simplified Matching Distillation (arXiv:2610.18472) distills the frozen Anima
rectified-flow DiT into a K-step deterministic student using only captions --
no images, no fake-score model, no adversarial loss. The teacher is the frozen
base with the student LoRA multiplier zeroed; the student is one LoRA adapter
on the shared frozen base (``networks/lora_anima.py``).

Per transition ``t_i -> t_{i+1}`` (paper Algorithm 2; dt < 0):

  - one with-grad student prediction ``v_th`` at the transition start;
  - Q teacher queries on the student's own path. With
    ``--smd_query_sampling raw_t`` (reference default) alpha ~ U(0,1) is
    sampled in raw-t space and warped to the query sigma; ``sigma``
    interpolates the warped sigma directly. Queries are sorted per sample
    (order is irrelevant to the loss);
  - per-transition loss ``0.5 * smd_w_rescale / K * (imitation - lambda * repulsion)``
    where the teacher target uses CFG ``--smd_cfg`` and the low-CFG reference
    ``ev`` uses ``--smd_ref_cfg``;
  - the rollout continues with the coarse student step (no Pivot, no grad);
    each transition backprops separately.

``--smd_step_granularity segment`` (reference default) takes one optimizer
step per transition and reuses the batch + rollout state across the K
transitions, so ``--max_train_steps`` counts segments; ``rollout`` takes one
step per full K-transition rollout (paper Algorithm 2's mean over transitions).

Pivot (Sec. 3.2): a training-only proxy predicts P basis coefficients ``a_m``
from the features of the ``--smd_pivot_hook_layer``-th block from the end,
bending the interior query path ``v = v_th + sum_m a_m b_m(alpha)`` while the
transition endpoints stay untouched (``B_m(0) = B_m(1) = 0``). Architecture:
copied student tail blocks -> the student's final AdaLN norm/modulation ->
a zero-initialized expanded output head (reference Pivot; no extra input
projection). ``--smd_basis polynomial`` is the paper Eq. 16/22 family
(reference ``taylor_path``, ``sqrt_n`` scaled); ``--smd_basis piecewise`` is
the Table 15 Qwen-Image / reference ``centered_dx`` family. Never evaluated at
inference; use ``--smd_no_pivot`` for the plain baseline.

The paper's image recipe (Table 15, Qwen-Image row) uses rank/alpha 256 LoRA
(``--network_dim 256 --network_alpha 256``), a full-rank Pivot with M=3 blocks
from hook -4, P=4, Q=2, CFG 5.0 / ref 1.0, lambda 0.2, LR 1e-4 (student and
pivot), and 1000 iterations with 100 warm-up. The shaped defaults here match
that recipe except the LoRA rank (repo default 32), the basis family (the
reference's generic ``taylor_path``; pass ``--smd_basis piecewise`` for the
Qwen-Image row / ``centered_dx``), and the final step size (1.0 for sampler
compatibility). AdamW comes from the shared optimizer factory, so pass
``--optimizer_args "weight_decay=0"`` for the paper's zero decay.

Deployment: CFG-free K Euler steps on the training schedule -- uniform raw
knots 1 -> 0 with the final segment scaled by ``--smd_final_step_size_scale``
(1.0 default = do_sample-compatible uniform grid), then the rational shift
``S(t) = shift*t / (1 + (shift-1)*t)``. Use the same scale + K + shift the
training used; guidance_scale = 1.0 at inference (CFG is baked in).

Run (single GPU; prompts file only)::

    python anima_train_smd_network.py \\
      --pretrained_model_name_or_path anima-base.safetensors \\
      --qwen3 /path/to/qwen3-0.6b.safetensors --vae /path/to/vae.safetensors \\
      --smd_prompts_file prompts.txt --resolution 512,512 \\
      --network_dim 256 --network_alpha 256 \\
      --train_batch_size 8 --max_train_steps 1000 --output_dir output/anima-smd-4step

Optimizer/LR/scheduler/batch/seed/save flags are shared with
``anima_train_network.py``. Outputs: ``{output_name}_step_<n>.safetensors``
(student LoRA, standard layout), ``{output_name}_ema_*`` (EMA weights,
recommended for inference), ``smd_state_step_<n>.safetensors`` +
``smd_optim_step_<n>.pt`` (resume via ``--smd_resume_state`` /
``--smd_resume_optim``), losses in ``smd_loss_log.csv``, preview grids in
``sample/`` (with ``--smd_eval_every``).
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
        "Simplified Matching Distillation few-step distillation (arXiv:2610.18472). Data-free: needs only "
        "--smd_prompts_file; optimizer/LR/scheduler/batch flags are shared with anima_train_network.py.",
    )
    g.add_argument("--smd_prompts_file", type=str, default=None,
                   help="Text file with one caption per line. REQUIRED.")
    g.add_argument("--smd_steps", type=int, default=4,
                   help="K: number of deterministic student steps / NFE (default: 4).")
    g.add_argument("--smd_shift", type=float, default=3.0,
                   help="Anima rational flow shift for the K-step schedule (default: 3.0).")
    g.add_argument("--smd_final_step_size_scale", type=float, default=1.0,
                   help="Scale of the LAST Euler segment; 1.0 = uniform grid matching the standard sampler. "
                        "A non-1.0 value must be mirrored in the deployment sampler.")
    g.add_argument("--smd_cfg", type=float, default=5.0,
                   help="Teacher CFG scale for the imitation target (paper image recipe: 5.0).")
    g.add_argument("--smd_neg_prompt", type=str, default="",
                   help="Teacher CFG negative prompt (baked into the CFG-free student).")
    g.add_argument("--smd_neg_file", type=str, default=None,
                   help="One negative prompt per line; overrides --smd_neg_prompt.")
    g.add_argument("--smd_queries", type=int, default=2,
                   help="Q: teacher query points per transition, alpha ~ U(0,1) (paper image recipe: 2).")
    g.add_argument("--smd_query_sampling", choices=["raw_t", "sigma"], default="raw_t",
                   help="How query points are placed inside a transition. raw_t (reference default): sample raw t, "
                        "warp it, and derive the basis alpha from the warped fraction. sigma: interpolate the warped "
                        "sigma linearly (basis alpha == sampling alpha).")
    g.add_argument("--smd_query_alpha_mode", choices=["random", "random_endpoint"], default="random",
                   help="random: alpha ~ U(0,1) (reference default). random_endpoint: each segment queries only "
                        "alpha=0 or alpha=1, chosen deterministically from (step, segment); requires Q=1.")
    g.add_argument("--smd_imitation_weight", type=float, default=1.0,
                   help="Weight of the imitation term.")
    g.add_argument("--smd_repulsion_lambda", type=float, default=0.2,
                   help="Repulsion weight lambda in [0, 1); 0 disables (paper image recipe: 0.2).")
    g.add_argument("--smd_repulsion_adaptive", action="store_true",
                   help="Reference adaptive repulsion gate: scale each sample's repulsion by "
                        "clamp((d(v,ev)-d(v,vT))/(d(vT,ev)+eps), 0, 1) (detached).")
    g.add_argument("--smd_ref_cfg", type=float, default=1.0,
                   help="CFG scale of the reference prediction ev at query states (1.0 = conditional branch).")
    g.add_argument("--smd_cfg_norm_rescale", action="store_true",
                   help="Reference cfg_norm_rescale: rescale the CFG-combined velocity to the conditional "
                        "velocity's per-token (2x2 patch) norm.")
    g.add_argument("--smd_w_rescale", type=float, default=30.0,
                   help="Loss normalizer: total = 0.5 * this / K * (imitation - lambda * repulsion).")
    g.add_argument("--smd_beta", type=float, default=1.0,
                   help="Velocity distance exponent: per-sample MSE ** (beta/2); 2 = MSE, 1 = smoothed L2.")
    g.add_argument("--smd_eps", type=float, default=1e-6,
                   help="Epsilon smoothing inside the velocity distance.")
    g.add_argument("--smd_teacher_window_alpha", type=float, default=0.0,
                   help="Reference teacher_window_alpha: when > 0 the student velocity matched against the teacher "
                        "is the average over [alpha, alpha+window] instead of the instantaneous value.")
    g.add_argument("--smd_no_pivot", action="store_true",
                   help="Train the plain on-policy baseline without the Pivot proxy (paper Sec. 3.1).")
    g.add_argument("--smd_no_repulsion", action="store_true",
                   help="Disable the repulsion term (same as --smd_repulsion_lambda 0).")
    # Pivot
    g.add_argument("--smd_pivot_hook_layer", type=int, default=-4,
                   help="Block index (negative) whose output features feed the Pivot (default: -4). "
                        "Must leave the copied M blocks after it.")
    g.add_argument("--smd_pivot_blocks", type=int, default=None,
                   help="M copied Pivot blocks (paper image recipe: 3). Default: all blocks after the hook.")
    g.add_argument("--smd_p", type=int, default=4,
                   help="P residual basis terms for the Pivot (paper image recipe: 4).")
    g.add_argument("--smd_basis", choices=["polynomial", "piecewise"], default="polynomial",
                   help="Pivot residual basis family. polynomial: B_m(t)=t(1-t)(2t-1)^(m-1) (paper Eq. 16/22; "
                        "reference basis_family='taylor_path'). piecewise: centred nodal velocities on P uniform "
                        "nodes (paper Table 15 Qwen-Image row; reference residual_basis='centered_dx').")
    g.add_argument("--smd_basis_scaling", choices=["sqrt_n", "none"], default="sqrt_n",
                   help="Reference residual_scale_mode for the polynomial basis: sqrt_n divides the coefficients "
                        "by sqrt(P) (reference default). Ignored by the piecewise basis (centered_dx).")
    g.add_argument("--smd_pivot_lr", type=float, default=1e-4,
                   help="Pivot optimizer LR (student uses --learning_rate).")
    g.add_argument("--smd_pivot_lora_rank", type=int, default=None,
                   help="If set, train the Pivot blocks via LoRA of this rank (the final norm/modulation and "
                        "output projection stay full-rank).")
    # Resume / eval / EMA (same conventions as the TDM script)
    g.add_argument("--smd_resume_state", type=str, default=None,
                   help="Resume student LoRA (+Pivot) from a smd_state_*.safetensors file.")
    g.add_argument("--smd_resume_optim", type=str, default=None,
                   help="Resume both optimizers (+schedulers) from a smd_optim_*.pt file.")
    g.add_argument("--smd_eval_every", type=int, default=0, help="Render K-step student previews every N steps (0 = disabled).")
    g.add_argument("--smd_preview_n", type=int, default=4, help="Preview images per eval.")
    g.add_argument("--smd_step_granularity", choices=["segment", "rollout"], default="segment",
                   help="Optimizer-step granularity. segment (reference default): one optimizer step per "
                        "transition, with the batch/latent reused across the K transitions of a rollout, so "
                        "--max_train_steps counts segments. rollout: one optimizer step per full K-transition "
                        "rollout (paper Algorithm 2 returns the mean over all transitions).")
    g.add_argument("--smd_no_ema", action="store_true",
                   help="Disable the student EMA shadow (default on; EMA drives previews and *_ema_* saves).")
    g.add_argument("--smd_ema_decay", type=float, default=0.999,
                   help="EMA decay for the student LoRA shadow, updated once per optimizer step.")
    g.add_argument("--smd_ema_warmup", type=int, default=0,
                   help="Optimizer steps before the EMA starts tracking.")
    g.add_argument("--smd_ema_update_every", type=int, default=1,
                   help="Update the EMA every N optimizer steps.")


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


def make_smd_raw_knots(steps: int, final_step_size_scale: float) -> torch.Tensor:
    """K+1 raw-t knots from 1.0 to 0.0 (SMD ``compute_raw_grid_knots``).

    Uniform except the last Euler segment, which is scaled by
    ``final_step_size_scale``. These are the un-warped times; the DiT never
    sees them directly -- see :func:`make_smd_schedule`.
    """
    fss = max(float(final_step_size_scale), 1e-4)
    base_seg = 1.0 / (steps - 1 + fss)
    raw = [1.0]
    for i in range(steps):
        seg = base_seg * (fss if i == steps - 1 else 1.0)
        raw.append(raw[-1] - seg)
    raw[-1] = 0.0
    return torch.tensor(raw, dtype=torch.float32)


def warp_t(raw_t: torch.Tensor, shift: float) -> torch.Tensor:
    """Anima/rectified-flow rational time shift ``S(t) = shift*t / (1 + (shift-1)*t)``."""
    if shift == 1.0:
        return raw_t
    return (raw_t * shift) / (1.0 + (shift - 1.0) * raw_t)


def make_smd_schedule(steps: int, shift: float, final_step_size_scale: float, device) -> torch.Tensor:
    """K+1 shifted sigmas from 1.0 to 0.0 on the SMD grid (paper Sec. 2/3).

    Raw knots are built by :func:`make_smd_raw_knots`, then Anima's rational
    shift is applied (same as sampling/do_sample). The resulting values are
    what both the student and the teacher pass as the timestep input to the
    DiT. ``--smd_query_sampling raw_t`` additionally needs the raw knots, so
    they are returned by the caller's separate ``make_smd_raw_knots`` call.
    """
    return warp_t(make_smd_raw_knots(steps, final_step_size_scale), shift).to(device)


class SMDBasis:
    """Zero-mean, endpoint-preserving residual basis for the training Pivot.

    Both families are the reference SMD bases evaluated on the Anima latent
    layout; they return a *unit-dt* path correction ``q`` (``dt`` is applied
    by the caller) and the instantaneous velocity residual ``r``:

        v(alpha) = v_base + r(alpha)
        x(alpha) = x_i + alpha*dt*v_base + dt*q(alpha)

    ``polynomial`` (paper Eq. 16/22, reference ``taylor_path``):
        B_m(t) = t(1-t)(2t-1)^(m-1),  q = sum_m a_m B_m,  r = sum_m a_m B_m'.
        ``sqrt_n`` scaling divides the coefficients by sqrt(P) (reference
        ``residual_scale_mode`` default).

    ``piecewise`` (paper Table 15 Qwen-Image row, reference ``centered_dx``):
        coefficients are residual *node values* on P uniform nodes, projected
        to zero trapezoidal mean; ``r`` is piecewise linear and ``q`` its exact
        piecewise-quadratic integral. No extra rescaling (matches reference).
    """

    def __init__(self, kind: str, p: int, scaling: str):
        if kind not in ("polynomial", "piecewise"):
            raise ValueError(f"Unknown --smd_basis={kind!r}; expected 'polynomial' or 'piecewise'.")
        if kind == "piecewise" and p < 2:
            raise ValueError(f"--smd_basis piecewise needs --smd_p >= 2, got {p}.")
        self.kind = kind
        self.p = int(p)
        # sqrt_n only applies to the analytic families in the reference.
        self.scale = (self.p ** -0.5) if (kind == "polynomial" and scaling == "sqrt_n") else 1.0
        if kind == "piecewise":
            h = 1.0 / (self.p - 1)
            w = torch.full((self.p,), h, dtype=torch.float32)
            w[0] = h / 2.0
            w[-1] = h / 2.0
            self._weights = w
            self._nodes = torch.linspace(0.0, 1.0, self.p, dtype=torch.float32)

    def build(self, flow: torch.Tensor) -> torch.Tensor:
        """Turn the Pivot's ``[B, P, C, H, W]`` output into basis coefficients."""
        if self.kind == "polynomial":
            return flow.float() * self.scale
        if flow.shape[1] != self.p:
            raise ValueError(f"Pivot produced {flow.shape[1]} slots, expected P={self.p}.")
        raw = flow.float()
        w = self._weights.to(device=raw.device).reshape(1, -1, *([1] * (raw.ndim - 2)))
        return raw - (w * raw).sum(dim=1, keepdim=True)

    def evaluate(self, coeffs: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(q_unit_dt, r)`` at per-sample ``alpha`` [B]."""
        if self.kind == "polynomial":
            return self._eval_polynomial(coeffs, alpha)
        return self._eval_piecewise(coeffs, alpha)

    def _eval_polynomial(self, coeffs: torch.Tensor, alpha: torch.Tensor):
        B, P = coeffs.shape[:2]
        a = alpha.to(coeffs.dtype).view(B, 1, 1, 1)
        two_a_m1 = 2.0 * a - 1.0
        factor = a * (1.0 - a)
        q = torch.zeros_like(coeffs[:, 0])
        r = torch.zeros_like(coeffs[:, 0])
        for m in range(1, P + 1):
            c = coeffs[:, m - 1]
            pw = two_a_m1 ** (m - 1)
            q = q + c * (factor * pw)
            b = (1.0 - 2.0 * a) * pw
            if m >= 2:
                b = b + 2.0 * (m - 1) * factor * two_a_m1 ** (m - 2)
            r = r + c * b
        return q, r

    @staticmethod
    def _gather(v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return v[torch.arange(v.shape[0], device=v.device), idx]

    def _interval(self, coeffs: torch.Tensor, alpha: torch.Tensor):
        B, P = coeffs.shape[:2]
        pos = alpha.to(torch.float32).clamp(0.0, 1.0) * (P - 1)
        idx = pos.floor().to(torch.long).clamp(0, P - 2)
        frac = (pos - idx).to(coeffs.dtype).view(B, 1, 1, 1)
        return idx, frac

    def _eval_piecewise(self, coeffs: torch.Tensor, alpha: torch.Tensor):
        B, P = coeffs.shape[:2]
        idx, frac = self._interval(coeffs, alpha)
        r_lo = self._gather(coeffs, idx)
        r_hi = self._gather(coeffs, idx + 1)
        r = r_lo + frac * (r_hi - r_lo)

        h = coeffs.new_tensor(1.0 / (P - 1))
        areas = 0.5 * h * (coeffs[:, :-1] + coeffs[:, 1:])          # [B, P-1, ...]
        prefix = torch.cat([torch.zeros_like(areas[:, :1]), areas.cumsum(dim=1)], dim=1)  # [B, P, ...]
        q = self._gather(prefix, idx) + h * (frac * r_lo + 0.5 * frac.square() * (r_hi - r_lo))
        # Exact zero at both transition endpoints (reference does the same).
        endpoint = (alpha <= 0.0) | (alpha >= 1.0)
        if endpoint.any():
            q = torch.where(endpoint.view(B, 1, 1, 1), torch.zeros_like(q), q)
        return q, r

    def windowed_r(self, coeffs: torch.Tensor, alpha0: torch.Tensor, alpha1: torch.Tensor) -> torch.Tensor:
        """Average ``r`` over ``[alpha0, alpha1]`` (reference ``windowed_velocity``)."""
        width = (alpha1 - alpha0).clamp(min=0.0)
        small = width < 1e-6
        width_safe = width.clamp(min=1e-6).to(coeffs.dtype).view(-1, 1, 1, 1)
        if self.kind == "polynomial":
            q1, _ = self._eval_polynomial(coeffs, alpha1)
            q0, _ = self._eval_polynomial(coeffs, alpha0)
        else:
            q1, _ = self._eval_piecewise(coeffs, alpha1)
            q0, _ = self._eval_piecewise(coeffs, alpha0)
        avg = (q1 - q0) / width_safe
        if small.any():
            _, r0 = self.evaluate(coeffs, alpha0)
            avg = torch.where(small.view(-1, 1, 1, 1), r0, avg)
        return avg


def sample_query_alphas(mode: str, iteration: int, segment_index: int, batch: int, count: int, device) -> torch.Tensor:
    """Reference SMD query-alpha sampling, sorted per sample (order is irrelevant to the loss)."""
    if mode == "random":
        return torch.rand(batch, count, device=device, dtype=torch.float32).sort(dim=1).values
    if mode != "random_endpoint":
        raise ValueError(f"Unknown --smd_query_alpha_mode={mode!r}.")
    if count != 1:
        raise ValueError("--smd_query_alpha_mode random_endpoint requires --smd_queries 1.")
    bucket = (int(iteration) * 2246822519 + int(segment_index) * 3266489917) % 1000
    endpoint = 1.0 if bucket >= 500 else 0.0
    return torch.full((batch, 1), endpoint, device=device, dtype=torch.float32)



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


def _pack_patch_tokens(v: torch.Tensor, p: int = 2) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/p)*(W/p), C*p*p] (Anima's spatial patch tokens)."""
    b, c, h, w = v.shape
    if h % p or w % p:
        raise ValueError(f"cfg_norm_rescale needs latent H,W divisible by patch_spatial={p}, got {h}x{w}.")
    return v.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), c * p * p)


def _unpack_patch_tokens(t: torch.Tensor, c: int, h: int, w: int, p: int = 2) -> torch.Tensor:
    """Inverse of :func:`_pack_patch_tokens`."""
    b = t.shape[0]
    return t.reshape(b, h // p, w // p, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


def cfg_combine(out_cond: torch.Tensor, out_uncond: torch.Tensor, cfg_scale: float,
                norm_rescale: bool = False, patch: int = 2) -> torch.Tensor:
    """Teacher CFG in velocity form (paper Eq. 4). 1.0 = conditional, 0.0 = unconditional.

    ``norm_rescale`` reproduces the reference ``cfg_norm_rescale``: the CFG
    combination is rescaled to the conditional prediction's per-token norm
    (``patch x patch`` tokens, matching Anima's patch layout).
    """
    if cfg_scale == 1.0:
        return out_cond
    if cfg_scale == 0.0:
        return out_uncond
    comb = out_uncond + float(cfg_scale) * (out_cond - out_uncond)
    if not norm_rescale:
        return comb
    c, h, w = out_cond.shape[1:]
    cond_p = _pack_patch_tokens(out_cond.float(), patch)
    comb_p = _pack_patch_tokens(comb.float(), patch)
    scale = cond_p.norm(dim=-1, keepdim=True) / comb_p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return _unpack_patch_tokens(comb_p * scale, c, h, w, patch).to(comb.dtype)


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

    Architecture (paper Fig. 36 / Sec. 3.2): features captured from the
    student block at ``hook_pos`` are processed by M Transformer blocks copied
    from the final M student blocks, then by the student's final AdaLN
    norm + modulation (``final_layer`` prefix) and a ZERO-INITIALIZED output
    projection producing P residual coefficient vectors. With the zero-init
    output the network starts exactly at the no-Pivot simple baseline, and no
    extra randomly-initialized input projection is introduced (reference
    ``Pivot`` = copied blocks + copied ``norm_out`` + expanded ``proj_out``).

    Anima's ``FinalLayer`` bundles ``layer_norm -> AdaLN -> linear``; we copy
    it wholesale (pretrained norm + modulation) and replace only ``linear``
    with the expanded, zero-init Pivot head.

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
        # Pre-trained init: exact copies of the natural student tail + final norm/modulation.
        self.blocks = nn.ModuleList([copy.deepcopy(blk) for blk in tail])
        self.final_layer = copy.deepcopy(dit.final_layer)
        base_out = self.final_layer.linear.out_features  # p*p*t*out_channels
        self.final_layer.linear = nn.Linear(
            self.final_layer.linear.in_features, base_out * basis_p, bias=False
        )
        nn.init.zeros_(self.final_layer.linear.weight)

    def trainables(self) -> list:
        """Trainable Pivot parameters (all in full mode; head only in LoRA mode)."""
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
        h = feat_B_T_H_W_D
        for blk in self.blocks:
            h = blk(h, emb_B_T_D, crossattn_emb, attn_params, use_fp32, **block_kwargs)
        # Student final AdaLN norm + modulation, then the expanded Pivot head.
        tokens = self.final_layer(
            h, emb_B_T_D,
            adaln_lora_B_T_3D=block_kwargs.get("adaln_lora_B_T_3D"),
            use_fp32=use_fp32,
        )  # [B, T, H', W', P * p*p*t * C]
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
            "ss_smd_query_sampling": g("smd_query_sampling"),
            "ss_smd_query_alpha_mode": g("smd_query_alpha_mode"),
            "ss_smd_imitation_weight": g("smd_imitation_weight"),
            "ss_smd_repulsion_lambda": g("smd_repulsion_lambda"),
            "ss_smd_repulsion_adaptive": bool(g("smd_repulsion_adaptive", False)),
            "ss_smd_ref_cfg": g("smd_ref_cfg"),
            "ss_smd_cfg_norm_rescale": bool(g("smd_cfg_norm_rescale", False)),
            "ss_smd_teacher_window_alpha": g("smd_teacher_window_alpha"),
            "ss_smd_w_rescale": g("smd_w_rescale"),
            "ss_smd_beta": g("smd_beta"),
            "ss_smd_eps": g("smd_eps"),
            "ss_smd_step_granularity": g("smd_step_granularity"),
            "ss_smd_pivot": bool(use_pivot),
            "ss_smd_pivot_mode": pivot_mode,
            "ss_smd_pivot_hook_layer": g("smd_pivot_hook_layer"),
            "ss_smd_pivot_blocks": g("smd_pivot_blocks"),
            "ss_smd_p": g("smd_p"),
            "ss_smd_basis": g("smd_basis"),
            "ss_smd_basis_scaling": g("smd_basis_scaling"),
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
    basis_p = int(args.smd_p)
    if basis_p < 1:
        raise ValueError(f"--smd_p must be >= 1, got {basis_p}")
    if args.smd_basis == "piecewise" and basis_p < 2:
        raise ValueError("--smd_basis piecewise needs --smd_p >= 2.")
    if args.smd_query_alpha_mode == "random_endpoint" and Q != 1:
        raise ValueError("--smd_query_alpha_mode random_endpoint requires --smd_queries 1.")
    window_alpha = float(args.smd_teacher_window_alpha)
    if window_alpha < 0.0:
        raise ValueError(f"--smd_teacher_window_alpha must be >= 0, got {window_alpha}")
    if window_alpha > 1.0:
        raise ValueError(f"--smd_teacher_window_alpha must be <= 1, got {window_alpha}")
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
            basis_p=basis_p,
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
            f"M={M} copied blocks + copied final norm/modulation, P={basis_p} {args.smd_basis} basis, "
            f"zero-init output projection."
        )
        if args.smd_pivot_lora_rank:
            rank_p = int(args.smd_pivot_lora_rank)
            logger.info(
                f"Pivot blocks train via LoRA rank={rank_p} (bases frozen); "
                "final norm/modulation + output projection stay full-rank."
            )
            pivot_lora_net = _lora_anima_mod.create_network(
                1.0, rank_p, float(rank_p), None, None, pivot, neuron_dropout=args.network_dropout,
                verbose="false",
            )
            pivot_lora_net.apply_to([], pivot, apply_text_encoder=False, apply_unet=True)
            # Freeze the copied block bases; the copied final norm/modulation and
            # the expanded output head stay full-rank (reference lora Pivot).
            pivot.requires_grad_(False)
            for p in pivot.final_layer.parameters():
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
    ts_raw = make_smd_raw_knots(K, args.smd_final_step_size_scale).to(device)  # raw t before the shift
    basis = SMDBasis(args.smd_basis, basis_p, args.smd_basis_scaling)
    logger.info(f"Student schedule (shift={args.smd_shift}, final_step_size_scale={args.smd_final_step_size_scale}): "
                f"{[round(float(v), 4) for v in ts.cpu().tolist()]}")
    logger.info(
        f"Query sampling: {args.smd_query_sampling}, alpha mode={args.smd_query_alpha_mode}, "
        f"teacher_window_alpha={window_alpha:g}, cfg_norm_rescale={bool(args.smd_cfg_norm_rescale)}, "
        f"adaptive_repulsion={bool(args.smd_repulsion_adaptive)}; "
        f"step granularity={args.smd_step_granularity}."
    )
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
    def run_transition(x, cond, uncond, seg_idx: int, iteration: int):
        """One training transition ``t_i -> t_{i+1}`` on the given rollout state.

        Returns ``(seg_loss, imitation, repulsion, x_next)``. The loss is
        backpropagated here (scaled by ``1/accum_steps``) so the autograd graph
        spans a single transition (reference ``per_segment_backward=True``).
        """
        t_start = float(ts[seg_idx].item())
        dt = float(ts[seg_idx + 1].item()) - t_start  # < 0
        if dt >= 0:
            raise RuntimeError(f"Non-decreasing schedule segment at {seg_idx}: dt={dt}")
        raw_start = float(ts_raw[seg_idx].item())
        raw_end = float(ts_raw[seg_idx + 1].item())
        x = x.detach()
        t_vec = torch.full((B,), t_start, dtype=torch.float32, device=device)
        alphas = sample_query_alphas(args.smd_query_alpha_mode, iteration, seg_idx, B, Q, device)

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
                coeffs = basis.build(pivot(
                    cap["feat"].to(x.dtype), cap["args"][1], cap["args"][2], attn_params, use_fp32, bkw
                ))  # [B, P, C, H, W]
        else:
            coeffs = None

        # (c) proxy query states + velocities (both carry grad).
        # Paper Sec. 3.2 (tau there == alpha here):
        # v(alpha) = v_th + r(alpha);
        # x_s = x_t + alpha*dt*v_th + dt*q(alpha) with B_m endpoints fixed.
        v_q_list, x_q_list, s_q_list = [], [], []
        for k in range(Q):
            alpha_k = alphas[:, k]  # sampling variable [B]
            if args.smd_query_sampling == "sigma":
                basis_alpha = alpha_k
                sigma_k = t_start + alpha_k * dt
            else:
                # Reference raw_t: interpolate raw t, warp it, then express the
                # query as a fraction of the warped transition.
                raw_k = raw_start + alpha_k * (raw_end - raw_start)
                sigma_k = warp_t(raw_k, float(args.smd_shift))
                basis_alpha = (sigma_k - t_start) / dt
            if coeffs is not None:
                q_k, r_k = basis.evaluate(coeffs, basis_alpha)  # [B, C, H, W] each
                if window_alpha > 0.0:
                    a1 = (basis_alpha + window_alpha).clamp(max=1.0)
                    r_k = basis.windowed_r(coeffs, basis_alpha, a1)
            else:
                q_k, r_k = torch.zeros(1, device=device), torch.zeros(1, device=device)
            v_q_k = v_theta + r_k
            ta4 = basis_alpha.view(B, 1, 1, 1)
            x_q_k = x.float() + ta4 * dt * v_theta + dt * q_k
            v_q_list.append(v_q_k)
            x_q_list.append(x_q_k)
            s_q_list.append(sigma_k)
        v_q_all = torch.stack(v_q_list)          # [Q, B, C, H, W] grad
        x_all = torch.cat(x_q_list, dim=0)       # [B*Q, C, H, W] grad
        s_all = torch.cat(s_q_list, dim=0)       # [B*Q] no grad

        # (d) frozen teacher queries at the SAME states (targets; teacher
        # outputs are detached per the paper). Separate cond/uncond branches
        # are combined with the CFG scales. The student adapter is zeroed so
        # the teacher is the plain base weights.
        set_role(student_net, "teacher")
        with torch.autocast(**autocast_kw), torch.no_grad():
            v_real_all, v_unc_all = None, None
            if need_cond:
                v_real_all = dit_predict_v(dit, x_all.detach(), s_all, cond, weight_dtype).float()
            if need_uncond:
                v_unc_all = dit_predict_v(dit, x_all.detach(), s_all, uncond, weight_dtype).float()
        norm_rescale = bool(args.smd_cfg_norm_rescale)
        patch = int(dit.patch_spatial)
        v_target_all = cfg_combine(v_real_all, v_unc_all, teacher_cfg, norm_rescale, patch).view(Q, B, *v_theta.shape[1:])
        v_ref_all = cfg_combine(v_real_all, v_unc_all, ref_cfg, norm_rescale, patch).view(Q, B, *v_theta.shape[1:])

        # (e) imitation + repulsion over the query points
        imitation, repulsion = None, None
        for k in range(Q):
            if w_imitation > 0:
                d_t = per_sample_velocity_distance(v_q_all[k], v_target_all[k], beta, eps).mean()
                imitation = d_t if imitation is None else imitation + d_t
            if lam > 0:
                d_r = per_sample_velocity_distance(v_q_all[k], v_ref_all[k], beta, eps)
                if args.smd_repulsion_adaptive:
                    # Reference gate: clamp((d(v,ev) - d(v,vT)) / (d(vT,ev)+eps), 0, 1).
                    d_pos = per_sample_velocity_distance(v_q_all[k], v_target_all[k], 1.0, 0.0)
                    d_neg = per_sample_velocity_distance(v_q_all[k], v_ref_all[k], 1.0, 0.0)
                    d_gap = per_sample_velocity_distance(v_target_all[k], v_ref_all[k], 1.0, 0.0)
                    gate = ((d_neg - d_pos) / (d_gap + eps)).clamp(0.0, 1.0).detach()
                    d_r = d_r * gate
                d_r = d_r.mean()
                repulsion = d_r if repulsion is None else repulsion + d_r
        imitation = imitation / Q if imitation is not None else torch.zeros((), device=device)
        repulsion = repulsion / Q if repulsion is not None else torch.zeros((), device=device)
        seg_loss = normalization * (w_imitation * imitation - lam * repulsion)

        # (f) per-transition backward.
        (seg_loss / accum_steps).backward()

        # (g) next rollout state follows the COARSE student update (no pivot,
        # no grad) -- paper Algorithm 2 line 17 / Algorithm 1 line 13.
        x_next = (x.float() + dt * v_theta.detach()).to(weight_dtype).detach()
        return seg_loss, imitation, repulsion, x_next

    step_granularity = args.smd_step_granularity
    if step_granularity == "segment":
        logger.info(
            f"Optimizer granularity: SEGMENT (one step per transition). A full rollout spans K={K} "
            f"optimizer steps and --max_train_steps={max_steps} counts segments."
        )
    else:
        logger.info(f"Optimizer granularity: ROLLOUT (one step per K={K}-transition rollout).")
    cycle_seg = 0
    slot_states = None
    pbar = tqdm(range(global_step, max_steps), initial=global_step, total=max_steps, desc="SMD")
    for opt_step in pbar:
        # Accumulation window: zero once, run `accum_steps` micro-batches,
        # step once. Losses are scaled by 1/accum so the accumulated grad
        # equals the mean over the effective (B*accum) batch.
        zero_all(student_net, *( [pivot_lora_net, pivot] if use_pivot else [] ))
        loss_sum, loss_imit_sum, loss_rep_sum = 0.0, 0.0, 0.0
        if step_granularity == "rollout":
            # Paper Algorithm 2: one optimizer step per full K-transition rollout.
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
                for i in range(K):
                    seg_loss, imitation, repulsion, x = run_transition(x, cond, uncond, i, opt_step)
                    loss_sum += seg_loss.item()
                    loss_imit_sum += float(imitation.item())
                    loss_rep_sum += float(repulsion.item())
        else:
            # Reference execution: one optimizer step per transition. Each
            # accumulation slot keeps its own batch + rollout state across the
            # K transitions, then starts a fresh rollout at cycle_seg == 0.
            if slot_states is None:
                slot_states = [None] * accum_steps
            if cycle_seg == 0:
                for _micro in range(accum_steps):
                    batch_caps = random.choices(prompts, k=B)
                    if neg_prompts is not None:
                        neg_caps = random.choices(neg_prompts, k=B)
                    else:
                        neg_caps = [neg_single] * B
                    with torch.no_grad():
                        with torch.autocast(**autocast_kw):
                            cond = encode_cond(tokenize_strategy, text_encoding_strategy, qwen3, batch_caps, weight_dtype)
                            use_blank_anchor = all(not s.strip() for s in neg_caps)
                            uncond = encode_cond(tokenize_strategy, text_encoding_strategy, qwen3, neg_caps, weight_dtype, uncond=use_blank_anchor) if need_uncond else None
                    slot_states[_micro] = {
                        "cond": cond,
                        "uncond": uncond,
                        "x": torch.randn(B, latent_ch, lat_h, lat_w, device=device, dtype=weight_dtype),
                    }
            for _micro in range(accum_steps):
                st = slot_states[_micro]
                seg_loss, imitation, repulsion, st["x"] = run_transition(
                    st["x"], st["cond"], st["uncond"], cycle_seg, opt_step
                )
                loss_sum += seg_loss.item()
                loss_imit_sum += float(imitation.item())
                loss_rep_sum += float(repulsion.item())
        if step_granularity == "segment":
            cycle_seg = (cycle_seg + 1) % K

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
