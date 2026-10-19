# DMAD (Distribution Matching as Adversarial Distillation) core for the Anima
# (rectified-flow DiT), following the MOSS implementation:
# ``RamTorchRun/moss/train_dmad.py`` (arXiv 2610.02188).
#
# ONE frozen DiT backbone carries the three LoRA roles of the moss trainer, via
# ``inject_dmad_lora`` on the Anima block projections:
#
#     None     teacher   : frozen base (+ the frozen, always-on ``--teacher_lora_path``
#                factors, applied unmerged in fp32 like ``attach_frozen_lora`` in
#                moss/model/lora.py); ``teacher_steps`` Euler on the
#                shift-dmad grid, no CFG (guidance is baked into the teacher)
#     default  student   : the same backbone plus a new LoRA (zero-init B, so the
#                student starts AT the teacher); "re-noise" sampler: predict
#                x0 = x - t v, renew the noise at the next t; k no-grad steps then
#                one with grad, k ~ U{0..steps-1}
#     disc     discriminator: the same backbone plus a disc LoRA; the final block's
#                image tokens go through a non-affine RMSNorm into two heads
#                (teacher, real), each Linear-SiLU-Linear to a per-token logit,
#                token mean.
#
# Per update (moss ``generator_step`` / ``disc_step`` / ``update_gap``):
#   generator: x0_G from the re-noise rollout; noise to t = shift(u), u ~
#     U(disc_u_range); ``L_G = -mean(lambda_T w(u) h_T + lambda_R h_R)`` with the
#     discriminator frozen (gradients flow through its input).
#   discriminator: balanced BCE per head on noisy (target, x0_G.detach()) pairs;
#     the teacher head scores teacher samples Q, the real head scores the
#     VAE-encoded real images; one concatenated forward for [Q, real, gen].
#   gap reweighting: per u band an EMA (fp64) of mean h_R(real) - mean h_R(teacher);
#     ``w = sigmoid((median - gap) / tau)``, normalized over the measured bands.
#
# Anima-specific notes: the DiT timesteps are the raw sigma in [0, 1]
# (``x_t = (1 - s) x0 + s eps`` with ``v = eps - x0``), latents are 4D
# ``(B, C, H, W)`` and get ``unsqueeze(2)`` inside every DiT call, and the
# conditioning tuple is ``(prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask)``.

import math
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

DISC_ROLE = "disc"
HEADS = ("teacher", "real")
LORA_PREFIX = "lora_unet"  # ComfyUI compatible, same as networks.lora_anima

# LoRA target: Anima block projections (self/cross attention qkv+out and the two
# MLP layers). Modulations, norms, embedders and the final layer stay frozen,
# matching the moss target ("layers.<i>.proj.<q|k|v|o|ffn_up_gate|ffn_down>")
# and networks.lora_anima's default excludes.
LORA_TARGET = r"blocks\.\d+\.(?:self_attn|cross_attn|mlp)\.(?:q_proj|k_proj|v_proj|output_proj|layer1|layer2)"


# ---------------------------------------------------------------------------
# schedule / flow helpers (moss_sigmas, shift_time, re-noise rollout)
# ---------------------------------------------------------------------------


def dmad_sigmas(steps: int, shift: float, training_shift: float = 1.0, total: int = 1000) -> torch.Tensor:
    """Descending sigmas (``steps`` or fewer after dedup) snapped to the training grid, plus a terminal 0.

    Port of ``moss_sigmas`` in moss/model/sampling.py: the shift grid
    ``s * u / (1 + (s - 1) * u)`` over ``steps`` points, each snapped to the
    (uniform, shift-``training_shift``) training grid of ``total`` timesteps.
    """
    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def shift_time(u: torch.Tensor, shift: float) -> torch.Tensor:
    """Schedule shift (``MOSSImageScheduler``) applied to a base time u in (0, 1)."""
    return shift * u / (1 + (shift - 1) * u)


def renoise_rollout(x0_fn, noise, schedule, k, noise_fn):
    """k no-grad re-noise steps from ``noise``; returns (x_k, t_k).

    ``x0_fn(x, t)`` predicts x0; step i: x0 = x0_fn(x, s_i), x = (1 - s_{i+1}) x0 + s_{i+1} eps with
    eps = noise_fn(shape). The caller runs step k (x0_fn(x_k, t_k)) with grad.
    """
    x = noise
    with torch.no_grad():
        for i in range(k):
            x0 = x0_fn(x, schedule[i])
            x = (1 - schedule[i + 1]) * x0 + schedule[i + 1] * noise_fn(x0.shape)
    return x, schedule[k]


def flow_add_noise(x0, eps, sigma):
    """Forward flow noising: ``x_t = (1 - s) * x0 + s * eps``.

    Args:
        x0: clean latents ``(B, C, H, W)`` or ``(B, C, 1, H, W)``.
        eps: noise of the same shape.
        sigma: per-sample noise level, shape ``(B,)`` or broadcastable.
    """
    while sigma.ndim < x0.ndim:
        sigma = sigma[(...,) + (None,) * (x0.ndim - sigma.ndim)]
    return (1.0 - sigma) * x0 + sigma * eps


def velocity_to_x0(noisy, pred_velocity, sigma):
    """Recover the clean estimate from a velocity prediction.

    With ``x_t = x0 + s * v`` where ``v = eps - x0``, ``x0 = x_t - s * v``.
    """
    while sigma.ndim < noisy.ndim:
        sigma = sigma[(...,) + (None,) * (noisy.ndim - sigma.ndim)]
    return noisy - sigma * pred_velocity


# ---------------------------------------------------------------------------
# generic training utilities (port of moss/train_utils.py pieces used here)
# ---------------------------------------------------------------------------


def finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {label}")


def allreduce_mean(tensors, bucket_bytes=256 * 2**20):
    """Average same-shaped tensors across ranks in flat buckets."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return tensors
    bucket, size = [], 0
    for tensor in tensors + [None]:
        if tensor is not None:
            bucket.append(tensor)
            size += tensor.numel() * tensor.element_size()
        if bucket and (tensor is None or size >= bucket_bytes):
            flat = torch.cat([t.reshape(-1) for t in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            offset = 0
            for t in bucket:
                t.copy_(flat[offset : offset + t.numel()].view_as(t))
                offset += t.numel()
            bucket, size = [], 0
    return tensors


def clip_step(params, opt, max_norm, label):
    """Grad all-reduce + clip + optimizer step (moss train_utils.clip_step)."""
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    allreduce_mean([p.grad for p in params])
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    finite(norm, f"{label} grad norm")
    opt.step()
    return norm


def broadcast_params(params, src=0):
    """Broadcast trainable parameters from ``src`` so all ranks start identical."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    for p in params:
        dist.broadcast(p, src=src)


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    """Karras et al. 2024 power-function EMA exponent for a relative std."""
    t = sigma_rel**-2
    return float(np.roots([1, 7, 16 - t, 12 - t]).real.max())


class PowerEMA:
    """EMA with beta_n = (1 - 1/n)^(gamma + 1); n counts updates from 1."""

    def __init__(self, params: dict, sigma_rel: float):
        self.gamma = sigma_rel_to_gamma(sigma_rel)
        self.shadow = {n: p.detach().float().clone() for n, p in params.items()}
        self.count = 0

    @torch.no_grad()
    def update(self, params: dict):
        self.count += 1
        beta = (1 - 1 / self.count) ** (self.gamma + 1)
        for n, p in params.items():
            self.shadow[n].lerp_(p.detach().float(), 1 - beta)

    def state_dict(self):
        return {"count": self.count, "gamma": self.gamma, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.count = state["count"]
        for n, value in state["shadow"].items():
            self.shadow[n].copy_(value)


# ---------------------------------------------------------------------------
# losses / heads / gap reweighting
# ---------------------------------------------------------------------------


def bce_balanced(target_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Balanced BCE: target and student classes weighted equally (optimum = log ratio)."""
    return F.softplus(-target_logits).mean() + F.softplus(student_logits).mean()


class DiscHeads(nn.Module):
    """Non-affine RMSNorm, then one Linear-SiLU-Linear head per target; mean over tokens.

    Port of ``DiscHeads`` in moss/train_dmad.py. ``tokens``: [B, N, D] final-block
    image tokens (Anima has no text tokens in the sequence, so all tokens are image).
    """

    def __init__(self, features: int, hidden: int, names=HEADS):
        super().__init__()
        self.heads = nn.ModuleDict({n: nn.Sequential(nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, 1)) for n in names})

    def forward(self, tokens: torch.Tensor) -> dict:
        x = tokens.float() * torch.rsqrt(tokens.float().square().mean(-1, keepdim=True) + 1e-6)
        x = x.to(tokens.dtype)
        return {n: head(x).squeeze(-1).float().mean(1) for n, head in self.heads.items()}


class GapReweighter:
    """DMAD gap-based teacher reweighting over equal-width bands of the base time u."""

    def __init__(self, bands=10, beta=0.99, tau=2.0, floor=1e-4):
        self.bands, self.beta, self.tau, self.floor = bands, beta, tau, floor
        self.gap = torch.full((bands,), float("nan"), dtype=torch.float64)

    def band(self, u: torch.Tensor) -> torch.Tensor:
        return (u.double() * self.bands).long().clamp(0, self.bands - 1)

    def update(self, u_real, real_logits, u_teacher, teacher_logits):
        """EMA of mean h_R(real) - mean h_R(teacher), only for bands seen in both."""
        br, bt = self.band(u_real.reshape(-1).cpu()), self.band(u_teacher.reshape(-1).cpu())
        real_logits = real_logits.detach().double().reshape(-1).cpu()
        teacher_logits = teacher_logits.detach().double().reshape(-1).cpu()
        for b in range(self.bands):
            r, t = real_logits[br == b], teacher_logits[bt == b]
            if len(r) and len(t):
                value = r.mean() - t.mean()
                old = self.gap[b]
                self.gap[b] = value if torch.isnan(old) else self.beta * old + (1 - self.beta) * value

    def band_weights(self) -> torch.Tensor:
        measured = ~torch.isnan(self.gap)
        if not measured.any():
            return torch.ones(self.bands, dtype=torch.float64)
        median = torch.quantile(self.gap[measured], 0.5)
        raw = torch.sigmoid((median - self.gap) / self.tau)
        raw[~measured] = 0.5
        return raw / raw[measured].mean().clamp_min(self.floor)

    def weights(self, u: torch.Tensor) -> torch.Tensor:
        """Per-sample teacher weight (no gradient)."""
        return self.band_weights()[self.band(u.cpu())].to(device=u.device, dtype=torch.float32)

    def state_dict(self):
        return {"gap": self.gap.clone()}

    def load_state_dict(self, state):
        self.gap = state["gap"].clone().double()


# ---------------------------------------------------------------------------
# role LoRA (port of moss/model/lora.py to Anima's Linear projections)
# ---------------------------------------------------------------------------


class DmadLoRALinear(nn.Module):
    """One frozen base Linear with a primary role adapter ("default", the student)
    plus ``extra_roles`` (e.g. "disc"); ``set_lora_role`` picks which applies,
    ``None`` = frozen base. ``attach_frozen_lora`` adds a frozen, always-on
    adapter that every role (``None`` included) sits on top of: an unmerged
    "merge" that weight casts cannot erase."""

    def __init__(self, base: nn.Module, rank: int, alpha: float, extra_roles: tuple = ()):
        super().__init__()
        if base.__class__.__name__ != "Linear":
            raise TypeError(f"DmadLoRALinear only supports nn.Linear, got {base.__class__.__name__} ({base})")
        self.in_features, self.out_features = base.in_features, base.out_features
        # registered buffers ride .to(device) and keep the base frozen
        self.register_buffer("base_weight", base.weight.detach(), persistent=False)
        self.register_buffer("base_bias", None if base.bias is None else base.bias.detach(), persistent=False)
        self.rank, self.scale = int(rank), float(alpha) / float(rank)
        self.extra_roles = tuple(extra_roles)
        self.active_role = "default"
        for role in ("default", *self.extra_roles):
            suffix = "" if role == "default" else f"_{role}"
            A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32, device=base.weight.device))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            setattr(self, f"lora_A{suffix}", A)
            setattr(
                self,
                f"lora_B{suffix}",
                nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32, device=base.weight.device)),
            )
        self.register_buffer("frozen_A", None, persistent=False)
        self.register_buffer("frozen_B", None, persistent=False)
        self.frozen_scale = 0.0

    @property
    def weight(self):
        return self.base_weight

    def adapter(self, role):
        if role is None:
            return None
        suffix = "" if role == "default" else f"_{role}"
        return getattr(self, f"lora_A{suffix}"), getattr(self, f"lora_B{suffix}")

    def forward(self, x):
        y = F.linear(x, self.base_weight, self.base_bias)
        if self.frozen_A is not None:
            # fp32 factors stay unmerged; cast at compute time (autocast does the
            # same thing when it is active)
            y = y + F.linear(F.linear(x, self.frozen_A.to(x.dtype)), self.frozen_B.to(x.dtype)) * self.frozen_scale
        ab = self.adapter(self.active_role)
        if ab is not None:
            A, B = ab
            y = y + F.linear(F.linear(x, A.to(x.dtype)), B.to(x.dtype)) * self.scale
        return y

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, roles={('default', *self.extra_roles)}"


def inject_dmad_lora(model: nn.Module, rank: int, alpha: float, target: str, extra_roles: tuple = ()) -> dict:
    """Replace every nn.Linear whose dotted name fully matches the regex ``target``."""
    import re

    pattern = re.compile(target)
    replaced = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and pattern.fullmatch(name):
            parent, _, leaf = name.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            layer = DmadLoRALinear(module, rank, alpha, extra_roles)
            if isinstance(owner, nn.ModuleDict):
                owner[leaf] = layer
            else:
                setattr(owner, leaf, layer)
            replaced[name] = layer
    if not replaced:
        raise ValueError(f"no nn.Linear matches {target!r}")
    return replaced


def set_lora_role(model: nn.Module, role):
    n = 0
    for m in model.modules():
        if isinstance(m, DmadLoRALinear):
            if role not in (None, "default") and role not in m.extra_roles:
                raise ValueError(f"role {role!r} was not injected")
            m.active_role = role
            n += 1
    if not n:
        raise ValueError("no DmadLoRALinear modules found")


def role_params(model: nn.Module, role: str = "default") -> dict:
    suffix = "" if role == "default" else f"_{role}"
    return {n: p for n, p in model.named_parameters() if n.endswith((f".lora_A{suffix}", f".lora_B{suffix}"))}


def kohya_lora_name(original_name: str) -> str:
    """Dotted module name -> kohya LoRA key stem (networks.lora_anima convention)."""
    return f"{LORA_PREFIX}_{original_name.replace('.', '_')}"


def frozen_factors_anima(model: nn.Module, state: dict, alpha=None) -> dict:
    """``{module: (A [r, in], B [out, r], alpha / r)}`` from a kohya anima LoRA
    state dict (``lora_unet_<module>.lora_down.weight`` / ``lora_up.weight`` /
    ``alpha``) or a generator export (``<module>.lora_A`` / ``<module>.lora_B``,
    needs ``alpha``). Only factors whose module is wrapped in ``model`` are kept."""
    wrapped = {}
    for name, m in model.named_modules():
        if isinstance(m, DmadLoRALinear):
            wrapped[kohya_lora_name(name)] = name
    out = {}
    for key in state:
        if key.endswith(".lora_down.weight"):
            stem = key[: -len(".lora_down.weight")]
            module = wrapped.get(stem)
            if module is None:
                continue
            A, B = state[key], state[f"{stem}.lora_up.weight"]
            a = float(state[f"{stem}.alpha"]) if f"{stem}.alpha" in state else alpha
            out[module] = (A, B, a / A.shape[0])
        elif key.endswith(".lora_A"):
            stem = key[: -len(".lora_A")]
            module = wrapped.get(stem)
            if module is None:
                continue
            if alpha is None:
                raise ValueError("an export-style checkpoint carries no alpha; pass alpha")
            out[module] = (state[key], state[f"{stem}.lora_B"], alpha / state[key].shape[0])
    if not out:
        raise ValueError("no LoRA factors matched the wrapped modules")
    return out


@torch.no_grad()
def attach_frozen_lora(model: nn.Module, state: dict, strength: float, alpha=None) -> int:
    """Frozen always-on adapter ``strength * scale * B @ A`` on the matching
    DmadLoRALinear modules (fp32 factors), as in moss. Returns the count."""
    factors = frozen_factors_anima(model, state, alpha)
    for name, (A, B, scale) in factors.items():
        m = model.get_submodule(name)
        m.frozen_A = A.detach().to(m.base_weight.device, torch.float32).clone()
        m.frozen_B = B.detach().to(m.base_weight.device, torch.float32).clone()
        m.frozen_scale = strength * scale
    return len(factors)


def kohya_state(factors: dict, rank_meta: str, prefix: str = LORA_PREFIX + "_") -> tuple:
    """{module: (A, B, alpha)} -> kohya tensors (fp32) and safetensors metadata."""
    out = {}
    for module, (A, B, alpha) in sorted(factors.items()):
        key = prefix + module.replace(".", "_")
        out[f"{key}.lora_down.weight"] = A.float().contiguous()
        out[f"{key}.lora_up.weight"] = B.float().contiguous()
        out[f"{key}.alpha"] = torch.tensor(float(alpha))
    return out, {"rank": rank_meta}


def combined_factors(teacher: dict, strength: float, student: dict) -> dict:
    """Exact single LoRA for ``strength * teacher + student``: concatenated factors,
    scales folded into up, alpha = rank (moss train_dmad.combined_factors)."""
    out = {}
    for module, (A_s, B_s, scale_s) in student.items():
        A_t, B_t, scale_t = teacher[module]
        A = torch.cat([A_t.float(), A_s.float()])
        B = torch.cat([B_t.float() * (strength * scale_t), B_s.float() * scale_s], 1)
        out[module] = (A, B, float(A.shape[0]))
    return out


# ---------------------------------------------------------------------------
# role-aware activation checkpointing
# ---------------------------------------------------------------------------


def enable_dmad_block_checkpointing(dit, cpu_offload: bool = False):
    """Non-reentrant gradient checkpointing that re-applies the LoRA role captured
    at forward time (moss ``role_checkpoint`` / ``block_call``).

    Replaces each Anima block's forward with a fully checkpointed role-aware
    version: the block's active role is captured when the block forward runs and
    re-applied when the checkpoint recomputes during backward — this is what lets
    the student forward and the discriminator forward share one checkpointed
    backbone without leaking roles across the two backward passes. With
    ``cpu_offload`` the activation stream between blocks is offloaded to CPU RAM
    (blocking transfers), mirroring anima's cpu_offload_checkpointing.
    """
    from torch.utils.checkpoint import checkpoint as ckpt

    from library import anima_models as _models

    for block in dit.blocks:
        block.gradient_checkpointing = False  # the internal per-block checkpointing is replaced here
        block.cpu_offload_checkpointing = False
        block.unsloth_offload_checkpointing = False
        orig_forward = block._forward

        def block_role(block):
            for m in block.modules():
                if isinstance(m, DmadLoRALinear):
                    return m.active_role
            raise ValueError("block has no DmadLoRALinear")

        def make_forward(block, orig):
            block_device = next(block.parameters()).device

            def wrapped_forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params, use_fp32=False,
                                rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None, extra_per_block_pos_emb=None):
                role = block_role(block)

                def run(*inputs):
                    if cpu_offload:
                        inputs = _models.to_device(inputs, block_device)
                        set_lora_role(block, role)
                        outputs = orig(*inputs)
                        return _models.to_cpu(outputs)
                    set_lora_role(block, role)
                    return orig(*inputs)

                args_in = (x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params, use_fp32,
                           rope_emb_L_1_1_D, adaln_lora_B_T_3D, extra_per_block_pos_emb)
                if cpu_offload:
                    args_in = _models.to_cpu(args_in)
                return ckpt(run, *args_in, use_reentrant=False)

            return wrapped_forward

        # plain instance-attribute function (NOT types.MethodType): a bound
        # method would prepend the block itself as x_B_T_H_W_D. Instance
        # attributes are not descriptors, so no implicit self is prepended.
        block.forward = make_forward(block, orig_forward)


# ---------------------------------------------------------------------------
# the trainer core (port of MossDMAD in moss/train_dmad.py)
# ---------------------------------------------------------------------------


class AnimaDmad:
    """DMAD few-step distillation of an Anima teacher, role LoRA on ONE frozen DiT.

    Mirrors ``MossDMAD``: online teacher sampling, re-noise student sampler,
    role-LoRA discriminator with teacher/real heads, gap reweighting, power EMA
    and the kohya LoRA exports. The Anima specifics: sigma timesteps in [0, 1],
    4D latents turned 5D inside ``velocity``/``disc_logits``, and the cond tuple
    ``(prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask)``.
    """

    label = "DMAD"

    def __init__(self, args, dit, device, rank=0, world=1, autocast_ctx=None):
        from safetensors.torch import load_file

        self.args, self.dit, self.device, self.rank, self.world = args, dit, device, rank, world
        self.autocast_ctx = autocast_ctx or (lambda: torch.autocast("cuda", enabled=False))

        # role LoRA on the block projections; the student role is primary
        inject_dmad_lora(dit, args.dmad_lora_rank, args.dmad_lora_alpha, LORA_TARGET, extra_roles=(DISC_ROLE,))

        # frozen always-on teacher LoRA (fp32 factors on top of base for every role)
        self.teacher_factors = None
        if args.teacher_lora_path:
            teacher_state = load_file(args.teacher_lora_path)
            self.teacher_factors = frozen_factors_anima(dit, teacher_state, args.teacher_lora_alpha)
            n = attach_frozen_lora(dit, teacher_state, args.teacher_lora_strength, args.teacher_lora_alpha)
            del teacher_state
            import logging

            logging.getLogger(__name__).info(f"attached frozen teacher LoRA: {n} modules, strength {args.teacher_lora_strength}")

        self.dim = int(dit.model_channels)
        self.heads = DiscHeads(self.dim, args.dmad_head_hidden).to(device)

        self.gen_named = dict(role_params(dit, "default"))
        self.disc_named = dict(role_params(dit, DISC_ROLE))
        self.disc_named.update({f"heads.{n}": p for n, p in self.heads.named_parameters()})
        self.fake_named = self.disc_named
        for p in list(self.gen_named.values()) + list(self.disc_named.values()):
            p.requires_grad_(True)

        self.ema = PowerEMA(self.gen_named, args.dmad_ema_sigma_rel)
        self.gap = GapReweighter(args.gap_bands, args.gap_beta, args.gap_tau, args.gap_floor)

        self.teacher_sigmas = dmad_sigmas(args.teacher_steps, args.dmad_flow_shift).tolist()
        self.student_sigmas = dmad_sigmas(args.student_steps, args.dmad_flow_shift).tolist()
        self.steps = args.student_steps
        if len(self.student_sigmas) != self.steps + 1 or len(self.teacher_sigmas) != args.teacher_steps + 1:
            raise ValueError("a step grid collapses after snapping to the training timesteps")

        lo, hi = args.disc_u_range
        assert 0 < lo < hi < 1, "disc_u_range must satisfy 0 < lo < hi < 1"

    # -- networks --------------------------------------------------------------------------------

    def autocast(self):
        return self.autocast_ctx()

    def _run_dit(self, role, x4d, t, cond, features=False):
        """One DiT forward under ``role``; 4D latents in, 4D velocity or tokens out."""
        set_lora_role(self.dit, role)
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = cond
        x5 = x4d.unsqueeze(2)  # (B, C, 1, H, W)
        b, _, _, h, w = x5.shape
        padding_mask = torch.zeros(b, 1, h, w, dtype=x4d.dtype, device=self.device)
        t = torch.as_tensor(t, device=self.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.reshape(1).expand(b)
        with self.autocast():
            out = self.dit(
                x5,
                t,
                prompt_embeds,
                padding_mask=padding_mask,
                source_attention_mask=attn_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                features=features,
            )
        if features:
            # out: [B, T, H, W, D] final-block tokens (all image tokens on Anima)
            return out.flatten(1, 3)
        return out.squeeze(2).float()

    def velocity(self, role, x4d, t, cond):
        return self._run_dit(role, x4d, t, cond, features=False)

    def disc_logits(self, x4d, t, cond):
        """Per-head token-mean logits from the disc-role backbone + heads."""
        tokens = self._run_dit(DISC_ROLE, x4d, t, cond, features=True)
        with self.autocast():
            return self.heads(tokens)

    @torch.no_grad()
    def teacher_sample(self, cond, noise, neg_cond=None):
        """Frozen teacher: ``teacher_steps`` Euler on the shift grid, no CFG (moss
        semantics; guidance is normally baked into the teacher / its LoRA).
        ``teacher_cfg > 1`` opts back into DMAD's guided Q with ``neg_cond``."""
        if float(self.args.teacher_cfg) > 1.0 and neg_cond is not None:
            with self.autocast():
                return euler_sample_x0(
                    self.dit, noise, *cond,
                    sigmas=self._teacher_sigmas(),
                    guidance_scale=float(self.args.teacher_cfg), neg_cond=neg_cond,
                )
        x = noise.float()
        for current, following in zip(self.teacher_sigmas[:-1], self.teacher_sigmas[1:]):
            x = x + (following - current) * self.velocity(None, x, current, cond)
        return x

    def _teacher_sigmas(self):
        return torch.tensor(self.teacher_sigmas, dtype=torch.float32, device=self.device)

    def student_x0(self, cond, noise, k):
        """x0 after k no-grad re-noise steps and one more step (with grad)."""
        x0_fn = lambda x, t: x - t * self.velocity("default", x, t, cond)
        noise_fn = lambda shape: torch.randn(shape, device=noise.device, dtype=torch.float32)
        x, t = renoise_rollout(x0_fn, noise, self.student_sigmas, k, noise_fn)
        return x0_fn(x, t)

    @torch.no_grad()
    def student_estimates(self, cond, noise):
        """The x0 estimate after each of the ``steps`` re-noise steps: [1-NFE, ..., steps-NFE]."""
        x, schedule, out = noise, self.student_sigmas, []
        for i in range(self.steps):
            x0 = x - schedule[i] * self.velocity("default", x, schedule[i], cond)
            out.append(x0)
            if i + 1 < self.steps:
                x = (1 - schedule[i + 1]) * x0 + schedule[i + 1] * torch.randn(
                    x0.shape, device=x0.device, dtype=torch.float32
                )
        return out

    def sample_u(self, n):
        lo, hi = self.args.disc_u_range
        return lo + (hi - lo) * torch.rand(n, device=self.device, dtype=torch.float32)

    def noisy(self, x0, t):
        tt = t.reshape(-1, 1, 1, 1).to(device=self.device, dtype=torch.float32)
        return (1 - tt) * x0 + tt * torch.randn_like(x0)

    def set_trainable(self, gen: bool, disc: bool):
        for p in self.gen_named.values():
            p.requires_grad_(gen)
        for p in self.disc_named.values():
            p.requires_grad_(disc)

    def set_model_mode(self, train: bool):
        self.dit.train(train)

    # -- one update --------------------------------------------------------------------------------
    def generator_step(self, cond, teacher_cond, noise, x_real, x_teacher, accum, train=True):
        """``train=False`` (disc warmup) only draws the student sample."""
        cfg = self.args
        k = random.randrange(self.steps)
        item = {"cond": cond, "teacher_cond": teacher_cond, "real": x_real, "teacher": x_teacher}
        if not train:
            with torch.no_grad():
                item["gen"] = self.student_x0(cond, noise, k)
            return item, {"k": k}
        self.set_trainable(gen=True, disc=False)
        x_gen = self.student_x0(cond, noise, k)
        u = self.sample_u(len(x_gen))
        t = shift_time(u, cfg.dmad_flow_shift)
        logits = self.disc_logits(self.noisy(x_gen, t), t, cond)
        weight = self.gap.weights(u)
        loss = -(cfg.lambda_teacher * weight * logits["teacher"] + cfg.lambda_real * logits["real"]).mean()
        finite(loss, "generator loss")
        (loss / accum).backward()
        item["gen"] = x_gen.detach()
        return item, {
            "loss_g": loss.item(),
            "k": k,
            "teacher_weight": weight.mean().item(),
            "g_h_teacher": logits["teacher"].mean().item(),
            "g_h_real": logits["real"].mean().item(),
            "x0_std": x_gen.detach().std().item(),
        }

    def disc_step(self, item, accum):
        cfg = self.args
        self.set_trainable(gen=False, disc=True)
        b = len(item["gen"])
        u = self.sample_u(b)
        t = shift_time(u, cfg.dmad_flow_shift)
        x = torch.cat([self.noisy(item[s], t) for s in ("teacher", "real", "gen")])
        # chunks: teacher (its own cond), real + gen (the batch's cond; in moss
        # all three share the same contexts since Q is sampled online there)
        cond3 = concat_cond([item["teacher_cond"], item["cond"], item["cond"]])
        logits = self.disc_logits(x, torch.cat([t, t, t]), cond3)
        h_t = dict(zip(("teacher", "real", "gen"), logits["teacher"].split(b)))
        h_r = dict(zip(("teacher", "real", "gen"), logits["real"].split(b)))
        loss_t, loss_r = bce_balanced(h_t["teacher"], h_t["gen"]), bce_balanced(h_r["real"], h_r["gen"])
        loss = loss_t + loss_r
        finite(loss, "discriminator loss")
        (loss / accum).backward()
        stats = {
            "loss_d": loss.item(),
            "loss_d_teacher": loss_t.item(),
            "loss_d_real": loss_r.item(),
            **{f"hT_{s}": h_t[s].mean().item() for s in h_t},
            **{f"hR_{s}": h_r[s].mean().item() for s in h_r},
        }
        return stats, torch.stack([u.detach(), h_r["real"].detach(), h_r["teacher"].detach()])

    def update_gap(self, local: torch.Tensor):
        """``local`` [3, n] (u, h_R real, h_R teacher) from this rank; every rank applies the gathered set."""
        if dist.is_available() and dist.is_initialized() and self.world > 1:
            parts = [torch.empty_like(local) for _ in range(self.world)]
            dist.all_gather(parts, local.contiguous())
            local = torch.cat(parts, 1)
        u, real, teacher = local
        self.gap.update(u, real, u, teacher)

    # -- EMA / exports / state --------------------------------------------------------------------

    @torch.no_grad()
    def swap_ema(self):
        for n, p in self.gen_named.items():
            backup = p.detach().clone()
            p.copy_(self.ema.shadow[n].to(p.dtype))
            self.ema.shadow[n].copy_(backup.float())

    def student_factors(self, values, role="default"):
        """{name.lora_A/B: tensor} (e.g. the EMA shadow) -> {module: (A, B, scale)}."""
        scale = self.args.dmad_lora_alpha / self.args.dmad_lora_rank
        suffix = "" if role == "default" else f"_{role}"
        return {
            n[: -len(f".lora_A{suffix}")]: (
                values[n].detach().float().cpu(),
                values[n[: -len(f".lora_A{suffix}")] + f".lora_B{suffix}"].detach().float().cpu(),
                scale,
            )
            for n in values
            if n.endswith(f".lora_A{suffix}")
        }

    def exports(self, ema_shadow):
        """{filename: tensors} of the kohya LoRA exports (moss naming), if any."""
        cfg = self.args
        rows = {}
        for stem, factors in (
            ("student_lora.safetensors", self.student_factors(role_factors_live(self.dit, "default"))),
            ("student_ema.safetensors", self.student_factors(ema_shadow, role="default")),
        ):
            rows[stem], _ = kohya_state(
                {k: (A, B, cfg.dmad_lora_alpha) for k, (A, B, _) in factors.items()}, str(cfg.dmad_lora_rank)
            )
        if self.teacher_factors is not None:
            strength = cfg.teacher_lora_strength
            ema_student = self.student_factors(ema_shadow, role="default")
            combined = combined_factors(self.teacher_factors, strength, ema_student)
            rank = next(iter(combined.values()))[0].shape[0]
            rows[f"data_lora_{self.steps}step.safetensors"], _ = kohya_state(combined, str(rank))
        disc_rows, _ = kohya_state(
            {k: v for k, v in role_factors_live(self.dit, DISC_ROLE).items()},
            str(cfg.dmad_lora_rank),
        )
        rows["disc_lora.safetensors"] = disc_rows
        heads_rows = {}
        for n, p in self.heads.named_parameters():
            heads_rows[f"heads.{n}"] = p.detach().float().contiguous().cpu()
        rows["disc_heads.safetensors"] = heads_rows
        return rows

    def extra_state(self):
        return {"gap": self.gap.state_dict(), "ema": self.ema.state_dict()}

    def load_extra_state(self, state):
        self.gap.load_state_dict(state["gap"])
        self.ema.load_state_dict(state["ema"])

    def identity(self):
        args = self.args
        return {
            "teacher_model_path": args.teacher_model_path,
            "teacher_lora_path": args.teacher_lora_path,
            "teacher_lora_strength": args.teacher_lora_strength,
            "teacher_steps": args.teacher_steps,
            "student_steps": args.student_steps,
            "dmad_flow_shift": args.dmad_flow_shift,
            "dmad_lora_rank": args.dmad_lora_rank,
            "dmad_lora_alpha": args.dmad_lora_alpha,
            "dmad_head_hidden": args.dmad_head_hidden,
            "disc_u_range": list(args.disc_u_range),
            "lambda_teacher": args.lambda_teacher,
            "lambda_real": args.lambda_real,
            "gap": [args.gap_bands, args.gap_beta, args.gap_tau, args.gap_floor],
        }


def role_factors_live(dit, role="default"):
    """Q: live role adapters keyed by module, from the current parameters."""
    params = role_params(dit, role)
    suffix = "" if role == "default" else f"_{role}"
    out = {}
    for name, p in params.items():
        if name.endswith(f".lora_A{suffix}"):
            module = name[: -len(f".lora_A{suffix}")]
            wrapper = dit.get_submodule(module)
            B = wrapper.get_parameter(f"lora_B{suffix}")
            out[module] = (p.detach().float().cpu(), B.detach().float().cpu(), wrapper.scale)
    return out


def concat_cond(conds):
    """Concat a list of cond tuples along the batch dim."""
    parts = list(zip(*conds))
    return tuple(torch.cat([torch.as_tensor(p) for p in ps], dim=0) for ps in parts)


# ---------------------------------------------------------------------------
# cached teacher latents (Q): LMDB in the DMAD format
# ---------------------------------------------------------------------------
#
# Keys: ``latents_shape`` -> "N C H W"; per row ``latents_{i}_data`` (fp16 raw)
# and ``prompts_{i}_data`` (utf-8). Written by anima_dmad_generate_teacher.py.


class AnimaTeacherLatentDataset(torch.utils.data.Dataset):
    """Random-access reader for cached Anima teacher latents (Q).

    Returns ``{"latents": Tensor[C, H, W] fp32, "prompt": str}``. The LMDB
    environment is opened lazily so the dataset pickles cleanly across
    DataLoader workers.
    """

    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self._env = None
        import lmdb

        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            shape = txn.get(b"latents_shape").decode()
        env.close()
        parts = tuple(map(int, shape.split()))
        assert len(parts) == 4, f"bad latents_shape: {shape!r}"
        self.length, self.latent_shape = parts[0], parts[1:]
        print(f"Teacher LMDB: {self.length} samples, latent shape {tuple(self.latent_shape)}")

    def _env_or_open(self):
        if self._env is None:
            import lmdb

            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        return self._env

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import numpy as np

        env = self._env_or_open()
        with env.begin() as txn:
            raw = txn.get(f"latents_{idx}_data".encode())
            prompt = txn.get(f"prompts_{idx}_data".encode()).decode()
        latent = np.frombuffer(raw, dtype=np.float16).reshape(self.latent_shape).copy()
        return {"latents": torch.from_numpy(latent).float(), "prompt": prompt}


# ---------------------------------------------------------------------------
# teacher Euler sampling (kept for ``anima_dmad_generate_teacher.py`` and for
# the optional CFG teacher; grid comes from ``dmad_sigmas`` when supplied)
# ---------------------------------------------------------------------------


@torch.no_grad()
def euler_sample_x0(dit, noise4d, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
                    steps=20, flow_shift=1.0, guidance_scale=1.0, neg_cond=None, sigmas=None):
    """Multi-step Euler sampling for rectified flow (no grad). Returns x0 latents (4D).

    The DiT runs in its own parameter dtype (timesteps included — a float32
    ``t`` against bf16 weights raises a dtype error); accumulation stays fp32.

    Args:
        guidance_scale: CFG scale (1.0 = no guidance). With > 1.0, ``neg_cond``
            must be a ``(neg_embeds, neg_attn, neg_t5_ids, neg_t5_mask)`` tuple;
            the guided velocity ``v_neg + cfg * (v_pos - v_neg)`` is integrated,
            baking the guidance into the samples.
        sigmas: optional explicit (descending) sigma grid ending at 0
            (e.g. ``dmad_sigmas(steps, shift)``); overrides steps/flow_shift.
    """
    device = noise4d.device
    try:
        model_dtype = next(dit.parameters()).dtype
    except StopIteration:
        model_dtype = noise4d.dtype
    if sigmas is None:
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
        if float(flow_shift) != 1.0:
            fs = float(flow_shift)
            sigmas = (sigmas * fs) / (1 + (fs - 1) * sigmas)
    sigmas = sigmas.reshape(-1).to(device=device, dtype=torch.float32)
    use_cfg = float(guidance_scale) > 1.0 and neg_cond is not None
    x = noise4d.unsqueeze(2).float()  # 5D float32 accumulator
    b, _, _, h, w = x.shape
    padding_mask = torch.zeros(b, 1, h, w, dtype=model_dtype, device=device)

    def run_dit(x_in, t_in, cond):
        c_emb, c_attn, c_t5, c_t5m = cond
        if torch.is_tensor(c_emb):
            c_emb = c_emb.to(model_dtype)
        return dit(x_in.to(model_dtype), t_in,
                   c_emb, padding_mask=padding_mask,
                   source_attention_mask=c_attn,
                   target_input_ids=c_t5,
                   target_attention_mask=c_t5m).float()

    s_list = sigmas.tolist()
    for i in range(len(s_list) - 1):
        s = s_list[i]
        t = torch.full((b,), s, device=device, dtype=model_dtype)
        if use_cfg:
            v_pos = run_dit(x, t, (prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask))
            v_neg = run_dit(x, t, neg_cond)
            v = v_neg + float(guidance_scale) * (v_pos - v_neg)
        else:
            v = run_dit(x, t, (prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask))
        x = x + v * (s_list[i + 1] - s)
    return x.squeeze(2).to(noise4d.dtype)
