# DMAD critic and helpers for the Anima (rectified-flow DiT) model.
#
# Port of ``DMAD/train/image/main/sd_guidance.py`` (SDXL) to the Anima DiT:
# a two-head GAN critic in place of a fake score model.
#
# Two binary heads read the same pooled backbone features:
#     real head    d_real    : real latents (T) vs generated latents (G)
#     teacher head d_teacher : teacher-sample latents (Q) vs generated latents (G)
# Critic loss:     BCE(real head; T=1, G=0) + BCE(teacher head; Q=1, G=0)
# Generator loss:  -d_teacher(G) (optionally re-weighted per noise band with
#                  --teacher_gap_route) and -d_real(G).
#
# Backbone: a copy of the Anima DiT (patch embedder, timestep embedder, RoPE
# embedder, LLM adapter and, by default, all transformer blocks — the paper's
# DiT discriminators read final-block tokens) with two MLP heads. Per-token
# logits are averaged, as in the Wan setup. Flow-matching noising
# ``x_t = (1 - s) * x0 + s * eps`` with ``s`` uniform in [0, critic_max_sigma)
# replaces the DDPM noising of the SDXL version. Gap routing uses ten
# equal-width bands over [0, 1) with band-based normalization, as in the
# MiniMax-H3 setup.

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

NUM_GAP_BANDS = 10
# Default number of gap-routing bands (ten equal-width bands over [0, 1),
# MiniMax-H3 setup). Overridable per critic via gap_num_bands / --gap_num_bands.
DEFAULT_GAP_BANDS = NUM_GAP_BANDS
# Floor for the band-based gap-weight normalization (MiniMax-H3 setup).
GAP_WEIGHT_FLOOR = 1e-4


# ---------------------------------------------------------------------------
# frozen-gain spectral norm (same idea as DMAD's _ScaledSN, extended to Linear)
# ---------------------------------------------------------------------------


class _ScaledSN(nn.Module):
    """weight -> gain * W / sigma_max(W), sigma_max from one power iteration per forward.

    The gain is frozen at the value the first forward computes, so the layer is
    unchanged at initialization. Works for both Conv2d (reshaped) and Linear.
    """

    def __init__(self, weight):
        super().__init__()
        w2 = weight.detach().reshape(weight.shape[0], -1)
        u = torch.randn(w2.shape[0], device=weight.device)
        v = torch.randn(w2.shape[1], device=weight.device)
        for _ in range(200):
            v = F.normalize(w2.t() @ u, dim=0)
            u = F.normalize(w2 @ v, dim=0)
        self.register_buffer("u", u.clone())
        self.register_buffer("v", v.clone())
        v1 = F.normalize(w2.t() @ u, dim=0)
        u1 = F.normalize(w2 @ v1, dim=0)
        self.register_buffer("gain", torch.sum(u1 * (w2 @ v1)).clone())

    def forward(self, w):
        w2 = w.reshape(w.shape[0], -1)
        with torch.no_grad():
            self.v.copy_(F.normalize(w2.t() @ self.u, dim=0))
            self.u.copy_(F.normalize(w2 @ self.v, dim=0))
        # clone: one backward graph can span several forwards, and the next
        # forward updates u, v in place
        u, v = self.u.clone(), self.v.clone()
        sigma = torch.sum(u * (w2 @ v)).clamp(min=1e-12)
        return w * (self.gain / sigma)


def apply_scaled_spectral_norm(module):
    """Frozen-gain spectral norm on every trainable, non-zero Conv2d/Linear in ``module``.

    Returns the number of wrapped layers.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.requires_grad:
            if m.weight.detach().abs().max() < 1e-8:  # zero-initialized layer
                continue
            parametrize.register_parametrization(m, "weight", _ScaledSN(m.weight))
            n += 1
    return n


# ---------------------------------------------------------------------------
# flow-matching helpers
# ---------------------------------------------------------------------------


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


class _RMSNormNoAffine(nn.Module):
    """Non-affine RMS normalization (MiniMax-H3 discriminator-head style)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def make_critic_head(dim):
    """Final-block token features ``[..., D]`` -> per-token logits.

    Two linear layers with a SiLU between them (MiniMax-H3 head style); the
    caller averages the per-token logits, as in the Wan setup.
    """
    return nn.Sequential(
        _RMSNormNoAffine(dim),
        nn.Linear(dim, dim),
        nn.SiLU(),
        nn.Linear(dim, 1),
    )


# ---------------------------------------------------------------------------
# critic
# ---------------------------------------------------------------------------


class AnimaDMADCritic(nn.Module):
    """Two-head DMAD critic on a copy of the Anima DiT (final-block tokens)."""

    def __init__(
        self,
        template_dit,
        critic_num_blocks=None,
        freeze_backbone=False,
        spectral_norm=False,
        teacher_gap_route=False,
        gap_tau=2.0,
        gap_num_bands=None,
        critic_max_sigma=1.0,
    ):
        super().__init__()
        self.critic_max_sigma = float(critic_max_sigma)
        self.teacher_gap_route = teacher_gap_route
        self.gap_tau = float(gap_tau)
        self.num_gap_bands = int(gap_num_bands) if gap_num_bands else DEFAULT_GAP_BANDS
        assert self.num_gap_bands >= 2, "gap_num_bands must be >= 2"
        self.gap_ema_beta = 0.99
        self.register_buffer("_gap_ema", torch.zeros(self.num_gap_bands))
        self.register_buffer("_gap_ready", torch.zeros(self.num_gap_bands))

        # Copy the conditioning / embedding front-end and transformer blocks so
        # the critic matches the generator's architecture exactly. By default
        # all blocks are kept: the paper's DiT discriminators (Wan, MiniMax-H3)
        # read final-block tokens. Fewer blocks (an encoder-style critic, as in
        # the SDXL/ImageNet setups) can be selected with critic_num_blocks.
        total_blocks = len(template_dit.blocks)
        if critic_num_blocks is None or critic_num_blocks <= 0:
            critic_num_blocks = total_blocks
        critic_num_blocks = min(int(critic_num_blocks), total_blocks)
        self.critic_num_blocks = critic_num_blocks

        self.x_embedder = copy.deepcopy(template_dit.x_embedder)
        self.t_embedder = copy.deepcopy(template_dit.t_embedder)
        self.t_embedding_norm = copy.deepcopy(template_dit.t_embedding_norm)
        self.pos_embedder = copy.deepcopy(template_dit.pos_embedder)
        self.extra_pos_embedder = (
            copy.deepcopy(template_dit.extra_pos_embedder) if getattr(template_dit, "extra_per_block_abs_pos_emb", False) else None
        )
        self.extra_per_block_abs_pos_emb = bool(getattr(template_dit, "extra_per_block_abs_pos_emb", False))
        if getattr(template_dit, "use_llm_adapter", False) and hasattr(template_dit, "llm_adapter"):
            self.llm_adapter = copy.deepcopy(template_dit.llm_adapter)
            self.use_llm_adapter = True
        else:
            self.llm_adapter = None
            self.use_llm_adapter = False

        self.blocks = nn.ModuleList(copy.deepcopy(template_dit.blocks[:critic_num_blocks]))
        self.attn_mode = getattr(template_dit, "attn_mode", "torch")
        self.split_attn = bool(getattr(template_dit, "split_attn", False))
        self.pos_emb_cls = getattr(template_dit, "pos_emb_cls", "rope3d")
        self.concat_padding_mask = bool(getattr(template_dit, "concat_padding_mask", True))

        dim = int(template_dit.model_channels)
        self.head_real = make_critic_head(dim)
        self.head_teacher = copy.deepcopy(self.head_real)
        for m in self.head_teacher.modules():  # re-initialize so the heads start different
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if freeze_backbone:
            for p in list(self.x_embedder.parameters()) + list(self.t_embedder.parameters()) + \
                    list(self.t_embedding_norm.parameters()) + list(self.blocks.parameters()):
                p.requires_grad_(False)
            if self.llm_adapter is not None:
                for p in self.llm_adapter.parameters():
                    p.requires_grad_(False)
        if spectral_norm:
            n = 0
            n += apply_scaled_spectral_norm(self.x_embedder)
            n += apply_scaled_spectral_norm(self.blocks)
            n += apply_scaled_spectral_norm(self.head_real)
            n += apply_scaled_spectral_norm(self.head_teacher)
            print(f"anima critic spectral norm: {n} layers")

    # -- feature extraction -------------------------------------------------

    def _prepare_embedded_sequence(self, x_B_C_T_H_W, padding_mask):
        if self.concat_padding_mask:
            from torchvision import transforms

            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        if self.extra_per_block_abs_pos_emb and self.extra_pos_embedder is not None:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=None)
        else:
            extra_pos_emb = None
        if "rope" in str(self.pos_emb_cls).lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=None), extra_pos_emb
        return x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D), None, extra_pos_emb

    def _backbone_features(
        self, x_5d, sigmas, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, padding_mask
    ):
        from library import attention as attention_lib

        crossattn_emb = prompt_embeds
        if self.use_llm_adapter and t5_input_ids is not None and self.llm_adapter is not None:
            crossattn_emb = self.llm_adapter(
                source_hidden_states=prompt_embeds,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )
            if t5_attn_mask is not None:
                crossattn_emb = crossattn_emb.masked_fill(~t5_attn_mask.bool().unsqueeze(-1), 0)

        x_B_T_H_W_D, rope_emb, extra_pos_emb = self._prepare_embedded_sequence(x_5d, padding_mask)

        t = sigmas.reshape(-1, 1).to(x_5d.dtype)
        t_emb, adaln_lora = self.t_embedder(t)
        t_emb = self.t_embedding_norm(t_emb)

        attn_params = attention_lib.AttentionParams.create_attention_params(self.attn_mode, self.split_attn)
        use_fp32 = x_B_T_H_W_D.dtype == torch.float16
        h = x_B_T_H_W_D
        for block in self.blocks:
            h = block(h, t_emb, crossattn_emb, attn_params, use_fp32,
                      rope_emb_L_1_1_D=rope_emb, adaln_lora_B_T_3D=adaln_lora,
                      extra_per_block_pos_emb=extra_pos_emb)
        # full token map [B, T, H, W, D]; heads score each token, the caller
        # averages the per-token logits (Wan setup)
        return h

    # -- public API ----------------------------------------------------------

    def sample_sigmas(self, batch_size, device, dtype=torch.float32):
        return torch.rand(batch_size, device=device, dtype=dtype) * self.critic_max_sigma

    def critic_logits(self, image4d, sigmas, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask):
        """(d_real, d_teacher), [N] each, on image noised to ``sigmas``.

        Per-token head logits are averaged over the token sequence (Wan setup).
        """
        eps = torch.randn_like(image4d)
        s = sigmas.reshape(-1, 1, 1, 1).to(image4d.dtype)
        noised = flow_add_noise(image4d, eps, s).unsqueeze(2)  # 4D -> 5D
        b, _, _, h, w = noised.shape
        padding_mask = torch.zeros(b, 1, h, w, dtype=noised.dtype, device=noised.device)
        tokens = self._backbone_features(noised, sigmas, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, padding_mask)
        tokens = tokens.float()
        d_real = self.head_real(tokens).reshape(b, -1).mean(dim=1)
        d_teacher = self.head_teacher(tokens).reshape(b, -1).mean(dim=1)
        return d_real, d_teacher

    # -- gap routing (ten equal-width bands over [0, 1), MiniMax-H3 setup) ----

    @torch.no_grad()
    def _update_gap_ema(self, dR_T, dR_Q, s_T, s_Q):
        nb = self.num_gap_bands
        dev = dR_T.device
        bt = (s_T.reshape(-1) * nb).long().clamp(0, nb - 1)
        bq = (s_Q.reshape(-1) * nb).long().clamp(0, nb - 1)
        sT = torch.zeros(nb, device=dev)
        cT = torch.zeros(nb, device=dev)
        sQ = torch.zeros(nb, device=dev)
        cQ = torch.zeros(nb, device=dev)
        sT.scatter_add_(0, bt, dR_T.float())
        cT.scatter_add_(0, bt, torch.ones_like(dR_T, dtype=torch.float))
        sQ.scatter_add_(0, bq, dR_Q.float())
        cQ.scatter_add_(0, bq, torch.ones_like(dR_Q, dtype=torch.float))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for t in (sT, cT, sQ, cQ):
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        valid = (cT > 0) & (cQ > 0)
        gap_now = sT / cT.clamp(min=1) - sQ / cQ.clamp(min=1)
        beta = self.gap_ema_beta
        fresh = valid & (self._gap_ready > 0)
        upd = torch.where(fresh, beta * self._gap_ema.to(dev) + (1 - beta) * gap_now, gap_now)
        self._gap_ema.copy_(torch.where(valid, upd, self._gap_ema.to(dev)))
        self._gap_ready.copy_(torch.where(valid, torch.ones_like(self._gap_ready.to(dev)), self._gap_ready.to(dev)))
        return torch.where(valid, gap_now, torch.full_like(gap_now, float("nan")))

    def _teacher_gap_weight(self, sigmas):
        """Per-sample teacher weights, normalized over bands (Wan/H3 setup).

        Raw band weights ``a_b = sigmoid((median gap - gap_b) / tau)`` are
        divided by ``C = max(mean a_b over measured bands, floor)`` (paper
        Eq. 8-9). Returns None (plain mean) until half of the bands have been
        measured. Statistics and weights are buffers / index ops, so no
        gradient flows through them during the generator update.
        """
        if not self.teacher_gap_route:
            return None
        nb = self.num_gap_bands
        ready = self._gap_ready > 0
        if int(ready.sum()) < max(2, (nb + 1) // 2):
            return None
        ema = self._gap_ema
        center = float(ema[ready].median())
        a_all = torch.sigmoid((center - ema) / self.gap_tau)
        c = max(float(a_all[ready].mean()), GAP_WEIGHT_FLOOR)
        band = (sigmas.reshape(-1) * nb).long().clamp(0, nb - 1)
        return (a_all[band] / c).to(sigmas.dtype)

    # -- losses ----------------------------------------------------------------

    def compute_generator_loss(self, fake_image, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask):
        # Paper Eq. 9: -1/B * sum(lambda_T * w_i * h_T + lambda_R * h_R);
        # the trainer applies lambda_T / lambda_R.
        s = self.sample_sigmas(fake_image.shape[0], fake_image.device)
        d_real, d_teacher = self.critic_logits(fake_image, s, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask)
        w = self._teacher_gap_weight(s)
        if w is None:
            teacher_loss = (-d_teacher).mean()
        else:
            teacher_loss = (-d_teacher * w).mean() + (-d_teacher).sum() * 0.0
        real_loss = (-d_real).mean()
        return {"gen_teacher_loss": teacher_loss, "gen_real_loss": real_loss}

    def compute_critic_loss(self, fake_image, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
                            real_image, real_cond, teacher_image, teacher_cond):
        n = max(real_image.shape[0], fake_image.shape[0], teacher_image.shape[0])
        s = self.sample_sigmas(n, fake_image.device)  # shared by the T / G / Q samples of each index
        s_T, s_G, s_Q = s[:real_image.shape[0]], s[:fake_image.shape[0]], s[:teacher_image.shape[0]]

        dR_T, _ = self.critic_logits(real_image.detach(), s_T, *real_cond)
        dR_G, dQ_G = self.critic_logits(fake_image.detach(), s_G, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask)
        dR_Q, dQ_Q = self.critic_logits(teacher_image.detach(), s_Q, *teacher_cond)

        loss_real = F.softplus(-dR_T).mean() + F.softplus(dR_G).mean()
        loss_teacher = F.softplus(-dQ_Q).mean() + F.softplus(dQ_G).mean()

        log_dict = {
            "critic_acc_real": (dR_T > 0).float().mean().detach(),
            "critic_acc_teacher": (dQ_Q > 0).float().mean().detach(),
            "critic_acc_fake": 0.5 * ((dR_G < 0).float().mean() + (dQ_G < 0).float().mean()).detach(),
        }
        gap = self._update_gap_ema(dR_T, dR_Q, s_T, s_Q)
        for b in range(self.num_gap_bands):
            if bool(self._gap_ready[b] > 0):
                log_dict[f"gap_ema_b{b}"] = float(self._gap_ema[b])
            if not torch.isnan(gap[b]):
                log_dict[f"gap_b{b}"] = float(gap[b])
        return {"critic_loss": loss_real + loss_teacher}, log_dict


@torch.no_grad()
def euler_sample_x0(dit, noise4d, prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
                    steps=20, flow_shift=1.0, guidance_scale=1.0, neg_cond=None):
    """Multi-step Euler sampling for rectified flow (no grad). Returns x0 latents (4D).

    Args:
        guidance_scale: CFG scale (1.0 = no guidance). With > 1.0, ``neg_cond``
            must be a ``(neg_embeds, neg_attn, neg_t5_ids, neg_t5_mask)`` tuple;
            the guided velocity ``v_neg + cfg * (v_pos - v_neg)`` is integrated,
            baking the guidance into the samples.
    """
    device, dtype = noise4d.device, noise4d.dtype
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    if float(flow_shift) != 1.0:
        fs = float(flow_shift)
        sigmas = (sigmas * fs) / (1 + (fs - 1) * sigmas)
    use_cfg = float(guidance_scale) > 1.0 and neg_cond is not None
    x = noise4d.unsqueeze(2).to(torch.float32)  # 5D
    b, _, _, h, w = x.shape
    padding_mask = torch.zeros(b, 1, h, w, dtype=torch.float32, device=device)
    for i in range(steps):
        s = sigmas[i]
        t = torch.full((b,), float(s), device=device, dtype=torch.float32)
        if use_cfg:
            n_emb, n_attn, n_t5, n_t5m = neg_cond
            v_pos = dit(x.to(dtype), t,
                        prompt_embeds.to(dtype) if torch.is_tensor(prompt_embeds) else prompt_embeds,
                        padding_mask=padding_mask.to(dtype),
                        source_attention_mask=attn_mask,
                        target_input_ids=t5_input_ids,
                        target_attention_mask=t5_attn_mask).float()
            v_neg = dit(x.to(dtype), t,
                        n_emb.to(dtype) if torch.is_tensor(n_emb) else n_emb,
                        padding_mask=padding_mask.to(dtype),
                        source_attention_mask=n_attn,
                        target_input_ids=n_t5,
                        target_attention_mask=n_t5m).float()
            v = v_neg + float(guidance_scale) * (v_pos - v_neg)
        else:
            v = dit(x.to(dtype), t,
                    prompt_embeds.to(dtype) if torch.is_tensor(prompt_embeds) else prompt_embeds,
                    padding_mask=padding_mask.to(dtype),
                    source_attention_mask=attn_mask,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask)
            v = v.float()
        x = x + v * (float(sigmas[i + 1]) - float(s))
    return x.squeeze(2).to(dtype)


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
