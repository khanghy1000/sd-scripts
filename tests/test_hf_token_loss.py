"""
Tests for the High-Frequency Token (HF) latent loss (library/hf_token_loss.py).

Port of `scripts/test_hf_latent_loss.py` from the reference implementation, adapted to
the sd_scripts train_network integration (spec: `plans/high-frequency-token-loss (1).md`).

Coverage:
  - Laplacian unit tests (constant -> 0, checkerboard -> interior 64*s^2)
  - tokenize shapes
  - hf_token_weights (flat -> 1, single token -> 1, RNG neutrality)
  - Off-mode bit-identity (scale == 0 is a no-op)
  - Decomposition: total_loss == mse + hf_scaled bit-exact
  - On-mode == first-principles reference across scale/exponent/patch x t-weighting
  - Flat-latent and single-token edges equal plain x0-MSE
  - Negative controls (uniform weights, weights from prediction, velocity error,
    missing-eps -> NaN)
  - Tweedie modes (flow, x0_direct, x0_residual_eps, vpred_ddpm, eps_ddpm)
  - Config validation (scale < 0, exponent <= 0 raise)
  - Patch divisibility assertion
  - CLI argument registration in train_network.setup_parser
  - Tuple-arity regression: sd3/lumina/hunyuan return a 5-tuple (noise) so the base
    process_batch unpack at train_network.py:945 does not break

All tensors run on CUDA (assumed available).
"""

import argparse
import math
import os
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from library.hf_token_loss import (
    HF_EPS,
    laplacian_energy,
    tokenize,
    hf_token_weights,
    hf_per_sample_loss,
    validate_hf_args,
    validate_hf_high_noise_gate_args,
    hf_sigma_from_timesteps,
    hf_one_sided_snr_gate,
    hf_snr_from_timesteps,
    hf_x0_hat,
    hf_apply_term,
)

DEVICE = torch.device("cuda")
B, C, H, W = 4, 8, 16, 16  # 16x16 latents: patch 2 -> N=64, patch 4 -> N=16


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


class FakeScheduler:
    """Minimal scheduler stub exposing config.num_train_timesteps + alphas_cumprod.

    alphas_cumprod follows sigma^2 = 1 - a_bar = (t/T)^2, i.e. a_bar = 1 - (t/T)^2.
    """

    def __init__(self, num_train_timesteps: int = 1000, device=DEVICE):
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps)
        t = torch.linspace(0.0, 1.0, num_train_timesteps, device=device)
        self.alphas_cumprod = (1.0 - t**2).to(torch.float64)


def make_flow_tensors(bs=B, ch=C, h=H, w=W, seed=0, perturb=0.0):
    """Return (clean, noise, sigmas, model_pred as velocity) for flow Tweedie.

    perturb > 0 adds Gaussian noise to the velocity so x0_hat != clean by an O(perturb)
    amount — needed by the on-mode/reference and negative-control tests (with perturb=0
    the x0 error is pure floating-point residue and comparisons are meaningless).
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    clean = torch.randn(bs, ch, h, w, device=DEVICE, dtype=torch.float64, generator=g)
    noise = torch.randn(bs, ch, h, w, device=DEVICE, dtype=torch.float64, generator=g)
    sigmas = torch.rand(bs, 1, 1, 1, device=DEVICE, dtype=torch.float64, generator=g)
    model_pred = noise - clean  # velocity: v = x1 - x0
    if perturb > 0.0:
        model_pred = model_pred + perturb * torch.randn_like(model_pred)
    return clean, noise, sigmas, model_pred


def reference_hf_term(
    model_pred, clean, noisy, sigmas, patch, exponent, scale, weighting=None
):
    """First-principles re-implementation of spec §2 using public ops only."""
    eps = 1e-6
    x0_hat = noisy - sigmas * model_pred

    # squared Laplacian with replication padding
    padded = F.pad(clean, (1, 1, 1, 1), mode="replicate")
    lap = (
        4.0 * clean
        - padded[:, :, :-2, 1:-1]
        - padded[:, :, 2:, 1:-1]
        - padded[:, :, 1:-1, :-2]
        - padded[:, :, 1:-1, 2:]
    )
    d = lap * lap

    # per-token mean detail
    N = (clean.shape[-2] // patch) * (clean.shape[-1] // patch)
    cols = F.unfold(d, kernel_size=patch, stride=patch).transpose(1, 2)
    d_token = cols.reshape(clean.shape[0], N, -1).mean(dim=-1)  # [B, N]

    # normalize (eps-guarded, mean-1 rescale)
    m = d_token.mean(dim=-1, keepdim=True)
    raw = ((d_token + eps) / (m + eps)) ** exponent
    w = raw / raw.mean(dim=-1, keepdim=True)

    # per-token squared x0 error
    err_cols = F.unfold(x0_hat - clean, kernel_size=patch, stride=patch).transpose(1, 2)
    per_token = err_cols.reshape(clean.shape[0], N, -1).square().mean(dim=-1)  # [B, N]

    per_sample = (w * per_token).mean(dim=-1)  # [B]
    if weighting is not None:
        per_sample = per_sample * weighting.detach().reshape(-1).to(dtype=per_sample.dtype)
    hf = per_sample.mean()
    return scale * hf


def plain_x0_mse(x0_hat, clean):
    return (x0_hat - clean).square().mean()


# ──────────────────────────────────────────────
# Laplacian unit tests (spec §8.10)
# ──────────────────────────────────────────────


class TestLaplacian:
    def test_constant_input_is_exactly_zero(self):
        # dyadic constant: the subtraction chain 4x - up - down - left - right is exact
        x = torch.full((2, 3, 8, 8), 0.5, device=DEVICE, dtype=torch.float64)
        lap = laplacian_energy(x)
        assert torch.equal(lap, torch.zeros_like(lap)), (
            "Replication padding must give exactly 0 for a (dyadic) constant input"
        )

    def test_arbitrary_constant_has_no_boundary_ring(self):
        # non-dyadic constants have an fp rounding residue (~ulp), but it must be uniform:
        # replication padding must NOT produce a spurious boundary ring (zero-padding would).
        x = torch.full((2, 3, 8, 8), 0.7, device=DEVICE, dtype=torch.float64)
        lap = laplacian_energy(x)
        assert torch.isfinite(lap).all()
        assert lap.abs().max().item() < 1e-14, (
            "constant input must be ~0 (fp rounding only), got max abs"
        )
        # uniform everywhere (interior == boundary == corners): no boundary ring
        assert torch.equal(lap, lap[0, 0, 0, 0].expand_as(lap)), (
            "no spurious boundary ring: residual must be uniform across the grid"
        )

    def test_checkerboard_interior_is_64_times_amplitude_squared(self):
        # true 2D checkerboard (parity of i+j) of amplitude s=1: interior response is
        # 8*s, squared 64
        h, w = 8, 8
        rows = torch.arange(h, device=DEVICE).view(1, 1, h, 1)
        cols = torch.arange(w, device=DEVICE).view(1, 1, 1, w)
        x = torch.where((rows + cols) % 2 == 0, 1.0, -1.0)
        x = x.expand(2, 3, h, w).to(torch.float64)
        lap = laplacian_energy(x)
        # interior only (exclude the boundary ring)
        interior = lap[:, :, 1:-1, 1:-1]
        assert torch.equal(interior, torch.full_like(interior, 64.0)), (
            "Checkerboard interior squared response must be 64 (8*s with s=1)"
        )
        # boundary is finite
        assert torch.isfinite(lap).all()

    def test_tokenize_shape(self):
        x = torch.randn(2, 4, 8, 8, device=DEVICE, dtype=torch.float64)
        t2 = tokenize(x, 2)
        assert t2.shape == (2, 16, 16)  # N=16 tokens, 4*2*2 features
        t4 = tokenize(x, 4)
        assert t4.shape == (2, 4, 64)


# ──────────────────────────────────────────────
# Weights: flat / single-token / RNG neutrality
# ──────────────────────────────────────────────


class TestWeights:
    def test_flat_latent_weights_are_exactly_one(self):
        clean = torch.zeros(2, 4, 8, 8, device=DEVICE, dtype=torch.float64)
        w = hf_token_weights(clean, patch=2, exponent=2.0)
        assert torch.equal(w, torch.ones_like(w)), "All-flat latent must give w == 1 exactly"

    def test_single_token_weights_are_exactly_one(self):
        clean = torch.randn(3, 4, 2, 2, device=DEVICE, dtype=torch.float64)  # N = 1
        w = hf_token_weights(clean, patch=2, exponent=3.0)
        assert torch.equal(w, torch.ones_like(w)), "Single-token grid must give w == 1 exactly"

    def test_per_sample_mean_is_one(self):
        clean = torch.randn(4, 8, 16, 16, device=DEVICE, dtype=torch.float64)
        for exponent in (0.5, 1.0, 2.0):
            w = hf_token_weights(clean, patch=2, exponent=exponent)
            means = w.mean(dim=-1)
            assert torch.allclose(means, torch.ones_like(means), rtol=1e-12), (
                f"per-sample mean must be exactly 1 (exponent={exponent})"
            )

    def test_rng_neutrality(self):
        clean = torch.randn(2, 4, 16, 16, device=DEVICE, dtype=torch.float64)
        state_before = torch.cuda.get_rng_state()
        _ = hf_token_weights(clean, patch=2, exponent=1.5)
        state_after = torch.cuda.get_rng_state()
        assert torch.equal(state_before, state_after), (
            "HF weight computation must not draw from the global RNG"
        )

    def test_detail_weights_favor_high_frequency_tokens(self):
        # left half flat, right half alternating columns -> right tokens get higher weight
        clean = torch.zeros(1, 1, 8, 16, device=DEVICE, dtype=torch.float64)
        idx = torch.arange(8, device=DEVICE).reshape(1, 8)
        checker = torch.where(idx % 2 == 0, 1.0, -1.0)
        clean[0, 0, :, 8:] = checker
        w = hf_token_weights(clean, patch=2, exponent=1.0)  # [1, N], N = 4*8 = 32
        w = w.reshape(4, 8)
        right = w[:, 4:].mean()
        left = w[:, :4].mean()
        assert right > left, "Checkerboard (detail) tokens must be weighted higher"


class TestHighNoiseSNRGate:
    def test_low_noise_is_exactly_unchanged(self):
        snr = torch.tensor([1e6, 9.0, 1.0 / 9.0], device=DEVICE, dtype=torch.float64)
        gate = hf_one_sided_snr_gate(snr, snr_cut=1.0 / 9.0, min_gate=0.1, power=1.0)
        assert torch.equal(gate, torch.ones_like(gate))

    def test_only_high_noise_is_attenuated_and_floored(self):
        snr = torch.tensor([1.0 / 9.0, 0.0625, 0.012345679, 0.0], device=DEVICE, dtype=torch.float64)
        gate = hf_one_sided_snr_gate(snr, snr_cut=1.0 / 9.0, min_gate=0.1, power=1.0)
        assert gate[0] == 1.0
        assert gate[1] < 1.0
        assert gate[2] < gate[1]
        assert gate[3] == 0.1
        assert torch.isfinite(gate).all()

    def test_power_controls_high_noise_attenuation(self):
        snr = torch.tensor([0.0625, 0.012345679], device=DEVICE, dtype=torch.float64)
        linear = hf_one_sided_snr_gate(snr, snr_cut=1.0 / 9.0, min_gate=0.0, power=1.0)
        squared = hf_one_sided_snr_gate(snr, snr_cut=1.0 / 9.0, min_gate=0.0, power=2.0)
        assert torch.all(squared <= linear)
        assert torch.all(squared < linear)

    def test_timestep_conversion_matches_flow_convention(self):
        scheduler = FakeScheduler(num_train_timesteps=1000)
        timesteps = torch.tensor([0.0, 750.0, 1000.0], device=DEVICE, dtype=torch.float64)
        sigma = hf_sigma_from_timesteps(timesteps, scheduler, timesteps_in_sigma=False)
        torch.testing.assert_close(
            sigma,
            torch.tensor([1e-6, 0.75, 1.0], device=DEVICE, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_sigma_timesteps_are_used_directly(self):
        sigma = hf_sigma_from_timesteps(
            torch.tensor([0.2, 0.8], device=DEVICE, dtype=torch.float64),
            timesteps_in_sigma=True,
        )
        torch.testing.assert_close(
            sigma,
            torch.tensor([0.2, 0.8], device=DEVICE, dtype=torch.float64),
        )

    def test_gate_validation(self):
        validate_hf_high_noise_gate_args(0.1, 1.0, 1.0 / 9.0)
        with pytest.raises(ValueError):
            validate_hf_high_noise_gate_args(0.1, 1.0, 0.0)
        with pytest.raises(ValueError):
            validate_hf_high_noise_gate_args(1.1, 1.0, 1.0 / 9.0)
        with pytest.raises(ValueError):
            validate_hf_high_noise_gate_args(0.1, 0.0, 1.0 / 9.0)

    def test_generic_snr_gate_leaves_high_snr_unchanged(self):
        snr = torch.tensor([10.0, 1.0, 0.1, 0.0], device=DEVICE, dtype=torch.float64)
        gate = hf_one_sided_snr_gate(snr, snr_cut=0.1, min_gate=0.1, power=1.0)
        assert gate[0] == 1.0
        assert gate[1] == 1.0
        assert gate[2] == 1.0
        assert gate[3] == 0.1

    def test_eps_mode_uses_zero_floor_for_inverse_snr(self):
        snr = torch.tensor([0.1, 0.01, 0.0], device=DEVICE, dtype=torch.float64)
        gate = hf_one_sided_snr_gate(snr, snr_cut=0.1, min_gate=0.0, power=1.0)
        expected = torch.tensor([1.0, 0.1, 0.0], device=DEVICE, dtype=torch.float64)
        torch.testing.assert_close(gate, expected)

    def test_ddpm_snr_from_timesteps(self):
        scheduler = FakeScheduler(num_train_timesteps=1000)
        timesteps = torch.tensor([0, 500, 900], device=DEVICE, dtype=torch.long)
        snr = hf_snr_from_timesteps(timesteps, "vpred_ddpm", scheduler)
        expected_alpha = scheduler.alphas_cumprod[timesteps]
        expected = expected_alpha / (1.0 - expected_alpha).clamp_min(1e-6)
        torch.testing.assert_close(snr, expected)

    def test_flow_snr_from_timesteps(self):
        scheduler = FakeScheduler(num_train_timesteps=1000)
        timesteps = torch.tensor([250.0, 750.0], device=DEVICE, dtype=torch.float64)
        snr = hf_snr_from_timesteps(timesteps, "flow", scheduler)
        expected_sigma = timesteps / 1000.0
        expected = ((1.0 - expected_sigma) / expected_sigma).square()
        torch.testing.assert_close(snr, expected)

    def test_x0_direct_uses_flow_snr_without_alpha_bar(self):
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        timesteps = torch.tensor([250.0, 750.0], device=DEVICE, dtype=torch.float64)
        snr = hf_snr_from_timesteps(timesteps, "x0_direct", scheduler)
        expected_sigma = timesteps / 1000.0
        expected = ((1.0 - expected_sigma) / expected_sigma).square()
        torch.testing.assert_close(snr, expected)


# ──────────────────────────────────────────────
# Config validation
# ──────────────────────────────────────────────


class TestValidation:
    def test_negative_scale_rejected(self):
        with pytest.raises(ValueError):
            validate_hf_args(-1.0, 1.0, 2)

    def test_nonpositive_exponent_rejected(self):
        for bad in (0.0, -0.5):
            with pytest.raises(ValueError):
                validate_hf_args(1.0, bad, 2)

    def test_nonpositive_patch_rejected(self):
        with pytest.raises(ValueError):
            validate_hf_args(1.0, 1.0, 0)

    def test_valid_args_pass(self):
        validate_hf_args(0.0, 1.0, 2)
        validate_hf_args(0.5, 2.0, 16)


# ──────────────────────────────────────────────
# Off-mode bit-identity + decomposition
# ──────────────────────────────────────────────


class TestApplyTerm:
    def test_off_mode_is_bit_identical_noop(self):
        clean, noise, sigmas, model_pred = make_flow_tensors()
        noisy = (1.0 - sigmas) * clean + sigmas * noise
        mse = plain_x0_mse(noisy - sigmas * model_pred, clean)

        out, hf_value = hf_apply_term(
            mse, model_pred, clean, noisy, sigmas * 1000.0, None,
            scale=0.0, exponent=2.0, patch=2, mode="flow",
            noise_scheduler=FakeScheduler(), timesteps_in_sigma=True,
        )
        assert hf_value is None
        assert out is mse, "off-mode must return the identical loss object"
        assert torch.equal(out, mse)

    def test_decomposition_bit_exact(self):
        clean, noise, sigmas, model_pred = make_flow_tensors()
        noisy = (1.0 - sigmas) * clean + sigmas * noise
        mse = plain_x0_mse(noisy - sigmas * model_pred, clean)

        new_loss, hf_scaled = hf_apply_term(
            mse, model_pred, clean, noisy, sigmas, None,
            scale=1.0, exponent=1.0, patch=2, mode="flow",
            noise_scheduler=FakeScheduler(), timesteps_in_sigma=True,
        )
        assert hf_scaled is not None
        assert torch.equal(new_loss, mse + hf_scaled), (
            "total_loss == mse + hf_scaled must hold bit-exactly"
        )

    @pytest.mark.parametrize("scale", [0.01, 1.0])
    @pytest.mark.parametrize("exponent", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("patch", [2, 4])
    def test_on_mode_matches_first_principles_reference(self, scale, exponent, patch):
        clean, noise, sigmas, model_pred = make_flow_tensors(perturb=0.2)
        noisy = (1.0 - sigmas) * clean + sigmas * noise
        weighting = torch.rand(B, 1, 1, 1, device=DEVICE, dtype=torch.float64)

        mse = plain_x0_mse(noisy - sigmas * model_pred, clean)
        _, hf_scaled = hf_apply_term(
            mse, model_pred, clean, noisy, sigmas, weighting,
            scale=scale, exponent=exponent, patch=patch, mode="flow",
            noise_scheduler=FakeScheduler(), timesteps_in_sigma=True,
        )
        expected = reference_hf_term(
            model_pred, clean, noisy, sigmas, patch, exponent, scale, weighting=weighting
        )
        assert hf_scaled is not None
        assert torch.allclose(hf_scaled, expected, rtol=1e-12, atol=0.0), (
            f"HF term must match first-principles reference (scale={scale}, exp={exponent}, patch={patch})"
        )

    def test_flat_latent_edge_equals_plain_x0_mse(self):
        clean = torch.zeros(B, C, H, W, device=DEVICE, dtype=torch.float64)
        noise = torch.randn(B, C, H, W, device=DEVICE, dtype=torch.float64)
        sigmas = torch.full((B, 1, 1, 1), 0.5, device=DEVICE, dtype=torch.float64)
        model_pred = noise - clean  # v
        noisy = (1.0 - sigmas) * clean + sigmas * noise
        x0_hat = noisy - sigmas * model_pred
        mse = plain_x0_mse(x0_hat, clean)

        for exponent in (1.0, 2.0):
            _, hf_scaled = hf_apply_term(
                mse, model_pred, clean, noisy, sigmas, None,
                scale=1.0, exponent=exponent, patch=2, mode="flow",
                noise_scheduler=FakeScheduler(), timesteps_in_sigma=True,
            )
            assert torch.isfinite(hf_scaled), "flat-latent HF term must be finite (eps guard)"
            # weights all 1 -> term == plain x0-MSE
            assert torch.allclose(hf_scaled, mse, rtol=1e-12, atol=0.0), (
                "flat latent: HF term must degenerate into plain x0-MSE"
            )

    def test_single_token_edge_equals_plain_x0_mse(self):
        bs, ch, p = 2, 4, 2
        clean = torch.randn(bs, ch, p, p, device=DEVICE, dtype=torch.float64)
        noise = torch.randn(bs, ch, p, p, device=DEVICE, dtype=torch.float64)
        sigmas = torch.rand(bs, 1, 1, 1, device=DEVICE, dtype=torch.float64)
        model_pred = noise - clean
        noisy = (1.0 - sigmas) * clean + sigmas * noise
        x0_hat = noisy - sigmas * model_pred
        mse = plain_x0_mse(x0_hat, clean)

        _, hf_scaled = hf_apply_term(
            mse, model_pred, clean, noisy, sigmas, None,
            scale=1.0, exponent=2.0, patch=p, mode="flow",
            noise_scheduler=FakeScheduler(), timesteps_in_sigma=True,
        )
        assert torch.isfinite(hf_scaled)
        assert torch.allclose(hf_scaled, mse, rtol=1e-12, atol=0.0), (
            "single-token grid: HF term must equal plain x0-MSE"
        )

    def test_patch_divisibility_asserted(self):
        clean = torch.randn(2, 4, 6, 6, device=DEVICE, dtype=torch.float64)  # 6 % 4 != 0
        noise = torch.randn(2, 4, 6, 6, device=DEVICE, dtype=torch.float64)
        sigmas = torch.full((2, 1, 1, 1), 0.5, device=DEVICE, dtype=torch.float64)
        model_pred = noise - clean
        noisy = (1.0 - sigmas) * clean + sigmas * noise
        mse = plain_x0_mse(noisy - sigmas * model_pred, clean)
        with pytest.raises(ValueError, match="hf_patch"):
            hf_apply_term(
                mse, model_pred, clean, noisy, sigmas, None,
                scale=1.0, exponent=1.0, patch=4, mode="flow",
                noise_scheduler=FakeScheduler(), timesteps_in_sigma=True,
            )


# ──────────────────────────────────────────────
# Negative controls (each must NOT match; spec §8.4)
# ──────────────────────────────────────────────


class TestNegativeControls:
    def setup_method(self):
        self.clean, self.noise, self.sigmas, self.model_pred = make_flow_tensors(seed=42, perturb=0.2)
        self.noisy = (1.0 - self.sigmas) * self.clean + self.sigmas * self.noise
        self.patch = 2

    def test_uniform_weights_do_not_match(self):
        real = hf_per_sample_loss(
            self.noisy - self.sigmas * self.model_pred, self.clean, self.patch, 1.0
        )
        w_uniform = torch.ones(B, 64, device=DEVICE, dtype=torch.float64)
        per_token = tokenize(
            (self.noisy - self.sigmas * self.model_pred) - self.clean, self.patch
        ).square().mean(dim=-1)
        uniform = (w_uniform * per_token).mean(dim=-1)
        assert not torch.allclose(real, uniform, rtol=1e-12, atol=0.0), (
            "detail weights must differ from uniform weights"
        )

    def test_weights_from_prediction_do_not_match(self):
        x0_hat = self.noisy - self.sigmas * self.model_pred
        w_clean = hf_token_weights(self.clean, self.patch, 1.0)
        w_pred = hf_token_weights(x0_hat.detach(), self.patch, 1.0)
        assert not torch.allclose(w_clean, w_pred, rtol=1e-12, atol=0.0), (
            "weights computed from the prediction must differ from clean-derived weights"
        )

    def test_velocity_error_weighting_does_not_match(self):
        x0_hat = self.noisy - self.sigmas * self.model_pred
        real = hf_per_sample_loss(x0_hat, self.clean, self.patch, 1.0)
        # apply the same clean-derived weights to the velocity error instead of x0 error
        w = hf_token_weights(self.clean, self.patch, 1.0)
        v_err = tokenize(self.model_pred - (self.noise - self.clean), self.patch).square().mean(dim=-1)
        vel_weighted = (w * v_err).mean(dim=-1)
        assert not torch.allclose(real, vel_weighted, rtol=1e-12, atol=0.0), (
            "weighting the velocity error must not match the x0-domain term"
        )

    def test_missing_eps_produces_nan_on_flat_latent(self):
        # negative control: the formula without eps gives 0/0 -> NaN on flat latents
        detail = torch.zeros(2, 16, device=DEVICE, dtype=torch.float64)
        mean_d = detail.mean(dim=-1, keepdim=True)
        raw = (detail / mean_d) ** 1.0
        assert torch.isnan(raw).any(), (
            "missing-eps formula must produce NaN on flat latents (negative control)"
        )
        # and the eps-guarded implementation stays finite
        clean = torch.zeros(2, 4, 8, 8, device=DEVICE, dtype=torch.float64)
        w = hf_token_weights(clean, patch=2, exponent=1.0)
        assert torch.isfinite(w).all()


# ──────────────────────────────────────────────
# Tweedie reconstruction (spec §2.2)
# ──────────────────────────────────────────────


class TestTweedie:
    def test_flow_mode_reconstructs_clean(self):
        clean, noise, sigmas, v = make_flow_tensors()
        noisy = (1.0 - sigmas) * clean + sigmas * noise
        x0_hat = hf_x0_hat(v, noisy, sigmas * 1000.0, "flow", FakeScheduler())
        assert torch.allclose(x0_hat, clean, rtol=1e-10, atol=1e-10)

    def test_flow_mode_sigma_range_timesteps(self):
        clean, noise, sigmas, v = make_flow_tensors()
        noisy = (1.0 - sigmas) * clean + sigmas * noise
        # timesteps already in [0,1] (Anima convention)
        x0_hat = hf_x0_hat(v, noisy, sigmas, "flow", FakeScheduler(), timesteps_in_sigma=True)
        assert torch.allclose(x0_hat, clean, rtol=1e-10, atol=1e-10)

    def test_x0_direct_mode_passthrough(self):
        clean, noise, sigmas, v = make_flow_tensors()
        pred = clean + 0.1 * torch.randn_like(clean)
        x0_hat = hf_x0_hat(pred, None, None, "x0_direct")
        assert torch.equal(x0_hat, pred.to(torch.float64))

    def test_x0_residual_eps_mode_reconstructs_clean(self):
        clean, noise, sigmas, _ = make_flow_tensors()
        eps_train = 5e-2
        noisy = (1.0 - sigmas) * clean + sigmas * noise
        v = (noisy - clean) / (sigmas + eps_train)
        x0_hat = hf_x0_hat(
            v, noisy, sigmas * 1000.0, "x0_residual_eps",
            FakeScheduler(), eps_train=eps_train,
        )
        assert torch.allclose(x0_hat, clean, rtol=1e-10, atol=1e-10)

    def test_vpred_ddpm_mode_reconstructs_clean(self):
        sched = FakeScheduler()
        g = torch.Generator(device=DEVICE).manual_seed(7)
        clean = torch.randn(2, 4, 8, 8, device=DEVICE, dtype=torch.float64, generator=g)
        eps = torch.randn(2, 4, 8, 8, device=DEVICE, dtype=torch.float64, generator=g)
        ts = torch.tensor([500, 250], device=DEVICE, dtype=torch.long)
        a_bar = sched.alphas_cumprod[ts].reshape(-1, 1, 1, 1)
        noisy = torch.sqrt(a_bar) * clean + torch.sqrt(1.0 - a_bar) * eps
        v = torch.sqrt(a_bar) * eps - torch.sqrt(1.0 - a_bar) * clean
        x0_hat = hf_x0_hat(v, noisy, ts, "vpred_ddpm", sched)
        assert torch.allclose(x0_hat, clean, rtol=1e-10, atol=1e-10)

    def test_eps_ddpm_mode_reconstructs_clean(self):
        sched = FakeScheduler()
        g = torch.Generator(device=DEVICE).manual_seed(11)
        clean = torch.randn(2, 4, 8, 8, device=DEVICE, dtype=torch.float64, generator=g)
        eps = torch.randn(2, 4, 8, 8, device=DEVICE, dtype=torch.float64, generator=g)
        ts = torch.tensor([800, 100], device=DEVICE, dtype=torch.long)
        a_bar = sched.alphas_cumprod[ts].reshape(-1, 1, 1, 1)
        noisy = torch.sqrt(a_bar) * clean + torch.sqrt(1.0 - a_bar) * eps
        x0_hat = hf_x0_hat(eps, noisy, ts, "eps_ddpm", sched)
        assert torch.allclose(x0_hat, clean, rtol=1e-10, atol=1e-10)

    def test_unknown_mode_rejected(self):
        clean, noise, sigmas, v = make_flow_tensors()
        with pytest.raises(ValueError, match="Unknown hf_prediction_mode"):
            hf_x0_hat(v, noise, sigmas, "bogus", FakeScheduler())


# ──────────────────────────────────────────────
# CLI registration (train_network.setup_parser)
# ──────────────────────────────────────────────


def test_cli_args_registered():
    train_network = pytest.importorskip("train_network")
    parser = train_network.setup_parser()

    # defaults
    ns = parser.parse_args([])
    assert ns.hf_scale == 0.0
    assert ns.hf_exponent == 1.0
    assert ns.hf_patch == 2
    assert ns.hf_high_noise_snr_cut == pytest.approx(1.0 / 9.0)
    assert ns.hf_high_noise_min_weight == 0.10
    assert ns.hf_high_noise_power == 1.0

    # explicit values
    ns = parser.parse_args([
        "--hf_scale", "0.5", "--hf_exponent", "2.0", "--hf_patch", "16",
        "--hf_high_noise_snr_cut", "0.05",
        "--hf_high_noise_min_weight", "0.2",
        "--hf_high_noise_power", "2.0",
    ])
    assert ns.hf_scale == 0.5
    assert ns.hf_exponent == 2.0
    assert ns.hf_patch == 16
    assert ns.hf_high_noise_snr_cut == 0.05
    assert ns.hf_high_noise_min_weight == 0.2
    assert ns.hf_high_noise_power == 2.0


# ──────────────────────────────────────────────
# Tuple-arity regression (sd3/lumina/hunyuan)
# ──────────────────────────────────────────────


def _variant_return_lines(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    path = os.path.join(root, filename)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    return [
        line.strip()
        for line in src.splitlines()
        if line.strip().startswith("return model_pred, target, timesteps, weighting")
    ]


@pytest.mark.parametrize(
    "filename",
    ["sd3_train_network.py", "lumina_train_network.py", "hunyuan_image_train_network.py"],
)
def test_get_noise_pred_and_target_returns_noise(filename):
    """Base process_batch unpacks 5 values; these trainers must return the noise 5th."""
    lines = _variant_return_lines(filename)
    assert lines, f"{filename}: no return model_pred, target, timesteps, weighting found"
    for line in lines:
        assert re.search(r",\s*noise\s*$", line), (
            f"{filename}: return must include the 5th element `noise`, got: {line}"
        )
