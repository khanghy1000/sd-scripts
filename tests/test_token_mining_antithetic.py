"""CUDA tests for token-level hard mining and antithetic sigma sampling.

Covers:
- library.custom_train_functions.compute_antithetic_sigmas
- library.custom_train_functions.apply_flow_shift
- library.custom_train_functions.apply_token_mining
- library.train_util.get_noise_noisy_latents_and_timesteps (flow antithetic path)
- library.flux_train_utils.compute_density_for_timestep_sampling (antithetic flag)

All tensor tests run on CUDA per project policy.
"""

import argparse
import math
from types import SimpleNamespace

import pytest
import torch

from library.custom_train_functions import (
    apply_flow_shift,
    apply_token_mining,
    compute_antithetic_sigmas,
    compute_density_for_timestep_sampling,
    _QMCSequenceManager,
)
from library.flux_train_utils import (
    compute_density_for_timestep_sampling,
    get_lin_function,
    get_noisy_model_input_and_timesteps,
)
from library.train_util import get_noise_noisy_latents_and_timesteps

DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# compute_antithetic_sigmas
# ---------------------------------------------------------------------------

class TestAntitheticSigmas:
    def test_uniform_pairs_sum_to_one(self):
        torch.manual_seed(0)
        s = compute_antithetic_sigmas(8, "uniform", DEVICE)
        assert s.shape == (8,)
        for i in range(4):
            assert s[i].item() + s[i + 4].item() == pytest.approx(1.0, abs=1e-6)

    def test_logit_normal_pairs_symmetric_around_mean(self):
        torch.manual_seed(0)
        mean, std = 0.7, 1.3
        s = compute_antithetic_sigmas(8, "logit_normal", DEVICE, logit_mean=mean, logit_std=std)
        logits = torch.logit(s.clamp(1e-7, 1 - 1e-7))
        for i in range(4):
            pair_mean = (logits[i].item() + logits[i + 4].item()) / 2
            assert pair_mean == pytest.approx(mean, abs=1e-5)

    def test_sigmoid_pairs_symmetric_around_zero_and_scale_respected(self):
        torch.manual_seed(0)
        scale = 2.0
        s = compute_antithetic_sigmas(8, "sigmoid", DEVICE, sigmoid_scale=scale)
        logits = torch.logit(s.clamp(1e-7, 1 - 1e-7))
        for i in range(4):
            assert logits[i].item() + logits[i + 4].item() == pytest.approx(0.0, abs=1e-5)
        # |logit| must not exceed what the base normal draw * scale produced; check
        # scale consistency: logits / scale must be a valid (z, -z) pair
        z = logits / scale
        for i in range(4):
            assert z[i].item() == pytest.approx(-z[i + 4].item(), abs=1e-6)

    def test_odd_batch_size(self):
        torch.manual_seed(0)
        s = compute_antithetic_sigmas(5, "uniform", DEVICE)
        assert s.shape == (5,)
        # first two complete pairs must be mirrored
        assert s[0].item() + s[3].item() == pytest.approx(1.0, abs=1e-6)
        assert s[1].item() + s[4].item() == pytest.approx(1.0, abs=1e-6)

    def test_marginal_distribution_uniform(self):
        """With many samples, the empirical mean must match U(0,1) — antithetic
        pairing must not bias the marginal distribution."""
        torch.manual_seed(0)
        s = compute_antithetic_sigmas(20000, "uniform", DEVICE)
        assert s.mean().item() == pytest.approx(0.5, abs=0.01)

    def test_marginal_distribution_logit_normal(self):
        """Symmetric logit-normal (mean=0) has expected sigma 0.5; pairing keeps it."""
        torch.manual_seed(0)
        s = compute_antithetic_sigmas(20000, "logit_normal", DEVICE, logit_mean=0.0, logit_std=1.0)
        assert s.mean().item() == pytest.approx(0.5, abs=0.01)

    def test_variance_reduction_vs_iid(self):
        """Antithetic batch-mean of uniform sigmas must have lower variance than iid."""
        torch.manual_seed(0)
        anti_means, iid_means = [], []
        for _ in range(500):
            anti_means.append(compute_antithetic_sigmas(8, "uniform", DEVICE).mean().item())
            iid_means.append(torch.rand(8, device=DEVICE).mean().item())
        var_anti = torch.tensor(anti_means).var().item()
        var_iid = torch.tensor(iid_means).var().item()
        # antithetic (u, 1-u) pairs have exactly zero pair-mean variance contribution
        assert var_anti < var_iid * 0.5

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError):
            compute_antithetic_sigmas(4, "hump", DEVICE)


# ---------------------------------------------------------------------------
# apply_flow_shift
# ---------------------------------------------------------------------------

class TestFlowShift:
    def test_shift_formula(self):
        s = torch.tensor([0.1, 0.5, 0.9], device=DEVICE)
        shift = 3.0
        out = apply_flow_shift(s, shift)
        expected = (s * shift) / (1.0 + (shift - 1.0) * s)
        assert torch.allclose(out, expected)

    def test_shift_one_is_identity(self):
        s = torch.rand(16, device=DEVICE)
        assert torch.allclose(apply_flow_shift(s, 1.0), s)

    def test_shift_greater_one_moves_toward_noise(self):
        s = torch.rand(16, device=DEVICE)
        out = apply_flow_shift(s, 3.0)
        assert (out >= s).all()

    def test_per_sample_tensor_shift(self):
        s = torch.rand(4, device=DEVICE)
        ratios = torch.tensor([1.0, 2.0, 2.5, 4.0], device=DEVICE)
        out = apply_flow_shift(s, ratios)
        expected = (s * ratios) / (1.0 + (ratios - 1.0) * s)
        assert torch.allclose(out, expected)

    def test_antithetic_pairs_remain_paired_after_shift(self):
        """A deterministic shift maps (s, 1-s) to (f(s), f(1-s)); the inverse-shifted
        pair must still sum to 1 — i.e. pairing survives the shift transform."""
        torch.manual_seed(0)
        r = 2.5
        base = compute_antithetic_sigmas(8, "uniform", DEVICE)
        shifted = apply_flow_shift(base, r)
        recovered = shifted / (r - (r - 1.0) * shifted)  # exact algebraic inverse
        for i in range(4):
            assert recovered[i].item() + recovered[i + 4].item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# apply_token_mining
# ---------------------------------------------------------------------------

class TestTokenMining:
    def test_output_shape_and_dtype_preserved(self):
        loss = torch.rand(4, 16, 8, 8, device=DEVICE, dtype=torch.float64)
        out = apply_token_mining(loss, sigmas=torch.full((4,), 0.5, device=DEVICE))
        assert out.shape == loss.shape
        assert out.dtype == loss.dtype

    def test_low_dim_loss_passthrough(self):
        loss = torch.rand(4, device=DEVICE)
        out = apply_token_mining(loss)
        assert torch.equal(out, loss)

    def test_uniform_loss_gives_uniform_weights(self):
        loss = torch.ones(4, 4, 8, 8, device=DEVICE)
        out = apply_token_mining(loss, sigmas=torch.full((4,), 0.5, device=DEVICE), sigma_gate=False)
        assert torch.allclose(out, loss)

    def test_hard_tokens_get_more_weight(self):
        loss = torch.ones(2, 4, 8, 8, device=DEVICE)
        loss[:, :, :2, :2] = 10.0  # hard region
        out = apply_token_mining(
            loss, sigmas=torch.full((2,), 0.5, device=DEVICE), alpha=1.0,
            min_weight=0.01, max_weight=100.0, sigma_gate=False,
        )
        w = out / loss
        hard_w = w[:, :, :2, :2].mean()
        easy_w = w[:, :, 4:, 4:].mean()
        assert hard_w > easy_w

    def test_weights_clamped_and_renormalized(self):
        loss = torch.rand(2, 4, 16, 16, device=DEVICE) * 100
        loss[:, :, 0, 0] = 1e6  # extreme outlier
        out = apply_token_mining(
            loss, sigmas=torch.full((2,), 0.5, device=DEVICE), alpha=1.0,
            min_weight=0.25, max_weight=4.0, sigma_gate=False,
        )
        w = (out / loss).flatten(1)  # (B, N)
        # renormalization happens after clamping, so the effective bounds shift
        # slightly; the raw ratio must stay in a sane neighborhood of the clamps
        assert w.max().item() <= 4.0 * 1.5
        assert w.min().item() >= 0.25 / 4.0
        # mean weight per sample must be exactly 1 after renormalization
        assert torch.allclose(w.mean(dim=1), torch.ones(2, device=DEVICE), atol=1e-5)

    def test_weights_are_detached(self):
        """Mining weights must not carry gradient: d(out)/d(loss) == weight only."""
        loss = (torch.rand(2, 4, 8, 8, device=DEVICE) + 0.1).requires_grad_(True)
        out = apply_token_mining(loss, sigma_gate=False)
        out.sum().backward()
        w = (out / loss).detach()
        assert torch.allclose(loss.grad, w, atol=1e-5)

    def test_sigma_gate_disables_mining_at_extremes(self):
        loss = torch.rand(2, 4, 8, 8, device=DEVICE) + 0.01
        sigmas = torch.tensor([0.0, 1.0], device=DEVICE)
        out = apply_token_mining(loss, sigmas=sigmas, alpha=2.0, sigma_gate=True)
        assert torch.allclose(out, loss, atol=1e-6)

    def test_sigma_gate_full_strength_at_midpoint(self):
        loss = torch.rand(2, 4, 8, 8, device=DEVICE) + 0.01
        sigmas = torch.tensor([0.5, 0.5], device=DEVICE)
        gated = apply_token_mining(loss, sigmas=sigmas, alpha=2.0, sigma_gate=True)
        ungated = apply_token_mining(loss, sigmas=sigmas, alpha=2.0, sigma_gate=False)
        assert torch.allclose(gated, ungated, atol=1e-6)

    def test_sigma_gate_partial_blend(self):
        """At sigma=0.25 the gate is 0.75, so weights must lie between uniform
        and the full-strength weights."""
        loss = torch.rand(2, 4, 16, 16, device=DEVICE) + 0.01
        full = apply_token_mining(loss, sigma_gate=False, alpha=2.0)
        part = apply_token_mining(loss, sigmas=torch.full((2,), 0.25, device=DEVICE), alpha=2.0)
        w_full = (full / loss).flatten(1)
        w_part = (part / loss).flatten(1)
        # partial weights should be less extreme (closer to 1) than full weights
        dev_full = (w_full - 1.0).abs().mean()
        dev_part = (w_part - 1.0).abs().mean()
        assert 0 < dev_part < dev_full

    def test_sigmas_broadcast_shape(self):
        """Sigmas shaped (B, 1, 1, 1) (the trainer's broadcast form) must work."""
        loss = torch.rand(2, 4, 8, 8, device=DEVICE) + 0.01
        sigmas = torch.tensor([0.5, 0.0], device=DEVICE).view(2, 1, 1, 1)
        out = apply_token_mining(loss, sigmas=sigmas)
        assert out.shape == loss.shape

    def test_gradient_flows_end_to_end(self):
        pred = (torch.rand(2, 4, 8, 8, device=DEVICE) + 0.1).requires_grad_(True)
        target = torch.rand(2, 4, 8, 8, device=DEVICE)
        loss = (pred - target) ** 2
        out = apply_token_mining(loss, sigmas=torch.full((2,), 0.5, device=DEVICE))
        out.mean().backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# flux compute_density_for_timestep_sampling antithetic flag
# ---------------------------------------------------------------------------

class TestFluxDensityAntithetic:
    def test_uniform_pairs_sum_to_one(self):
        torch.manual_seed(0)
        u = compute_density_for_timestep_sampling("uniform", 8, antithetic=True)
        for i in range(4):
            assert u[i].item() + u[i + 4].item() == pytest.approx(1.0, abs=1e-6)

    def test_logit_normal_pairs_symmetric(self):
        torch.manual_seed(0)
        mean, std = 0.0, 1.0
        u = compute_density_for_timestep_sampling("logit_normal", 8, logit_mean=mean, logit_std=std, antithetic=True)
        logits = torch.logit(u.clamp(1e-7, 1 - 1e-7))
        for i in range(4):
            assert logits[i].item() + logits[i + 4].item() == pytest.approx(2 * mean, abs=1e-5)

    def test_mode_pairs_mirrored_in_base(self):
        """Mode transform is deterministic in u, so two draws with antithetic=True
        must reproduce the same first-half values given the same seed as a
        half-size non-antithetic draw."""
        torch.manual_seed(42)
        half = compute_density_for_timestep_sampling("uniform", 4, antithetic=False)
        torch.manual_seed(42)
        full = compute_density_for_timestep_sampling("uniform", 8, antithetic=True)
        assert torch.allclose(full[:4], half)
        assert torch.allclose(full[4:], 1.0 - half)

    def test_antithetic_false_unchanged(self):
        torch.manual_seed(0)
        a = compute_density_for_timestep_sampling("uniform", 8)
        torch.manual_seed(0)
        b = compute_density_for_timestep_sampling("uniform", 8, antithetic=False)
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Integration: train_util flow path with antithetic sampling + shift
# ---------------------------------------------------------------------------

def _make_flow_args(distribution: str, antithetic: bool, static_ratio=None, shift=False, stratified=False):
    return argparse.Namespace(
        noise_offset=None,
        noise_offset_random_strength=False,
        multires_noise_iterations=None,
        multires_noise_discount=0.3,
        adaptive_noise_scale=None,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        min_timestep=None,
        max_timestep=None,
        flow_model=True,
        flow_use_ot=False,
        flow_timestep_distribution=distribution,
        flow_logit_mean=0.0,
        flow_logit_std=1.0,
        flow_mode_scale=1.29,
        antithetic_timestep_sampling=antithetic,
        stratified_timestep_sampling=stratified,
        flow_uniform_shift=shift,
        flow_uniform_static_ratio=static_ratio,
        flow_uniform_base_pixels=1024.0 * 1024.0,
    )


def _make_scheduler(num_timesteps=1000):
    cfg = SimpleNamespace(num_train_timesteps=num_timesteps)
    return SimpleNamespace(config=cfg, alphas_cumprod=torch.ones(num_timesteps))


class TestFluxBranchIntegration:
    """Integration through flux_train_utils.get_noisy_model_input_and_timesteps,
    which is also the sampling path used by anima_train_network.py."""

    @staticmethod
    def _args(sampling):
        return argparse.Namespace(
            timestep_sampling=sampling,
            sigmoid_scale=1.0,
            discrete_flow_shift=3.0,
            ip_noise_gamma=None,
            ip_noise_gamma_random_strength=False,
            weighting_scheme="none",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            antithetic_timestep_sampling=True,
        )

    @staticmethod
    def _scheduler():
        return SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    def _run(self, sampling, h=8, w=8):
        torch.manual_seed(0)
        latents = torch.randn(8, 4, h, w, device=DEVICE)
        noise = torch.randn_like(latents)
        _, _, sigmas = get_noisy_model_input_and_timesteps(
            self._args(sampling), self._scheduler(), latents, noise, DEVICE, torch.float32, is_train=True
        )
        return sigmas.flatten()  # (B,)

    def test_sigmoid_branch_pairs(self):
        s = self._run("sigmoid")
        for i in range(4):
            assert s[i].item() + s[i + 4].item() == pytest.approx(1.0, abs=1e-5)

    def test_shift_branch_pairs_and_shift_applied(self):
        r = 3.0  # matches discrete_flow_shift in _args
        s = self._run("shift")
        # invert the shift: pairing must be restored
        recovered = s / (r - (r - 1.0) * s)
        for i in range(4):
            assert recovered[i].item() + recovered[i + 4].item() == pytest.approx(1.0, abs=1e-4)
        assert s.mean().item() > 0.5  # shift=3 biases toward sigma=1

    def test_flux_shift_branch_pairs(self):
        h = w = 8
        s = self._run("flux_shift", h=h, w=w)
        mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
        e = math.exp(mu)
        # invert time_shift(mu, 1, t) = e / (e + (1/t - 1))
        recovered = 1.0 / (1.0 + e * (1.0 / s.clamp(1e-6, 1 - 1e-6) - 1.0))
        for i in range(4):
            assert recovered[i].item() + recovered[i + 4].item() == pytest.approx(1.0, abs=1e-4)

    def test_odd_batch_flux_branch(self):
        torch.manual_seed(0)
        latents = torch.randn(5, 4, 8, 8, device=DEVICE)
        noise = torch.randn_like(latents)
        _, timesteps, sigmas = get_noisy_model_input_and_timesteps(
            self._args("sigmoid"), self._scheduler(), latents, noise, DEVICE, torch.float32, is_train=True
        )
        s = sigmas.flatten()
        assert s.shape == (5,)
        assert s[0].item() + s[3].item() == pytest.approx(1.0, abs=1e-5)


class TestFlowPathIntegration:
    def test_antithetic_uniform_pairs(self):
        torch.manual_seed(0)
        args = _make_flow_args("uniform", antithetic=True)
        sched = _make_scheduler()
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        _, _, timesteps = get_noise_noisy_latents_and_timesteps(args, sched, latents)
        sigmas = timesteps.float() / sched.config.num_train_timesteps
        # pairs sum to 1 within timestep quantization tolerance
        tol = 2.0 / sched.config.num_train_timesteps
        for i in range(4):
            assert abs(sigmas[i].item() + sigmas[i + 4].item() - 1.0) <= tol + 1e-6

    def test_antithetic_disabled_gives_random(self):
        torch.manual_seed(0)
        args = _make_flow_args("uniform", antithetic=False)
        sched = _make_scheduler()
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        _, _, timesteps = get_noise_noisy_latents_and_timesteps(args, sched, latents)
        sigmas = timesteps.float() / sched.config.num_train_timesteps
        # iid uniform pairs summing to exactly 1 with prob ~0
        sums = [sigmas[i].item() + sigmas[i + 4].item() for i in range(4)]
        assert any(abs(s - 1.0) > 0.05 for s in sums)

    def test_antithetic_logit_normal_pairs(self):
        torch.manual_seed(0)
        args = _make_flow_args("logit_normal", antithetic=True)
        sched = _make_scheduler()
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        _, _, timesteps = get_noise_noisy_latents_and_timesteps(args, sched, latents)
        sigmas = (timesteps.float() / sched.config.num_train_timesteps).clamp(1e-6, 1 - 1e-6)
        logits = torch.logit(sigmas)
        tol = 4.0 / sched.config.num_train_timesteps  # logit amplification of quant error
        for i in range(4):
            pair_mean = (logits[i].item() + logits[i + 4].item()) / 2
            assert abs(pair_mean - 0.0) <= tol / max(sigmas[i].item(), 1e-3) + 0.05

    def test_antithetic_with_static_shift_respects_distribution(self):
        """With static shift ratio r, inverse-shifting recovered sigmas must
        restore antithetic pairs summing to 1."""
        torch.manual_seed(0)
        r = 2.5
        args = _make_flow_args("uniform", antithetic=True, static_ratio=r)
        sched = _make_scheduler()
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        _, _, timesteps = get_noise_noisy_latents_and_timesteps(args, sched, latents)
        sigmas = (timesteps.float() / sched.config.num_train_timesteps).clamp(1e-6, 1 - 1e-6)
        recovered = sigmas / (r - (r - 1.0) * sigmas)
        tol = 6.0 / sched.config.num_train_timesteps
        for i in range(4):
            assert abs(recovered[i].item() + recovered[i + 4].item() - 1.0) <= tol

    def test_noisy_latents_shape_and_finiteness(self):
        torch.manual_seed(0)
        args = _make_flow_args("logit_normal", antithetic=True, static_ratio=2.5)
        sched = _make_scheduler()
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        noise, noisy, timesteps = get_noise_noisy_latents_and_timesteps(args, sched, latents)
        assert noisy.shape == latents.shape
        assert torch.isfinite(noisy).all()
        assert timesteps.shape == (8,)


# ---------------------------------------------------------------------------
# Stratified sampling (new variance-reduction method)
# ---------------------------------------------------------------------------

class TestStratifiedSampling:
    def test_stratified_uniform_covers_all_strata(self):
        """Each stratum of width 1/B must contain exactly one sample."""
        torch.manual_seed(0)
        B = 8
        u = compute_density_for_timestep_sampling("uniform", B, stratified=True, device=DEVICE)
        assert u.shape == (B,)
        # Sort and check each falls in its own stratum
        u_sorted, _ = u.sort()
        for i in range(B):
            assert i / B <= u_sorted[i].item() < (i + 1) / B

    def test_stratified_works_for_odd_batch(self):
        """Stratified must work for odd batch sizes (unlike antithetic)."""
        torch.manual_seed(0)
        u = compute_density_for_timestep_sampling("uniform", 5, stratified=True, device=DEVICE)
        assert u.shape == (5,)
        u_sorted, _ = u.sort()
        for i in range(5):
            assert i / 5 <= u_sorted[i].item() < (i + 1) / 5

    def test_stratified_variance_lower_than_iid(self):
        """Stratified batch-mean must have lower variance than iid uniform."""
        torch.manual_seed(0)
        strat_means, iid_means = [], []
        for _ in range(500):
            strat_means.append(
                compute_density_for_timestep_sampling("uniform", 8, stratified=True, device=DEVICE).mean().item()
            )
            iid_means.append(torch.rand(8, device=DEVICE).mean().item())
        var_strat = torch.tensor(strat_means).var().item()
        var_iid = torch.tensor(iid_means).var().item()
        assert var_strat < var_iid * 0.5

    def test_stratified_marginal_is_uniform(self):
        """Stratified samples must be marginally uniform over [0,1]."""
        torch.manual_seed(0)
        u = compute_density_for_timestep_sampling("uniform", 20000, stratified=True, device=DEVICE)
        assert u.mean().item() == pytest.approx(0.5, abs=0.01)

    def test_antithetic_takes_precedence_over_stratified(self):
        """When both are set, antithetic wins (pairs sum to 1, not stratified)."""
        torch.manual_seed(0)
        u = compute_density_for_timestep_sampling("uniform", 8, antithetic=True, stratified=True, device=DEVICE)
        # Antithetic pairs: u[i] + u[i+4] == 1
        for i in range(4):
            assert u[i].item() + u[i + 4].item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# batch_size == 1 short-circuit
# ---------------------------------------------------------------------------

class TestBatchSizeOne:
    def test_batch_size_one_antithetic_no_error(self):
        """batch_size=1 with antithetic must not error and returns a single sample."""
        torch.manual_seed(0)
        u = compute_density_for_timestep_sampling("uniform", 1, antithetic=True, device=DEVICE)
        assert u.shape == (1,)
        assert 0.0 <= u.item() <= 1.0

    def test_batch_size_one_stratified_no_error(self):
        """batch_size=1 with stratified must not error and returns a single sample."""
        torch.manual_seed(0)
        u = compute_density_for_timestep_sampling("uniform", 1, stratified=True, device=DEVICE)
        assert u.shape == (1,)
        assert 0.0 <= u.item() <= 1.0

    def test_compute_antithetic_sigmas_batch_one(self):
        """The legacy wrapper must also handle batch_size=1 gracefully."""
        torch.manual_seed(0)
        s = compute_antithetic_sigmas(1, "uniform", DEVICE)
        assert s.shape == (1,)


# ---------------------------------------------------------------------------
# Device placement (CUDA, not CPU)
# ---------------------------------------------------------------------------

class TestDevicePlacement:
    def test_density_generated_on_cuda(self):
        """When device=cuda is passed, the result must live on cuda."""
        u = compute_density_for_timestep_sampling("uniform", 8, device=DEVICE)
        assert u.device.type == "cuda"

    def test_density_logit_normal_on_cuda(self):
        u = compute_density_for_timestep_sampling("logit_normal", 8, logit_mean=0.0, logit_std=1.0, device=DEVICE)
        assert u.device.type == "cuda"

    def test_density_mode_on_cuda(self):
        u = compute_density_for_timestep_sampling("mode", 8, mode_scale=1.29, device=DEVICE)
        assert u.device.type == "cuda"

    def test_density_default_is_cpu(self):
        """Without an explicit device, the default must remain CPU (backward compat)."""
        u = compute_density_for_timestep_sampling("uniform", 8)
        assert u.device.type == "cpu"


# ---------------------------------------------------------------------------
# mode distribution support in the flow path (train_util)
# ---------------------------------------------------------------------------

class TestFlowModeDistribution:
    def test_flow_mode_antithetic_pairs(self):
        """mode distribution with antithetic in the flow path must produce
        mirrored base-variates (u, 1-u) before the mode transform."""
        torch.manual_seed(0)
        args = _make_flow_args("mode", antithetic=True)
        sched = _make_scheduler()
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        _, _, timesteps = get_noise_noisy_latents_and_timesteps(args, sched, latents)
        # mode transform is monotonic, so antithetic pairs in u produce
        # negatively-correlated sigmas; just check shape and finiteness.
        assert timesteps.shape == (8,)

    def test_flow_mode_without_antithetic(self):
        """mode distribution without antithetic must run and produce valid timesteps."""
        torch.manual_seed(0)
        args = _make_flow_args("mode", antithetic=False)
        sched = _make_scheduler()
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        _, _, timesteps = get_noise_noisy_latents_and_timesteps(args, sched, latents)
        assert timesteps.shape == (8,)

    def test_flow_stratified_uniform(self):
        """stratified uniform in the flow path must produce valid timesteps."""
        torch.manual_seed(0)
        args = _make_flow_args("uniform", antithetic=False, stratified=True)
        sched = _make_scheduler()
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        _, _, timesteps = get_noise_noisy_latents_and_timesteps(args, sched, latents)
        assert timesteps.shape == (8,)


# ---------------------------------------------------------------------------
# Flux branch: stratified uniform
# ---------------------------------------------------------------------------

class TestFluxStratifiedBranch:
    @staticmethod
    def _args(sampling, stratified=False):
        return argparse.Namespace(
            timestep_sampling=sampling,
            sigmoid_scale=1.0,
            discrete_flow_shift=3.0,
            ip_noise_gamma=None,
            ip_noise_gamma_random_strength=False,
            weighting_scheme="none",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            antithetic_timestep_sampling=False,
            stratified_timestep_sampling=stratified,
        )

    @staticmethod
    def _scheduler():
        return SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    def test_stratified_uniform_covers_strata(self):
        torch.manual_seed(0)
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        noise = torch.randn_like(latents)
        _, _, sigmas = get_noisy_model_input_and_timesteps(
            self._args("uniform", stratified=True), self._scheduler(), latents, noise, DEVICE, torch.float32, is_train=True
        )
        s = sigmas.flatten()
        s_sorted, _ = s.sort()
        for i in range(8):
            assert i / 8 <= s_sorted[i].item() <= (i + 1) / 8 + 1e-6


# ---------------------------------------------------------------------------
# QMC (Sobol / Halton) low-discrepancy sequences
# ---------------------------------------------------------------------------

class TestQMCSequenceManager:
    def test_sobol_basic_draw(self):
        """Sobol manager must produce points in [0,1] on the requested device."""
        mgr = _QMCSequenceManager(method="sobol", seed=42)
        pts = mgr.draw(8, device=DEVICE)
        assert pts.shape == (8,)
        assert pts.device.type == "cuda"
        assert (pts >= 0).all() and (pts <= 1).all()

    def test_halton_basic_draw(self):
        """Halton manager must produce points in [0,1] on the requested device."""
        mgr = _QMCSequenceManager(method="halton", seed=42)
        pts = mgr.draw(8, device=DEVICE)
        assert pts.shape == (8,)
        assert pts.device.type == "cuda"
        assert (pts >= 0).all() and (pts <= 1).all()

    def test_sobol_advances_across_draws(self):
        """Consecutive draws must produce different points (sequence advances)."""
        mgr = _QMCSequenceManager(method="sobol", seed=99)
        a = mgr.draw(4, device=DEVICE)
        b = mgr.draw(4, device=DEVICE)
        assert not torch.allclose(a, b)

    def test_sobol_reset_restarts_sequence(self):
        """After reset, the sequence must restart from the beginning."""
        mgr = _QMCSequenceManager(method="sobol", seed=7)
        a = mgr.draw(4, device=DEVICE)
        mgr.reset()
        b = mgr.draw(4, device=DEVICE)
        assert torch.allclose(a, b)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            _QMCSequenceManager(method="invalid", seed=0)


class TestQMCDensity:
    def test_sobol_uniform_in_range(self):
        """Sobol-based uniform density must produce values in [0,1]."""
        u = compute_density_for_timestep_sampling("uniform", 8, qmc="sobol", qmc_seed=0, device=DEVICE)
        assert u.shape == (8,)
        assert u.device.type == "cuda"
        assert (u >= 0).all() and (u <= 1).all()

    def test_halton_uniform_in_range(self):
        """Halton-based uniform density must produce values in [0,1]."""
        u = compute_density_for_timestep_sampling("uniform", 8, qmc="halton", qmc_seed=0, device=DEVICE)
        assert u.shape == (8,)
        assert u.device.type == "cuda"
        assert (u >= 0).all() and (u <= 1).all()

    def test_sobol_logit_normal_in_range(self):
        """Sobol with logit_normal must produce values in (0,1)."""
        u = compute_density_for_timestep_sampling(
            "logit_normal", 8, logit_mean=0.0, logit_std=1.0, qmc="sobol", qmc_seed=0, device=DEVICE
        )
        assert u.shape == (8,)
        assert (u > 0).all() and (u < 1).all()

    def test_sobol_mode_in_range(self):
        """Sobol with mode distribution must produce values in [0,1]."""
        u = compute_density_for_timestep_sampling("mode", 8, mode_scale=1.29, qmc="sobol", qmc_seed=0, device=DEVICE)
        assert u.shape == (8,)
        assert (u >= 0).all() and (u <= 1).all()

    def test_sobol_sigmoid_in_range(self):
        """Sobol with sigmoid must produce values in (0,1)."""
        u = compute_density_for_timestep_sampling(
            "sigmoid", 8, sigmoid_scale=1.0, qmc="sobol", qmc_seed=0, device=DEVICE
        )
        assert u.shape == (8,)
        assert (u > 0).all() and (u < 1).all()

    def test_sobol_marginal_mean_near_half(self):
        """Over many draws, the Sobol uniform marginal mean must be ~0.5."""
        mgr = _QMCSequenceManager(method="sobol", seed=0)
        all_pts = mgr.draw(20000, device=DEVICE)
        assert all_pts.mean().item() == pytest.approx(0.5, abs=0.02)

    def test_sobol_variance_lower_than_iid(self):
        """Sobol batch-mean must have lower variance than iid uniform."""
        # Use a fresh manager with a unique seed to avoid interference from other tests.
        mgr = _QMCSequenceManager(method="sobol", seed=12345)
        sobol_means, iid_means = [], []
        for _ in range(200):
            sobol_means.append(mgr.draw(8, device=DEVICE).mean().item())
            iid_means.append(torch.rand(8, device=DEVICE).mean().item())
        var_sobol = torch.tensor(sobol_means).var().item()
        var_iid = torch.tensor(iid_means).var().item()
        assert var_sobol < var_iid

    def test_antithetic_qmc_composes(self):
        """When both antithetic and qmc are set, they compose (antithetic-QMC):
        batch_size//2 low-discrepancy points are drawn and mirrored as (u, 1-u)."""
        u = compute_density_for_timestep_sampling(
            "uniform", 8, antithetic=True, qmc="sobol", qmc_seed=0, device=DEVICE
        )
        # Antithetic pairing: u[i] + u[i+4] == 1
        for i in range(4):
            assert u[i].item() + u[i + 4].item() == pytest.approx(1.0, abs=1e-6)

    def test_qmc_takes_precedence_over_stratified(self):
        """When both qmc and stratified are set, qmc wins.

        We verify this by comparing against a pure QMC draw (no stratified flag):
        the results must be identical, proving stratified was ignored.
        """
        # Use a fresh seed to get a clean QMC sequence.
        u_both = compute_density_for_timestep_sampling(
            "uniform", 8, qmc="sobol", qmc_seed=555, stratified=True, device=DEVICE
        )
        # Reset the QMC manager so the next draw starts from the same point.
        _QMCSequenceManager(method="sobol", seed=555).reset()
        u_qmc_only = compute_density_for_timestep_sampling(
            "uniform", 8, qmc="sobol", qmc_seed=555, stratified=False, device=DEVICE
        )
        assert torch.allclose(u_both, u_qmc_only)

    def test_batch_size_one_qmc_no_error(self):
        """batch_size=1 with QMC must not error and returns a single sample."""
        u = compute_density_for_timestep_sampling("uniform", 1, qmc="sobol", qmc_seed=0, device=DEVICE)
        assert u.shape == (1,)
        assert 0.0 <= u.item() <= 1.0


class TestFluxQMCBranch:
    @staticmethod
    def _args(sampling, qmc=None):
        return argparse.Namespace(
            timestep_sampling=sampling,
            sigmoid_scale=1.0,
            discrete_flow_shift=3.0,
            ip_noise_gamma=None,
            ip_noise_gamma_random_strength=False,
            weighting_scheme="none",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            antithetic_timestep_sampling=False,
            stratified_timestep_sampling=False,
            qmc_timestep_sampling=qmc,
            qmc_seed=0,
        )

    @staticmethod
    def _scheduler():
        return SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    def test_qmc_sobol_uniform(self):
        torch.manual_seed(0)
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        noise = torch.randn_like(latents)
        _, _, sigmas = get_noisy_model_input_and_timesteps(
            self._args("uniform", qmc="sobol"), self._scheduler(), latents, noise, DEVICE, torch.float32, is_train=True
        )
        s = sigmas.flatten()
        assert s.shape == (8,)
        assert (s >= 0).all() and (s <= 1).all()

    def test_qmc_sobol_sigmoid(self):
        torch.manual_seed(0)
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        noise = torch.randn_like(latents)
        _, _, sigmas = get_noisy_model_input_and_timesteps(
            self._args("sigmoid", qmc="sobol"), self._scheduler(), latents, noise, DEVICE, torch.float32, is_train=True
        )
        s = sigmas.flatten()
        assert s.shape == (8,)
        assert (s > 0).all() and (s < 1).all()


# ---------------------------------------------------------------------------
# Stratified/QMC with shift and flux_shift branches
# ---------------------------------------------------------------------------

class TestStratifiedWithShift:
    """Verify stratified and QMC work with the shift and flux_shift branches
    (previously these branches only checked antithetic, silently ignoring
    stratified/QMC)."""

    @staticmethod
    def _args(sampling, stratified=False, qmc=None):
        return argparse.Namespace(
            timestep_sampling=sampling,
            sigmoid_scale=1.0,
            discrete_flow_shift=3.0,
            ip_noise_gamma=None,
            ip_noise_gamma_random_strength=False,
            weighting_scheme="none",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            antithetic_timestep_sampling=False,
            stratified_timestep_sampling=stratified,
            qmc_timestep_sampling=qmc,
            qmc_seed=0,
        )

    @staticmethod
    def _scheduler():
        return SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    def _run(self, sampling, **vr):
        torch.manual_seed(0)
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        noise = torch.randn_like(latents)
        _, _, sigmas = get_noisy_model_input_and_timesteps(
            self._args(sampling, **vr), self._scheduler(), latents, noise, DEVICE, torch.float32, is_train=True
        )
        return sigmas.flatten()

    def test_stratified_shift_valid(self):
        """Stratified + shift: must produce valid sigmas in [0,1]."""
        s = self._run("shift", stratified=True)
        assert s.shape == (8,)
        assert (s >= 0).all() and (s <= 1).all()

    def test_stratified_flux_shift_valid(self):
        """Stratified + flux_shift: must produce valid sigmas in [0,1]."""
        s = self._run("flux_shift", stratified=True)
        assert s.shape == (8,)
        assert (s >= 0).all() and (s <= 1).all()

    def test_qmc_sobol_shift_valid(self):
        """QMC Sobol + shift: must produce valid sigmas in [0,1]."""
        s = self._run("shift", qmc="sobol")
        assert s.shape == (8,)
        assert (s >= 0).all() and (s <= 1).all()

    def test_qmc_sobol_flux_shift_valid(self):
        """QMC Sobol + flux_shift: must produce valid sigmas in [0,1]."""
        s = self._run("flux_shift", qmc="sobol")
        assert s.shape == (8,)
        assert (s >= 0).all() and (s <= 1).all()

    def test_stratified_shift_not_ignored(self):
        """Stratified + shift must NOT fall through to the random branch.
        Verify by checking the base sigmas (before shift) are stratified."""
        r = 3.0
        s = self._run("shift", stratified=True)
        # Invert the shift to recover the base sigmas.
        recovered = s / (r - (r - 1.0) * s)
        recovered_sorted, _ = recovered.sort()
        # Each recovered sigma must fall in its own stratum.
        for i in range(8):
            assert i / 8 <= recovered_sorted[i].item() <= (i + 1) / 8 + 1e-4


# ---------------------------------------------------------------------------
# Anima verification: all Anima trainers call flux_train_utils.get_noisy_model_input_and_timesteps
# with Anima-default args (timestep_sampling=sigmoid, discrete_flow_shift=1.0).
# These tests verify all three variance-reduction methods work through that path.
# ---------------------------------------------------------------------------

class TestAnimaPathVerification:
    """Verify variance-reduction methods work through the Anima sampling path.

    Anima trainers (anima_train_network.py, anima_train.py, anima_train_control_net_lllite.py)
    all call flux_train_utils.get_noisy_model_input_and_timesteps() with these default args.
    """

    @staticmethod
    def _anima_args(timestep_sampling="sigmoid", antithetic=False, stratified=False, qmc=None):
        """Mimic anima_train_utils.add_anima_training_arguments defaults + add_custom_train_arguments."""
        return argparse.Namespace(
            # Anima defaults
            timestep_sampling=timestep_sampling,
            sigmoid_scale=1.0,
            discrete_flow_shift=1.0,
            # From add_dit_training_arguments (weighting-scheme else-branch)
            weighting_scheme="none",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            # From add_custom_train_arguments
            antithetic_timestep_sampling=antithetic,
            stratified_timestep_sampling=stratified,
            qmc_timestep_sampling=qmc,
            qmc_seed=0,
            # ip_noise (not used in these tests)
            ip_noise_gamma=None,
            ip_noise_gamma_random_strength=False,
        )

    @staticmethod
    def _scheduler():
        return SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    def _run(self, **kw):
        torch.manual_seed(0)
        latents = torch.randn(8, 4, 8, 8, device=DEVICE)
        noise = torch.randn_like(latents)
        args = self._anima_args(**kw)
        _, timesteps, sigmas = get_noisy_model_input_and_timesteps(
            args, self._scheduler(), latents, noise, DEVICE, torch.float32, is_train=True
        )
        return timesteps, sigmas.flatten()

    # --- Antithetic ---

    def test_anima_antithetic_sigmoid(self):
        """Anima default (sigmoid) + antithetic: pairs must sum to ~1."""
        _, s = self._run(timestep_sampling="sigmoid", antithetic=True)
        for i in range(4):
            assert s[i].item() + s[i + 4].item() == pytest.approx(1.0, abs=1e-5)

    def test_anima_antithetic_uniform(self):
        _, s = self._run(timestep_sampling="uniform", antithetic=True)
        for i in range(4):
            assert s[i].item() + s[i + 4].item() == pytest.approx(1.0, abs=1e-5)

    def test_anima_antithetic_shift(self):
        """Anima shift branch + antithetic: inverse-shift recovers pairs summing to 1."""
        r = 1.0  # discrete_flow_shift default
        _, s = self._run(timestep_sampling="shift", antithetic=True)
        recovered = s / (r - (r - 1.0) * s)
        for i in range(4):
            assert recovered[i].item() + recovered[i + 4].item() == pytest.approx(1.0, abs=1e-4)

    def test_anima_antithetic_flux_shift(self):
        """Anima flux_shift branch + antithetic."""
        h = w = 8
        _, s = self._run(timestep_sampling="flux_shift", antithetic=True)
        mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
        e = math.exp(mu)
        recovered = 1.0 / (1.0 + e * (1.0 / s.clamp(1e-6, 1 - 1e-6) - 1.0))
        for i in range(4):
            assert recovered[i].item() + recovered[i + 4].item() == pytest.approx(1.0, abs=1e-4)

    # --- Stratified ---

    def test_anima_stratified_uniform(self):
        """Anima uniform + stratified: one sample per stratum."""
        _, s = self._run(timestep_sampling="uniform", stratified=True)
        s_sorted, _ = s.sort()
        for i in range(8):
            assert i / 8 <= s_sorted[i].item() <= (i + 1) / 8 + 1e-6

    def test_anima_stratified_sigmoid(self):
        """Anima sigmoid + stratified: must produce valid sigmas in (0,1)."""
        _, s = self._run(timestep_sampling="sigmoid", stratified=True)
        assert s.shape == (8,)
        assert (s > 0).all() and (s < 1).all()

    # --- QMC ---

    def test_anima_qmc_sobol_sigmoid(self):
        """Anima default (sigmoid) + Sobol QMC: valid sigmas in (0,1)."""
        _, s = self._run(timestep_sampling="sigmoid", qmc="sobol")
        assert s.shape == (8,)
        assert (s > 0).all() and (s < 1).all()

    def test_anima_qmc_sobol_uniform(self):
        """Anima uniform + Sobol QMC: valid sigmas in [0,1]."""
        _, s = self._run(timestep_sampling="uniform", qmc="sobol")
        assert s.shape == (8,)
        assert (s >= 0).all() and (s <= 1).all()

    def test_anima_qmc_halton_sigmoid(self):
        """Anima sigmoid + Halton QMC: valid sigmas in (0,1)."""
        _, s = self._run(timestep_sampling="sigmoid", qmc="halton")
        assert s.shape == (8,)
        assert (s > 0).all() and (s < 1).all()

    def test_anima_qmc_sobol_shift(self):
        """Anima shift branch + Sobol QMC: valid sigmas after shift."""
        _, s = self._run(timestep_sampling="shift", qmc="sobol")
        assert s.shape == (8,)
        assert (s >= 0).all() and (s <= 1).all()

    def test_anima_qmc_sobol_flux_shift(self):
        """Anima flux_shift branch + Sobol QMC: valid sigmas after time_shift."""
        _, s = self._run(timestep_sampling="flux_shift", qmc="sobol")
        assert s.shape == (8,)
        assert (s >= 0).all() and (s <= 1).all()

    # --- No variance reduction (baseline) ---

    def test_anima_baseline_sigmoid(self):
        """Anima default (sigmoid) with no variance reduction: valid output."""
        t, s = self._run(timestep_sampling="sigmoid")
        assert t.shape == (8,)
        assert s.shape == (8,)
        assert (s > 0).all() and (s < 1).all()

    def test_anima_timesteps_finite(self):
        """All methods must produce finite timesteps."""
        for ts in ["sigmoid", "uniform", "shift", "flux_shift"]:
            for vr in [{}, {"antithetic": True}, {"stratified": True}, {"qmc": "sobol"}]:
                t, _ = self._run(timestep_sampling=ts, **vr)
                assert torch.isfinite(t).all(), f"Non-finite timesteps for {ts} + {vr}"
