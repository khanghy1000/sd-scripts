"""Tests for the multiscale MSE x0-prediction ("anchor") loss (library/anchor_loss.py).

Covers the invariants of plans/multiscale-mse-x0-pred-loss.md §6 plus the x0
extraction hop, the optional SNR gate, config validation, and the trainer
integration surface. All tensor tests run on CUDA.
"""

import math
import types

import pytest
import torch

from library import anchor_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tests require a GPU")

DEVICE = torch.device("cuda")


@pytest.fixture(autouse=True)
def _clear_caches():
    anchor_loss._KERNEL_CACHE.clear()
    anchor_loss._BAND_ENERGY_CACHE.clear()
    yield
    anchor_loss._KERNEL_CACHE.clear()
    anchor_loss._BAND_ENERGY_CACHE.clear()


def _make_inputs(batch=2, channels=3, height=64, width=64, dtype=torch.float32, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    clean = torch.randn(batch, channels, height, width, generator=generator, dtype=torch.float32)
    clean = clean.to(device=DEVICE, dtype=dtype)
    return clean


# ---------------------------------------------------------------------------
# Pyramid primitives
# ---------------------------------------------------------------------------


def test_sigma_zero_blur_is_exact_identity():
    x = _make_inputs(dtype=torch.bfloat16)
    out = anchor_loss._gaussian_blur(x, 0.0)
    assert out is x  # short-circuit: bit-identical, same object


def test_gaussian_kernel_is_dc_preserving():
    kernel = anchor_loss._gaussian_kernel(anchor_loss.PYRAMID_BLUR_SIGMA, DEVICE, torch.float32)
    assert kernel.shape[0] == 7  # radius = ceil(3 * 1.0) = 3
    assert abs(kernel.sum().item() - 1.0) < 1e-6
    # binomial [1,4,6,4,1]/16 shape: center is the max, symmetric
    assert kernel.argmax() == 3
    assert torch.allclose(kernel, kernel.flip(0))


def test_pyramid_linearity():
    a, b = 1.7, -0.6
    x = _make_inputs(seed=1)
    y = _make_inputs(seed=2)
    pyr_xy = anchor_loss._laplacian_pyramid(a * x + b * y, 3)
    pyr_x = anchor_loss._laplacian_pyramid(x, 3)
    pyr_y = anchor_loss._laplacian_pyramid(y, 3)
    for band_xy, band_x, band_y in zip(pyr_xy, pyr_x, pyr_y):
        expected = a * band_x + b * band_y
        torch.testing.assert_close(band_xy, expected, rtol=1e-4, atol=1e-5)


def test_one_pyramid_on_delta_identity():
    pred = _make_inputs(seed=3)
    clean = _make_inputs(seed=4)
    delta_bands = anchor_loss._laplacian_pyramid(pred - clean, 3)
    pred_bands = anchor_loss._laplacian_pyramid(pred, 3)
    clean_bands = anchor_loss._laplacian_pyramid(clean, 3)
    for band_d, band_p, band_c in zip(delta_bands, pred_bands, clean_bands):
        torch.testing.assert_close(band_d, band_p - band_c, rtol=1e-3, atol=1e-4)


def test_effective_levels_ladder():
    # reflect padding needs min(dim) >= 4 per blurred grid
    assert anchor_loss._effective_levels(64, 64, 4) == 4
    assert anchor_loss._effective_levels(64, 64, 8) == 5  # 64->32->16->8->4, then 2 < 4
    assert anchor_loss._effective_levels(8, 4, 4) == 1  # second grid would be 2 px tall
    assert anchor_loss._effective_levels(4, 4, 4) == 1
    assert anchor_loss._effective_levels(2, 2, 4) == 0  # residual band only
    assert anchor_loss._effective_levels(64, 32, 2) == 2


def test_band_energy_cache_keying():
    e1 = anchor_loss._band_energies(32, 32, 3, DEVICE, torch.float32)
    e2 = anchor_loss._band_energies(32, 32, 3, DEVICE, torch.float32)
    assert e1 is e2  # cached
    e3 = anchor_loss._band_energies(32, 32, 3, DEVICE, torch.bfloat16)
    assert e3 is not e1
    e4 = anchor_loss._band_energies(16, 32, 3, DEVICE, torch.float32)
    assert e4 is not e1
    for e in e1:
        assert e.dtype == torch.float32
        assert e.device.type == DEVICE.type
        assert e.dim() == 0
        assert torch.isfinite(e) and e.item() > 0


# ---------------------------------------------------------------------------
# Per-sample loss properties
# ---------------------------------------------------------------------------


def test_frequency_evenness_on_white_noise():
    """For unit-variance white-noise deltas, every whitened band MSE ≈ 1 (§6.5)."""
    h = w = 128
    levels = 4
    eff = anchor_loss._effective_levels(h, w, levels)
    white = torch.randn(1, 1, h, w, dtype=torch.float32, device=DEVICE)
    pyramid = anchor_loss._laplacian_pyramid(white, eff)
    energies = anchor_loss._band_energies(h, w, eff, DEVICE, torch.float32)
    whitened = [band.square().mean().item() / e.item() for band, e in zip(pyramid, energies)]
    assert len(whitened) == eff + 1
    for k, value in enumerate(whitened):
        assert 0.5 < value < 2.0, f"band {k} whitened MSE {value} not ~1"


def test_zero_delta_gives_zero_loss():
    clean = _make_inputs(dtype=torch.bfloat16)
    loss = anchor_loss.anchor_per_sample_loss(clean.clone(), clean, levels=4)
    assert loss.shape == (clean.shape[0],)
    assert loss.dtype == torch.float32
    assert loss.abs().max().item() == 0.0


def test_rng_neutrality():
    """Enabling the term must not advance the global RNG stream (§6.4)."""
    clean = _make_inputs(dtype=torch.bfloat16)
    pred = clean + torch.randn_like(clean).to(clean.dtype)

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()

    loss = anchor_loss.anchor_per_sample_loss(pred, clean, levels=4)
    assert torch.isfinite(loss).all()

    assert torch.equal(cpu_state, torch.get_rng_state())
    assert torch.equal(cuda_state, torch.cuda.get_rng_state())


def test_gradient_flows_only_into_prediction():
    clean = _make_inputs(dtype=torch.bfloat16)
    pred = (clean + 0.1 * torch.randn_like(clean).to(clean.dtype)).requires_grad_(True)

    loss = anchor_loss.anchor_per_sample_loss(pred, clean, levels=4)
    loss.sum().backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0
    assert not clean.requires_grad


def test_small_grids_are_graceful():
    for h, w in [(4, 4), (8, 4), (2, 2), (6, 6)]:
        clean = _make_inputs(height=h, width=w, dtype=torch.bfloat16)
        pred = clean + 0.1
        loss = anchor_loss.anchor_per_sample_loss(pred, clean, levels=4)
        assert loss.shape == (clean.shape[0],)
        assert torch.isfinite(loss).all()


def test_bf16_and_fp32_agree():
    clean32 = _make_inputs(seed=7, dtype=torch.float32)
    pred32 = clean32 + 0.1 * torch.randn_like(clean32)
    loss32 = anchor_loss.anchor_per_sample_loss(pred32, clean32, levels=4)

    clean16 = clean32.to(torch.bfloat16)
    pred16 = pred32.to(torch.bfloat16)
    loss16 = anchor_loss.anchor_per_sample_loss(pred16, clean16, levels=4)

    torch.testing.assert_close(loss16, loss32, rtol=5e-2, atol=5e-3)


# ---------------------------------------------------------------------------
# x0 extraction hop (via hf_token_loss.hf_x0_hat)
# ---------------------------------------------------------------------------


def test_x0_extraction_flow_sigma_timesteps():
    clean = _make_inputs(seed=8)
    noise = _make_inputs(seed=9)
    t = torch.tensor([0.25, 0.8], device=DEVICE)
    noisy = t.view(-1, 1, 1, 1) * noise + (1 - t.view(-1, 1, 1, 1)) * clean
    v = noise - clean  # flow-matching velocity target
    x0 = hf_x0_mode(v, noisy, t, "flow", timesteps_in_sigma=True)
    torch.testing.assert_close(x0, clean.double(), rtol=1e-4, atol=1e-4)


def hf_x0_mode(pred, noisy, timesteps, mode, timesteps_in_sigma, scheduler=None):
    from library.hf_token_loss import hf_x0_hat

    return hf_x0_hat(
        pred, noisy, timesteps, mode, noise_scheduler=scheduler, timesteps_in_sigma=timesteps_in_sigma
    )


def test_x0_extraction_flow_discrete_timesteps():
    from diffusers import DDPMScheduler

    scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    clean = _make_inputs(seed=10)
    noise = _make_inputs(seed=11)
    t = torch.tensor([250, 800], device=DEVICE)
    sigma = t.float() / 1000.0
    noisy = sigma.view(-1, 1, 1, 1) * noise + (1 - sigma.view(-1, 1, 1, 1)) * clean
    v = noise - clean
    x0 = hf_x0_mode(v, noisy, t, "flow", timesteps_in_sigma=False, scheduler=scheduler)
    torch.testing.assert_close(x0, clean.double(), rtol=1e-4, atol=1e-4)


def test_x0_extraction_ddpm_eps_and_vpred():
    from diffusers import DDPMScheduler

    scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    acp = scheduler.alphas_cumprod.to(DEVICE)
    clean = _make_inputs(seed=12)
    noise = _make_inputs(seed=13)
    t = torch.tensor([300, 700], device=DEVICE)
    a_bar = acp[t.long()].view(-1, 1, 1, 1)
    sqrt_a, sqrt_1ma = a_bar.sqrt(), (1 - a_bar).sqrt()
    noisy = sqrt_a * clean + sqrt_1ma * noise

    x0_eps = hf_x0_mode(noise, noisy, t, "eps_ddpm", False, scheduler=scheduler)
    torch.testing.assert_close(x0_eps, clean.double(), rtol=1e-3, atol=1e-3)

    v = sqrt_a * noise - sqrt_1ma * clean
    x0_v = hf_x0_mode(v, noisy, t, "vpred_ddpm", False, scheduler=scheduler)
    torch.testing.assert_close(x0_v, clean.double(), rtol=1e-3, atol=1e-3)


def test_per_sample_from_prediction_flow_end_to_end():
    clean = _make_inputs(seed=14, dtype=torch.bfloat16)
    noise = _make_inputs(seed=15, dtype=torch.bfloat16)
    t = torch.tensor([0.3, 0.9], device=DEVICE)
    noisy = (t.view(-1, 1, 1, 1) * noise + (1 - t.view(-1, 1, 1, 1)) * clean).detach()
    v = (noise - clean).requires_grad_(True)

    loss = anchor_loss.anchor_per_sample_from_prediction(
        v, clean, noisy, t, "flow", timesteps_in_sigma=True, levels=4
    )
    assert loss.shape == (2,)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


# ---------------------------------------------------------------------------
# Optional soft SNR gate
# ---------------------------------------------------------------------------


def test_snr_weights_flow_formula_and_normalization():
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=DEVICE)
    w = anchor_loss.anchor_snr_weights(t, "flow", timesteps_in_sigma=True)
    expected = (1 - t.double()).square() / (t.double().square() + 1e-6)
    expected = expected / expected.mean().clamp_min(1e-8)
    torch.testing.assert_close(w.double(), expected, rtol=1e-5, atol=1e-6)
    assert w.dtype == torch.float32
    assert abs(w.mean().item() - 1.0) < 1e-6


def test_snr_weights_degenerate_batches_are_finite():
    # t = 0 only: huge but finite weights, mean 1
    w0 = anchor_loss.anchor_snr_weights(torch.zeros(4, device=DEVICE), "flow", timesteps_in_sigma=True)
    assert torch.isfinite(w0).all()
    assert abs(w0.mean().item() - 1.0) < 1e-5

    # all t = 1: all-zero raw weights => zero weights, finite, no NaN (§3.7)
    w1 = anchor_loss.anchor_snr_weights(torch.ones(4, device=DEVICE), "flow", timesteps_in_sigma=True)
    assert torch.isfinite(w1).all()
    assert w1.abs().max().item() == 0.0


def test_snr_weights_ddpm_schedule():
    from diffusers import DDPMScheduler

    scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    t = torch.tensor([0, 250, 500, 750, 999], device=DEVICE)
    w = anchor_loss.anchor_snr_weights(t, "eps_ddpm", noise_scheduler=scheduler)
    acp = scheduler.alphas_cumprod.to(DEVICE)
    expected = acp[t.long()] / (1 - acp[t.long()]).clamp_min(1e-6)
    expected = expected / expected.mean().clamp_min(1e-8)
    torch.testing.assert_close(w.double(), expected.double(), rtol=1e-5, atol=1e-6)
    assert abs(w.mean().item() - 1.0) < 1e-5


def test_snr_weighting_preserves_batch_average():
    """Batch-mean-1 normalization: a constant term's average is unchanged (§6.7)."""
    per_sample = torch.full((8,), 1.234, device=DEVICE)
    t = torch.linspace(0.05, 0.95, 8, device=DEVICE)
    w = anchor_loss.anchor_snr_weights(t, "flow", timesteps_in_sigma=True)
    weighted_mean = (per_sample * w).mean().item()
    assert abs(weighted_mean - per_sample.mean().item()) < 1e-4


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_validate_anchor_args():
    anchor_loss.validate_anchor_args(0.0, 1)  # off, valid
    anchor_loss.validate_anchor_args(0.5, 4)
    with pytest.raises(ValueError):
        anchor_loss.validate_anchor_args(-0.1, 4)
    with pytest.raises(ValueError):
        anchor_loss.validate_anchor_args(0.5, 0)


# ---------------------------------------------------------------------------
# Trainer integration surface
# ---------------------------------------------------------------------------


def _make_args(**kwargs):
    base = dict(
        anchor_scale=0.5,
        anchor_levels=3,
        anchor_snr_weighting=True,
        flow_model=False,
        v_parameterization=False,
    )
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def test_trainer_defaults_are_off():
    import train_network

    trainer = train_network.NetworkTrainer()
    assert trainer.anchor_scale == 0.0
    assert trainer.anchor_levels == 4
    assert trainer.anchor_snr_weighting is False
    assert trainer.anchor_prediction_mode is None
    assert trainer.anchor_timesteps_in_sigma is False
    assert trainer._anchor_noisy_latents is None
    assert trainer.anchor_loss_value is None
    assert trainer._anchor_warned_missing is False


def test_setup_anchor_objective_mode_resolution():
    import train_network

    trainer = train_network.NetworkTrainer()
    trainer.setup_anchor_objective(_make_args())
    assert trainer.anchor_prediction_mode == "eps_ddpm"
    assert trainer.anchor_scale == 0.5
    assert trainer.anchor_levels == 3
    assert trainer.anchor_snr_weighting is True

    trainer = train_network.NetworkTrainer()
    trainer.setup_anchor_objective(_make_args(flow_model=True))
    assert trainer.anchor_prediction_mode == "flow"

    trainer = train_network.NetworkTrainer()
    trainer.setup_anchor_objective(_make_args(v_parameterization=True))
    assert trainer.anchor_prediction_mode == "vpred_ddpm"

    # Subclass preset (e.g. Anima) is never clobbered.
    trainer = train_network.NetworkTrainer()
    trainer.anchor_prediction_mode = "flow"
    trainer.anchor_timesteps_in_sigma = True
    trainer.setup_anchor_objective(_make_args())
    assert trainer.anchor_prediction_mode == "flow"
    assert trainer.anchor_timesteps_in_sigma is True


def test_setup_anchor_objective_validation():
    import train_network

    trainer = train_network.NetworkTrainer()
    with pytest.raises(ValueError):
        trainer.setup_anchor_objective(_make_args(anchor_scale=-1.0))
    with pytest.raises(ValueError):
        trainer.setup_anchor_objective(_make_args(anchor_levels=0))


def test_generate_step_logs_includes_anchor():
    import train_network

    trainer = train_network.NetworkTrainer()
    args = types.SimpleNamespace(optimizer_type="AdamW", network_train_unet_only=False)

    class FakeScheduler:
        def get_last_lr(self):
            return [1e-4]

    logs = trainer.generate_step_logs(args, 0.5, 0.5, FakeScheduler(), None, current_anchor_loss=0.125)
    assert logs["loss/current_anchor"] == 0.125

    logs = trainer.generate_step_logs(args, 0.5, 0.5, FakeScheduler(), None)
    assert "loss/current_anchor" not in logs


def test_parser_has_anchor_arguments():
    import train_network

    parser = train_network.setup_parser()
    args = parser.parse_args([])
    assert args.anchor_scale == 0.0
    assert args.anchor_levels == 4
    assert args.anchor_snr_weighting is False

    args = parser.parse_args(["--anchor_scale", "0.3", "--anchor_levels", "5", "--anchor_snr_weighting"])
    assert args.anchor_scale == 0.3
    assert args.anchor_levels == 5
    assert args.anchor_snr_weighting is True
