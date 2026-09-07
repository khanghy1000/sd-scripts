"""Variance benchmarks: paper-original vs VRAM-optimized adaptive sampling settings.

Quantifies the numerical divergence introduced by each approximation:

1. **bf16 sweep accumulation** (default, ``fp32_eval=False``) vs the paper-original
   fp32 upcast path (``fp32_eval=True``).
2. **Strided evaluation grid** (``eval_stride > 1``) vs the paper-original full
   grid (``eval_stride = 1``).
3. **Chunk-size invariance** (memory-only change; expected ~zero variance).

Each end-to-end test runs both configurations through identical simulated training
(same data, same model trajectory, same REINFORCE actions) and reports:
- relative error of the delta approximation used for the policy update
- KL divergence between the resulting Beta sampling distributions
- drift of the sampled-timestep mean

Assertions use loose bounds so the tests act as regression guards; the printed
tables (run with ``pytest -s``) are the primary output.

Requires CUDA (per project test policy).
"""

import math

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from library.adaptive_timestep_sampler import (
    AdaptiveTimestepManager,
    TimestepSamplerNetwork,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T = 100  # small timestep count for fast sweeps


@pytest.fixture
def device():
    return torch.device("cuda")


def _make_scheduler(device, T=T):
    """Mock DDPM scheduler with a cosine alphas_cumprod schedule."""
    scheduler = MagicMock()
    t = torch.linspace(0, 1, T, device=device)
    scheduler.alphas_cumprod = torch.cos(t * math.pi / 2) ** 2
    scheduler.config = MagicMock()
    scheduler.config.num_train_timesteps = T
    return scheduler


def _make_model(device, dtype=torch.float32, seed=42):
    """Deterministic tiny conv model standing in for the diffusion model."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    conv = nn.Conv2d(4, 4, 3, padding=1, bias=True)
    with torch.no_grad():
        for p in conv.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
    return conv.to(device=device, dtype=dtype)


def _perturb_model(model, cycle):
    """Deterministic simulated optimizer step: shrink weights a bit each cycle.

    Produces non-zero, smooth per-timestep deltas between the before/after
    sweeps of Algorithm 2, mimicking a real optimizer update.
    """
    eps = 0.01 + 0.005 * math.sin(cycle)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(1.0 - eps)


def _beta_kl(a1, b1, a2, b2):
    """KL(Beta(a1,b1) || Beta(a2,b2)) in closed form (element-wise)."""
    log_B1 = torch.lgamma(a1) + torch.lgamma(b1) - torch.lgamma(a1 + b1)
    log_B2 = torch.lgamma(a2) + torch.lgamma(b2) - torch.lgamma(a2 + b2)
    return (
        (log_B2 - log_B1)
        + (a1 - a2) * torch.digamma(a1)
        + (b1 - b2) * torch.digamma(b1)
        + (a2 - a1 + b2 - b1) * torch.digamma(a1 + b1)
    )


def _make_manager(device, scheduler, seed=123, **overrides):
    """Create a manager with a deterministically-initialized sampler network.

    Seeds the global RNG before construction so that nn.Linear bias init
    (which uses the global RNG) is also reproducible — two managers created
    with the same seed get bitwise-identical networks.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    net = TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2)
    with torch.no_grad():
        for p in net.parameters():
            if p.ndim > 1:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.05)
    net = net.to(device)
    kwargs = dict(
        sampler_network=net,
        noise_scheduler=scheduler,
        device=device,
        dtype=torch.float32,
        learning_rate=1e-2,
        entropy_coeff=1e-2,
        update_freq=1,  # update every cycle
        queue_size=10,
        num_selected=3,
        model_type="ddpm",
    )
    kwargs.update(overrides)
    return AdaptiveTimestepManager(**kwargs)


def _fixed_data(device, n_cycles, batch=2, seed=7):
    """Pre-generate identical (x, noise) pairs for every cycle."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    data = []
    for _ in range(n_cycles):
        x = torch.randn(batch, 4, 8, 8, generator=gen).to(device)
        noise = torch.randn(batch, 4, 8, 8, generator=gen).to(device)
        data.append((x, noise))
    return data


def _run_cycle(manager, model, model_dtype, x, noise, action_t):
    """Run one full Algorithm 2 cycle (before sweep, perturb, after sweep, update).

    The REINFORCE action is pinned to ``action_t`` so policy differences between
    two managers come only from their delta approximations and network drift.
    """
    def model_fn(xt, ts, wd):
        return model(xt.to(model_dtype)).to(wd)

    losses_before = manager.compute_per_timestep_losses(x, noise, model_fn, model_dtype)
    _perturb_model(model, manager._cycle_counter)
    losses_after = manager.compute_per_timestep_losses(x, noise, model_fn, model_dtype)

    deltas = losses_before - losses_after
    delta_approx, selected = manager.compute_delta_approximation(
        model_fn, x, noise, model_dtype, losses_before,
        full_batch_latents=None, full_batch_noise=None,
    )

    manager.sample_timesteps(x, T)  # refresh cached action (then pin it)
    manager._cached_t_continuous = torch.full(
        (x.shape[0],), action_t, device=x.device
    )
    manager.update_sampler(delta_approx, x)
    manager._cycle_counter += 1
    return deltas, delta_approx, selected


# ---------------------------------------------------------------------------
# 1. Chunk-size invariance (memory-only change — expect ~zero variance)
# ---------------------------------------------------------------------------

class TestChunkSizeInvariance:

    def test_chunk_sizes_produce_identical_losses(self, device):
        scheduler = _make_scheduler(device)
        model = _make_model(device)
        x = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x)

        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        losses = {}
        for chunk in (7, 16, 64):
            mgr = _make_manager(device, scheduler, eval_chunk_size=chunk, fp32_eval=True)
            losses[chunk] = mgr.compute_per_timestep_losses(x, noise, model_fn, torch.float32)

        d1 = (losses[7] - losses[16]).abs().max().item()
        d2 = (losses[16] - losses[64]).abs().max().item()
        print(f"\n[chunk-size invariance] max |diff| 7 vs 16: {d1:.3e}, 16 vs 64: {d2:.3e}")
        assert d1 < 1e-5 and d2 < 1e-5, "chunk size must not change results"


# ---------------------------------------------------------------------------
# 2. bf16 accumulation vs fp32 (paper original)
# ---------------------------------------------------------------------------

class TestBF16AccumulationVariance:

    def test_per_timestep_loss_relative_error(self, device):
        """bf16 accumulation vs fp32 upcast on a single sweep (bf16 model + bf16 latents)."""
        scheduler = _make_scheduler(device)
        model = _make_model(device, dtype=torch.bfloat16)
        # bf16 latents, as produced by the VAE-cache in real training
        x = torch.randn(1, 4, 8, 8, device=device).to(torch.bfloat16)
        noise = torch.randn_like(x)

        def model_fn(xt, ts, wd):
            return model(xt.to(torch.bfloat16)).to(wd)

        mgr_fp32 = _make_manager(device, scheduler, fp32_eval=True, eval_chunk_size=16)
        mgr_bf16 = _make_manager(device, scheduler, fp32_eval=False, eval_chunk_size=16)

        l_fp32 = mgr_fp32.compute_per_timestep_losses(x, noise, model_fn, torch.bfloat16)
        l_bf16 = mgr_bf16.compute_per_timestep_losses(x, noise, model_fn, torch.bfloat16)

        rel = ((l_bf16 - l_fp32).abs() / l_fp32.clamp_min(1e-12))
        print(
            f"\n[bf16 vs fp32 sweep] relative error of per-timestep losses: "
            f"mean={rel.mean().item():.3e}, max={rel.max().item():.3e}"
        )
        assert rel.max().item() < 5e-2

    def test_batch_loss_relative_error(self, device):
        """bf16 vs fp32 on the |S|-loop batch path (bf16 latents)."""
        scheduler = _make_scheduler(device)
        model = _make_model(device, dtype=torch.bfloat16)
        x = torch.randn(4, 4, 8, 8, device=device).to(torch.bfloat16)
        noise = torch.randn_like(x)
        selected = torch.tensor([10, 55, 90], device=device)

        def model_fn(xt, ts, wd):
            return model(xt.to(torch.bfloat16)).to(wd)

        mgr_fp32 = _make_manager(device, scheduler, fp32_eval=True)
        mgr_bf16 = _make_manager(device, scheduler, fp32_eval=False)

        l_fp32 = mgr_fp32.compute_per_timestep_losses_for_batch(
            x, noise, model_fn, torch.bfloat16, selected
        )
        l_bf16 = mgr_bf16.compute_per_timestep_losses_for_batch(
            x, noise, model_fn, torch.bfloat16, selected
        )
        rel = abs(l_bf16.item() - l_fp32.item()) / max(abs(l_fp32.item()), 1e-12)
        print(
            f"\n[bf16 vs fp32 batch path] fp32={l_fp32.item():.6f}, "
            f"bf16={l_bf16.item():.6f}, rel err={rel:.3e}"
        )
        assert rel < 5e-2

    def test_end_to_end_beta_divergence(self, device):
        """20 simulated training cycles: bf16 (approx) vs fp32 (paper) managers.

        Uses bf16 latents and a bf16 sampler network, mirroring real bf16 training.
        """
        n_cycles = 20
        scheduler = _make_scheduler(device)
        # Two independent but identically-initialized bf16 models
        model_a = _make_model(device, dtype=torch.bfloat16)
        model_b = _make_model(device, dtype=torch.bfloat16)
        model_b.load_state_dict(model_a.state_dict())

        mgr_fp32 = _make_manager(device, scheduler, seed=123, dtype=torch.bfloat16, fp32_eval=True)
        mgr_bf16 = _make_manager(device, scheduler, seed=123, dtype=torch.bfloat16, fp32_eval=False)
        mgr_fp32._cycle_counter = 0
        mgr_bf16._cycle_counter = 0

        # bf16 latents for every cycle
        data = [(x.to(torch.bfloat16), n.to(torch.bfloat16)) for x, n in _fixed_data(device, n_cycles)]
        rel_errs = []
        for k, (x, noise) in enumerate(data):
            _, delta_fp32, _ = _run_cycle(mgr_fp32, model_a, torch.bfloat16, x, noise, action_t=0.4)
            _, delta_bf16, _ = _run_cycle(mgr_bf16, model_b, torch.bfloat16, x, noise, action_t=0.4)
            denom = max(abs(delta_fp32.item()), 1e-12)
            rel_errs.append(abs(delta_bf16.item() - delta_fp32.item()) / denom)

        a1, b1 = mgr_fp32.sampler_network(data[-1][0])
        a2, b2 = mgr_bf16.sampler_network(data[-1][0])
        kl = _beta_kl(a1.detach().float(), b1.detach().float(), a2.detach().float(), b2.detach().float()).mean().item()

        print(
            f"\n[bf16 vs fp32 | {n_cycles} cycles]\n"
            f"  delta_approx relative error: mean={sum(rel_errs)/len(rel_errs):.3e}, "
            f"max={max(rel_errs):.3e}\n"
            f"  Beta params after training: fp32 a={a1.float().mean().item():.4f} b={b1.float().mean().item():.4f} | "
            f"bf16 a={a2.float().mean().item():.4f} b={b2.float().mean().item():.4f}\n"
            f"  KL(Beta_fp32 || Beta_bf16) = {kl:.3e} nats"
        )
        assert sum(rel_errs) / len(rel_errs) < 0.1
        assert kl < 0.5


# ---------------------------------------------------------------------------
# 3. Strided eval grid vs full grid (paper original)
# ---------------------------------------------------------------------------

class TestStrideApproximationVariance:

    def test_grid_point_losses_match_full_grid(self, device):
        """Strided sweep must reproduce the full-grid losses at the grid points."""
        scheduler = _make_scheduler(device)
        model = _make_model(device)
        x = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x)

        def model_fn(xt, ts, wd):
            return model(xt.to(wd))

        mgr_full = _make_manager(device, scheduler, eval_stride=1, fp32_eval=True)
        mgr_strided = _make_manager(device, scheduler, eval_stride=4, fp32_eval=True)

        l_full = mgr_full.compute_per_timestep_losses(x, noise, model_fn, torch.float32)
        l_strided = mgr_strided.compute_per_timestep_losses(x, noise, model_fn, torch.float32)

        grid = mgr_strided._eval_timesteps.to(device)
        diff = (l_strided - l_full[grid.long()]).abs().max().item()
        print(f"\n[stride grid correctness] max |l_strided - l_full[grid]| = {diff:.3e}")
        assert diff < 1e-4, "strided grid must match full grid at shared timesteps"

    def test_delta_approx_and_selection_report(self, device):
        """Single Algorithm 2 cycle: compare selected timesteps and delta_approx."""
        scheduler = _make_scheduler(device)
        model_a = _make_model(device)
        model_b = _make_model(device)
        model_b.load_state_dict(model_a.state_dict())

        mgr_full = _make_manager(device, scheduler, seed=123, eval_stride=1, fp32_eval=True)
        mgr_s4 = _make_manager(device, scheduler, seed=123, eval_stride=4, fp32_eval=True)
        mgr_full._cycle_counter = 0
        mgr_s4._cycle_counter = 0

        (x, noise) = _fixed_data(device, 1)[0]

        _, delta_full, sel_full = _run_cycle(mgr_full, model_a, torch.float32, x, noise, action_t=0.4)
        _, delta_s4, sel_s4 = _run_cycle(mgr_s4, model_b, torch.float32, x, noise, action_t=0.4)

        # Nearest-full-selection distance for each strided pick
        dists = []
        for s in sel_s4.tolist():
            dists.append(min(abs(s - f) for f in sel_full.tolist()))
        rel = abs(delta_s4.item() - delta_full.item()) / max(abs(delta_full.item()), 1e-12)

        print(
            f"\n[stride=4 vs full | single cycle]\n"
            f"  full-grid selected timesteps:    {sorted(sel_full.tolist())}\n"
            f"  strided-grid selected timesteps: {sorted(sel_s4.tolist())}\n"
            f"  max distance to nearest full pick: {max(dists)}\n"
            f"  delta_approx: full={delta_full.item():.6e}, strided={delta_s4.item():.6e}, "
            f"rel err={rel:.3e}"
        )
        assert max(dists) <= 8, "strided picks should land near full-grid picks"
        assert rel < 0.5

    def test_end_to_end_beta_divergence_stride4(self, device):
        """20 simulated training cycles: stride=4 (approx) vs stride=1 (paper)."""
        n_cycles = 20
        scheduler = _make_scheduler(device)
        model_a = _make_model(device)
        model_b = _make_model(device)
        model_b.load_state_dict(model_a.state_dict())

        mgr_full = _make_manager(device, scheduler, seed=123, eval_stride=1, fp32_eval=True)
        mgr_s4 = _make_manager(device, scheduler, seed=123, eval_stride=4, fp32_eval=True)
        mgr_full._cycle_counter = 0
        mgr_s4._cycle_counter = 0

        data = _fixed_data(device, n_cycles)
        rel_errs = []
        for k, (x, noise) in enumerate(data):
            _, delta_full, _ = _run_cycle(mgr_full, model_a, torch.float32, x, noise, action_t=0.4)
            _, delta_s4, _ = _run_cycle(mgr_s4, model_b, torch.float32, x, noise, action_t=0.4)
            denom = max(abs(delta_full.item()), 1e-12)
            rel_errs.append(abs(delta_s4.item() - delta_full.item()) / denom)

        x_last = data[-1][0]
        a1, b1 = mgr_full.sampler_network(x_last)
        a2, b2 = mgr_s4.sampler_network(x_last)
        kl = _beta_kl(a1.detach(), b1.detach(), a2.detach(), b2.detach()).mean().item()

        # Sampled-timestep mean drift under each policy
        with torch.no_grad():
            t_full = a1 / (a1 + b1)
            t_s4 = a2 / (a2 + b2)
        mean_drift = abs(t_full.mean().item() - t_s4.mean().item())

        print(
            f"\n[stride=4 vs full | {n_cycles} cycles]\n"
            f"  delta_approx relative error: mean={sum(rel_errs)/len(rel_errs):.3e}, "
            f"max={max(rel_errs):.3e}\n"
            f"  Beta params after training: full a={a1.mean().item():.4f} b={b1.mean().item():.4f} | "
            f"stride4 a={a2.mean().item():.4f} b={b2.mean().item():.4f}\n"
            f"  KL(Beta_full || Beta_stride4) = {kl:.3e} nats\n"
            f"  policy mean drift |E[t]_full - E[t]_stride4| = {mean_drift:.4f} (t in [0,1])"
        )
        assert sum(rel_errs) / len(rel_errs) < 0.5
        assert kl < 1.0
