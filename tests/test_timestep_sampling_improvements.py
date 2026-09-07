"""Tests for timestep sampling improvements (CUDA).

Covers:
- Sobol draw_base2 alignment in _QMCSequenceManager (padding keeps every
  power-of-two draw a balanced base-2 Sobol block, reproducible across
  state_dict save/restore).
- Jittered/stratified-QMC hybrid ("sobol_jittered", "halton_jittered"):
  exactly one point per equal stratum per batch, composing with the
  distribution transform and antithetic pairing.
- Antithetic noise pairing (--antithetic_noise_pairing):
  apply_antithetic_noise_pairing / maybe_apply_antithetic_noise_pairing.

All torch tensors are created on CUDA per project convention.
"""

import argparse
import math
from types import SimpleNamespace

import pytest
import torch

from library.custom_train_functions import (
    _QMCSequenceManager,
    add_custom_train_arguments,
    apply_antithetic_noise_pairing,
    compute_density_for_timestep_sampling,
    maybe_apply_antithetic_noise_pairing,
)

DEVICE = torch.device("cuda")


@pytest.fixture(autouse=True)
def _clear_qmc_instances():
    _QMCSequenceManager.clear_instances()
    yield
    _QMCSequenceManager.clear_instances()


# ---------------------------------------------------------------------------
# Sobol draw_base2 alignment
# ---------------------------------------------------------------------------
class TestSobolBase2Alignment:
    def test_consecutive_pow2_draws_match_single_block(self):
        """Two draw(8) calls from a fresh sequence must equal one draw(16)."""
        mgr_a = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        first = mgr_a.draw(8, device=DEVICE)
        second = mgr_a.draw(8, device=DEVICE)

        _QMCSequenceManager.clear_instances()
        mgr_b = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        block = mgr_b.draw(16, device=DEVICE)

        assert torch.allclose(first, block[:8])
        assert torch.allclose(second, block[8:])

    def test_alignment_padding_after_non_pow2_draw(self):
        """After a non-power-of-two draw, the manager pads so the next pow2
        draw is a balanced base-2 block (positions 8..15 of the sequence)."""
        mgr_a = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        mgr_a.draw(6, device=DEVICE)  # count = 6 (not a multiple of 8)
        pts = mgr_a.draw(8, device=DEVICE)  # pad 2 (count -> 8), then base2 block 8..15
        assert mgr_a._draw_count == 16  # padding is counted

        _QMCSequenceManager.clear_instances()
        mgr_b = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        block = mgr_b.draw(16, device=DEVICE)

        assert torch.allclose(pts, block[8:16])

    def test_alignment_state_dict_roundtrip(self):
        """Save/restore must reproduce the exact sequence including padding."""
        mgr_a = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        mgr_a.draw(6, device=DEVICE)
        mgr_a.draw(8, device=DEVICE)  # pad 2 -> base2 block at positions 8..15
        state = mgr_a.state_dict()
        assert state["draw_count"] == 16
        expected_next = mgr_a.draw(8, device=DEVICE)  # positions 16..23

        _QMCSequenceManager.clear_instances()
        mgr_b = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        mgr_b.load_state_dict(state)
        actual = mgr_b.draw(8, device=DEVICE)

        assert torch.allclose(expected_next, actual)
        assert mgr_b._draw_count == 24

    def test_points_in_unit_interval_on_device(self):
        mgr = _QMCSequenceManager(method="sobol", seed=1, rank=0)
        pts = mgr.draw(32, device=DEVICE)
        assert pts.device.type == "cuda"
        assert pts.dtype == torch.float32
        assert torch.all(pts >= 0.0) and torch.all(pts < 1.0)

    def test_low_discrepancy_coverage(self):
        """Cumulative Sobol points should fill [0,1] far more evenly than
        pseudo-random: every 1/16 bin is non-empty after 64 aligned points."""
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        pts = mgr.draw(64, device=DEVICE)
        bins = (pts * 16).long()
        assert bins.unique().numel() == 16


# ---------------------------------------------------------------------------
# Jittered / stratified-QMC hybrid
# ---------------------------------------------------------------------------
class TestJitteredQMC:
    @pytest.mark.parametrize("method", ["sobol_jittered", "halton_jittered"])
    def test_one_point_per_stratum(self, method):
        pytest.importorskip("scipy") if "halton" in method else None
        n = 8
        mgr = _QMCSequenceManager(method=method, seed=0, rank=0)
        pts = mgr.draw(n, device=DEVICE)
        assert pts.shape == (n,)
        assert torch.all(pts >= 0.0) and torch.all(pts < 1.0)
        # Exactly one point in each stratum [k/n, (k+1)/n)
        strata = (pts * n).long()
        assert torch.equal(torch.sort(strata).values, torch.arange(n, device=DEVICE))

    def test_batches_advance_and_differ(self):
        mgr = _QMCSequenceManager(method="sobol_jittered", seed=0, rank=0)
        a = mgr.draw(8, device=DEVICE)
        b = mgr.draw(8, device=DEVICE)
        assert not torch.allclose(a, b)

    def test_jitter_stays_low_discrepancy_across_batches(self):
        """Across several batches, the union of jittered points covers the
        unit interval better than pure iid (which would miss many 16-bin bins)."""
        mgr = _QMCSequenceManager(method="sobol_jittered", seed=0, rank=0)
        pts = torch.cat([mgr.draw(8, device=DEVICE) for _ in range(4)])
        coarse = (pts * 16).long().clamp(0, 15)
        # iid would leave ~2-4 of the 16 bins empty; jittered does better
        assert coarse.unique().numel() >= 14

    def test_compute_density_uniform_jittered_is_stratified(self):
        n = 8
        u = compute_density_for_timestep_sampling(
            weighting_scheme="uniform", batch_size=n, qmc="sobol_jittered", qmc_seed=0, device=DEVICE
        )
        strata = (u * n).long()
        assert torch.equal(torch.sort(strata).values, torch.arange(n, device=DEVICE))

    def test_jittered_composes_with_antithetic(self):
        n = 8
        u = compute_density_for_timestep_sampling(
            weighting_scheme="uniform",
            batch_size=n,
            qmc="sobol_jittered",
            qmc_seed=0,
            antithetic=True,
            device=DEVICE,
        )
        assert u.shape == (n,)
        n_pairs = n // 2
        # Mirrored pair structure: second half is 1 - first half
        assert torch.allclose(u[:n_pairs] + u[n_pairs:], torch.ones(n_pairs, device=DEVICE), atol=1e-6)

    def test_jittered_composes_with_logit_normal_and_shift(self):
        from library.custom_train_functions import apply_flow_shift

        n = 16
        sig = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=n,
            logit_mean=0.0,
            logit_std=1.0,
            qmc="sobol_jittered",
            qmc_seed=0,
            device=DEVICE,
        )
        shifted = apply_flow_shift(sig, 0.8)
        assert torch.all(shifted > 0.0) and torch.all(shifted < 1.0)
        # sigma' = s*sigma / (1 + (s-1)*sigma) is monotonic; s=0.8 < 1 lowers sigma
        assert shifted.mean() < sig.mean()
        assert torch.all(shifted < sig)


# ---------------------------------------------------------------------------
# Antithetic noise pairing
# ---------------------------------------------------------------------------
class TestAntitheticNoisePairing:
    def test_pair_structure_even_batch(self):
        noise = torch.randn(8, 4, 4, 4, device=DEVICE)
        paired = apply_antithetic_noise_pairing(noise)
        n_pairs = 4
        assert torch.equal(paired[:n_pairs], noise[:n_pairs])
        assert torch.equal(paired[n_pairs:], -noise[:n_pairs])

    def test_pair_structure_odd_batch(self):
        noise = torch.randn(7, 4, 4, 4, device=DEVICE)
        paired = apply_antithetic_noise_pairing(noise)
        n_pairs = 4
        assert torch.equal(paired[:n_pairs], noise[:n_pairs])
        # Last mirrored pair is truncated: only 3 mirrors
        assert torch.equal(paired[n_pairs:], -noise[:3])

    def test_batch_size_one_noop(self):
        noise = torch.randn(1, 4, 4, 4, device=DEVICE)
        assert torch.equal(apply_antithetic_noise_pairing(noise), noise)

    def test_preserves_shape_dtype_device(self):
        noise = torch.randn(6, 16, 8, 8, device=DEVICE, dtype=torch.float16)
        paired = apply_antithetic_noise_pairing(noise)
        assert paired.shape == noise.shape
        assert paired.dtype == noise.dtype
        assert paired.device == noise.device

    def test_marginal_distribution_preserved(self):
        """Mirrored noise has the same per-sample marginal: each entry is
        either eps or -eps, both N(0,1). The second moment equals that of the
        base draws, and the paired batch mean is exactly zero (even batch)."""
        torch.manual_seed(0)
        noise = torch.randn(4096, 64, device=DEVICE)
        paired = apply_antithetic_noise_pairing(noise)
        assert torch.allclose(paired.pow(2).mean(), noise[:2048].pow(2).mean(), atol=1e-6)
        # Paired batch mean is exactly zero: perfect variance cancellation
        assert paired.mean().abs() < 1e-7

    def test_maybe_guard_respects_flag_and_is_train(self):
        noise = torch.randn(4, 2, device=DEVICE)
        args_off = SimpleNamespace(antithetic_noise_pairing=False)
        args_on = SimpleNamespace(antithetic_noise_pairing=True)
        assert torch.equal(maybe_apply_antithetic_noise_pairing(args_off, noise), noise)
        assert torch.equal(maybe_apply_antithetic_noise_pairing(args_on, noise, is_train=False), noise)
        paired = maybe_apply_antithetic_noise_pairing(args_on, noise)
        assert torch.equal(paired[2:], -noise[:2])

    def test_pairing_order_matches_timestep_pairing(self):
        """The density function places base draws first and mirrors second;
        noise pairing must use the same layout so sample i pairs with i+n/2."""
        n = 8
        u = compute_density_for_timestep_sampling(
            weighting_scheme="uniform", batch_size=n, antithetic=True, device=DEVICE
        )
        n_pairs = n // 2
        assert torch.allclose(u[:n_pairs] + u[n_pairs:], torch.ones(n_pairs, device=DEVICE), atol=1e-6)
        noise = torch.randn(n, 4, device=DEVICE)
        paired = apply_antithetic_noise_pairing(noise)
        # Same (i, i + n_pairs) pairing on both axes
        assert torch.equal(paired[n_pairs:], -noise[:n_pairs])


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------
class TestCLIArguments:
    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        add_custom_train_arguments(parser)
        return parser.parse_args(argv)

    def test_qmc_jittered_choices_accepted(self):
        args = self._parse(["--qmc_timestep_sampling", "sobol_jittered"])
        assert args.qmc_timestep_sampling == "sobol_jittered"
        args = self._parse(["--qmc_timestep_sampling", "halton_jittered"])
        assert args.qmc_timestep_sampling == "halton_jittered"

    def test_antithetic_noise_pairing_flag(self):
        assert self._parse([]).antithetic_noise_pairing is False
        assert self._parse(["--antithetic_noise_pairing"]).antithetic_noise_pairing is True

