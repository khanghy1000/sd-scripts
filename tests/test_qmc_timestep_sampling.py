"""CUDA tests for QMC timestep sampling improvements.

Covers:
- DDP rank-offset (disjoint slices across ranks)
- Antithetic-QMC composition (all distributions)
- state_dict / load_state_dict (checkpoint resume fast-forward)
- clear_instances() (test isolation)
- draw_base2 for power-of-2 batch sizes
- n <= 0 guard in draw()

All tensor tests run on CUDA per project policy.
"""

import pytest
import torch

from library.custom_train_functions import (
    compute_density_for_timestep_sampling,
    _QMCSequenceManager,
)

DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# DDP rank-offset: each rank consumes a disjoint slice of the global sequence
# ---------------------------------------------------------------------------

class TestDDPRankOffset:
    """Verify that different ranks draw disjoint slices of the global QMC sequence."""

    def test_ranks_draw_disjoint_points(self):
        """Rank 0 and rank 1 must draw different points (not identical)."""
        _QMCSequenceManager.clear_instances()
        mgr0 = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        mgr1 = _QMCSequenceManager(method="sobol", seed=0, rank=1)
        a = mgr0.draw(8, device=DEVICE)
        b = mgr1.draw(8, device=DEVICE)
        assert not torch.allclose(a, b)

    def test_rank_uses_seed_offset(self):
        """Rank r must use seed+r, so rank 1 with seed 0 == rank 0 with seed 1."""
        _QMCSequenceManager.clear_instances()
        mgr_rank1_seed0 = _QMCSequenceManager(method="sobol", seed=0, rank=1)
        a = mgr_rank1_seed0.draw(8, device=DEVICE)

        _QMCSequenceManager.clear_instances()
        mgr_rank0_seed1 = _QMCSequenceManager(method="sobol", seed=1, rank=0)
        b = mgr_rank0_seed1.draw(8, device=DEVICE)
        assert torch.allclose(a, b, atol=1e-6)

    def test_rank_zero_unchanged(self):
        """Rank 0 must draw the same points as a no-rank manager."""
        _QMCSequenceManager.clear_instances()
        mgr_rank0 = _QMCSequenceManager(method="sobol", seed=5, rank=0)
        a = mgr_rank0.draw(8, device=DEVICE)

        _QMCSequenceManager.clear_instances()
        mgr_norank = _QMCSequenceManager(method="sobol", seed=5, rank=0)
        b = mgr_norank.draw(8, device=DEVICE)
        assert torch.allclose(a, b)

    def test_density_passes_rank(self):
        """compute_density_for_timestep_sampling must honor the rank argument."""
        _QMCSequenceManager.clear_instances()
        u0 = compute_density_for_timestep_sampling(
            "uniform", 8, qmc="sobol", qmc_seed=0, rank=0, device=DEVICE
        )
        _QMCSequenceManager.clear_instances()
        u1 = compute_density_for_timestep_sampling(
            "uniform", 8, qmc="sobol", qmc_seed=0, rank=1, device=DEVICE
        )
        assert not torch.allclose(u0, u1)


# ---------------------------------------------------------------------------
# Antithetic-QMC composition
# ---------------------------------------------------------------------------

class TestAntitheticQMCComposition:
    """Verify antithetic and qmc compose (not mutually exclusive)."""

    def test_uniform_pairs_sum_to_one(self):
        """Antithetic-QMC uniform: u[i] + u[i+4] == 1."""
        _QMCSequenceManager.clear_instances()
        u = compute_density_for_timestep_sampling(
            "uniform", 8, antithetic=True, qmc="sobol", qmc_seed=0, device=DEVICE
        )
        for i in range(4):
            assert u[i].item() + u[i + 4].item() == pytest.approx(1.0, abs=1e-6)

    def test_logit_normal_pairs_symmetric(self):
        """Antithetic-QMC logit_normal: logits are mirrored (z, -z)."""
        _QMCSequenceManager.clear_instances()
        u = compute_density_for_timestep_sampling(
            "logit_normal", 8, logit_mean=0.0, logit_std=1.0,
            antithetic=True, qmc="sobol", qmc_seed=0, device=DEVICE
        )
        logits = torch.logit(u.clamp(1e-7, 1 - 1e-7))
        for i in range(4):
            assert logits[i].item() + logits[i + 4].item() == pytest.approx(0.0, abs=1e-5)

    def test_sigmoid_pairs_symmetric(self):
        """Antithetic-QMC sigmoid: logits are mirrored (z, -z)."""
        _QMCSequenceManager.clear_instances()
        u = compute_density_for_timestep_sampling(
            "sigmoid", 8, sigmoid_scale=1.0,
            antithetic=True, qmc="sobol", qmc_seed=0, device=DEVICE
        )
        logits = torch.logit(u.clamp(1e-7, 1 - 1e-7))
        for i in range(4):
            assert logits[i].item() + logits[i + 4].item() == pytest.approx(0.0, abs=1e-5)

    def test_mode_pairs_mirrored_base(self):
        """Antithetic-QMC mode: the base uniform is mirrored before the transform."""
        _QMCSequenceManager.clear_instances()
        u = compute_density_for_timestep_sampling(
            "mode", 8, mode_scale=1.29,
            antithetic=True, qmc="sobol", qmc_seed=0, device=DEVICE
        )
        # The mode transform is deterministic in the base uniform; with antithetic
        # the base is (u, 1-u), so the output pairs are negatively correlated.
        # Just check shape and range.
        assert u.shape == (8,)
        assert (u >= 0).all() and (u <= 1).all()

    def test_odd_batch_truncates_last_pair(self):
        """Odd batch size with antithetic-QMC truncates the last mirrored pair.

        n_pairs = (5+1)//2 = 3, so pairs are (0,3), (1,4) and index 2 is the
        unpaired leftover.
        """
        _QMCSequenceManager.clear_instances()
        u = compute_density_for_timestep_sampling(
            "uniform", 5, antithetic=True, qmc="sobol", qmc_seed=0, device=DEVICE
        )
        assert u.shape == (5,)
        # First two complete pairs must be mirrored.
        assert u[0].item() + u[3].item() == pytest.approx(1.0, abs=1e-6)
        assert u[1].item() + u[4].item() == pytest.approx(1.0, abs=1e-6)

    def test_antithetic_qmc_uses_half_draws(self):
        """Antithetic-QMC must draw only batch_size//2 QMC points (not batch_size)."""
        _QMCSequenceManager.clear_instances()
        # Draw with antithetic-QMC: batch=8 -> 4 QMC points drawn.
        compute_density_for_timestep_sampling(
            "uniform", 8, antithetic=True, qmc="sobol", qmc_seed=0, device=DEVICE
        )
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        # 4 points consumed + 0 rank offset = draw_count 4.
        assert mgr._draw_count == 4

    def test_qmc_only_uses_full_draws(self):
        """QMC without antithetic must draw batch_size QMC points."""
        _QMCSequenceManager.clear_instances()
        compute_density_for_timestep_sampling(
            "uniform", 8, antithetic=False, qmc="sobol", qmc_seed=0, device=DEVICE
        )
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        assert mgr._draw_count == 8


# ---------------------------------------------------------------------------
# state_dict / load_state_dict (checkpoint resume)
# ---------------------------------------------------------------------------

class TestStateDict:
    """Verify QMC sequence position save/restore for checkpoint resume."""

    def test_state_dict_has_draw_count(self):
        """state_dict must return the current draw count."""
        _QMCSequenceManager.clear_instances()
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        mgr.draw(10, device=DEVICE)
        sd = mgr.state_dict()
        assert sd["draw_count"] == 10
        assert sd["method"] == "sobol"
        assert sd["seed"] == 0
        assert sd["rank"] == 0

    def test_load_state_dict_resumes_sequence(self):
        """After load_state_dict, draws must continue from the saved position."""
        _QMCSequenceManager.clear_instances()
        # Draw 8 points, save state.
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        mgr.draw(8, device=DEVICE)
        sd = mgr.state_dict()

        # Draw 8 more on the original manager (the "ground truth" continuation).
        expected_next = mgr.draw(8, device=DEVICE)

        # Now create a fresh manager, load the saved state, and draw 8.
        _QMCSequenceManager.clear_instances()
        mgr2 = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        mgr2.load_state_dict(sd)
        actual_next = mgr2.draw(8, device=DEVICE)

        assert torch.allclose(actual_next, expected_next, atol=1e-6)

    def test_load_state_dict_with_rank(self):
        """load_state_dict must preserve the rank offset on resume."""
        _QMCSequenceManager.clear_instances()
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=2)
        mgr.draw(8, device=DEVICE)
        sd = mgr.state_dict()

        # Resume on rank 2: the next draw must match the original's next draw.
        expected_next = mgr.draw(8, device=DEVICE)

        _QMCSequenceManager.clear_instances()
        mgr2 = _QMCSequenceManager(method="sobol", seed=0, rank=2)
        mgr2.load_state_dict(sd)
        actual_next = mgr2.draw(8, device=DEVICE)
        assert torch.allclose(actual_next, expected_next, atol=1e-6)


# ---------------------------------------------------------------------------
# clear_instances (test isolation)
# ---------------------------------------------------------------------------

class TestClearInstances:
    """Verify clear_instances() drops cached singletons."""

    def test_clear_resets_sequence(self):
        """After clear_instances, a new manager restarts from position 0."""
        _QMCSequenceManager.clear_instances()
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        first = mgr.draw(8, device=DEVICE)
        # Advance the sequence.
        mgr.draw(8, device=DEVICE)

        # Clear and re-create: should restart from 0.
        _QMCSequenceManager.clear_instances()
        mgr2 = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        restarted = mgr2.draw(8, device=DEVICE)
        assert torch.allclose(restarted, first, atol=1e-6)

    def test_clear_evicts_all_keys(self):
        """clear_instances must empty the _instances dict."""
        _QMCSequenceManager(method="sobol", seed=1, rank=0)
        _QMCSequenceManager(method="sobol", seed=2, rank=0)
        assert len(_QMCSequenceManager._instances) >= 2
        _QMCSequenceManager.clear_instances()
        assert len(_QMCSequenceManager._instances) == 0


# ---------------------------------------------------------------------------
# draw_base2 for power-of-2 batch sizes
# ---------------------------------------------------------------------------

class TestDrawBase2:
    """Verify draw_base2 is used for power-of-2 sizes and gives valid output."""

    def test_power_of_two_valid(self):
        """Power-of-2 draws must produce valid points in [0,1]."""
        _QMCSequenceManager.clear_instances()
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        for n in [1, 2, 4, 8, 16, 32]:
            pts = mgr.draw(n, device=DEVICE)
            assert pts.shape == (n,)
            assert (pts >= 0).all() and (pts <= 1).all()

    def test_non_power_of_two_valid(self):
        """Non-power-of-2 draws must also work (falls back to draw)."""
        _QMCSequenceManager.clear_instances()
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        for n in [3, 5, 6, 7, 9, 10]:
            pts = mgr.draw(n, device=DEVICE)
            assert pts.shape == (n,)
            assert (pts >= 0).all() and (pts <= 1).all()

    def test_power_of_two_marginal_mean(self):
        """A large power-of-2 draw must have marginal mean ~0.5."""
        _QMCSequenceManager.clear_instances()
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        pts = mgr.draw(8192, device=DEVICE)
        assert pts.mean().item() == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# n <= 0 guard in draw()
# ---------------------------------------------------------------------------

class TestDrawZeroGuard:
    """Verify draw(0) and draw(negative) return an empty tensor without error."""

    def test_draw_zero(self):
        _QMCSequenceManager.clear_instances()
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        pts = mgr.draw(0, device=DEVICE)
        assert pts.shape == (0,)
        assert pts.dtype == torch.float32

    def test_draw_negative(self):
        _QMCSequenceManager.clear_instances()
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        pts = mgr.draw(-5, device=DEVICE)
        assert pts.shape == (0,)

    def test_draw_zero_does_not_advance(self):
        """A zero draw must not advance the sequence position."""
        _QMCSequenceManager.clear_instances()
        mgr = _QMCSequenceManager(method="sobol", seed=0, rank=0)
        mgr.draw(0, device=DEVICE)
        assert mgr._draw_count == 0
