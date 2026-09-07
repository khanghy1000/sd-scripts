"""
Tests for Latent Wavelet Diffusion (LWD) wavelet masking utilities.

Tests:
  - compute_wavelet_attention_map: output shape, value range [0,1], dtype preservation
  - get_wavelet_mask: binary output, timestep gating, shape correctness
  - setup_wavelet_dwt: DWT module initialization
  - End-to-end: mask application to loss tensor
"""

import pytest
import torch
import sys
import os

# Add parent directory to path so we can import library modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.train_util import (
    setup_wavelet_dwt,
    compute_wavelet_attention_map,
    get_wavelet_mask,
)


@pytest.fixture
def dwt_module():
    """Create a DWT module for testing."""
    return setup_wavelet_dwt(torch.device("cpu"))


class TestSetupWaveletDwt:
    """Tests for setup_wavelet_dwt()."""

    def test_returns_dwt_module(self, dwt_module):
        """DWT module should be a valid PyTorch module."""
        assert dwt_module is not None
        assert hasattr(dwt_module, "forward") or callable(dwt_module)

    def test_module_is_eval_mode(self, dwt_module):
        """DWT module should be in eval mode (no training)."""
        assert not dwt_module.training

    def test_module_params_no_grad(self, dwt_module):
        """DWT module parameters should not require gradients."""
        for param in dwt_module.parameters():
            assert not param.requires_grad

    def test_raises_import_error_without_pytorch_wavelets(self, monkeypatch):
        """Should raise ImportError if pytorch_wavelets is not installed."""
        import importlib
        # Simulate missing pytorch_wavelets by making the import fail
        monkeypatch.setitem(sys.modules, "pytorch_wavelets", None)
        with pytest.raises(ImportError, match="pytorch-wavelets"):
            # Force re-execution of the function by calling it
            # The import is inside the function body, so it should fail
            from library.train_util import setup_wavelet_dwt as setup_fn
            setup_fn(torch.device("cpu"))


class TestComputeWaveletAttentionMap:
    """Tests for compute_wavelet_attention_map()."""

    def test_output_shape_matches_input_spatial(self, dwt_module):
        """Output should have shape (B, H, W) matching input spatial dims."""
        B, C, H, W = 2, 4, 16, 16
        latent = torch.randn(B, C, H, W)
        attn_map = compute_wavelet_attention_map(latent, dwt_module)
        assert attn_map.shape == (B, H, W)

    def test_output_shape_batch_size_1(self, dwt_module):
        """Should work with batch size 1."""
        latent = torch.randn(1, 8, 32, 32)
        attn_map = compute_wavelet_attention_map(latent, dwt_module)
        assert attn_map.shape == (1, 32, 32)

    def test_output_range_zero_to_one(self, dwt_module):
        """Output values should be in [0, 1] after min-max normalization."""
        latent = torch.randn(3, 16, 64, 64)
        attn_map = compute_wavelet_attention_map(latent, dwt_module)
        assert attn_map.min() >= 0.0 - 1e-6
        assert attn_map.max() <= 1.0 + 1e-6

    def test_output_contains_both_extremes(self, dwt_module):
        """For random input, output should contain values close to both 0 and 1."""
        latent = torch.randn(4, 8, 32, 32)
        attn_map = compute_wavelet_attention_map(latent, dwt_module)
        # With random data, we expect min close to 0 and max close to 1
        assert attn_map.min() < 0.1
        assert attn_map.max() > 0.9

    def test_dtype_preservation(self, dwt_module):
        """Output dtype should match input dtype."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            latent = torch.randn(2, 4, 16, 16, dtype=dtype)
            attn_map = compute_wavelet_attention_map(latent, dwt_module)
            assert attn_map.dtype == dtype

    def test_uniform_input_gives_uniform_output(self, dwt_module):
        """Uniform input (constant value) should produce uniform attention map."""
        # A constant latent has zero HF energy everywhere
        latent = torch.ones(2, 4, 16, 16) * 3.14
        attn_map = compute_wavelet_attention_map(latent, dwt_module)
        # All values should be the same (min == max, normalized to 0)
        assert attn_map.std() < 1e-6

    def test_no_grad_computation(self, dwt_module):
        """When called under torch.no_grad(), output should not have grad."""
        latent = torch.randn(2, 4, 16, 16, requires_grad=True)
        with torch.no_grad():
            attn_map = compute_wavelet_attention_map(latent, dwt_module)
        assert not attn_map.requires_grad

    def test_different_channel_counts(self, dwt_module):
        """Should work with different channel counts (C >= 4 typical for VAE latents)."""
        for C in [4, 8, 16, 32]:
            latent = torch.randn(1, C, 16, 16)
            attn_map = compute_wavelet_attention_map(latent, dwt_module)
            assert attn_map.shape == (1, 16, 16)

    def test_high_energy_at_edges(self, dwt_module):
        """A latent with a sharp edge should show high energy at the edge location."""
        B, C, H, W = 1, 4, 32, 32
        latent = torch.zeros(B, C, H, W)
        # Create a sharp vertical edge in the middle
        latent[:, :, :, W // 2:] = 10.0
        attn_map = compute_wavelet_attention_map(latent, dwt_module)
        # Energy should be concentrated near the edge, not at the far left
        center_energy = attn_map[0, :, W // 4].mean()
        edge_energy = attn_map[0, :, W // 2].mean()
        assert edge_energy > center_energy


class TestGetWaveletMask:
    """Tests for get_wavelet_mask()."""

    def test_output_shape(self, dwt_module):
        """Mask should have shape (B, 1, H, W)."""
        B, H, W = 2, 32, 32
        A = torch.rand(B, H, W)
        timesteps = torch.randint(0, 1000, (B,)).float()
        mask = get_wavelet_mask(A, l=0.3, T=1000, timesteps=timesteps)
        assert mask.shape == (B, 1, H, W)

    def test_binary_output(self, dwt_module):
        """Mask values should be exactly 0.0 or 1.0."""
        B, H, W = 4, 16, 16
        A = torch.rand(B, H, W)
        timesteps = torch.randint(0, 1000, (B,)).float()
        mask = get_wavelet_mask(A, l=0.3, T=1000, timesteps=timesteps)
        unique_vals = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_low_timestep_gives_all_ones(self, dwt_module):
        """At timestep 0 (no noise), all regions should be supervised (mask=1)."""
        B, H, W = 2, 16, 16
        A = torch.rand(B, H, W)
        timesteps = torch.zeros(B)
        mask = get_wavelet_mask(A, l=0.3, T=1000, timesteps=timesteps)
        # t=0: T*(A + l) >= 0 is always true
        assert mask.sum() == mask.numel()

    def test_high_timestep_with_zero_l_bound(self, dwt_module):
        """At max timestep with l=0, only high-energy regions should be supervised."""
        B, H, W = 1, 32, 32
        A = torch.zeros(B, H, W)
        A[0, 0, 0] = 1.0  # Single high-energy location
        timesteps = torch.tensor([1000.0])
        mask = get_wavelet_mask(A, l=0.0, T=1000, timesteps=timesteps)
        # T*(A + 0) >= 1000: only where A >= 1.0
        # The high-energy spot should be 1, rest should be 0
        assert mask[0, 0, 0, 0] == 1.0
        # Most of the mask should be 0
        assert mask.sum() < mask.numel()

    def test_l_bound_ensures_minimum_supervision(self, dwt_module):
        """Higher l_bound should result in more regions being supervised."""
        B, H, W = 1, 32, 32
        A = torch.rand(B, H, W)
        timesteps = torch.tensor([500.0])
        
        mask_low_l = get_wavelet_mask(A, l=0.1, T=1000, timesteps=timesteps)
        mask_high_l = get_wavelet_mask(A, l=0.5, T=1000, timesteps=timesteps)
        
        # Higher l_bound -> more mask=1
        assert mask_high_l.sum() >= mask_low_l.sum()

    def test_dtype_preservation(self, dwt_module):
        """Mask dtype should match input attention map dtype."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            A = torch.rand(2, 16, 16, dtype=dtype)
            timesteps = torch.tensor([500.0, 300.0])
            mask = get_wavelet_mask(A, l=0.3, T=1000, timesteps=timesteps)
            assert mask.dtype == dtype

    def test_per_sample_timestep_gating(self, dwt_module):
        """Each sample should be masked according to its own timestep."""
        B, H, W = 2, 16, 16
        A = torch.full((B, H, W), 0.5)  # Uniform attention
        # Sample 0: low timestep -> more mask=1
        # Sample 1: high timestep -> fewer mask=1
        timesteps = torch.tensor([100.0, 900.0])
        mask = get_wavelet_mask(A, l=0.1, T=1000, timesteps=timesteps)
        
        mask_sample_0 = mask[0].sum()
        mask_sample_1 = mask[1].sum()
        assert mask_sample_0 >= mask_sample_1

    def test_flow_matching_timesteps_not_all_ones(self, dwt_module):
        """Flow-matching timesteps in [0, 1] must NOT produce an all-ones mask.

        Regression test for the bug where flow-matching trainers (Flux/SD3/Anima/
        Lumina/Hunyuan) divide timesteps by 1000 before returning them, so
        get_wavelet_mask compared t in [0,1] against T*(A+l) in [300,1300] and
        the mask was always 1.0 (wavelet_mask_ratio stuck at 1.0).
        """
        B, H, W = 2, 32, 32
        A = torch.rand(B, H, W)
        # Flow-matching convention: timesteps in [0, 1]
        timesteps = torch.tensor([0.5, 0.9])
        mask = get_wavelet_mask(A, l=0.3, T=1000, timesteps=timesteps, flow_matching=True)
        # The mask must NOT be all ones (that was the bug)
        assert mask.sum() < mask.numel()
        # And it must contain some zeros
        assert (mask == 0.0).any()

    def test_flow_matching_matches_ddpm_scale(self, dwt_module):
        """A flow-matching timestep t in [0,1] must yield the same mask as t*T in [0,T].

        Eq. 6 is scale-invariant: M_t = 1 iff T*(A+l) >= t iff (A+l) >= t/T.
        So t=0.5 (flow) should equal t=500 (DDPM) for T=1000.
        """
        B, H, W = 2, 32, 32
        torch.manual_seed(42)
        A = torch.rand(B, H, W)
        l = 0.3
        T = 1000

        # Flow-matching convention
        t_flow = torch.tensor([0.25, 0.75])
        mask_flow = get_wavelet_mask(A, l=l, T=T, timesteps=t_flow, flow_matching=True)

        # Equivalent DDPM convention
        t_ddpm = t_flow * T
        mask_ddpm = get_wavelet_mask(A, l=l, T=T, timesteps=t_ddpm, flow_matching=False)

        assert torch.equal(mask_flow, mask_ddpm)

    def test_flow_matching_high_timestep_masks_low_energy(self, dwt_module):
        """At high flow-matching timestep with l=0, low-energy regions must be masked out."""
        B, H, W = 1, 32, 32
        A = torch.zeros(B, H, W)
        A[0, 0, 0] = 1.0  # single high-energy location
        # Flow-matching t=1.0 (max noise)
        timesteps = torch.tensor([1.0])
        mask = get_wavelet_mask(A, l=0.0, T=1000, timesteps=timesteps, flow_matching=True)
        # Only the high-energy spot (A=1) satisfies T*(1+0) >= 1000
        assert mask[0, 0, 0, 0] == 1.0
        # Most of the mask should be 0
        assert mask.sum() < mask.numel()

    def test_flow_matching_ratio_varies_with_timestep(self, dwt_module):
        """The mask ratio should decrease as the flow-matching timestep increases.

        This directly tests the reported symptom: ratio was stuck at 1.0 regardless
        of timestep. After the fix, higher timesteps must yield lower ratios.
        """
        B, H, W = 1, 64, 64
        torch.manual_seed(0)
        A = torch.rand(B, H, W)
        l = 0.3
        T = 1000

        ratios = []
        for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            t = torch.tensor([t_val])
            mask = get_wavelet_mask(A, l=l, T=T, timesteps=t, flow_matching=True)
            ratios.append(mask.mean().item())

        # Ratio at low timestep should be higher than at high timestep
        assert ratios[0] > ratios[-1]
        # And the highest-timestep ratio must be strictly less than 1.0
        assert ratios[-1] < 1.0

    def test_flow_matching_false_with_unit_timesteps_is_all_ones(self, dwt_module):
        """Without flow_matching=True, unit-scale timesteps must stay all-ones.

        This documents the original bug behavior: if a flow-matching trainer
        forgets to pass flow_matching=True, the mask silently degrades to
        all-ones (ratio=1.0). The explicit flag is required for correctness.
        """
        B, H, W = 1, 32, 32
        A = torch.rand(B, H, W)
        timesteps = torch.tensor([0.9])  # unit-scale, but flag omitted
        mask = get_wavelet_mask(A, l=0.3, T=1000, timesteps=timesteps, flow_matching=False)
        # Bug behavior: threshold in [300,1300] >> t=0.9 -> all ones
        assert mask.sum() == mask.numel()


class TestEndToEndMaskApplication:
    """Test that the wavelet mask can be applied to a loss tensor correctly."""

    def test_mask_broadcasts_to_loss(self, dwt_module):
        """Mask (B, 1, H, W) should broadcast correctly with loss (B, C, H, W)."""
        B, C, H, W = 2, 4, 16, 16
        latent = torch.randn(B, C, H, W)
        
        attn_map = compute_wavelet_attention_map(latent, dwt_module)
        timesteps = torch.tensor([100.0, 500.0])
        mask = get_wavelet_mask(attn_map, l=0.3, T=1000, timesteps=timesteps)
        
        # Simulate a loss tensor
        loss = torch.ones(B, C, H, W)
        masked_loss = loss * mask
        
        # Where mask is 0, loss should be 0
        # Where mask is 1, loss should be 1
        assert masked_loss.shape == loss.shape

    def test_mask_preserves_gradient_flow(self, dwt_module):
        """Applying mask should not break gradient computation on the loss."""
        B, C, H, W = 2, 4, 16, 16
        pred = torch.randn(B, C, H, W, requires_grad=True)
        target = torch.randn(B, C, H, W)
        
        # Compute loss
        loss = (pred - target) ** 2  # element-wise
        
        # Compute mask
        latent = torch.randn(B, 4, H, W)  # noisy latent (different from pred)
        with torch.no_grad():
            attn_map = compute_wavelet_attention_map(latent, dwt_module)
            timesteps = torch.tensor([100.0, 500.0])
            mask = get_wavelet_mask(attn_map, l=0.3, T=1000, timesteps=timesteps)
        
        # Apply mask and compute final loss
        masked_loss = (loss * mask).mean()
        masked_loss.backward()
        
        # Gradients should exist and not be NaN
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_zero_mask_gives_zero_loss(self, dwt_module):
        """If mask is all zeros, the masked loss should be zero."""
        B, C, H, W = 2, 4, 16, 16
        loss = torch.randn(B, C, H, W).abs()
        
        # Create a zero mask
        mask = torch.zeros(B, 1, H, W)
        masked_loss = (loss * mask).mean()
        
        assert masked_loss.item() == pytest.approx(0.0)

    def test_full_mask_preserves_loss(self, dwt_module):
        """If mask is all ones, the masked loss should equal the original loss."""
        B, C, H, W = 2, 4, 16, 16
        loss = torch.randn(B, C, H, W).abs()
        
        # Create a full mask
        mask = torch.ones(B, 1, H, W)
        masked_loss = (loss * mask).mean()
        original_loss = loss.mean()
        
        assert masked_loss.item() == pytest.approx(original_loss.item(), rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
