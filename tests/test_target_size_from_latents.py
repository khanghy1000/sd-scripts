"""Regression tests for cached-latent size conditioning (BaseDataset.__getitem__).

Verifies that the latent-derived pixel target size uses the correct axes
(``shape[-1]`` = W, ``shape[-2]`` = H for channels-first latents), matching the
image-path convention ``(W, H)`` -> stored ``(H, W)`` that SDXL size
embeddings and ControlNet crop math consume. Previously the formula used
``(shape[2], shape[1])`` which produced ``(H*8, C*8)`` (e.g. (32, 512) instead
of (512, 512) for SD1.5).
"""

import torch

from library.train_util import target_size_from_latents

VAE_SCALE = 8


def _latent(bucket_wh, channels, ndim):
    """Latent for a bucket (W, H) with the given channel count, on CUDA."""
    bucket_w, bucket_h = bucket_wh
    h_lat, w_lat = bucket_h // VAE_SCALE, bucket_w // VAE_SCALE
    if ndim == 4:
        return torch.zeros(2, channels, h_lat, w_lat, device="cuda")
    return torch.zeros(2, channels, 1, h_lat, w_lat, device="cuda")


def _target_sizes_hw(target_size):
    """Same conversion as BaseDataset.__getitem__ (stored as (H, W))."""
    return (int(target_size[1]), int(target_size[0]))


def test_target_size_width_first_4d():
    """target_size is (W, H); non-square buckets must not be transposed."""
    lat = _latent((1152, 896), channels=4, ndim=4)
    assert target_size_from_latents(lat) == (1152, 896)


def test_target_size_width_first_5d():
    """5D latents (B, C, 1, H, W) use the same trailing (H, W) invariant."""
    lat = _latent((1152, 896), channels=16, ndim=5)
    assert target_size_from_latents(lat) == (1152, 896)


def test_sd15_square_bucket():
    lat = _latent((512, 512), channels=4, ndim=4)
    assert target_size_from_latents(lat) == (512, 512)


def test_sdxl_square_bucket():
    lat = _latent((1024, 1024), channels=4, ndim=4)
    assert target_size_from_latents(lat) == (1024, 1024)


def test_anima_square_bucket_4d_and_5d():
    assert target_size_from_latents(_latent((1024, 1024), 16, 4)) == (1024, 1024)
    assert target_size_from_latents(_latent((1024, 1024), 16, 5)) == (1024, 1024)


def test_stored_target_sizes_hw_matches_consumer_hw_convention():
    """target_sizes_hw must be (h, w) — what SDXL get_size_embeddings expects."""
    lat = _latent((1152, 896), channels=4, ndim=4)
    target_size = target_size_from_latents(lat)
    assert _target_sizes_hw(target_size) == (896, 1152)  # (h, w)

    lat_anima = _latent((1024, 1024), channels=16, ndim=5)
    assert _target_sizes_hw(target_size_from_latents(lat_anima)) == (1024, 1024)


def test_flip_math_uses_width_as_first_component():
    """Flipped crop left = target_width - absolute_right (crop_ltrb[2])."""
    bucket_wh = (1152, 896)
    target_size = target_size_from_latents(_latent(bucket_wh, channels=4, ndim=4))
    crop_ltrb = (100, 50, 1000, 800)  # absolute (left, top, right, bottom)
    flipped_left = target_size[0] - crop_ltrb[2]
    # Mirror of the crop window [left, right) in a width-1152 image:
    assert flipped_left == 1152 - 1000


def test_alpha_mask_fallback_shape_matches_image_path():
    """Missing alpha masks are ones((H, W)) — consistent with the image path.

    Image path: torch.ones((image.shape[1], image.shape[2])) == (H, W).
    Latent path must produce the same (H, W) pixel mask.
    """
    lat = _latent((1152, 896), channels=4, ndim=4)
    mask = torch.ones((lat.shape[-2] * VAE_SCALE, lat.shape[-1] * VAE_SCALE), dtype=torch.float32)
    assert tuple(mask.shape) == (896, 1152)

    lat_5d = _latent((1152, 896), channels=16, ndim=5)
    mask_5d = torch.ones((lat_5d.shape[-2] * VAE_SCALE, lat_5d.shape[-1] * VAE_SCALE), dtype=torch.float32)
    assert tuple(mask_5d.shape) == (896, 1152)
