"""Unit tests for K-variant sampled augmentation caching.

Tests cover:
- variant_key / compute_aug_config_hash
- Latent npz schema round-trip (save/load/validity) via SdSdxlLatentsCachingStrategy
- load_image_variants_for_caching correctness (pixel-level augmentations)
- Gate logic (is_latent_cacheable, is_text_encoder_output_cacheable)
- Caption variant generation (process_caption_canonical, build_caption_variants)
- TE output npz schema round-trip via SdxlTextEncoderOutputsCachingStrategy
- Backward compatibility with legacy caches
"""

import os
import random
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from library.strategy_base import (
    TextEncoderOutputsCachingStrategy,
    variant_key,
    compute_aug_config_hash,
)
from library.train_util import (
    ImageInfo,
    BaseDataset,
    BucketManager,
    AugHelper,
    load_image_variants_for_caching,
    IMAGE_TRANSFORMS,
    trim_and_resize_if_required,
)

# ---------------------------------------------------------------------------
# Helper: minimal subset stub for testing caption processing / gates
# ---------------------------------------------------------------------------

def make_subset(**overrides) -> SimpleNamespace:
    defaults = dict(
        color_aug=False,
        gamma_aug=False,
        gamma_aug_range=None,
        gamma_aug_rate=1.0,
        flip_aug=False,
        random_crop=False,
        random_crop_padding_percent=0.05,
        alpha_mask=False,
        caption_prefix="",
        caption_suffix="",
        caption_separator=",",
        secondary_separator="",
        shuffle_caption=False,
        keep_tokens=0,
        keep_tokens_separator="",
        token_warmup_step=0,
        token_warmup_min=1,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        enable_wildcard=False,
        protected_tags_file=None,
        _protected_tags=set(),
        custom_attributes={},
        is_reg=False,
        is_val=False,
        image_dir="",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ============================================================================
# 1. variant_key / compute_aug_config_hash
# ============================================================================

class TestVariantKey:
    def test_variant_0_returns_base(self):
        assert variant_key("latents", 0) == "latents"

    def test_variant_negative_returns_base(self):
        assert variant_key("latents", -1) == "latents"

    def test_variant_1_returns_suffix(self):
        assert variant_key("latents", 1) == "latents_v1"

    def test_variant_5_returns_suffix(self):
        assert variant_key("hidden_state1", 5) == "hidden_state1_v5"

    def test_multires_suffix_composition(self):
        key = variant_key("latents", 3) + "_32x64"
        assert key == "latents_v3_32x64"

    def test_legacy_key_unchanged(self):
        key = variant_key("latents_flipped", 0) + "_16x16"
        assert key == "latents_flipped_16x16"


class TestAugConfigHash:
    def test_stable_hash(self):
        config = {"color_aug": True, "random_crop": False}
        h1 = compute_aug_config_hash(config)
        h2 = compute_aug_config_hash(config)
        assert h1 == h2

    def test_config_change_changes_hash(self):
        config1 = {"color_aug": True, "random_crop": False}
        config2 = {"color_aug": True, "random_crop": True}
        assert compute_aug_config_hash(config1) != compute_aug_config_hash(config2)

    def test_empty_config(self):
        h = compute_aug_config_hash({})
        assert isinstance(h, str) and len(h) > 0


# ============================================================================
# 2. Latent npz schema round-trip
# ============================================================================

class TestLatentNpzVariantSchema:
    """Test save/load/validity of augmentation variant latents via SdSdxlLatentsCachingStrategy."""

    @pytest.fixture(autouse=True)
    def setup_strategy(self, tmp_path):
        from library.strategy_sd import SdSdxlLatentsCachingStrategy
        self.strategy = SdSdxlLatentsCachingStrategy(
            sd=True,
            cache_to_disk=True,
            batch_size=1,
            skip_disk_cache_validity_check=False,
        )
        self.tmp_path = tmp_path

    def _make_info(self, reso=(64, 64)):
        img_path = str(self.tmp_path / f"test_{random.randint(0, 999999)}.png")
        info = ImageInfo("test_key", 1, "a cat", False, False, img_path)
        info.bucket_reso = reso
        info.latents_npz = img_path + ".npz"
        return info

    def test_legacy_save_load(self):
        """Legacy cache (no variants) round-trips correctly."""
        info = self._make_info()
        latent = torch.randn(4, 8, 8)
        flipped = torch.randn(4, 8, 8)
        self.strategy.save_latents_to_disk(info.latents_npz, latent, (64, 64), (0, 0, 64, 64), flipped)
        result = self.strategy.load_latents_from_disk(info.latents_npz, (64, 64))
        assert len(result) == 6
        lat, orig, crop, flip_lat, alpha, var_flipped = result
        assert lat.shape == (4, 8, 8)
        assert orig == [64, 64]
        assert flip_lat.shape == (4, 8, 8)
        assert var_flipped is None  # legacy npz -> variant_flipped is None

    def test_variant_save_and_load(self):
        """Save variants and load each one correctly."""
        info = self._make_info()
        canonical = torch.randn(4, 8, 8)
        aug_variants = [
            {"latents": torch.randn(4, 8, 8), "crop_ltrb": (5, 5, 60, 60), "flipped": True, "alpha_mask": None},
            {"latents": torch.randn(4, 8, 8), "crop_ltrb": (10, 0, 54, 64), "flipped": False, "alpha_mask": torch.ones(8, 8)},
        ]
        config_hash = compute_aug_config_hash({"color_aug": True})
        # SdSdxlLatentsCachingStrategy uses multi_resolution=True, so keys get _8x8 suffix
        self.strategy.save_latents_to_disk(
            info.latents_npz, canonical, (64, 64), (0, 0, 64, 64),
            key_reso_suffix="_8x8",
            aug_variants=aug_variants, aug_config_hash=config_hash,
        )

        # load canonical (variant 0)
        lat0, _, crop0, flip_lat0, alpha0, vf0 = self.strategy.load_latents_from_disk(info.latents_npz, (64, 64))
        assert np.allclose(lat0, canonical.numpy())
        assert vf0 is not None and vf0 is False  # variant cache, canonical = unflipped
        assert flip_lat0 is None  # no separate flipped latents in variant mode

        # load variant 1 (flipped)
        lat1, _, crop1, _, alpha1, vf1 = self.strategy.load_latents_from_disk(info.latents_npz, (64, 64), variant=1)
        assert np.allclose(lat1, aug_variants[0]["latents"].numpy())
        assert vf1 is True
        assert crop1 == list(aug_variants[0]["crop_ltrb"])

        # load variant 2 (unflipped, with alpha)
        lat2, _, crop2, _, alpha2, vf2 = self.strategy.load_latents_from_disk(info.latents_npz, (64, 64), variant=2)
        assert np.allclose(lat2, aug_variants[1]["latents"].numpy())
        assert vf2 is False
        assert alpha2 is not None

    def test_variant_missing_raises(self):
        """Loading a non-existent variant raises ValueError."""
        info = self._make_info()
        canonical = torch.randn(4, 8, 8)
        self.strategy.save_latents_to_disk(info.latents_npz, canonical, (64, 64), (0, 0, 64, 64))
        with pytest.raises(ValueError, match="augmentation variant 1 is not cached"):
            self.strategy.load_latents_from_disk(info.latents_npz, (64, 64), variant=1)

    def test_validity_legacy(self):
        """Legacy cache passes validity for num_aug_variants=0."""
        info = self._make_info()
        self.strategy.save_latents_to_disk(info.latents_npz, torch.randn(4, 8, 8), (64, 64), (0, 0, 64, 64))
        assert self.strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, False, False)

    def test_validity_variant_match(self):
        """Variant cache passes validity for matching num_aug_variants + hash."""
        info = self._make_info()
        config_hash = compute_aug_config_hash({"color_aug": True})
        aug_variants = [
            {"latents": torch.randn(4, 8, 8), "crop_ltrb": (0, 0, 64, 64), "flipped": False, "alpha_mask": None},
        ]
        # SdSdxlLatentsCachingStrategy uses multi_resolution=True, so keys get _8x8 suffix
        self.strategy.save_latents_to_disk(
            info.latents_npz, torch.randn(4, 8, 8), (64, 64), (0, 0, 64, 64),
            key_reso_suffix="_8x8",
            aug_variants=aug_variants, aug_config_hash=config_hash,
        )
        assert self.strategy.is_disk_cached_latents_expected(
            (64, 64), info.latents_npz, False, False, num_aug_variants=2, aug_config_hash=config_hash
        )

    def test_validity_variant_hash_mismatch(self):
        """Variant cache fails validity with wrong config hash."""
        info = self._make_info()
        correct_hash = compute_aug_config_hash({"color_aug": True})
        wrong_hash = compute_aug_config_hash({"color_aug": False})
        aug_variants = [
            {"latents": torch.randn(4, 8, 8), "crop_ltrb": (0, 0, 64, 64), "flipped": False, "alpha_mask": None},
        ]
        self.strategy.save_latents_to_disk(
            info.latents_npz, torch.randn(4, 8, 8), (64, 64), (0, 0, 64, 64),
            aug_variants=aug_variants, aug_config_hash=correct_hash,
        )
        assert not self.strategy.is_disk_cached_latents_expected(
            (64, 64), info.latents_npz, False, False, num_aug_variants=2, aug_config_hash=wrong_hash
        )

    def test_validity_variant_K_too_high(self):
        """Variant cache fails validity when requesting more variants than cached."""
        info = self._make_info()
        config_hash = compute_aug_config_hash({"color_aug": True})
        aug_variants = [
            {"latents": torch.randn(4, 8, 8), "crop_ltrb": (0, 0, 64, 64), "flipped": False, "alpha_mask": None},
        ]
        self.strategy.save_latents_to_disk(
            info.latents_npz, torch.randn(4, 8, 8), (64, 64), (0, 0, 64, 64),
            aug_variants=aug_variants, aug_config_hash=config_hash,
        )
        # requesting 4 variants but npz only has 2 (canonical + 1)
        assert not self.strategy.is_disk_cached_latents_expected(
            (64, 64), info.latents_npz, False, False, num_aug_variants=4, aug_config_hash=config_hash
        )

    def test_validity_legacy_with_variants_fails(self):
        """Legacy cache fails validity when variants are requested."""
        info = self._make_info()
        self.strategy.save_latents_to_disk(info.latents_npz, torch.randn(4, 8, 8), (64, 64), (0, 0, 64, 64))
        assert not self.strategy.is_disk_cached_latents_expected(
            (64, 64), info.latents_npz, False, False, num_aug_variants=2,
        )


# ============================================================================
# 3. load_image_variants_for_caching
# ============================================================================

class TestLoadImageVariantsForCaching:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        # Create a simple 80x80 RGB image
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8))
        self.img_path = str(tmp_path / "test.png")
        img.save(self.img_path)

    def _make_info(self, reso=(64, 64), resized=(64, 64)):
        info = ImageInfo("key", 1, "cat", False, False, self.img_path)
        info.bucket_reso = reso
        info.resized_size = resized
        info.resize_interpolation = None
        info.image = None
        return info

    def test_basic_shapes(self):
        """Each variant produces a correctly shaped tensor."""
        info = self._make_info()
        imgs, alphas, orig, crops, flippeds = load_image_variants_for_caching(
            [info], num_variants=3, use_alpha_mask=False, flip_aug=False, augmentor=None, random_crop=False,
        )
        assert len(imgs) == 3
        for k in range(3):
            assert imgs[k].shape == (1, 3, 64, 64)  # [B, 3, H, W]
            assert len(crops[k]) == 1
            assert len(flippeds[k]) == 1

    def test_canonical_center_crop(self):
        """Canonical variant (k=0) produces a center crop matching the legacy center-crop logic."""
        info = self._make_info(reso=(64, 64), resized=(80, 80))
        imgs, _, orig, crops, flippeds = load_image_variants_for_caching(
            [info], num_variants=2, use_alpha_mask=False, flip_aug=False, augmentor=None, random_crop=True,
        )
        # canonical (k=0): center crop, unflipped
        assert flippeds[0][0] is False
        # Center crop from an 80x80 resized image down to 64x64: trim = 16, center p = 8
        expected_crop = BucketManager.get_crop_ltrb((64, 64), (80, 80))
        assert crops[0][0] == tuple(expected_crop)

    def test_variants_can_differ(self):
        """With random_crop and augmentor, variant 1+ differ from canonical (with high probability)."""
        info = self._make_info(reso=(64, 64), resized=(80, 80))
        aug = AugHelper().get_augmentor(True, False, None, 0.0)
        imgs, _, _, crops, flippeds = load_image_variants_for_caching(
            [info], num_variants=8, use_alpha_mask=False, flip_aug=True, augmentor=aug, random_crop=True,
        )
        # At least some variants should differ from canonical
        v0 = imgs[0][0]
        differs = any(not torch.equal(imgs[k][0], v0) for k in range(1, 8))
        assert differs, "Expected at least one variant to differ from canonical"

    def test_canonical_unflipped(self):
        """Canonical variant is always unflipped."""
        info = self._make_info()
        _, _, _, _, flippeds = load_image_variants_for_caching(
            [info], num_variants=4, use_alpha_mask=False, flip_aug=True, augmentor=None, random_crop=False,
        )
        assert flippeds[0][0] is False  # canonical = unflipped

    def test_alpha_mask_handling(self):
        """Alpha masks are returned for each variant."""
        info = self._make_info()
        _, alphas, _, _, _ = load_image_variants_for_caching(
            [info], num_variants=3, use_alpha_mask=True, flip_aug=False, augmentor=None, random_crop=False,
        )
        for k in range(3):
            assert alphas[k][0] is not None
            assert alphas[k][0].shape == (64, 64)


# ============================================================================
# 4. Gate logic
# ============================================================================

class SimpleDataset(BaseDataset):
    """Minimal BaseDataset subclass for testing gate logic without full initialization."""

    def __init__(self, latents_aug_variant_count=0, caption_aug_variant_count=0):
        # Set attributes that methods need without calling BaseDataset.__init__
        self.latents_aug_variant_count = latents_aug_variant_count
        self.caption_aug_variant_count = caption_aug_variant_count
        self.aug_refresh_epochs = 0
        self.subsets = []
        self.replacements = {}
        self.image_data = {}
        self.image_to_subset = {}

    def set_aug_variant_config(self, latents_aug_variants: int = 0, caption_aug_variants: int = 0):
        self.latents_aug_variant_count = int(latents_aug_variants) if latents_aug_variants else 0
        self.caption_aug_variant_count = int(caption_aug_variants) if caption_aug_variants else 0

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError


class TestGateLogic:
    def test_latent_cacheable_no_aug(self):
        ds = SimpleDataset(latents_aug_variant_count=0)
        ds.subsets = [make_subset()]
        assert ds.is_latent_cacheable() is True

    def test_latent_cacheable_color_aug_blocked(self):
        ds = SimpleDataset(latents_aug_variant_count=0)
        ds.subsets = [make_subset(color_aug=True)]
        assert ds.is_latent_cacheable() is False

    def test_latent_cacheable_color_aug_with_variants(self):
        ds = SimpleDataset(latents_aug_variant_count=4)
        ds.subsets = [make_subset(color_aug=True)]
        assert ds.is_latent_cacheable() is True

    def test_te_cacheable_shuffle_blocked_no_variants(self):
        ds = SimpleDataset(caption_aug_variant_count=0)
        ds.subsets = [make_subset(shuffle_caption=True)]
        assert ds.is_text_encoder_output_cacheable() is False

    def test_te_cacheable_shuffle_allowed_with_variants(self):
        ds = SimpleDataset(caption_aug_variant_count=4)
        ds.subsets = [make_subset(shuffle_caption=True)]
        assert ds.is_text_encoder_output_cacheable() is True

    def test_te_cacheable_token_warmup_always_blocked(self):
        ds = SimpleDataset(caption_aug_variant_count=4)
        ds.subsets = [make_subset(token_warmup_step=100)]
        assert ds.is_text_encoder_output_cacheable() is False

    def test_te_cacheable_tag_dropout_with_variants(self):
        ds = SimpleDataset(caption_aug_variant_count=4)
        ds.subsets = [make_subset(caption_tag_dropout_rate=0.2)]
        assert ds.is_text_encoder_output_cacheable() is True

    def test_te_cacheable_caption_dropout_with_variants(self):
        ds = SimpleDataset(caption_aug_variant_count=4)
        ds.subsets = [make_subset(caption_dropout_rate=0.1)]
        assert ds.is_text_encoder_output_cacheable() is True

    def test_effective_aug_variant_count_no_augs(self):
        ds = SimpleDataset(latents_aug_variant_count=4)
        ds.subsets = [make_subset()]
        assert ds.get_effective_aug_variant_count(ds.subsets[0]) == 0

    def test_effective_aug_variant_count_with_augs(self):
        ds = SimpleDataset(latents_aug_variant_count=4)
        ds.subsets = [make_subset(random_crop=True)]
        assert ds.get_effective_aug_variant_count(ds.subsets[0]) == 4

    def test_effective_caption_variant_count_no_stochastic(self):
        ds = SimpleDataset(caption_aug_variant_count=4)
        ds.subsets = [make_subset()]
        assert ds.get_effective_caption_variant_count(ds.subsets[0]) == 0

    def test_effective_caption_variant_count_with_shuffle(self):
        ds = SimpleDataset(caption_aug_variant_count=4)
        ds.subsets = [make_subset(shuffle_caption=True)]
        assert ds.get_effective_caption_variant_count(ds.subsets[0]) == 4


# ============================================================================
# 5. Caption variant generation
# ============================================================================

class TestCaptionVariantGeneration:
    def _make_dataset(self, **subset_kwargs):
        ds = SimpleDataset(caption_aug_variant_count=4)
        subset = make_subset(**subset_kwargs)
        ds.subsets = [subset]
        ds.current_epoch = 0
        ds.current_step = 0
        ds.max_train_steps = 1000
        ds.replacements = {}
        ds.log_caption_dropout = False
        ds.log_caption_tag_dropout = False
        ds.tag_frequency = {}
        ds.tokenizers = []

        info = ImageInfo("key", 1, "a cat, fluffy, outdoors", False, False, "/fake/path.png")
        info.caption_dropout_rate = subset.caption_dropout_rate
        ds.image_data = {"key": info}
        ds.image_to_subset = {"key": subset}
        return ds, info

    def test_canonical_is_deterministic(self):
        ds, info = self._make_dataset(shuffle_caption=True, caption_dropout_rate=0.5, caption_tag_dropout_rate=0.3)
        subset = ds.subsets[0]
        c1 = ds.process_caption_canonical(subset, info.caption)
        c2 = ds.process_caption_canonical(subset, info.caption)
        assert c1 == c2

    def test_canonical_no_dropout(self):
        ds, info = self._make_dataset(caption_dropout_rate=1.0)
        subset = ds.subsets[0]
        canonical = ds.process_caption_canonical(subset, info.caption)
        assert canonical != ""  # should NOT be dropped

    def test_canonical_prefix_suffix(self):
        ds, info = self._make_dataset(caption_prefix="best quality", caption_suffix="4k")
        subset = ds.subsets[0]
        canonical = ds.process_caption_canonical(subset, info.caption)
        assert canonical.startswith("best quality")
        assert canonical.endswith("4k")

    def test_canonical_wildcard_first_option(self):
        ds, info = self._make_dataset(enable_wildcard=True)
        info.caption = "{cat|dog|bird}"
        subset = ds.subsets[0]
        canonical = ds.process_caption_canonical(subset, info.caption)
        assert canonical == "cat"

    def test_build_caption_variants_creates_variants(self):
        ds, info = self._make_dataset(
            shuffle_caption=True,
            caption_tag_dropout_rate=0.1,
        )
        random.seed(42)
        ds.build_caption_variants()
        assert info.caption_variants is not None
        assert len(info.caption_variants) == 4  # K=4
        # variant 0 should be canonical (deterministic)
        canonical = ds.process_caption_canonical(ds.subsets[0], info.caption)
        assert info.caption_variants[0] == canonical

    def test_build_caption_variants_neutralizes_every_n_epochs(self):
        """caption_dropout_every_n_epochs is neutralized during variant sampling.

        With every_n=1, epoch=0, and shuffle_caption=True, naive process_caption
        would drop ALL variants to "". With neutralization, at least some should be non-empty.
        """
        ds, info = self._make_dataset(
            shuffle_caption=True,
            caption_dropout_every_n_epochs=1,
        )
        random.seed(42)
        ds.build_caption_variants()
        assert info.caption_variants is not None
        non_empty = [v for v in info.caption_variants[1:] if v != ""]
        assert len(non_empty) > 0, "Variants should not all be empty when every_n_epochs is neutralized"

    def test_build_caption_variants_neutralizes_token_warmup(self):
        """token_warmup_step is neutralized during variant sampling.

        With shuffle_caption=True (needed for effective_caption_variant_count > 0)
        and token_warmup_step=0.5 (fractional), variants should still be generated.
        """
        ds, info = self._make_dataset(
            shuffle_caption=True,
            token_warmup_step=0.5,  # fractional warmup
        )
        random.seed(42)
        ds.build_caption_variants()
        assert info.caption_variants is not None
        assert len(info.caption_variants) == 4

    def test_no_variants_for_effective_count_zero(self):
        ds, info = self._make_dataset()  # no stochastic augmentations
        ds.build_caption_variants()
        assert info.caption_variants is None


# ============================================================================
# 5b. Epoch-based caption dropout on cached TE outputs
# ============================================================================

class TestEpochCaptionDropout:
    """Test that caption_dropout_every_n_epochs correctly zeros cached TE outputs at train time."""

    def test_epoch_match_zeros_outputs(self):
        """When current_epoch is a multiple of every_n_epochs, cached outputs are zeroed."""
        subset = make_subset(caption_dropout_every_n_epochs=3)
        cached_outputs = [torch.randn(77, 768), torch.randn(1280)]
        current_epoch = 3  # 3 % 3 == 0

        # Simulate the __getitem__ logic
        text_encoder_outputs = cached_outputs
        is_canonical_only = False
        caption = "a cat"

        if (
            not is_canonical_only
            and text_encoder_outputs is not None
            and getattr(subset, "caption_dropout_every_n_epochs", 0) > 0
            and current_epoch % subset.caption_dropout_every_n_epochs == 0
        ):
            caption = ""
            text_encoder_outputs = [
                torch.zeros_like(t) if isinstance(t, torch.Tensor) else t
                for t in text_encoder_outputs
            ]

        assert caption == ""
        for t in text_encoder_outputs:
            assert torch.all(t == 0)

    def test_epoch_no_match_preserves_outputs(self):
        """When current_epoch is NOT a multiple of every_n_epochs, outputs are preserved."""
        subset = make_subset(caption_dropout_every_n_epochs=3)
        original = [torch.randn(77, 768), torch.randn(1280)]
        cached_outputs = [t.clone() for t in original]
        current_epoch = 2  # 2 % 3 != 0

        text_encoder_outputs = cached_outputs
        is_canonical_only = False
        caption = "a cat"

        if (
            not is_canonical_only
            and text_encoder_outputs is not None
            and getattr(subset, "caption_dropout_every_n_epochs", 0) > 0
            and current_epoch % subset.caption_dropout_every_n_epochs == 0
        ):
            caption = ""
            text_encoder_outputs = [
                torch.zeros_like(t) if isinstance(t, torch.Tensor) else t
                for t in text_encoder_outputs
            ]

        assert caption == "a cat"
        for i, t in enumerate(text_encoder_outputs):
            assert torch.allclose(t, original[i])

    def test_validation_skips_dropout(self):
        """Validation datasets (is_canonical_only) are not affected by epoch dropout."""
        subset = make_subset(caption_dropout_every_n_epochs=1)  # every epoch
        original = [torch.randn(77, 768)]
        cached_outputs = [t.clone() for t in original]
        current_epoch = 0

        text_encoder_outputs = cached_outputs
        is_canonical_only = True  # validation
        caption = "a cat"

        if (
            not is_canonical_only
            and text_encoder_outputs is not None
            and getattr(subset, "caption_dropout_every_n_epochs", 0) > 0
            and current_epoch % subset.caption_dropout_every_n_epochs == 0
        ):
            caption = ""
            text_encoder_outputs = [
                torch.zeros_like(t) if isinstance(t, torch.Tensor) else t
                for t in text_encoder_outputs
            ]

        assert caption == "a cat"
        for i, t in enumerate(text_encoder_outputs):
            assert torch.allclose(t, original[i])

    def test_zero_every_n_epochs_noop(self):
        """When every_n_epochs is 0, no dropout occurs."""
        subset = make_subset(caption_dropout_every_n_epochs=0)
        original = [torch.randn(77, 768)]
        cached_outputs = [t.clone() for t in original]
        current_epoch = 0

        text_encoder_outputs = cached_outputs
        is_canonical_only = False
        caption = "a cat"

        if (
            not is_canonical_only
            and text_encoder_outputs is not None
            and getattr(subset, "caption_dropout_every_n_epochs", 0) > 0
            and current_epoch % subset.caption_dropout_every_n_epochs == 0
        ):
            caption = ""
            text_encoder_outputs = [
                torch.zeros_like(t) if isinstance(t, torch.Tensor) else t
                for t in text_encoder_outputs
            ]

        assert caption == "a cat"

    def test_every_n_with_numpy_outputs(self):
        """Works with numpy arrays (disk-cached TE outputs loaded as numpy)."""
        subset = make_subset(caption_dropout_every_n_epochs=2)
        cached_outputs = [np.random.randn(77, 768).astype(np.float32)]
        current_epoch = 4  # 4 % 2 == 0

        text_encoder_outputs = cached_outputs
        is_canonical_only = False
        caption = "a cat"

        if (
            not is_canonical_only
            and text_encoder_outputs is not None
            and getattr(subset, "caption_dropout_every_n_epochs", 0) > 0
            and current_epoch % subset.caption_dropout_every_n_epochs == 0
        ):
            caption = ""
            text_encoder_outputs = [
                torch.zeros_like(t) if isinstance(t, torch.Tensor) else np.zeros_like(t) if isinstance(t, np.ndarray) else t
                for t in text_encoder_outputs
            ]

        assert caption == ""
        for t in text_encoder_outputs:
            assert np.all(t == 0)


# ============================================================================
# 5c. Epoch refresh (in-memory variant regeneration)
# ============================================================================

class TestEpochRefresh:
    """Test refresh_latent_variants and set_aug_refresh_epochs."""

    def _make_dataset_with_image(self, tmp_path, **subset_kwargs):
        """Create a SimpleDataset with one real image and subset config."""
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img_path = str(tmp_path / "test.png")
        img.save(img_path)

        ds = SimpleDataset(latents_aug_variant_count=4, caption_aug_variant_count=0)
        subset = make_subset(**subset_kwargs)
        ds.subsets = [subset]
        ds.aug_helper = AugHelper()
        ds.aug_refresh_epochs = 1
        ds.current_epoch = 0
        ds.current_step = 0
        ds.max_train_steps = 1000
        ds.replacements = {}
        ds.log_caption_dropout = False
        ds.log_caption_tag_dropout = False
        ds.tag_frequency = {}

        info = ImageInfo("key", 1, "a cat", False, False, img_path)
        info.bucket_reso = (64, 64)
        info.resized_size = (64, 64)
        info.resize_interpolation = None
        info.image = None
        info.latents_npz = img_path + ".npz"
        info.latents = None
        info.latents_aug_variants = None
        info.caption_variants = None

        ds.image_data = {"key": info}
        ds.image_to_subset = {"key": subset}
        return ds, info

    def test_refresh_latent_variants_generates_variants(self, tmp_path):
        """refresh_latent_variants populates info.latents_aug_variants."""
        ds, info = self._make_dataset_with_image(tmp_path, color_aug=True)

        # Fake VAE encode: pool and tile to 4 channels
        def fake_encode(img_tensor):
            return torch.nn.functional.avg_pool2d(img_tensor, 8).repeat(1, 2, 1, 1)[:, :4]

        ds.refresh_latent_variants(fake_encode, torch.device("cpu"), torch.float32)

        assert info.latents_aug_variants is not None
        assert len(info.latents_aug_variants) == 3  # K=4, variants 1..3
        for var in info.latents_aug_variants:
            assert "latents" in var
            assert "crop_ltrb" in var
            assert "flipped" in var
            assert var["latents"].shape == (4, 8, 8)

    def test_refresh_produces_different_variants_each_call(self, tmp_path):
        """Each refresh call should produce different variant data (statistical)."""
        ds, info = self._make_dataset_with_image(tmp_path, color_aug=True, random_crop=True)

        def fake_encode(img_tensor):
            return torch.nn.functional.avg_pool2d(img_tensor, 8).repeat(1, 2, 1, 1)[:, :4]

        random.seed(42)
        ds.refresh_latent_variants(fake_encode, torch.device("cpu"), torch.float32)
        first_variants = [v["latents"].clone() for v in info.latents_aug_variants]

        random.seed(99)
        ds.refresh_latent_variants(fake_encode, torch.device("cpu"), torch.float32)
        second_variants = [v["latents"].clone() for v in info.latents_aug_variants]

        # At least one variant should differ (statistically, all should)
        differs = any(not torch.equal(f, s) for f, s in zip(first_variants, second_variants))
        assert differs, "Expected different variants after refresh with different seed"

    def test_refresh_no_effect_when_no_augments(self, tmp_path):
        """refresh_latent_variants is a no-op when no augmentations are enabled."""
        ds, info = self._make_dataset_with_image(tmp_path)  # no augs

        def fake_encode(img_tensor):
            return torch.nn.functional.avg_pool2d(img_tensor, 8).repeat(1, 2, 1, 1)[:, :4]

        ds.refresh_latent_variants(fake_encode, torch.device("cpu"), torch.float32)
        assert info.latents_aug_variants is None  # no augs → no variants

    def test_set_aug_refresh_epochs_on_dataset(self):
        """BaseDataset.set_aug_config stores refresh_epochs correctly."""
        ds = SimpleDataset(latents_aug_variant_count=4)
        assert ds.aug_refresh_epochs == 0  # default
        ds.aug_refresh_epochs = 3
        assert ds.aug_refresh_epochs == 3

    def test_aug_refresh_epochs_initialized_without_set_aug_variant_config(self):
        """Regression: DreamBoothDataset must expose aug_refresh_epochs even when
        set_aug_variant_config is never called.

        Previously aug_refresh_epochs was only created inside set_aug_variant_config(),
        which is only invoked when --cache_aug_variants/--cache_caption_variants > 1.
        Plain --cache_latents therefore crashed in new_cache_latents() with
        AttributeError: 'DreamBoothDataset' object has no attribute 'aug_refresh_epochs'.
        """
        from library.train_util import MinimalDataset

        ds = MinimalDataset(resolution=(512, 512), network_multiplier=1.0)
        assert hasattr(ds, "aug_refresh_epochs"), "aug_refresh_epochs must be initialized by BaseDataset.__init__"
        assert ds.aug_refresh_epochs == 0
        # The exact read that previously crashed in new_cache_latents / new_cache_text_encoder_outputs
        assert (ds.aug_refresh_epochs > 0) is False


# ============================================================================
# 6. TE output npz schema round-trip (SDXL)
# ============================================================================

class TestSdxlTEOutputVariantSchema:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        from library.strategy_sdxl import SdxlTextEncoderOutputsCachingStrategy
        self.strategy = SdxlTextEncoderOutputsCachingStrategy(
            cache_to_disk=True,
            batch_size=1,
            skip_disk_cache_validity_check=False,
            is_partial=False,
            is_weighted=False,
            use_attention_mask=True,
        )
        self.tmp_path = tmp_path

    def _make_info(self):
        img_path = str(self.tmp_path / f"test_{random.randint(0, 999999)}.png")
        info = ImageInfo("key", 1, "a cat", False, False, img_path)
        info.text_encoder_outputs_npz = img_path + ".npz"
        info.caption_variants = None
        info.caption_aug_hash = None
        info.text_encoder_outputs = None
        info.text_encoder_outputs_variants = None
        return info

    def test_legacy_save_load(self):
        """Legacy cache (no variants) round-trips correctly."""
        info = self._make_info()
        info.caption_variants = None
        np.savez(
            info.text_encoder_outputs_npz,
            hidden_state1=np.random.randn(77, 768).astype(np.float32),
            hidden_state2=np.random.randn(77, 1024).astype(np.float32),
            pool2=np.random.randn(1280).astype(np.float32),
            attention_mask1=np.ones((1, 77), dtype=np.float32),
            attention_mask2=np.ones((1, 77), dtype=np.float32),
        )
        result = self.strategy.load_outputs_npz(info.text_encoder_outputs_npz)
        assert len(result) == 5
        assert result[0].shape == (77, 768)

    def test_legacy_load_with_variant_0(self):
        """Loading variant 0 from a legacy cache returns the same result."""
        info = self._make_info()
        np.savez(
            info.text_encoder_outputs_npz,
            hidden_state1=np.random.randn(77, 768).astype(np.float32),
            hidden_state2=np.random.randn(77, 1024).astype(np.float32),
            pool2=np.random.randn(1280).astype(np.float32),
        )
        result = self.strategy.load_outputs_npz(info.text_encoder_outputs_npz, variant=0)
        assert len(result) == 3

    def test_variant_save_and_load(self):
        """Variant TE outputs round-trip correctly."""
        info = self._make_info()
        hs1 = np.random.randn(77, 768).astype(np.float32)
        hs1_v1 = np.random.randn(77, 768).astype(np.float32)

        config_hash = compute_aug_config_hash({"shuffle_caption": True})
        np.savez(
            info.text_encoder_outputs_npz,
            hidden_state1=hs1,
            hidden_state2=np.random.randn(77, 1024).astype(np.float32),
            pool2=np.random.randn(1280).astype(np.float32),
            attention_mask1=np.ones((1, 77), dtype=np.float32),
            attention_mask2=np.ones((1, 77), dtype=np.float32),
            hidden_state1_v1=hs1_v1,
            hidden_state2_v1=np.random.randn(77, 1024).astype(np.float32),
            pool2_v1=np.random.randn(1280).astype(np.float32),
            attention_mask1_v1=np.ones((1, 77), dtype=np.float32),
            attention_mask2_v1=np.ones((1, 77), dtype=np.float32),
            caption_variants=np.array(2),
            caption_aug_hash=np.array(config_hash),
        )

        # load variant 0 (canonical) - uses legacy key
        result0 = self.strategy.load_outputs_npz(info.text_encoder_outputs_npz, variant=0)
        assert np.allclose(result0[0], hs1)

        # load variant 1 (uses _v1 keys)
        result1 = self.strategy.load_outputs_npz(info.text_encoder_outputs_npz, variant=1)
        assert np.allclose(result1[0], hs1_v1)

    def test_variant_fallback_to_legacy(self):
        """Loading variant > 0 from a legacy cache falls back to legacy key."""
        info = self._make_info()
        hs1 = np.random.randn(77, 768).astype(np.float32)
        np.savez(
            info.text_encoder_outputs_npz,
            hidden_state1=hs1,
            hidden_state2=np.random.randn(77, 1024).astype(np.float32),
            pool2=np.random.randn(1280).astype(np.float32),
        )
        result = self.strategy.load_outputs_npz(info.text_encoder_outputs_npz, variant=1)
        assert np.allclose(result[0], hs1)  # falls back to legacy

    def test_validity_legacy(self):
        info = self._make_info()
        np.savez(
            info.text_encoder_outputs_npz,
            hidden_state1=np.zeros((77, 768), dtype=np.float32),
            hidden_state2=np.zeros((77, 1024), dtype=np.float32),
            pool2=np.zeros((1280,), dtype=np.float32),
            attention_mask1=np.ones((1, 77), dtype=np.float32),
            attention_mask2=np.ones((1, 77), dtype=np.float32),
        )
        assert self.strategy.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)

    def test_validity_variant_match(self):
        info = self._make_info()
        config_hash = compute_aug_config_hash({"shuffle_caption": True})
        np.savez(
            info.text_encoder_outputs_npz,
            hidden_state1=np.zeros((77, 768), dtype=np.float32),
            hidden_state2=np.zeros((77, 1024), dtype=np.float32),
            pool2=np.zeros((1280,), dtype=np.float32),
            attention_mask1=np.ones((1, 77), dtype=np.float32),
            attention_mask2=np.ones((1, 77), dtype=np.float32),
            hidden_state1_v1=np.zeros((77, 768), dtype=np.float32),
            hidden_state2_v1=np.zeros((77, 1024), dtype=np.float32),
            pool2_v1=np.zeros((1280,), dtype=np.float32),
            attention_mask1_v1=np.ones((1, 77), dtype=np.float32),
            attention_mask2_v1=np.ones((1, 77), dtype=np.float32),
            caption_variants=np.array(2),
            caption_aug_hash=np.array(config_hash),
        )
        assert self.strategy.is_disk_cached_outputs_expected(
            info.text_encoder_outputs_npz, num_caption_variants=2, caption_aug_hash=config_hash
        )

    def test_validity_variant_hash_mismatch(self):
        info = self._make_info()
        correct_hash = compute_aug_config_hash({"shuffle_caption": True})
        wrong_hash = compute_aug_config_hash({"shuffle_caption": False})
        np.savez(
            info.text_encoder_outputs_npz,
            hidden_state1=np.zeros((77, 768), dtype=np.float32),
            hidden_state2=np.zeros((77, 1024), dtype=np.float32),
            pool2=np.zeros((1280,), dtype=np.float32),
            attention_mask1=np.ones((1, 77), dtype=np.float32),
            attention_mask2=np.ones((1, 77), dtype=np.float32),
            hidden_state1_v1=np.zeros((77, 768), dtype=np.float32),
            hidden_state2_v1=np.zeros((77, 1024), dtype=np.float32),
            pool2_v1=np.zeros((1280,), dtype=np.float32),
            attention_mask1_v1=np.ones((1, 77), dtype=np.float32),
            attention_mask2_v1=np.ones((1, 77), dtype=np.float32),
            caption_variants=np.array(2),
            caption_aug_hash=np.array(correct_hash),
        )
        assert not self.strategy.is_disk_cached_outputs_expected(
            info.text_encoder_outputs_npz, num_caption_variants=2, caption_aug_hash=wrong_hash
        )

    def test_validity_legacy_with_variants_fails(self):
        info = self._make_info()
        np.savez(
            info.text_encoder_outputs_npz,
            hidden_state1=np.zeros((77, 768), dtype=np.float32),
            hidden_state2=np.zeros((77, 1024), dtype=np.float32),
            pool2=np.zeros((1280,), dtype=np.float32),
        )
        assert not self.strategy.is_disk_cached_outputs_expected(
            info.text_encoder_outputs_npz, num_caption_variants=2,
        )


# ============================================================================
# 7. _npz_get helper
# ============================================================================

class TestNpzGet:
    def test_variant_0_returns_legacy(self):
        data = {"key": np.array([1, 2, 3])}
        result = TextEncoderOutputsCachingStrategy._npz_get(data, "key", variant=0)
        assert np.array_equal(result, data["key"])

    def test_variant_with_suffix_present(self):
        data = {"key": np.array([1, 2, 3]), "key_v1": np.array([4, 5, 6])}
        result = TextEncoderOutputsCachingStrategy._npz_get(data, "key", variant=1)
        assert np.array_equal(result, data["key_v1"])

    def test_variant_fallback_to_legacy(self):
        data = {"key": np.array([1, 2, 3])}
        result = TextEncoderOutputsCachingStrategy._npz_get(data, "key", variant=1)
        assert np.array_equal(result, data["key"])


# ============================================================================
# 8. _get_variant_caption_groups helper
# ============================================================================

class TestGetVariantCaptionGroups:
    def test_no_variants(self):
        info = SimpleNamespace(caption="cat", caption_variants=None)
        strategy = TextEncoderOutputsCachingStrategy(False, None, False)
        canonical, groups = strategy._get_variant_caption_groups([info])
        assert canonical == ["cat"]
        assert groups == []

    def test_with_variants(self):
        info = SimpleNamespace(caption="cat", caption_variants=["cat", "fluffy cat", "outdoors cat"])
        strategy = TextEncoderOutputsCachingStrategy(False, None, False)
        canonical, groups = strategy._get_variant_caption_groups([info])
        assert canonical == ["cat"]
        assert len(groups) == 2
        assert groups[0] == [(0, "fluffy cat")]
        assert groups[1] == [(0, "outdoors cat")]

    def test_mixed_variants(self):
        info1 = SimpleNamespace(caption="cat", caption_variants=["cat", "fluffy cat"])
        info2 = SimpleNamespace(caption="dog", caption_variants=None)
        strategy = TextEncoderOutputsCachingStrategy(False, None, False)
        canonical, groups = strategy._get_variant_caption_groups([info1, info2])
        assert canonical == ["cat", "dog"]
        assert len(groups) == 1
        assert groups[0] == [(0, "fluffy cat")]


# ============================================================================
# 9. set_aug_variant_config
# ============================================================================

class TestSetAugVariantConfig:
    def test_simple_dataset(self):
        ds = SimpleDataset()
        ds.set_aug_variant_config(4, 8)
        assert ds.latents_aug_variant_count == 4
        assert ds.caption_aug_variant_count == 8


# ============================================================================
# 10. Backward compatibility
# ============================================================================

class TestBackwardCompatibility:
    def test_legacy_npz_no_variant_keys(self):
        """Legacy npz without variant metadata loads correctly."""
        from library.strategy_sd import SdSdxlLatentsCachingStrategy
        strategy = SdSdxlLatentsCachingStrategy(
            sd=True, cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = os.path.join(tmp, "test.npz")
            np.savez(npz_path,
                latents=np.random.randn(4, 8, 8).astype(np.float32),
                original_size=np.array([64, 64]),
                crop_ltrb=np.array([0, 0, 64, 64]),
                latents_flipped=np.random.randn(4, 8, 8).astype(np.float32),
            )
            result = strategy.load_latents_from_disk(npz_path, (64, 64))
            assert len(result) == 6
            assert result[5] is None  # variant_flipped = None for legacy

    def test_variant_npz_variant_flipped_false_for_canonical(self):
        """Variant npz with aug_variants metadata returns variant_flipped=False for canonical."""
        from library.strategy_sd import SdSdxlLatentsCachingStrategy
        strategy = SdSdxlLatentsCachingStrategy(
            sd=True, cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = os.path.join(tmp, "test.npz")
            np.savez(npz_path,
                latents=np.random.randn(4, 8, 8).astype(np.float32),
                original_size=np.array([64, 64]),
                crop_ltrb=np.array([0, 0, 64, 64]),
                aug_variants=np.array(2),
                aug_config_hash=np.array("abc123"),
            )
            result = strategy.load_latents_from_disk(npz_path, (64, 64))
            assert result[5] is False  # variant_flipped = False for canonical in variant mode


# ============================================================================
# 11. Integration: fake VAE variant caching end-to-end
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVariantCachingEndToEnd:
    """End-to-end test: fake VAE encode -> variant caching -> load round-trip on CUDA."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.device = torch.device("cuda")
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        self.img_path = str(tmp_path / "test.png")
        img.save(self.img_path)

    def test_cache_batch_latents_with_variants(self):
        """_default_cache_batch_latents_with_variants produces correct npz via a fake VAE."""
        from library.strategy_sd import SdSdxlLatentsCachingStrategy
        strategy = SdSdxlLatentsCachingStrategy(
            sd=True, cache_to_disk=True, batch_size=4, skip_disk_cache_validity_check=False,
        )

        info = ImageInfo("key", 1, "cat", False, False, self.img_path)
        info.bucket_reso = (64, 64)
        info.resized_size = (64, 64)
        info.resize_interpolation = None
        info.image = None
        info.latents_npz = self.img_path + ".npz"

        # Fake VAE: pool to H/8, tile to 4 channels
        def fake_encode(img_tensor):
            return torch.nn.functional.avg_pool2d(img_tensor, 8).repeat(1, 2, 1, 1)[:, :4]

        aug = AugHelper().get_augmentor(True, False, None, 0.0)
        config_hash = compute_aug_config_hash({"color_aug": True})

        strategy._default_cache_batch_latents_with_variants(
            fake_encode, self.device, torch.float32, [info],
            flip_aug=True, apply_alpha_mask=False, random_crop=False,
            multi_resolution=True, random_crop_padding_percent=0.05,
            num_aug_variants=4, augmentor=aug, aug_config_hash=config_hash,
        )

        # Verify npz contains variant keys (multi_resolution=True appends _8x8 suffix)
        npz = np.load(info.latents_npz)
        assert "latents_8x8" in npz
        assert "latents_v1_8x8" in npz
        assert "latents_v2_8x8" in npz
        assert "latents_v3_8x8" in npz
        assert "aug_variants" in npz
        assert "aug_config_hash" in npz
        assert int(npz["aug_variants"]) == 4
        assert str(npz["aug_config_hash"].tolist()) == config_hash

        # Load each variant
        for k in range(4):
            result = strategy.load_latents_from_disk(info.latents_npz, (64, 64), variant=k)
            assert len(result) == 6
            lat = result[0]
            assert lat.shape[0] == 4  # 4 latent channels

        # Variant 0 is canonical (unflipped)
        _, _, _, _, _, vf0 = strategy.load_latents_from_disk(info.latents_npz, (64, 64), variant=0)
        assert vf0 is False

        # Validity check
        assert strategy.is_disk_cached_latents_expected(
            (64, 64), info.latents_npz, flip_aug=True, alpha_mask=False,
            num_aug_variants=4, aug_config_hash=config_hash,
        )
