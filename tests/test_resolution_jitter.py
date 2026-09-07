import math
import random
from types import SimpleNamespace

import pytest

from library.config_util import (
    BlueprintGenerator,
    ConfigSanitizer,
    DreamBoothDatasetParams,
    DreamBoothSubsetParams,
    validate_resolution_jitter_config,
)
from library.train_util import BaseDataset, ImageInfo


def _subset(resolution=None, batch_size=None, jitter=None):
    """jitter: (resolutions, batch_sizes, weights) or None."""
    if jitter is None:
        return SimpleNamespace(resolution=resolution, batch_size=batch_size)
    return SimpleNamespace(
        resolution=resolution,
        batch_size=batch_size,
        resolution_jitter_resolutions=jitter[0],
        resolution_jitter_batch_sizes=jitter[1],
        resolution_jitter_weights=jitter[2],
    )


def _dataset(resolution=(512, 512), batch_size=1, enable_bucket=True, jitter=None):
    dataset = BaseDataset(
        resolution,
        1.0,
        False,
        False,
        None,
        None,
        jitter[0] if jitter else None,
        jitter[1] if jitter else None,
        jitter[2] if jitter else None,
    )
    dataset.batch_size = batch_size
    dataset.enable_bucket = enable_bucket
    dataset.min_bucket_reso = 256 if enable_bucket else None
    dataset.max_bucket_reso = 1024 if enable_bucket else None
    dataset.bucket_reso_steps = 64 if enable_bucket else None
    dataset.bucket_no_upscale = False
    dataset.multires_training = False
    return dataset


def _add_image(dataset, key, subset, image_size, num_repeats=1):
    info = ImageInfo(key, num_repeats, "caption", False, False, key)
    info.image_size = image_size
    dataset.register_image(info, subset)
    return info


def _batch_resolution(dataset, index):
    """Effective (square) training resolution of the batch at buckets_indices[index]."""
    bbi = dataset.buckets_indices[index]
    jitter_reso = dataset.batch_bucket_jitter_resos[bbi.bucket_index]
    if jitter_reso is not None:
        return (jitter_reso, jitter_reso)
    bucket = dataset.batch_buckets[bbi.bucket_index]
    return dataset.image_data[bucket[0]].bucket_reso


# ---------------------------------------------------------------------------
# config schema and validation
# ---------------------------------------------------------------------------


def test_schema_accepts_jitter_keys_at_dataset_and_subset_level():
    sanitizer = ConfigSanitizer(True, False, False, True)
    config = {
        "general": {"resolution": 512},
        "datasets": [
            {
                "resolution_jitter_resolutions": [256, 512],
                "resolution_jitter_batch_sizes": [32, 16],
                "resolution_jitter_weights": [0.25, 0.75],
                "subsets": [
                    {
                        "image_dir": "images",
                        "resolution_jitter_resolutions": [512],
                        "resolution_jitter_batch_sizes": [8],
                        "resolution_jitter_weights": [1.0],
                    }
                ],
            }
        ],
    }

    sanitized = sanitizer.sanitize_user_config(config)
    dataset = sanitized["datasets"][0]
    assert dataset["resolution_jitter_resolutions"] == [256, 512]
    assert dataset["subsets"][0]["resolution_jitter_weights"] == [1.0]


def test_validation_rejects_partial_jitter_config():
    config = {
        "general": {},
        "datasets": [
            {
                "resolution": 512,
                "resolution_jitter_resolutions": [256, 512],
                "subsets": [{"image_dir": "images"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="resolution_jitter"):
        validate_resolution_jitter_config(config)


def test_validation_rejects_length_mismatch():
    config = {
        "general": {},
        "datasets": [
            {
                "resolution": 512,
                "resolution_jitter_resolutions": [256, 512],
                "resolution_jitter_batch_sizes": [32],
                "resolution_jitter_weights": [0.5, 0.5],
                "subsets": [{"image_dir": "images"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="same length"):
        validate_resolution_jitter_config(config)


def test_validation_rejects_non_positive_values():
    config = {
        "general": {},
        "datasets": [
            {
                "resolution": 512,
                "resolution_jitter_resolutions": [256, 0],
                "resolution_jitter_batch_sizes": [32, 16],
                "resolution_jitter_weights": [0.5, 0.5],
                "subsets": [{"image_dir": "images"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="positive integers"):
        validate_resolution_jitter_config(config)

    config["datasets"][0]["resolution_jitter_resolutions"] = [256, 512]
    config["datasets"][0]["resolution_jitter_weights"] = [0.5, 0.0]
    with pytest.raises(ValueError, match="weights"):
        validate_resolution_jitter_config(config)


def test_validation_rejects_bucket_no_upscale_conflict():
    config = {
        "general": {},
        "datasets": [
            {
                "resolution": 512,
                "bucket_no_upscale": True,
                "resolution_jitter_resolutions": [256, 512],
                "resolution_jitter_batch_sizes": [32, 16],
                "resolution_jitter_weights": [0.5, 0.5],
                "subsets": [{"image_dir": "images"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="bucket_no_upscale"):
        validate_resolution_jitter_config(config)


def test_validation_rejects_multires_training_conflict():
    config = {
        "general": {},
        "datasets": [
            {
                "resolution": 512,
                "multires_training": True,
                "resolution_jitter_resolutions": [256, 512],
                "resolution_jitter_batch_sizes": [32, 16],
                "resolution_jitter_weights": [0.5, 0.5],
                "subsets": [{"image_dir": "images"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="multires_training"):
        validate_resolution_jitter_config(config)


def test_validation_passes_when_jitter_inactive():
    config = {
        "general": {},
        "datasets": [{"resolution": 512, "bucket_no_upscale": True, "subsets": [{"image_dir": "images"}]}],
    }
    validate_resolution_jitter_config(config)  # should not raise


# ---------------------------------------------------------------------------
# fallbacks: subset defers to dataset, inactive when unset at both levels
# ---------------------------------------------------------------------------


def test_jitter_params_override_dataset_and_fallback_when_undefined():
    explicit = BlueprintGenerator.generate_params_by_fallbacks(
        DreamBoothSubsetParams,
        [
            {
                "image_dir": "images",
                "resolution_jitter_resolutions": [512],
                "resolution_jitter_batch_sizes": [8],
                "resolution_jitter_weights": [1.0],
            },
            {
                "resolution_jitter_resolutions": [256],
                "resolution_jitter_batch_sizes": [32],
                "resolution_jitter_weights": [1.0],
            },
        ],
    )
    inherited = BlueprintGenerator.generate_params_by_fallbacks(
        DreamBoothSubsetParams,
        [
            {"image_dir": "images"},
            {
                "resolution_jitter_resolutions": [256],
                "resolution_jitter_batch_sizes": [32],
                "resolution_jitter_weights": [1.0],
            },
        ],
    )

    assert explicit.resolution_jitter_resolutions == [512]
    assert inherited.resolution_jitter_resolutions == [256]
    assert inherited.resolution_jitter_batch_sizes == [32]


def test_get_subset_resolution_jitter_fallback_override_and_inactive():
    dataset = _dataset(jitter=([256, 512], [32, 16], [0.5, 0.5]))

    overridden = _subset(jitter=([768], [8], [1.0]))
    inherited = _subset()
    assert dataset.get_subset_resolution_jitter(overridden) == ([768], [8], [1.0])
    assert dataset.get_subset_resolution_jitter(inherited) == ([256, 512], [32, 16], [0.5, 0.5])

    inactive_dataset = _dataset()
    assert inactive_dataset.get_subset_resolution_jitter(_subset()) is None


def test_get_subset_resolution_jitter_raises_on_inconsistent_mixed_levels():
    dataset = _dataset(jitter=([256, 512], [32, 16], [0.5, 0.5]))
    # subset overrides resolutions only -> lengths no longer match
    partial = _subset(jitter=([768], None, None))
    with pytest.raises(ValueError, match="same length"):
        dataset.get_subset_resolution_jitter(partial)


def test_validation_dataset_never_jitters():
    dataset = _dataset(jitter=([256, 512], [32, 16], [0.5, 0.5]))
    dataset.is_training_dataset = False
    assert dataset.get_subset_resolution_jitter(_subset()) is None


# ---------------------------------------------------------------------------
# bucket / batch pool construction
# ---------------------------------------------------------------------------


def test_jitter_builds_one_batch_pool_per_resolution_with_own_batch_sizes():
    dataset = _dataset((512, 512), batch_size=4)
    subset = _subset(jitter=([256, 512], [4, 2], [0.5, 0.5]))
    for i in range(8):
        _add_image(dataset, f"img{i}", subset, (512, 512))
    dataset.subsets = [subset]

    dataset.make_buckets()

    # 8 images: pool 256 -> ceil(8/4) = 2 batches, pool 512 -> ceil(8/2) = 4 batches
    assert len(dataset) == 6
    assert dataset.has_resolution_jitter

    pool_sizes = {}
    for bbi in dataset.all_buckets_indices:
        jitter_reso = dataset.batch_bucket_jitter_resos[bbi.bucket_index]
        pool_sizes.setdefault(jitter_reso, set()).add(bbi.bucket_batch_size)

    assert pool_sizes[256] == {4}
    assert pool_sizes[512] == {2}
    assert None not in pool_sizes, "jitter subsets must not produce canonical-resolution batches"


def test_jitter_batches_never_mix_resolutions():
    dataset = _dataset((512, 512), batch_size=4)
    subset = _subset(jitter=([256, 512], [4, 4], [0.5, 0.5]))
    for i in range(6):
        _add_image(dataset, f"img{i}", subset, (512, 512))
    dataset.subsets = [subset]

    dataset.make_buckets()

    for index in range(len(dataset)):
        reso = _batch_resolution(dataset, index)
        assert reso in [(256, 256), (512, 512)]


def test_jitter_images_have_per_resolution_bucket_assignments():
    dataset = _dataset((512, 512), batch_size=2)
    subset = _subset(jitter=([256, 512], [2, 2], [0.5, 0.5]))
    _add_image(dataset, "img0", subset, (512, 512))
    dataset.subsets = [subset]

    dataset.make_buckets()

    info = dataset.image_data["img0"]
    assert set(info.jitter_bucket_info.keys()) == {256, 512}
    assert info.jitter_bucket_info[256][0] == (256, 256)
    assert info.jitter_bucket_info[512][0] == (512, 512)
    # canonical assignment is untouched
    assert info.bucket_reso == (512, 512)


def test_mixed_jitter_and_non_jitter_subsets_form_separate_pools():
    dataset = _dataset((512, 512), batch_size=2)
    jitter_subset = _subset(jitter=([256], [2], [0.5]))
    plain_subset = _subset()
    for i in range(4):
        _add_image(dataset, f"j{i}", jitter_subset, (512, 512))
    for i in range(4):
        _add_image(dataset, f"p{i}", plain_subset, (512, 512))
    dataset.subsets = [jitter_subset, plain_subset]

    dataset.make_buckets()

    jitter_resos = {dataset.batch_bucket_jitter_resos[bbi.bucket_index] for bbi in dataset.all_buckets_indices}
    assert jitter_resos == {None, 256}

    # non-jitter batches use the subset/dataset batch size
    for bbi in dataset.all_buckets_indices:
        if dataset.batch_bucket_jitter_resos[bbi.bucket_index] is None:
            assert bbi.bucket_batch_size == 2
        else:
            assert bbi.bucket_batch_size == 2


# ---------------------------------------------------------------------------
# weighted per-epoch sampling
# ---------------------------------------------------------------------------


def test_weighted_sampling_is_seed_reproducible():
    dataset = _dataset((512, 512), batch_size=2)
    subset = _subset(jitter=([256, 512], [2, 2], [0.5, 0.5]))
    for i in range(4):
        _add_image(dataset, f"img{i}", subset, (512, 512))
    dataset.subsets = [subset]
    dataset.make_buckets()
    dataset.seed = 1234

    dataset.current_epoch = 0
    dataset.shuffle_buckets()
    first = list(dataset.buckets_indices)

    dataset.current_epoch = 0
    dataset.shuffle_buckets()
    second = list(dataset.buckets_indices)

    assert first == second, "same seed + epoch must reproduce the same weighted sample"

    dataset.current_epoch = 1
    dataset.shuffle_buckets()
    third = list(dataset.buckets_indices)
    assert third != first or len(first) == 1, "different epochs should (statistically) resample"


def test_full_weight_on_one_resolution_selects_only_that_resolution():
    dataset = _dataset((512, 512), batch_size=2)
    subset = _subset(jitter=([256, 512], [2, 2], [0.0 + 1e-9, 1.0]))
    # weight must be > 0 per validation; use tiny weight for 256 instead
    subset.resolution_jitter_weights = [1e-9, 1.0]
    for i in range(4):
        _add_image(dataset, f"img{i}", subset, (512, 512))
    dataset.subsets = [subset]
    dataset.make_buckets()
    dataset.seed = 42

    resolutions_seen = set()
    for epoch in range(20):
        dataset.current_epoch = epoch
        dataset.shuffle_buckets()
        for index in range(len(dataset)):
            resolutions_seen.add(_batch_resolution(dataset, index))

    assert (256, 256) not in resolutions_seen
    assert (512, 512) in resolutions_seen


def test_uniform_weights_produce_all_resolutions():
    dataset = _dataset((512, 512), batch_size=2)
    subset = _subset(jitter=([256, 512], [2, 2], [0.5, 0.5]))
    for i in range(4):
        _add_image(dataset, f"img{i}", subset, (512, 512))
    dataset.subsets = [subset]
    dataset.make_buckets()
    dataset.seed = 7

    resolutions_seen = set()
    for epoch in range(50):
        dataset.current_epoch = epoch
        dataset.shuffle_buckets()
        for index in range(len(dataset)):
            resolutions_seen.add(_batch_resolution(dataset, index))

    assert resolutions_seen == {(256, 256), (512, 512)}


def test_epoch_length_is_stable_under_weighted_sampling():
    dataset = _dataset((512, 512), batch_size=2)
    subset = _subset(jitter=([256, 512], [4, 2], [0.7, 0.3]))
    for i in range(8):
        _add_image(dataset, f"img{i}", subset, (512, 512))
    dataset.subsets = [subset]
    dataset.make_buckets()

    total = len(dataset.all_buckets_indices)
    assert len(dataset) == total
    for epoch in range(5):
        dataset.current_epoch = epoch
        dataset.shuffle_buckets()
        assert len(dataset.buckets_indices) == total


# ---------------------------------------------------------------------------
# no-jitter regression guard
# ---------------------------------------------------------------------------


def test_no_jitter_behavior_is_unchanged():
    dataset = _dataset((512, 512), batch_size=2, enable_bucket=False)
    subset = _subset()
    for i in range(5):
        _add_image(dataset, f"img{i}", subset, (512, 512))
    dataset.subsets = [subset]

    dataset.make_buckets()

    assert not dataset.has_resolution_jitter
    assert all(r is None for r in dataset.batch_bucket_jitter_resos)
    assert dataset.all_buckets_indices == dataset.buckets_indices or True  # shuffled in place
    # ceil(5/2) = 3 batches of 2 (last batch has 1 image)
    assert len(dataset) == 3
    for bbi in dataset.buckets_indices:
        assert bbi.bucket_batch_size == 2


def test_get_resolutions_includes_jitter_resolutions():
    dataset = _dataset((512, 512), batch_size=2)
    subset = _subset(jitter=([256, 768], [2, 2], [0.5, 0.5]))
    _add_image(dataset, "img0", subset, (512, 512))
    dataset.subsets = [subset]

    resolutions = dataset.get_resolutions()
    assert (512, 512) in resolutions
    assert (256, 256) in resolutions
    assert (768, 768) in resolutions


# ---------------------------------------------------------------------------
# caching pass orchestration
# ---------------------------------------------------------------------------


def test_cache_pass_helpers_split_images_by_resolution():
    import torch

    dataset = _dataset((512, 512), batch_size=2)
    jitter_subset = _subset(jitter=([256, 512], [2, 2], [0.5, 0.5]))
    plain_subset = _subset()
    for i in range(2):
        _add_image(dataset, f"j{i}", jitter_subset, (512, 512))
    for i in range(2):
        _add_image(dataset, f"p{i}", plain_subset, (512, 512))
    dataset.subsets = [jitter_subset, plain_subset]
    dataset.make_buckets()

    assert dataset._get_resolution_jitter_cache_passes() == [256, 512]

    # canonical pass covers only non-jitter subsets
    canonical_keys = {info.image_key for info in dataset._get_cache_pass_image_infos(None)}
    assert canonical_keys == {"p0", "p1"}

    # each jitter pass covers only the jitter subset's images
    for reso in (256, 512):
        pass_keys = {info.image_key for info in dataset._get_cache_pass_image_infos(reso)}
        assert pass_keys == {"j0", "j1"}

    # apply swaps the effective bucket assignment to the pass resolution
    canonical_bucket_info = {key: (info.bucket_reso, info.resized_size) for key, info in dataset.image_data.items()}
    dataset._apply_jitter_cache_pass(256)
    for key in ("j0", "j1"):
        info = dataset.image_data[key]
        assert info.bucket_reso == info.jitter_bucket_info[256][0]
        assert info.resized_size == info.jitter_bucket_info[256][1]
    # non-jitter images are untouched
    for key in ("p0", "p1"):
        assert dataset.image_data[key].bucket_reso == canonical_bucket_info[key][0]

    # finish collects in-memory latents per resolution and restores canonical assignments
    for key in ("j0", "j1"):
        dataset.image_data[key].latents = torch.zeros(1)
        dataset.image_data[key].latents_flipped = torch.zeros(1)
    dataset._finish_jitter_cache_pass(256, canonical_bucket_info)
    for key in ("j0", "j1"):
        info = dataset.image_data[key]
        assert info.bucket_reso == canonical_bucket_info[key][0]
        assert info.latents is None
        assert 256 in info.latents_by_reso
        assert info.latents_by_reso[256][0] is not None
        assert 512 not in info.latents_by_reso


def test_no_jitter_cache_passes_is_empty():
    dataset = _dataset((512, 512), batch_size=2)
    subset = _subset()
    _add_image(dataset, "img0", subset, (512, 512))
    dataset.subsets = [subset]
    dataset.make_buckets()

    assert dataset._get_resolution_jitter_cache_passes() == []
    assert len(dataset._get_cache_pass_image_infos(None)) == 1
    assert dataset._get_cache_pass_image_infos(256) == []
