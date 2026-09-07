from dataclasses import asdict
from types import SimpleNamespace

from PIL import Image

from library.config_util import (
    BlueprintGenerator,
    ConfigSanitizer,
    DreamBoothDatasetParams,
    DreamBoothSubsetParams,
)
from library.train_util import BaseDataset, DreamBoothDataset, DreamBoothSubset, ImageInfo


def _subset(resolution=None, min_bucket_reso=None, max_bucket_reso=None):
    return SimpleNamespace(
        resolution=resolution,
        min_bucket_reso=min_bucket_reso,
        max_bucket_reso=max_bucket_reso,
    )


def _dataset(resolution=(768, 768), enable_bucket=True):
    dataset = BaseDataset(resolution, 1.0, False, False)
    dataset.batch_size = 1
    dataset.enable_bucket = enable_bucket
    dataset.min_bucket_reso = 256 if enable_bucket else None
    dataset.max_bucket_reso = 1024 if enable_bucket else None
    dataset.bucket_reso_steps = 64 if enable_bucket else None
    dataset.bucket_no_upscale = False
    dataset.multires_training = False
    return dataset


def _add_image(dataset, key, subset, image_size):
    info = ImageInfo(key, 1, "caption", False, False, key)
    info.image_size = image_size
    dataset.register_image(info, subset)
    return info


def test_subset_schema_accepts_resolution_and_bucket_bounds():
    sanitizer = ConfigSanitizer(True, False, False, True)
    config = {
        "general": {"resolution": 768},
        "datasets": [
            {
                "resolution": 512,
                "subsets": [
                    {
                        "image_dir": "images",
                        "resolution": [1024, 768],
                        "min_bucket_reso": 512,
                        "max_bucket_reso": 1536,
                    }
                ],
            }
        ],
    }

    sanitized = sanitizer.sanitize_user_config(config)
    subset = sanitized["datasets"][0]["subsets"][0]
    assert subset["resolution"] == (1024, 768)
    assert subset["min_bucket_reso"] == 512
    assert subset["max_bucket_reso"] == 1536


def test_subset_params_override_dataset_and_fallback_when_undefined():
    explicit = BlueprintGenerator.generate_params_by_fallbacks(
        DreamBoothSubsetParams,
        [{"resolution": (1024, 768), "min_bucket_reso": 512}, {"resolution": (768, 768), "min_bucket_reso": 256, "max_bucket_reso": 1024}],
    )
    inherited = BlueprintGenerator.generate_params_by_fallbacks(
        DreamBoothSubsetParams,
        [{"image_dir": "images"}, {"resolution": (768, 768), "min_bucket_reso": 256, "max_bucket_reso": 1024}],
    )

    assert explicit.resolution == (1024, 768)
    assert explicit.min_bucket_reso == 512
    assert explicit.max_bucket_reso == 1024
    assert inherited.resolution == (768, 768)
    assert inherited.min_bucket_reso == 256
    assert inherited.max_bucket_reso == 1024


def test_subset_bucket_settings_are_used_independently():
    dataset = _dataset()
    low = _subset((512, 512), 256, 1024)
    high = _subset((1280, 1280), 512, 1536)
    low_info = _add_image(dataset, "low", low, (512, 512))
    high_info = _add_image(dataset, "high", high, (1280, 1280))
    dataset.subsets = [low, high]

    dataset.make_buckets()

    assert low_info.bucket_reso == (512, 512)
    assert high_info.bucket_reso == (1280, 1280)
    assert (512, 512) in dataset.bucket_manager.resos
    assert (1280, 1280) in dataset.bucket_manager.resos


def test_subset_resolution_is_used_without_bucketing():
    dataset = _dataset((768, 768), enable_bucket=False)
    low = _subset((256, 256))
    high = _subset((512, 512))
    low_info = _add_image(dataset, "low", low, (256, 256))
    high_info = _add_image(dataset, "high", high, (512, 512))
    dataset.subsets = [low, high]

    dataset.make_buckets()

    assert low_info.bucket_reso == (256, 256)
    assert high_info.bucket_reso == (512, 512)


def test_dreambooth_dataset_constructs_with_subset_overrides(tmp_path):
    low_dir = tmp_path / "low"
    high_dir = tmp_path / "high"
    low_dir.mkdir()
    high_dir.mkdir()
    Image.new("RGB", (512, 512)).save(low_dir / "low.png")
    Image.new("RGB", (1024, 1024)).save(high_dir / "high.png")

    low_params = DreamBoothSubsetParams(image_dir=str(low_dir), resolution=(512, 512))
    high_params = DreamBoothSubsetParams(
        image_dir=str(high_dir),
        resolution=(1024, 1024),
        min_bucket_reso=512,
        max_bucket_reso=1536,
    )
    dataset_params = DreamBoothDatasetParams(
        resolution=(768, 768),
        enable_bucket=True,
        min_bucket_reso=256,
        max_bucket_reso=1024,
    )
    dataset = DreamBoothDataset(
        [DreamBoothSubset(**asdict(low_params)), DreamBoothSubset(**asdict(high_params))],
        is_training_dataset=True,
        **asdict(dataset_params),
    )

    dataset.make_buckets()

    low_info = next(info for info in dataset.image_data.values() if info.absolute_path.endswith("low.png"))
    high_info = next(info for info in dataset.image_data.values() if info.absolute_path.endswith("high.png"))
    assert low_info.bucket_reso == (512, 512)
    assert high_info.bucket_reso == (1024, 1024)


def test_dataset_group_resolution_enumeration_uses_effective_subset_values():
    dataset = _dataset()
    inherited = _subset(None, None, None)
    overridden = _subset((1024, 1024), 512, 1536)
    dataset.subsets = [inherited, overridden]

    assert dataset.get_resolutions() == [(768, 768), (1024, 1024)]
