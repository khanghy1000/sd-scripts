"""Tests for compressed, reduced-precision NPZ cache serialization."""

import zipfile

import numpy as np

from library.cache_utils import load_npz, save_npz


def test_save_npz_uses_deflate_compression(tmp_path):
    path = tmp_path / "compressed.npz"
    save_npz(path, {"latents": np.zeros((128, 128), dtype=np.float32)}, cache_dtype="fp32")

    with zipfile.ZipFile(path) as archive:
        assert archive.infolist()
        assert all(entry.compress_type == zipfile.ZIP_DEFLATED for entry in archive.infolist())


def test_fp16_cache_round_trip(tmp_path):
    path = tmp_path / "fp16.npz"
    values = np.array([[-1.25, 0.0, 3.5]], dtype=np.float32)
    save_npz(path, {"hidden_state": values}, cache_dtype="fp16")

    with np.load(path) as raw:
        assert raw["hidden_state"].dtype == np.float16
    loaded = load_npz(path)
    assert loaded["hidden_state"].dtype == np.float16
    np.testing.assert_allclose(loaded["hidden_state"], values, rtol=1e-3, atol=1e-3)


def test_bf16_cache_round_trip_decodes_to_float32(tmp_path):
    path = tmp_path / "bf16.npz"
    values = np.array([1.0, -2.75, 1000.25], dtype=np.float32)
    save_npz(path, {"hidden_state": values}, cache_dtype="bf16")

    with np.load(path) as raw:
        assert raw["hidden_state"].dtype == np.uint16
    loaded = load_npz(path)
    assert loaded["hidden_state"].dtype == np.float32
    np.testing.assert_allclose(loaded["hidden_state"], values, rtol=8e-3, atol=8e-3)


def test_auto_falls_back_to_fp32_when_fp16_would_overflow(tmp_path):
    path = tmp_path / "auto.npz"
    values = np.array([70000.0], dtype=np.float32)
    save_npz(path, {"latents": values}, cache_dtype="auto")

    with np.load(path) as raw:
        assert raw["latents"].dtype == np.float32
    np.testing.assert_array_equal(load_npz(path)["latents"], values)


def test_legacy_uncompressed_npz_still_loads(tmp_path):
    path = tmp_path / "legacy.npz"
    values = np.array([1, 2, 3], dtype=np.float32)
    np.savez(path, latents=values, original_size=np.array([64, 64]))

    loaded = load_npz(path)
    np.testing.assert_array_equal(loaded["latents"], values)
    np.testing.assert_array_equal(loaded["original_size"], [64, 64])
