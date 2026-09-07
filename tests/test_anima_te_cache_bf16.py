"""Tests for compact BF16 end-to-end Anima text-encoder output caching.

Regression coverage for the fix that stops upcasting cached ``prompt_embeds``
from bf16 to fp32 before saving. The cache must stay bf16 from the text encoder
through disk, the dataloader batch, and up to the weight_dtype cast at the model
boundary, halving disk/RAM/queue footprint along the way.
"""

import numpy as np
import pytest
import torch

from library.cache_utils import load_npz, save_npz
from library.strategy_anima import AnimaTextEncoderOutputsCachingStrategy
from library.train_util import convert_te_output_for_batch

_SEQ = 32
_DIM = 128


def _make_strategy() -> AnimaTextEncoderOutputsCachingStrategy:
    return AnimaTextEncoderOutputsCachingStrategy(
        cache_to_disk=False, batch_size=1, skip_disk_cache_validity_check=True
    )


def _make_bf16_prompt_embeds(seed: int = 0) -> torch.Tensor:
    """Simulate the Qwen3 TE output: a CPU bf16 [1, seq, dim] tensor."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, _SEQ, _DIM, generator=g, dtype=torch.bfloat16)


def _make_cache_kwargs(prompt_embeds) -> dict:
    return {
        "prompt_embeds": prompt_embeds,
        "attn_mask": np.ones((1, _SEQ), dtype=np.int64),
        "t5_input_ids": np.arange(_SEQ, dtype=np.int32).reshape(1, -1),
        "t5_attn_mask": np.ones((1, _SEQ), dtype=np.int64),
        "caption_dropout_rate": torch.tensor(0.1, dtype=torch.float32),
    }


def test_bf16_cache_is_stored_compactly_on_disk(tmp_path):
    """A bf16 prompt_embeds tensor must be written as its 2-byte uint16 layout."""
    path = tmp_path / "anima_te.npz"
    save_npz(path, _make_cache_kwargs(_make_bf16_prompt_embeds()), cache_dtype="auto")

    with np.load(path) as raw:
        # bf16 is stored as uint16 bit patterns (numpy has no native bf16).
        assert raw["prompt_embeds"].dtype == np.uint16
        assert raw["prompt_embeds"].itemsize == 2
        assert "prompt_embeds" in raw["__cache_dtypes__"].tolist()


def test_bf16_cache_round_trip_preserves_dtype_and_values(tmp_path):
    """load_outputs_npz reconstructs a bf16 torch tensor, not a widened fp32 array."""
    path = tmp_path / "anima_te.npz"
    original = _make_bf16_prompt_embeds()
    save_npz(path, _make_cache_kwargs(original), cache_dtype="auto")

    result = _make_strategy().load_outputs_npz(str(path))

    assert isinstance(result[0], torch.Tensor), "bf16 cache must load back as a torch tensor"
    assert result[0].dtype == torch.bfloat16, f"expected bf16, got {result[0].dtype}"
    assert result[0].shape == (1, _SEQ, _DIM)
    assert result[0].is_cpu
    # bf16 round-trip is lossless (values were already rounded at encode time).
    torch.testing.assert_close(result[0], original, rtol=0, atol=0)
    # Non-floating entries keep their historical numpy form and values.
    assert isinstance(result[1], np.ndarray)
    np.testing.assert_array_equal(result[1], np.ones((1, _SEQ), dtype=np.int64))
    np.testing.assert_array_equal(result[2], np.arange(_SEQ, dtype=np.int32).reshape(1, -1))
    assert float(result[4]) == pytest.approx(0.1, abs=1e-6)


def test_legacy_fp32_cache_loads_unchanged(tmp_path):
    """Old fp32 caches (no bf16 metadata) must load exactly as before: fp32 numpy."""
    path = tmp_path / "legacy_te.npz"
    fp32_np = np.random.RandomState(0).randn(1, _SEQ, _DIM).astype(np.float32)
    save_npz(path, _make_cache_kwargs(fp32_np), cache_dtype="fp32")

    result = _make_strategy().load_outputs_npz(str(path))

    assert isinstance(result[0], np.ndarray)
    assert result[0].dtype == np.float32
    np.testing.assert_array_equal(result[0], fp32_np)


def test_load_npz_return_dtypes_metadata(tmp_path):
    """load_npz(return_dtypes=True) reports which entries were stored as bf16."""
    path = tmp_path / "meta.npz"
    save_npz(path, _make_cache_kwargs(_make_bf16_prompt_embeds()), cache_dtype="auto")

    data, dtypes = load_npz(str(path), return_dtypes=True)
    assert dtypes["prompt_embeds"] == "bf16"
    assert data["prompt_embeds"].dtype == np.float32  # widened decode


def test_convert_te_output_for_batch_preserves_bf16():
    """The batch-stack converter keeps bf16 tensors compact and is a no-op otherwise."""
    bf16_tensor = torch.zeros(1, _SEQ, _DIM, dtype=torch.bfloat16)
    converted = convert_te_output_for_batch(bf16_tensor)
    assert converted is bf16_tensor  # no copy, no widen
    assert converted.dtype == torch.bfloat16

    fp32_np = np.zeros((1, _SEQ, _DIM), dtype=np.float32)
    assert convert_te_output_for_batch(fp32_np).dtype == torch.float32

    fp16_np = np.zeros((1, _SEQ, _DIM), dtype=np.float16)
    assert convert_te_output_for_batch(fp16_np).dtype == torch.float32  # historical

    int_np = np.ones((1, _SEQ), dtype=np.int64)
    assert convert_te_output_for_batch(int_np).dtype == torch.float32  # historical


def test_bf16_batch_flows_to_cuda_without_fp32_intermediate():
    """Full batch path: bf16 cache -> stack -> CUDA weight_dtype cast, no fp32 stop."""
    weight_dtype = torch.bfloat16
    samples = [_make_bf16_prompt_embeds(seed=i) for i in range(2)]

    # Mimic none_or_stack_elements(text_encoder_outputs_list, convert_te_output_for_batch)
    stacked = torch.stack([convert_te_output_for_batch(s) for s in samples])
    assert stacked.dtype == torch.bfloat16, "batch must remain bf16 through assembly"

    # Mimic AnimaNetworkTrainer.get_noise_pred_and_target's model-boundary cast.
    on_device = stacked.to(device="cuda", dtype=weight_dtype)
    assert on_device.device.type == "cuda"
    assert on_device.dtype == weight_dtype
    # Casting bf16->bf16 is value-preserving; verify against the CPU originals.
    torch.testing.assert_close(
        on_device.cpu(), torch.stack(samples), rtol=0, atol=0
    )

    # fp16 training path: the same bf16 batch is cast to fp16 on the device.
    fp16_device = stacked.to(device="cuda", dtype=torch.float16)
    assert fp16_device.dtype == torch.float16
    torch.testing.assert_close(
        fp16_device.cpu().float(), torch.stack(samples).float(), rtol=8e-3, atol=8e-3
    )
