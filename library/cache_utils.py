"""Utilities shared by disk caches.

NPZ supports compression transparently through NumPy, but it does not have a
native bfloat16 dtype on all supported NumPy versions.  BF16 cache entries are
therefore stored as their uint16 bit representation with per-entry metadata.

"""

import json
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch


CACHE_DTYPE_CHOICES = ("auto", "fp16", "bf16", "fp32")
CACHE_DTYPE_METADATA_KEY = "__cache_dtypes__"


def normalize_cache_dtype(cache_dtype: str) -> str:
    """Validate and normalize a cache floating-point dtype policy."""
    # Some downstream callers construct ``argparse.Namespace``-like objects
    # with MagicMock attributes in tests. Treat absent/non-string optional
    # values as the backwards-compatible default rather than rejecting them.
    if not isinstance(cache_dtype, str):
        return "auto"
    normalized = str(cache_dtype).lower()
    if normalized not in CACHE_DTYPE_CHOICES:
        raise ValueError(f"cache dtype must be one of {CACHE_DTYPE_CHOICES}, got: {cache_dtype}")
    return normalized


def _to_numpy(value: Any) -> Tuple[np.ndarray, Any]:
    """Return a CPU NumPy array and the original torch dtype, when available."""
    if isinstance(value, torch.Tensor):
        source_dtype = value.dtype
        tensor = value.detach().to("cpu")
        if source_dtype == torch.bfloat16:
            return tensor.float().numpy(), source_dtype
        return tensor.numpy(), source_dtype
    return np.asarray(value), None


def _choose_auto_dtype(array: np.ndarray, source_dtype: Any) -> str:
    """Choose a compact safe dtype for an array.

    Auto preserves the source precision for ordinary FP32/FP64 arrays.  It
    still preserves compact FP16/BF16 sources, and therefore never changes
    the numerical behavior of existing caches unless a compact policy is
    explicitly requested.
    """
    if source_dtype == torch.bfloat16:
        return "bf16"
    if source_dtype == torch.float16 or array.dtype == np.float16:
        return "fp16"
    return "fp32"


def encode_cache_array(value: Any, cache_dtype: str, key: str) -> Tuple[np.ndarray, str | None]:
    """Convert one cache value and return its stored dtype metadata."""
    policy = normalize_cache_dtype(cache_dtype)
    array, source_dtype = _to_numpy(value)
    if not np.issubdtype(array.dtype, np.floating):
        return array, None

    target = _choose_auto_dtype(array, source_dtype) if policy == "auto" else policy
    if target == "fp16":
        finite = np.isfinite(array)
        if np.any(finite & (np.abs(array) > np.finfo(np.float16).max)):
            if policy == "auto":
                target = "fp32"
            else:
                raise ValueError(f"{key} contains values outside the finite FP16 range")

    if target == "fp16":
        return array.astype(np.float16), target
    if target == "fp32":
        return array.astype(np.float32), target

    # NumPy cannot consistently represent BF16. Store the exact BF16 bits and
    # reconstruct them on load. The metadata tells the reader which entries
    # contain encoded BF16 values.
    contiguous = np.ascontiguousarray(array.astype(np.float32, copy=False))
    bf16 = torch.from_numpy(contiguous).to(dtype=torch.bfloat16)
    raw_bits = bf16.view(dtype=torch.uint16).numpy().copy()
    return raw_bits, target


def save_npz(path: str, arrays: Mapping[str, Any], cache_dtype: str = "auto") -> None:
    """Save a compressed NPZ cache, encoding floating arrays per policy."""
    policy = normalize_cache_dtype(cache_dtype)
    encoded: Dict[str, np.ndarray] = {}
    dtype_metadata: Dict[str, str] = {}
    for key, value in arrays.items():
        if key == CACHE_DTYPE_METADATA_KEY:
            continue
        encoded_value, actual_dtype = encode_cache_array(value, policy, key)
        encoded[key] = encoded_value
        if actual_dtype is not None:
            dtype_metadata[key] = actual_dtype

    if dtype_metadata:
        encoded[CACHE_DTYPE_METADATA_KEY] = np.array(json.dumps(dtype_metadata, sort_keys=True))
    np.savez_compressed(path, **encoded)


def _decode_bf16(array: np.ndarray) -> np.ndarray:
    raw_bits = np.ascontiguousarray(array.astype(np.uint16, copy=False))
    return torch.from_numpy(raw_bits).view(dtype=torch.bfloat16).float().numpy()


def load_npz(path: str, return_dtypes: bool = False) -> Dict[str, np.ndarray]:
    """Load compressed or uncompressed NPZ and decode cache BF16 entries.

    Args:
        path: Path to the npz file.
        return_dtypes: If True, also return the per-key stored dtype metadata
            as a second dict. This lets callers reconstruct compact BF16
            tensors: NumPy has no native BF16, so decoded BF16 entries are
            returned as widened fp32 arrays, and the metadata is the only way
            to know they originated as BF16.

    Returns:
        ``(data, dtypes)`` when ``return_dtypes`` is True, otherwise ``data``.
    """
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files if key != CACHE_DTYPE_METADATA_KEY}
        if CACHE_DTYPE_METADATA_KEY not in archive:
            return (data, {}) if return_dtypes else data
        metadata = json.loads(str(archive[CACHE_DTYPE_METADATA_KEY].tolist()))

    for key, dtype in metadata.items():
        if key in data and dtype == "bf16":
            data[key] = _decode_bf16(data[key])
    return (data, metadata) if return_dtypes else data
