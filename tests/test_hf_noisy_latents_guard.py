"""
Tests for the `_hf_noisy_latents` storage guard.

The High-Frequency Token loss is opt-in (`hf_scale > 0`). Storing a detached copy
of the noisy latent/model input on every training step when the feature is disabled
pins a full (B, C, H, W) tensor for no benefit (see review of train_network.py).

This module verifies, for every model type that sets `_hf_noisy_latents`:

  - Runtime: the base `NetworkTrainer.get_noise_pred_and_target` only stores the
    tensor when `is_train and self.hf_scale > 0.0` (CUDA).
  - Source: the assignment in each trainer file is guarded by `self.hf_scale > 0.0`
    (regression guard against the guard being silently dropped).

All tensors run on CUDA (assumed available).
"""

import contextlib
import os
from types import SimpleNamespace

import pytest
import torch


DEVICE = torch.device("cuda")


# ──────────────────────────────────────────────
# Runtime (functional) test — base NetworkTrainer
# ──────────────────────────────────────────────


def _make_trainer(hf_scale):
    train_network = pytest.importorskip("train_network")
    trainer = train_network.NetworkTrainer()
    trainer.hf_scale = hf_scale
    trainer.adaptive_manager = None
    trainer.tlora_enabled = False
    trainer.tlora_use_network_method = False
    return trainer


def _call_get_noise_pred_and_target(trainer, monkeypatch, is_train):
    import library.train_util as train_util

    B, C, H, W = 2, 4, 8, 8
    latents = torch.randn(B, C, H, W, device=DEVICE, dtype=torch.float32)
    noise = torch.randn(B, C, H, W, device=DEVICE, dtype=torch.float32)
    noisy_latents = latents + noise
    timesteps = torch.randint(0, 1000, (B,), device=DEVICE)
    noise_pred = torch.randn_like(noisy_latents)

    monkeypatch.setattr(
        train_util,
        "get_noise_noisy_latents_and_timesteps",
        lambda *args, **kwargs: (noise, noisy_latents, timesteps),
    )
    trainer.call_unet = lambda *args, **kwargs: noise_pred

    args = SimpleNamespace(v_parameterization=False, flow_model=False, gradient_checkpointing=False)
    accelerator = SimpleNamespace(autocast=lambda: contextlib.nullcontext(), device=DEVICE)
    noise_scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    trainer._hf_noisy_latents = None
    out = trainer.get_noise_pred_and_target(
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch={},
        text_encoder_conds=[None],
        text_encoder_masks=[None, None],
        unet=None,
        network=None,
        weight_dtype=torch.float32,
        train_unet=False,
        is_train=is_train,
    )
    assert len(out) == 5
    return trainer, noisy_latents


def test_hf_noisy_latents_not_stored_when_scale_zero(monkeypatch):
    trainer, _ = _call_get_noise_pred_and_target(_make_trainer(0.0), monkeypatch, is_train=True)
    assert trainer._hf_noisy_latents is None


def test_hf_noisy_latents_stored_when_scale_positive(monkeypatch):
    trainer, noisy_latents = _call_get_noise_pred_and_target(_make_trainer(0.5), monkeypatch, is_train=True)
    assert trainer._hf_noisy_latents is not None
    assert torch.equal(trainer._hf_noisy_latents, noisy_latents)


def test_hf_noisy_latents_not_stored_during_validation(monkeypatch):
    trainer, _ = _call_get_noise_pred_and_target(_make_trainer(0.5), monkeypatch, is_train=False)
    assert trainer._hf_noisy_latents is None


# ──────────────────────────────────────────────
# Source-level regression guard — all model types
# ──────────────────────────────────────────────


def _assignment_guard_lines(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    with open(os.path.join(root, filename), encoding="utf-8") as f:
        lines = f.readlines()

    guards = []
    for i, line in enumerate(lines):
        # Skip the init/skip assignments that are pure `None`.
        if "self._hf_noisy_latents = None" in line:
            continue
        if "self._hf_noisy_latents = " in line:
            # Walk back to the nearest controlling `if` (through comments/blank lines).
            guard = None
            for j in range(i - 1, max(i - 8, -1), -1):
                stripped = lines[j].strip()
                if stripped.startswith("if "):
                    guard = stripped
                    break
                if stripped and not stripped.startswith("#"):
                    break
            guards.append((filename, i + 1, guard))
    return guards


@pytest.mark.parametrize(
    "filename",
    [
        "train_network.py",
        "flux_train_network.py",
        "hunyuan_image_train_network.py",
        "lumina_train_network.py",
        "sd3_train_network.py",
        "anima_train_network.py",
    ],
)
def test_hf_noisy_latents_assignment_is_guarded(filename):
    entries = _assignment_guard_lines(filename)
    assert entries, f"{filename}: no `_hf_noisy_latents = ...` store found"
    for fname, lineno, guard in entries:
        assert guard is not None, (
            f"{fname}:{lineno}: `_hf_noisy_latents` assignment is not under an `if` guard"
        )
        assert "is_train" in guard and "self.hf_scale > 0.0" in guard, (
            f"{fname}:{lineno}: guard must be `is_train and self.hf_scale > 0.0`, got: {guard}"
        )
