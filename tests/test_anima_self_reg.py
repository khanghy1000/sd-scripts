"""CPU unit tests for Anima-only self-regularization caption helpers.

Covers `build_self_reg_anchor_caption` and `parse_self_reg_trigger_words` in
anima_train_network.py:
- exact-tag, case-insensitive trigger stripping (no substring removal),
- filler occupying the vacated trigger slot,
- shuffle keeping anchor/main tag sets consistent,
- multi-word comma-separated triggers.
"""

import random

import pytest

from anima_train_network import build_self_reg_anchor_caption, parse_self_reg_trigger_words


def test_trigger_stripped_case_insensitive_exact_tag():
    anchor = build_self_reg_anchor_caption("MyChara, 1girl, solo", "mychara", "")
    assert anchor == "1girl, solo"


def test_only_exact_tags_removed():
    # "mychara2" merely contains the trigger as a substring: it must stay.
    anchor = build_self_reg_anchor_caption("mychara, mychara2, solo", "mychara", "")
    assert anchor == "mychara2, solo"


def test_multi_word_triggers():
    anchor = build_self_reg_anchor_caption("sks woman, 1girl, solo", "sks woman", "")
    assert anchor == "1girl, solo"


def test_multiple_comma_separated_triggers():
    anchor = build_self_reg_anchor_caption("charA, charB, 1girl", "chara, charb", "")
    assert anchor == "1girl"


def test_filler_occupies_vacated_slot():
    anchor = build_self_reg_anchor_caption("mychara, 1girl, solo", "mychara", "person")
    assert anchor == "person, 1girl, solo"


def test_filler_with_multiple_words():
    anchor = build_self_reg_anchor_caption("mychara, solo", "mychara", "a person")
    assert anchor == "a person, solo"


def test_no_trigger_in_caption_still_valid_hold_target():
    anchor = build_self_reg_anchor_caption("1girl, solo", "mychara", "person")
    assert anchor == "person, 1girl, solo"


def test_empty_caption_and_empty_filler():
    assert build_self_reg_anchor_caption("", "mychara", "") == ""
    assert build_self_reg_anchor_caption("mychara", "mychara", "") == ""


def test_whitespace_and_empty_tags_dropped():
    anchor = build_self_reg_anchor_caption("  mychara ,, 1girl , ", "mychara", "")
    assert anchor == "1girl"


def test_main_caption_untouched_by_helper():
    caption = "mychara, 1girl, solo"
    build_self_reg_anchor_caption(caption, "mychara", "person", shuffle=True)
    assert caption == "mychara, 1girl, solo"


def test_shuffle_keeps_anchor_main_consistent():
    random.seed(0)
    caption_tags = ["mychara", "1girl", "solo", "long hair", "blue eyes", "school uniform"]
    caption = ", ".join(caption_tags)
    anchor = build_self_reg_anchor_caption(caption, "mychara", "person", shuffle=True)
    anchor_tags = [t.strip() for t in anchor.split(",")]
    # Same multiset as filler + caption-minus-trigger, only the order may differ.
    assert sorted(anchor_tags) == sorted(["person", "1girl", "solo", "long hair", "blue eyes", "school uniform"])
    assert anchor_tags[0] == "person"  # filler keeps the trigger slot occupied


def test_parse_trigger_words():
    assert parse_self_reg_trigger_words("mychara, sks woman,, ") == ["mychara", "sks woman"]
    assert parse_self_reg_trigger_words("") == []
    assert parse_self_reg_trigger_words(None) == []


# ---------------------------------------------------------------------------
# Step-mode decision, validation and batch-mutation logic (CPU, stubbed TE)
# ---------------------------------------------------------------------------

from types import SimpleNamespace

import torch

from anima_train_network import AnimaNetworkTrainer


def _make_args(**kw):
    d = dict(
        self_reg_weight=1.0,
        self_reg_trigger_word="mychara",
        self_reg_filler="person",
        self_reg_noise=0.0,
        self_reg_batched=False,
        self_reg_shuffle_tags=False,
        cache_text_encoder_outputs=False,
        train_batch_size=1,
        contrastive_flow_matching=False,
        ileco=False,
        addift=False,
        network_train_unet_only=True,
    )
    d.update(kw)
    return SimpleNamespace(**d)


def _make_batch(captions):
    return {
        "captions": list(captions),
        "loss_weights": torch.ones(len(captions)),
        "alpha_masks": torch.ones(len(captions), 8, 8),
    }


@pytest.fixture()
def trainer(monkeypatch):
    t = AnimaNetworkTrainer()
    captured = {}

    def _fake_encode(self, args, accelerator, text_encoders, anchor_captions, tokenize_strategy,
                     text_encoding_strategy, weight_dtype):
        captured["anchor_captions"] = list(anchor_captions)
        b = len(anchor_captions)
        return [
            torch.zeros(b, 2),
            torch.zeros(b, 2),
            torch.zeros(b, 2, dtype=torch.long),
            torch.zeros(b, 2, dtype=torch.long),
        ]

    monkeypatch.setattr(AnimaNetworkTrainer, "encode_self_reg_conds", _fake_encode)
    t._captured = captured
    return t


def _prepare(trainer, args, batch, is_train=True):
    return trainer.prepare_self_reg_step(
        args, SimpleNamespace(device=torch.device("cpu")), ["fake-te"], batch, None, None,
        torch.float32, is_train=is_train,
    )


def test_alternating_modes_single_batch(trainer):
    args = _make_args()
    batch = _make_batch(["mychara, 1girl, solo"])
    modes = [(_prepare(trainer, args, batch) or {}).get("mode") for _ in range(4)]
    assert modes == ["hold", None, "hold", None]
    assert trainer._self_reg_step == 4


def test_anchor_caption_passed_to_encoder(trainer):
    args = _make_args()
    batch = _make_batch(["mychara, 1girl, solo"])
    _prepare(trainer, args, batch)
    assert trainer._captured["anchor_captions"] == ["person, 1girl, solo"]


def test_together_mode_every_step(trainer):
    args = _make_args(self_reg_batched=True, train_batch_size=4)
    batch = _make_batch(["mychara, 1girl"] * 4)
    for _ in range(3):
        ctx = _prepare(trainer, args, batch)
        assert ctx["mode"] == "together"
        assert ctx["half"] == 2


def test_batched_falls_back_to_alternating_with_batch_1(trainer):
    args = _make_args(self_reg_batched=True, train_batch_size=1)
    batch = _make_batch(["mychara, 1girl"])
    modes = [(_prepare(trainer, args, batch) or {}).get("mode") for _ in range(2)]
    assert modes == ["hold", None]
    assert trainer._self_reg_batched_warned is True


def test_validation_step_skipped(trainer):
    args = _make_args()
    batch = _make_batch(["mychara, 1girl"])
    assert _prepare(trainer, args, batch, is_train=False) is None
    assert trainer._self_reg_step == 0


def test_disabled_weight_skipped(trainer):
    args = _make_args(self_reg_weight=0.0)
    batch = _make_batch(["mychara, 1girl"])
    assert _prepare(trainer, args, batch) is None
    assert trainer._self_reg_step == 0


def test_cached_te_outputs_raise(trainer):
    args = _make_args(cache_text_encoder_outputs=True)
    batch = _make_batch(["mychara, 1girl"])
    with pytest.raises(ValueError):
        _prepare(trainer, args, batch)


def test_together_mutations_slice_and_restore(trainer):
    args = _make_args(self_reg_batched=True, train_batch_size=4)
    batch = _make_batch(["mychara, 1girl"] * 4)
    orig_loss_weights = batch["loss_weights"]
    orig_alpha_masks = batch["alpha_masks"]
    ctx = _prepare(trainer, args, batch)
    saved = trainer.apply_self_reg_batch_mutations(args, batch, ctx)
    assert batch["loss_weights"].shape[0] == 2
    assert batch["alpha_masks"].shape[0] == 2
    trainer.restore_self_reg_batch_mutations(args, batch, saved)
    assert batch["loss_weights"] is orig_loss_weights
    assert batch["alpha_masks"] is orig_alpha_masks


def test_together_gates_adaptive_sampler(trainer):
    args = _make_args(self_reg_batched=True, train_batch_size=4)
    batch = _make_batch(["mychara, 1girl"] * 4)
    ctx = _prepare(trainer, args, batch)
    trainer._adaptive_update_pending = True
    saved = trainer.apply_self_reg_batch_mutations(args, batch, ctx)
    assert trainer._adaptive_update_pending is False
    trainer.restore_self_reg_batch_mutations(args, batch, saved)
    assert trainer._adaptive_update_pending is True


def test_hold_disables_cfm_temporarily(trainer):
    args = _make_args(contrastive_flow_matching=True)
    batch = _make_batch(["mychara, 1girl"])
    ctx = _prepare(trainer, args, batch)
    assert ctx["mode"] == "hold"
    saved = trainer.apply_self_reg_batch_mutations(args, batch, ctx)
    assert args.contrastive_flow_matching is False
    trainer.restore_self_reg_batch_mutations(args, batch, saved)
    assert args.contrastive_flow_matching is True


def test_validate_args_missing_trigger():
    t = AnimaNetworkTrainer()
    with pytest.raises(AssertionError):
        t.validate_self_reg_args(_make_args(self_reg_trigger_word=""))


def test_validate_args_bad_noise():
    t = AnimaNetworkTrainer()
    with pytest.raises(AssertionError):
        t.validate_self_reg_args(_make_args(self_reg_noise=1.5))
    with pytest.raises(AssertionError):
        t.validate_self_reg_args(_make_args(self_reg_noise=-0.1))


def test_validate_args_cached_te_rejected():
    t = AnimaNetworkTrainer()
    with pytest.raises(ValueError):
        t.validate_self_reg_args(_make_args(cache_text_encoder_outputs=True))


def test_validate_args_rejects_ileco_addift():
    t = AnimaNetworkTrainer()
    with pytest.raises(AssertionError):
        t.validate_self_reg_args(_make_args(ileco=True))
    with pytest.raises(AssertionError):
        t.validate_self_reg_args(_make_args(addift=True))


def test_validate_args_forces_unet_only():
    t = AnimaNetworkTrainer()
    args = _make_args(network_train_unet_only=False)
    t.validate_self_reg_args(args)
    assert args.network_train_unet_only is True


def test_validate_args_dataset_group_batch_size(caplog):
    t = AnimaNetworkTrainer()
    # CLI train_batch_size is 1 (default), but dataset has batch_size = 4
    args = _make_args(self_reg_batched=True, train_batch_size=1)
    dataset = SimpleNamespace(batch_size=4, subsets=[])
    dataset_group = SimpleNamespace(datasets=[dataset])

    with caplog.at_level("WARNING"):
        t.validate_self_reg_args(args, dataset_group)
    assert "Self-regularization needs a batch of two or more" not in caplog.text


def test_validate_args_dataset_group_batch_size_warns_when_all_single(caplog):
    t = AnimaNetworkTrainer()
    args = _make_args(self_reg_batched=True, train_batch_size=1)
    dataset = SimpleNamespace(batch_size=1, subsets=[])
    dataset_group = SimpleNamespace(datasets=[dataset])

    with caplog.at_level("WARNING"):
        t.validate_self_reg_args(args, dataset_group)
    assert "Self-regularization needs a batch of two or more" in caplog.text


def test_validate_args_subset_batch_size_override(caplog):
    t = AnimaNetworkTrainer()
    args = _make_args(self_reg_batched=True, train_batch_size=1)
    subset = SimpleNamespace(batch_size=4)
    dataset = SimpleNamespace(batch_size=1, subsets=[subset])
    dataset_group = SimpleNamespace(datasets=[dataset])

    with caplog.at_level("WARNING"):
        t.validate_self_reg_args(args, dataset_group)
    assert "Self-regularization needs a batch of two or more" not in caplog.text


# ---------------------------------------------------------------------------
# Forward-path integration on CPU with a fake DiT (no weights needed)
# ---------------------------------------------------------------------------

import contextlib


class _FakeNetwork:
    def __init__(self):
        self.multiplier = 1.0
        self.calls = []

    def set_multiplier(self, value):
        self.calls.append(value)
        self.multiplier = value


class _FakeAnima(torch.nn.Module):
    """DiT stub: output = noisy input + prompt_mean * lora_scale * multiplier."""

    def __init__(self, net, lora_scale=1.0):
        super().__init__()
        self.net = net
        self.lora_scale = lora_scale

    def forward(self, x, ts, prompt_embeds, padding_mask=None, target_input_ids=None,
                target_attention_mask=None, source_attention_mask=None):
        shift = prompt_embeds.float().mean() * self.lora_scale * self.net.multiplier
        return (x.float() + shift).to(x.dtype)


def _forward_args(**kw):
    d = dict(
        flow_use_ot=False,
        timestep_sampling="uniform",
        sigmoid_scale=1.0,
        weighting_scheme="none",
        loss_type="l2",
        loss_scale=1.0,
        self_reg_weight=1.0,
        self_reg_trigger_word="mychara",
        self_reg_filler="person",
        self_reg_noise=0.0,
        self_reg_batched=False,
        min_snr_gamma=0,
        gradient_checkpointing=False,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
    )
    d.update(kw)
    return SimpleNamespace(**d)


def _forward_setup(trainer, batch_size, lora_scale=1.0, mode="hold"):
    net = _FakeNetwork()
    anima = _FakeAnima(net, lora_scale=lora_scale)
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=contextlib.nullcontext)
    scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    main_conds = [
        torch.full((batch_size, 8, 4), 2.0),
        torch.ones(batch_size, 8),
        torch.ones(batch_size, 8, dtype=torch.long),
        torch.ones(batch_size, 8),
    ]
    anchor_conds = [
        torch.full((batch_size, 8, 4), 0.5),
        torch.ones(batch_size, 8),
        torch.ones(batch_size, 8, dtype=torch.long),
        torch.ones(batch_size, 8),
    ]
    trainer._self_reg_ctx = {
        "mode": mode,
        "half": batch_size // 2 if mode == "together" else None,
        "anchor_conds": anchor_conds,
    }
    batch = {"captions": ["mychara, 1girl"] * batch_size, "loss_weights": torch.ones(batch_size)}
    latents = torch.randn(batch_size, 4, 8, 8)
    return net, anima, accelerator, scheduler, main_conds, batch, latents


def test_hold_forward_shapes_and_multiplier_restored():
    trainer = AnimaNetworkTrainer()
    args = _forward_args()
    net, anima, accelerator, scheduler, main_conds, batch, latents = _forward_setup(trainer, 2)
    pred, target, ts, weighting, noise = trainer.get_self_reg_noise_pred_and_target(
        args, accelerator, scheduler, latents, batch, main_conds, anima, net, torch.float32,
    )
    assert pred.shape == (2, 4, 8, 8)
    assert target.shape == (2, 4, 8, 8)
    assert ts.shape == (2,)
    assert weighting.shape == (2, 1, 1, 1)  # broadcast shape, same as the main path
    assert float(weighting.mean()) == pytest.approx(1.0)  # scheme none * weight 1
    assert net.multiplier == pytest.approx(1.0)  # restored after teacher branch
    assert 0.0 in net.calls  # teacher ran with LoRA off
    assert trainer._hf_noisy_latents is None
    assert trainer._anchor_noisy_latents is None
    assert trainer._noisy_latents is None


def test_hold_zero_lora_gives_zero_loss():
    # Zero-init LoRA (no shift at any multiplier) -> anchor loss ~= 0.
    trainer = AnimaNetworkTrainer()
    args = _forward_args()
    net, anima, accelerator, scheduler, main_conds, batch, latents = _forward_setup(
        trainer, 2, lora_scale=0.0
    )
    pred, target, ts, weighting, noise = trainer.get_self_reg_noise_pred_and_target(
        args, accelerator, scheduler, latents, batch, main_conds, anima, net, torch.float32,
    )
    loss = torch.nn.functional.mse_loss(pred.double(), target.double(), reduction="none")
    assert float(loss.mean()) == pytest.approx(0.0, abs=1e-6)


def test_hold_nonzero_lora_gives_nonzero_loss():
    trainer = AnimaNetworkTrainer()
    args = _forward_args()
    net, anima, accelerator, scheduler, main_conds, batch, latents = _forward_setup(trainer, 2)
    pred, target, ts, weighting, noise = trainer.get_self_reg_noise_pred_and_target(
        args, accelerator, scheduler, latents, batch, main_conds, anima, net, torch.float32,
    )
    loss = torch.nn.functional.mse_loss(pred.double(), target.double(), reduction="none")
    assert float(loss.mean()) > 0.0


def test_hold_pure_noise_anchor_runs():
    trainer = AnimaNetworkTrainer()
    args = _forward_args(self_reg_noise=1.0)
    net, anima, accelerator, scheduler, main_conds, batch, latents = _forward_setup(trainer, 1)
    pred, target, *_ = trainer.get_self_reg_noise_pred_and_target(
        args, accelerator, scheduler, latents, batch, main_conds, anima, net, torch.float32,
    )
    assert pred.shape == (1, 4, 8, 8)


def test_together_returns_half_main_and_stash():
    trainer = AnimaNetworkTrainer()
    args = _forward_args()
    net, anima, accelerator, scheduler, main_conds, batch, latents = _forward_setup(
        trainer, 4, mode="together"
    )
    # Production slices per-sample batch entries to the first half before the
    # forward (see apply_self_reg_batch_mutations); mirror that here.
    batch["loss_weights"] = batch["loss_weights"][:2]
    pred, target, ts, weighting, noise = trainer.get_self_reg_noise_pred_and_target(
        args, accelerator, scheduler, latents, batch, main_conds, anima, net, torch.float32,
    )
    assert pred.shape == (2, 4, 8, 8)  # truncated to half
    assert target.shape == (2, 4, 8, 8)
    assert ts.shape == (2,)
    assert trainer._self_reg_anchor_stash is not None
    assert float(trainer._self_reg_anchor_stash) > 0.0
    assert net.multiplier == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# process_batch override end-to-end (stubbed base returns the real 3-tuple)
# ---------------------------------------------------------------------------

import train_network


def _call_process_batch(trainer, args, batch, stub_stash=None):
    def fake_base_process_batch(self, batch, *pos, **kw):
        if stub_stash is not None:
            trainer._self_reg_anchor_stash = stub_stash
        return (torch.tensor(1.0), torch.tensor(1.0), None)

    orig = train_network.NetworkTrainer.process_batch
    train_network.NetworkTrainer.process_batch = fake_base_process_batch
    try:
        return trainer.process_batch(
            batch, ["fake-te"], None, None, None, None, torch.float32, torch.float32,
            SimpleNamespace(device=torch.device("cpu")), args, None, None,
            is_train=True, train_text_encoder=False, train_unet=True, edm2_model=None,
        )
    finally:
        train_network.NetworkTrainer.process_batch = orig


def test_process_batch_hold_returns_triple(trainer):
    args = _make_args()
    batch = _make_batch(["mychara, 1girl, solo"])
    out = _call_process_batch(trainer, args, batch)
    assert isinstance(out, tuple) and len(out) == 3
    final, pre, scaled = out
    assert float(final) == pytest.approx(1.0)
    assert float(pre) == pytest.approx(1.0)
    assert scaled is None
    assert trainer._self_reg_ema == pytest.approx(1.0)
    assert trainer._self_reg_loss_value == pytest.approx(1.0)
    # Per-step state is cleared and the batch is untouched.
    assert trainer._self_reg_ctx is None
    assert trainer._self_reg_anchor_stash is None
    assert batch["loss_weights"].shape[0] == 1


def test_process_batch_together_adds_stash_to_both_losses(trainer):
    args = _make_args(self_reg_batched=True, train_batch_size=4)
    batch = _make_batch(["mychara, 1girl"] * 4)
    orig_loss_weights = batch["loss_weights"]
    orig_alpha_masks = batch["alpha_masks"]
    final, pre, scaled = _call_process_batch(trainer, args, batch, stub_stash=torch.tensor(0.5))
    assert float(final) == pytest.approx(1.5)
    assert float(pre) == pytest.approx(1.5)  # logging value carries the held term too
    assert scaled is None
    assert trainer._self_reg_ema == pytest.approx(0.5)
    assert batch["loss_weights"] is orig_loss_weights
    assert batch["alpha_masks"] is orig_alpha_masks
    assert trainer._self_reg_ctx is None
    assert trainer._self_reg_anchor_stash is None


def test_process_batch_together_missing_stash_raises(trainer):
    args = _make_args(self_reg_batched=True, train_batch_size=4)
    batch = _make_batch(["mychara, 1girl"] * 4)
    with pytest.raises(ValueError):
        _call_process_batch(trainer, args, batch, stub_stash=None)
    # State is still cleaned up and the batch restored after the error.
    assert trainer._self_reg_ctx is None
    assert batch["loss_weights"].shape[0] == 4


def test_process_batch_main_step_passes_through(trainer):
    args = _make_args()
    batch = _make_batch(["mychara, 1girl"])
    # First call is hold (step 1); second is an even main step -> plain passthrough.
    _call_process_batch(trainer, args, batch)
    out = _call_process_batch(trainer, args, batch)
    final, pre, scaled = out
    assert float(final) == pytest.approx(1.0)
    assert trainer._self_reg_step == 2
