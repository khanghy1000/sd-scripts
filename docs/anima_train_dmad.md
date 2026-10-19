# Anima DMAD training (`anima_train_dmad.py`)

DMAD (Distribution Matching as Adversarial Distillation, arXiv:2610.02188)
for the Anima rectified-flow DiT, ported from the MOSS training implementation
(`RamTorchRun/moss/train_dmad.py`, MOSS-Image-SFT method pieces) and integrated
with this repo's dataset / accelerator / caching / sampling infrastructure
(same as `anima_train.py`). The core lives in `library/anima_dmad.py`; the
teacher-sample cache producer is `anima_dmad_generate_teacher.py`.

## How it works: one frozen DiT, three LoRA roles

There is no separate critic copy. **One frozen DiT backbone** (`--teacher_model_path`
if given, else `--pretrained_model_name_or_path`) carries three role adapters,
injected by `inject_dmad_lora` on every block projection (`self_attn` /
`cross_attn` q/k/v/output and the `mlp.layer1`/`layer2` Linears):

| role | adapter | purpose |
| --- | --- | --- |
| `None` | frozen base (+ frozen always-on `--teacher_lora_path`) | the **teacher**: `--teacher_steps` Euler steps on the shift-`--dmad_flow_shift` grid, no CFG (guidance is baked into the teacher / its LoRA) |
| `default` | **student** LoRA (zero-init B: the student starts AT the teacher) | re-noise sampler: k ~ U{0..steps-1} no-grad re-noise steps, then one step with grad → `x0_G` |
| `disc` | **discriminator** LoRA | same backbone reads final-block image tokens through RMSNorm + two heads (teacher/real), each Linear-SiLU-Linear → per-token logit, token mean |

Backbone weights are fp32 master weights by default (cast with
`--full_bf16`/`--full_fp16` if you want the backbone in half precision).

Every update (moss `generator_step` / `disc_step` / `update_gap`) does:

1. **Generator turn** (accumulated over `--gradient_accumulation_steps`
   micro-batches): sample `Q` (see below), render `x0_G` with the student
   re-noise sampler; noise it to `t = shift(u)`, `u ~ U(--disc_u_range)`; the
   disc is frozen and `L_G = -mean(lambda_T * w(u) * h_T + lambda_R * h_R)`
   backpropagates **through the frozen disc forward** into the student role
   parameters (`--lambda_teacher` / `--lambda_real`, default 1.0).
   Plain AdamW on betas `--dmad_adam_beta1/2` (default `0, 0.99` as in the moss
   config) and weight decay `--dmad_weight_decay` (0); lr `--learning_rate`,
   ramped by `--dmad_warmup_steps`.
2. **Discriminator turn**: one concatenated forward of
   `[noisy(Q), noisy(T), noisy(x0_G.detach())]` with per-sample
   `t = shift(u)`, `u ~ U(--disc_u_range)`; balanced BCE per head
   `bce(h_T[Q], h_T[G]) + bce(h_R[T], h_R[G])`; lr `--disc_lr` (4e-5).
3. **Gap reweighting**: per `--gap_bands` equal-width bands, an fp64 EMA
   (`--gap_beta`) of `mean h_R(T) - mean h_R(Q)`;
   `w = sigmoid((median - gap) / --gap_tau)` normalized over measured bands
   (`--gap_floor`); updated with an all-gather across ranks. `w` carries no
   gradient. Teacher-head weighting in the generator loss uses these band
   weights.
4. **Power EMA** of the student role (`--dmad_ema_sigma_rel`, default 0.1,
   Karras-style), for the `*_ema_` export and previews.

`--disc_warmup_steps` (0) trains only the discriminator for the first updates;
the generator LR ramps with its EMA-updates count.

### Q: teacher samples

- **Cache (fast, default recipe)**: `--teacher_latent_path` = LMDB built by
  `anima_dmad_generate_teacher.py` (latents + prompt per row). Rows cycle
  independently of the real-data order; the cached Q's own prompt is used for
  its conditioning (so its teacher-head chunk is self-consistent).
- **Online (moss-exact)**: without the cache, `Q = teacher_sample(noise)`
  from the shared backbone in teacher role — `--teacher_steps` no-grad Euler
  steps per caption, no CFG. `--teacher_cfg > 1` opts back into guided sampling
  (needs `--teacher_negative_prompt`; uses the repo Euler sampler with baked
  guidance).

## Exported files (kohya anima LoRA format, `lora_unet_*` keys)

Written on every save step / epoch boundary and at the end
(`<output_name>` defaults to `anima-dmad`):

- `<base>_student_<step>.safetensors` — live student LoRA (applied **on top of**
  the teacher LoRA at `--teacher_lora_strength`).
- `<base>_student_ema_<step>.safetensors` — EMA of the student LoRA (usually
  the better sampler weights).
- `<base>_data_lora_<steps>step_<step>.safetensors` — only with
  `--teacher_lora_path`: ONE exact-rank LoRA = `strength * teacher + student(EMA)`
  (scales folded into the up factor, `alpha = rank`); use at strength 1 on the
  stock backbone.
- `<base>_disc_<step>.safetensors` + `<base>_disc_heads_<step>.safetensors` —
  discriminator role LoRA and its heads (for resuming analysis / tooling).

Each lora file carries SAI-style metadata (`ss_network_dim`, `ss_steps`, ...).

## 1. Dataset preparation

Real images use the standard dataset pipeline (same as `anima_train.py`); see
the [dataset configuration guide](./config_README-en.md).

```
data/
  train/            # real images (jpg/png/webp), any size; buckets handle the rest
prompts.txt         # prompt corpus for the teacher LMDB (Q)
sample_prompts.toml # sampling prompts
dataset.toml        # dataset config
```

Notes:

- **Resolution/buckets.** Cached `Q` latents may differ in size from `T` —
  exact match is not required, but keep buckets near the teacher generation
  resolution so `T/G/Q` cover the same detail regime.
- **Resolution steps.** Bucket sizes must be multiples of 16; the pipeline
  asserts this.
- **Captions.** Real-data supervision (`d_real`) works best with clean,
  descriptive captions. Keep any caption shuffling/dropout compatible with
  text-encoder caching (`--cache_text_encoder_outputs` asserts it).
- **Caching.** Pass `--cache_latents --cache_text_encoder_outputs` (add
  `_to_disk` variants to reuse across runs).
- **`prompts.txt`** (cache producer input) can be a different corpus from the
  training captions; Q rows are sampled independently during training.

### `sample_prompts.toml`

The distillated student samples without CFG (`scale = 1.0`) and with
`sample_steps = --student_steps`:

```toml
[[prompts]]
prompt = "a photo of a cat in a sunlit garden, detailed fur"
negative_prompt = ""
width = 1024
height = 1024
sample_steps = 4
scale = 1.0
seed = 42
```

Previews use the **EMA student** (moss-style swap around the sampling call).

## 2. Teacher cache (Q): pre-generate once

```bash
# one process per GPU (shards), resumable per chunk
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python anima_dmad_generate_teacher.py --mode generate \
    --shard $i --num_shards 4 \
    --dit /path/to/teacher_dit.safetensors \
    --qwen3 /path/to/qwen3-0.6b.safetensors \
    --prompts prompts.txt --out_dir teacher_chunks \
    --num_samples 50000 --batch_size 8 \
    --width 1024 --height 1024 \
    --teacher_steps 8 --teacher_cfg 4.0 --negative_prompt "" \
    --teacher_flow_shift 3.0 --seed 42 &
done; wait

python anima_dmad_generate_teacher.py --mode merge \
  --out_dir teacher_chunks --lmdb_path teacher_q_lmdb
```

Comments:

- The LMDB stores `latents_{i}_data` (fp16 latents) + `prompts_{i}_data`
  (utf-8 prompt) + `latents_shape`; training reads rows randomly each
  iteration. `--dmad_teacher_batch_size` (default 8) need not match the train
  batch size.
- CFG used at generation is **baked into Q**: the student learns the guided
  distribution directly and samples with CFG 1. Pin the teacher settings
  (steps / CFG / LoRA) — the cache defines your teacher.
- Size estimate: one 1024px latent is 16x128x128 fp16 = 512 KiB → 50k samples
  ≈ 25 GiB.

Without the cache, training samples the teacher online (costs
`--teacher_steps` frozen forwards per micro-batch; `--teacher_cfg > 1` doubles
that).

## 3. Worked example: 4-step CFG-1 student

**Step 1 — generate Q** (Section 2; here a 30-step CFG-4 teacher with a
frozen teacher LoRA if you have one, applied in the producer script).

**Step 2 — `dataset.toml`** (Section 1) and `sample_prompts.toml` with
`scale = 1.0`, `sample_steps = 4`.

**Step 3 — distill:**

```bash
accelerate launch --num_processes=1 anima_train_dmad.py \
  --dataset_config dataset.toml \
  --pretrained_model_name_or_path /models/anima_base.safetensors \
  --vae /models/anima_vae.safetensors \
  --qwen3 /models/qwen3-0.6b.safetensors \
  --output_dir output --output_name anima-4step-cfg1 \
  --save_every_n_steps 500 --sample_every_n_steps 500 --sample_at_first \
  --seed 42 --mixed_precision bf16 \
  --learning_rate 2.5e-6 --max_grad_norm 1.0 \
  --max_train_steps 4000 --gradient_checkpointing \
  --cache_latents --cache_text_encoder_outputs \
  --student_steps 4 --dmad_flow_shift 3.0 \
  --dmad_lora_rank 128 --dmad_lora_alpha 128 --dmad_head_hidden 1024 \
  --disc_lr 4e-5 --dmad_ema_sigma_rel 0.1 \
  --teacher_latent_path teacher_q_lmdb --dmad_teacher_batch_size 8 \
  --teacher_lora_path /models/teacher_lora.safetensors --teacher_lora_strength 1.0
```

Defaults (from the moss config): lr 2.5e-6, betas (0, 0.99), rank/alpha 128,
heads 1024, `--disc_u_range 0.001 0.999`, gap bands 10 / tau 2 / beta 0.99,
EMA sigma_rel 0.1, `--teacher_steps 8`, `--dmad_flow_shift 3.0`. One-step
distillation works with the defaults (`--student_steps 1` is the default).

Removed-flags note: the old single-critic copy flags
(`--critic_num_blocks`, `--critic_max_sigma`, `--conditioning_sigma`,
`--backward_simulation`, `--teacher_gap_route`, `--teacher_loss_weight` /
`--real_loss_weight` / `--critic_loss_weight`, `--generator_spectral_norm`,
`--generator_ema_kimg` / `--generator_ema_rampup`, `--dmad_critic_lora_dim`,
`--network_module` / `--network_dim` / `--network_alpha`,
`--num_denoising_steps` as a separate arg — now a deprecated alias of
`--student_steps`) are gone; the moss algorithm needs the shared role-LoRA
backbone. `--optimizer_type` / `--lr_scheduler` are ignored (plain AdamW,
no scheduler, both optimizers).

## 4. Memory / performance notes

- Backbone + student/disc role LoRAs + heads + VAE + Qwen3 resident; cached-Q
  mode adds only a small LMDB loader, online teacher mode is the
  memory-cheapest but slowest option.
- `--gradient_checkpointing` uses the role-aware per-block checkpointing
  (`enable_dmad_block_checkpointing`): the LoRA role is captured at forward
  time and re-applied when the checkpoint recomputes during backward — student
  and disc can share the checkpointed backbone without role leaks. Verify
  parity with CPU test `tests/test_anima_dmad.py`.
- `--cpu_offload_checkpointing` additionally offloads activations between
  blocks to CPU RAM (slower).
- `--blocks_to_swap <n>` works (block-swap + `--cpu_offload_checkpointing`
  are mutually exclusive).
- Multi-GPU: the DiT is NOT wrapped by accelerate/DDP; role LoRA gradients are
  averaged manually (`allreduce_mean` buckets) and parameters broadcast from
  rank 0 at start (moss does the same). The dataset loader is prepared by
  accelerate; the cached-Q loader cycles independently per rank.
- Not supported: `--deepspeed`, `--fused_backward_pass`, `--resume`
  (warned and ignored; restart runs from scratch or reinitialize weights with
  the exported files).
