# Anima DMAD training (`anima_train_dmad.py`)

DMAD (Distribution Matching as Adversarial Distillation, arXiv:2610.02188)
for the Anima rectified-flow DiT, ported from `DMAD/train/image` (SDXL:
`main/train_sd.py` + `main/sd_guidance.py`). Critic implementation lives in
`library/anima_dmad.py`; teacher-sample caching in
`anima_dmad_generate_teacher.py`.

## How it works

Each iteration does two updates (paper App. B: one discriminator update per
generator update, AdamW with (beta1, beta2) = (0, 0.99)):

1. **Generator turn** — the student DiT renders latents `G` (1-step at
   `--conditioning_sigma`, or multi-step Euler with `--num_denoising_steps`
   and optionally `--backward_simulation`). The critic is frozen; the
   generator minimizes the linear loss `-d_teacher(G) * --teacher_loss_weight`
   plus `-d_real(G) * --real_loss_weight` (paper Eq. 6/9; linear, not
   non-saturating).
2. **Critic turn** — the critic (DiT copy reading final-block tokens +
   real/teacher MLP heads, per-token logits averaged as in the Wan setup)
   minimizes balanced BCE `BCE(T=1, G=0) + BCE(Q=1, G=0)` (paper Eq. 5) on
   real latents `T`, detached `G`, and teacher samples `Q`.

`Q` comes from a **pre-generated LMDB cache** (`--teacher_latent_path`, built
by `anima_dmad_generate_teacher.py`; no online teacher evaluation, as in the
paper) or, as a fallback, from on-the-fly multi-step Euler sampling with the
frozen teacher DiT (`--teacher_model_path`, `--teacher_steps`,
`--teacher_cfg`).

Critic noising is flow-matching `x_t = (1-s)*x0 + s*eps` with `s` uniform in
`[0, --critic_max_sigma)`; `T`/`G`/`Q` samples of each index share one noise
level. `--teacher_gap_route` re-weights the teacher term per noise band from
the online gap `E[d_real(T)] - E[d_real(Q)]` (EMA decay 0.99, paper Eq. 7-9):
ten equal-width bands over `[0, 1)`, median-centered sigmoid weights with
`--gap_tau` (default 2, as for the paper's Wan/MiniMax-H3 DiTs), normalized
over measured bands with a 1e-4 floor. Weights carry no gradient.
The generator keeps an EMA (half-life `--generator_ema_kimg` thousand images,
ramp-up `--generator_ema_rampup`; defaults 1000 / 0.05 as in the paper).

The generator trains full-finetune by default or LoRA (`--network_module
networks.lora_anima --network_dim ...`); the critic trains fully or with LoRA
on its backbone (`--dmad_critic_lora_dim ...`) while its heads always train
fully.

## 1. Dataset preparation

Real images `T` use the standard dataset pipeline (same as `anima_train.py`).
See also the [dataset configuration guide](./config_README-en.md).

```
data/
  train/            # real images (jpg/png/webp), any size; buckets handle the rest
    00001.jpg
    00002.jpg
    ...
  reg/              # optional regularization images
prompts.txt         # one prompt per line, used for teacher generation (Q)
sample_prompts.toml # sampling prompts (see below)
dataset.toml        # dataset config
```

### 1.1 `dataset.toml`

DreamBooth-style (captions from `.txt` files next to images, or filename):

```toml
[general]
shuffle_caption = false
# keep Dropout at 0 for distillation so T conditioning is deterministic
caption_dropout_rate = 0

[[datasets]]
resolution = 1024
batch_size = 4
bucket_no_upscale = false

  [[datasets.subsets]]
  image_dir = "data/train"
  # caption_extension = ".txt"   # uncomment for image.txt captions
  # num_repeats = 1
```

Or with an explicit metadata file (`image_dir` + `metadata_file` with
`{"image_path": ..., "caption": ...}` rows):

```toml
[[datasets]]
resolution = [1024, 1024]
batch_size = 4

  [[datasets.subsets]]
  image_dir = "data/train"
  metadata_file = "data/meta.json"
```

Guidance:

- **Resolution/buckets.** The critic is resolution-flexible (RoPE crop), and
  cached `Q` latents may differ in size from `T` — exact match is not
  required. Still, keep training buckets near the teacher generation
  resolution (e.g. 1024 with `bucket_no_upscale = false`) so `T`/`G`/`Q`
  cover the same detail regime.
- **Resolution steps.** Bucket sizes must be multiples of 16
  (VAE downscale 8 x patch size 2); the script asserts this.
- **Captions.** Real-data supervision (`d_real`) works best with clean,
  descriptive captions. `shuffle_caption` / tag dropout must stay compatible
  with text-encoder caching (see errors at startup otherwise).
- **Caching.** Always pass `--cache_latents --cache_text_encoder_outputs`
  (add `_to_disk` variants to reuse across runs). First run writes
  `*_anima.npz` / `*_anima_te.npz` next to the images.
- **`prompts.txt`.** Any prompt corpus in the target domain works (the paper
  uses LAION captions). It does not need to match the real dataset 1:1;
  `Q` rows are sampled independently during training, but sharing the domain
  keeps the teacher distribution relevant.

### 1.2 `sample_prompts.toml`

Used for `--sample_prompts`. For a guidance-baked student (Section 4), sample
with `scale = 1.0` (no CFG):

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

## 2. Teacher cache (Q): pre-generate once

Generate teacher latents with the multi-step teacher (here: 30 steps, CFG 4):

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
    --teacher_steps 30 --teacher_cfg 4.0 --negative_prompt "" \
    --teacher_flow_shift 3.0 --seed 42 &
done; wait

# pack chunks into one LMDB
python anima_dmad_generate_teacher.py --mode merge \
  --out_dir teacher_chunks --lmdb_path teacher_q_lmdb
```

Notes:

- Sample noise is seeded by global index (`seed + i`), so keep
  `--num_shards`/`--batch_size` fixed to reproduce a cache. Completed chunks
  are skipped on re-runs.
- If `prompts.txt` is shorter than `--num_samples`, prompts cycle with
  different seeds.
- Size estimate: one 1024px latent is 16x128x128 fp16 = 512 KiB, so 50k
  samples take ~25 GiB.
- The LMDB stores `latents_{i}_data` + `prompts_{i}_data` (same key layout as
  the DMAD SDXL latents). Training reads rows randomly each iteration;
  `--dmad_teacher_batch_size` (default 8) need not match the train batch size.
- CFG used here is **baked into Q**: the student learns the guided
  distribution directly and samples with CFG 1 (Section 4).

Without `--teacher_latent_path`, training falls back to on-the-fly teacher
sampling (`--teacher_steps`, `--teacher_cfg`, `--teacher_negative_prompt`),
which costs that many frozen forwards per iteration and keeps the teacher DiT
resident. The cache removes both costs.

## 3. LoRA

Generator LoRA (paper H3 setup: rank 128, alpha 128, on attention + MLPs):

```bash
  --network_module networks.lora_anima --network_dim 128 --network_alpha 128 \
  --network_args "train_llm_adapter=true"   # optional: also adapt the LLM bridge
```

- Base DiT stays frozen; checkpoints are LoRA safetensors
  (`<output_name>_<step>.safetensors`) plus `<output_name>_ema_<step>.safetensors`
  (EMA of the LoRA). Resume with `--network_weights <ema-or-live>.safetensors`
  (add `--dim_from_weights` if the rank differs from `--network_dim`).
- At inference, merge or load the LoRA (e.g. `load_anima_model(...,
  lora_weights_list=[...])`, or convert with
  `networks/convert_anima_lora_to_comfy.py` for ComfyUI).
- `--generator_spectral_norm` is not supported with generator LoRA.
- Sampling during training automatically includes the live LoRA.

Critic LoRA (backbone only; heads always train fully):

```bash
  --dmad_critic_lora_dim 128 --dmad_critic_lora_alpha 128
```

Combine with `--critic_freeze_backbone` semantics automatically (backbone
frozen except LoRA). Critic LoRA weights save to
`<output_name>_critic_lora_<step>.safetensors` alongside the full critic state.

## 4. Worked example: 30-step CFG-4 teacher into a 4-step CFG-1 student

Goal: a student that renders in 4 Euler steps with `scale = 1.0`, matching a
30-step CFG-4 teacher. Guidance is baked in through `Q`, never applied to the
student.

**Step 1 — generate Q with the teacher (30 steps, CFG 4.0):**

```bash
python anima_dmad_generate_teacher.py --mode generate \
  --dit /models/anima_base.safetensors \
  --qwen3 /models/qwen3-0.6b.safetensors \
  --prompts prompts.txt --out_dir teacher_chunks \
  --num_samples 50000 --batch_size 8 \
  --width 1024 --height 1024 \
  --teacher_steps 30 --teacher_cfg 4.0 --negative_prompt "" \
  --teacher_flow_shift 3.0 --seed 42 --dtype bf16

python anima_dmad_generate_teacher.py --mode merge \
  --out_dir teacher_chunks --lmdb_path teacher_q_lmdb
```

**Step 2 — prepare `dataset.toml`** as in Section 1.1 (1024px buckets,
`caption_dropout_rate = 0`) and cache latents/text-encoder outputs.

**Step 3 — distill (4 steps, backward simulation, LoRA rank 128):**

```bash
accelerate launch --num_processes=1 anima_train_dmad.py \
  --dataset_config dataset.toml \
  --pretrained_model_name_or_path /models/anima_base.safetensors \
  --vae /models/anima_vae.safetensors \
  --qwen3 /models/qwen3-0.6b.safetensors \
  --output_dir output --output_name anima-4step-cfg1 \
  --save_every_n_steps 500 --save_every_n_epochs 1 \
  --sample_prompts sample_prompts.toml --sample_every_n_steps 500 \
  --seed 42 --mixed_precision bf16 \
  --optimizer_type AdamW --optimizer_args betas="0,0.99" \
  --learning_rate 4e-5 --lr_scheduler constant_with_warmup --lr_warmup_steps 500 \
  --max_train_steps 800 --gradient_checkpointing \
  --cache_latents --cache_text_encoder_outputs \
  --network_module networks.lora_anima --network_dim 128 --network_alpha 128 \
  --teacher_latent_path teacher_q_lmdb --dmad_teacher_batch_size 8 \
  --num_denoising_steps 4 --backward_simulation --dmad_flow_shift 3.0 \
  --guidance_lr 4e-5 --teacher_loss_weight 1 --real_loss_weight 1 \
  --teacher_gap_route \
  --dmad_critic_lora_dim 128 --dmad_critic_lora_alpha 128 \
  --generator_ema_kimg 1000 --generator_ema_rampup 0.05
```

Why these values:

- `--num_denoising_steps 4 --backward_simulation`: the student learns all 4
  inference steps (its own sampler rolled out no-grad, one grad update per
  iter — the 4-step recipe of the paper's SDXL run).
- No CFG anywhere on the student side: sampling prompts use `scale = 1.0`,
  and the loss has no guidance term. The CFG-4 look comes entirely from `Q`.
- `--learning_rate/--guidance_lr 4e-5` with `loss_weight 1`: the LoRA-scale
  convention of the paper's H3 run (full-finetune SDXL uses 5e-7/5e-3; scale
  to your setup). `--critic_loss_weight` stays at its default 1e-2 (the value
  behind the paper's SDXL results); tune it if the critic over/under-powers
  the generator (watch `critic_acc_*`: healthy training keeps them away from
  0/1 extremes).
- `--dmad_critic_lora_dim 128`: LoRA critic backbone as in the H3 run; drop
  it for a fully-trained critic (more VRAM).

**Step 4 — sample with CFG 1:**

Use `sample_prompts.toml` with `scale = 1.0`, `sample_steps = 4` (Section
1.2), or run inference with `guidance_scale=1.0` and the merged/attached
LoRA. If outputs look under-guided, the teacher cache (CFG strength at
generation time) is the knob — not student-side CFG.

## 5. Memory / performance notes

- Cached-Q mode loads no teacher DiT (the paper's setup). Full mode holds
  generator + frozen teacher + critic + VAE + Qwen3.
- The critic defaults to a full DiT copy; `--critic_num_blocks` (e.g. half)
  selects a cheaper encoder-style critic as in the SDXL/ImageNet setups.
- Use `--gradient_checkpointing`, `--blocks_to_swap` (generator only), small
  batch sizes, and latent/text-encoder caching on small GPUs.
- Multi-GPU: the real-data loader is sharded by accelerate; the cached-Q
  loader cycles independently per rank (same order) — a close approximation
  of independent sampling that keeps every rank fed without IPC.
- Not supported yet: `--deepspeed`, `--fused_backward_pass`, critic resume
  from checkpoint (generator LoRA resume via `--network_weights` works).
