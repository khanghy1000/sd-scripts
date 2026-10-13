# Generate and cache Anima teacher samples (Q) for DMAD training.
#
# Mirrors ``DMAD/train/image/main/sdxl/generate_teacher_latents.py``: teacher
# samples are produced once (multi-step Euler + CFG) and cached in an LMDB that
# ``anima_train_dmad.py --teacher_latent_path`` consumes, so no online teacher
# evaluation happens during training (paper App. B).
#
# Step 1, one process per GPU (resumable; completed chunks are skipped). The
# noise of each sample is seeded by its global index, so keep --num_shards and
# --batch_size fixed to reproduce a dataset:
#     for i in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$i python anima_dmad_generate_teacher.py --mode generate --shard $i --num_shards 4 \
#             --dit ... --qwen3 ... --prompts prompts.txt --out_dir teacher_chunks --num_samples 50000 &
#     done; wait
# Step 2, pack the chunks into one LMDB:
#     python anima_dmad_generate_teacher.py --mode merge --out_dir teacher_chunks \
#         --lmdb_path teacher_q_lmdb
#
# LMDB format (same keys as the DMAD SDXL latents): ``latents_shape`` holds
# "N C H W"; row i holds ``latents_{i}_data`` (fp16 raw, Anima VAE latent
# space: 16 channels, 1/8 of the pixel resolution) and ``prompts_{i}_data``.

import argparse
import glob
import os

import numpy as np
import torch

from library.device_utils import init_ipex

init_ipex()

from library import anima_utils, strategy_anima
from library.anima_dmad import euler_sample_x0
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Pre-generate Anima DMAD teacher latents (Q)")
    p.add_argument("--mode", choices=["generate", "merge"], default="generate")
    p.add_argument("--dit", type=str, default=None, help="Anima DiT checkpoint (teacher)")
    p.add_argument("--qwen3", type=str, default=None, help="Qwen3-0.6B model (dir or safetensors)")
    p.add_argument("--t5_tokenizer_path", type=str, default=None)
    p.add_argument("--qwen3_max_token_length", type=int, default=512)
    p.add_argument("--t5_max_token_length", type=int, default=512)
    p.add_argument("--prompts", type=str, default=None, help="txt file, one prompt per line (generate mode)")
    p.add_argument("--out_dir", type=str, required=True, help="folder for per-shard chunk files / chunk source")
    p.add_argument("--lmdb_path", type=str, default=None, help="merge mode: output LMDB")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--chunk_size", type=int, default=512, help="samples per chunk file (resume unit)")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--teacher_steps", type=int, default=30)
    p.add_argument("--teacher_cfg", type=float, default=4.0, help="CFG scale baked into Q (1.0 = no guidance)")
    p.add_argument("--negative_prompt", type=str, default="", help="negative prompt for CFG")
    p.add_argument("--teacher_flow_shift", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn_mode", type=str, default="torch")
    p.add_argument("--split_attn", action="store_true")
    return p.parse_args()


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    assert prompts, f"no prompts in {path}"
    print(f"{len(prompts)} prompts from {path}")
    return prompts


def chunk_path(out_dir, shard, chunk_idx):
    return os.path.join(out_dir, f"shard{shard:02d}_chunk{chunk_idx:05d}.npz")


def chunk_complete(cpath, expect_n):
    if not os.path.exists(cpath):
        return False
    try:
        with np.load(cpath) as z:
            return z["latents"].shape[0] == expect_n
    except Exception:
        return False


def run_generate(args):
    assert args.dit and args.qwen3 and args.prompts, "--dit, --qwen3 and --prompts are required in generate mode"
    assert torch.cuda.is_available(), "teacher generation needs a CUDA GPU"
    device = torch.device("cuda")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    os.makedirs(args.out_dir, exist_ok=True)

    prompts = load_prompts(args.prompts)
    my_indices = list(range(args.shard, args.num_samples, args.num_shards))

    logger.info("Loading Qwen3 text encoder...")
    qwen3_te, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=dtype, device="cpu")
    t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer,
        qwen3_max_length=args.qwen3_max_token_length, t5_max_length=args.t5_max_token_length)
    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()

    logger.info("Loading teacher DiT...")
    dit = anima_utils.load_anima_model("cuda", args.dit, args.attn_mode, args.split_attn, "cuda", dtype)
    dit.requires_grad_(False)
    dit.eval()
    qwen3_te.to(device)
    qwen3_te.eval()

    latent_h, latent_w = args.height // 8, args.width // 8
    use_cfg = float(args.teacher_cfg) > 1.0

    # negative conditioning is prompt-independent: encode once
    neg_cond = None
    if use_cfg:
        with torch.no_grad():
            neg_tokens = tokenize_strategy.tokenize([args.negative_prompt])
            neg_cond = text_encoding_strategy.encode_tokens(tokenize_strategy, [qwen3_te], neg_tokens)
            neg_cond = (
                neg_cond[0].to(device, dtype),
                neg_cond[1].to(device),
                neg_cond[2].to(device, dtype=torch.long),
                neg_cond[3].to(device),
            )

    n_chunks = (len(my_indices) + args.chunk_size - 1) // args.chunk_size
    for ci in range(n_chunks):
        cpath = chunk_path(args.out_dir, args.shard, ci)
        idxs = my_indices[ci * args.chunk_size:(ci + 1) * args.chunk_size]
        if chunk_complete(cpath, len(idxs)):
            print(f"skip {os.path.basename(cpath)} (complete)", flush=True)
            continue
        lats, texts = [], []
        for b0 in range(0, len(idxs), args.batch_size):
            bidx = idxs[b0:b0 + args.batch_size]
            batch_prompts = [prompts[i % len(prompts)] for i in bidx]
            with torch.no_grad():
                tokens = tokenize_strategy.tokenize(batch_prompts)
                pos = text_encoding_strategy.encode_tokens(tokenize_strategy, [qwen3_te], tokens)
                pos = (pos[0].to(device, dtype), pos[1].to(device),
                       pos[2].to(device, dtype=torch.long), pos[3].to(device))
                if use_cfg:
                    n = len(bidx)
                    neg = (neg_cond[0].expand(n, -1, -1).contiguous(), neg_cond[1].expand(n, -1).contiguous(),
                           neg_cond[2].expand(n, -1).contiguous(), neg_cond[3].expand(n, -1).contiguous())
                else:
                    neg = None
                noise = torch.stack([
                    torch.randn(16, latent_h, latent_w, dtype=torch.float32,
                                generator=torch.Generator().manual_seed(args.seed + i))
                    for i in bidx
                ]).to(device, dtype)
                q = euler_sample_x0(dit, noise, *pos, steps=args.teacher_steps,
                                    flow_shift=args.teacher_flow_shift,
                                    guidance_scale=args.teacher_cfg, neg_cond=neg)
            lats.append(q.float().cpu().numpy().astype(np.float16))
            texts.extend(batch_prompts)
        np.savez(cpath + ".tmp.npz", latents=np.concatenate(lats, axis=0),
                 prompt_idx=np.array(idxs, dtype=np.int64), prompts=np.array(texts, dtype=object))
        os.replace(cpath + ".tmp.npz", cpath)
        print(f"shard {args.shard}: wrote chunk {ci + 1}/{n_chunks}", flush=True)


def run_merge(args):
    import lmdb

    assert args.lmdb_path, "--lmdb_path required for merge"
    chunks = sorted(glob.glob(os.path.join(args.out_dir, "shard*_chunk*.npz")))
    assert chunks, f"no chunks in {args.out_dir}"
    total, shape = 0, None
    for c in chunks:
        with np.load(c, allow_pickle=True) as z:
            total += z["latents"].shape[0]
            shape = z["latents"].shape[1:]
    print(f"{len(chunks)} chunks, {total} samples, latent shape {shape}")
    os.makedirs(args.lmdb_path, exist_ok=True)
    # LMDB needs headroom; enforce a minimum so small caches also merge.
    map_size = max(int(total * np.prod(shape) * 2 * 1.3), 1024**3)
    env = lmdb.open(args.lmdb_path, map_size=map_size)
    seen = np.zeros(total, dtype=bool)
    with env.begin(write=True) as txn:
        txn.put(b"latents_shape", f"{total} {' '.join(map(str, shape))}".encode())
        for c in chunks:
            with np.load(c, allow_pickle=True) as z:
                for lat, text, gi in zip(z["latents"], z["prompts"], z["prompt_idx"]):
                    gi = int(gi)
                    assert not seen[gi], f"duplicate index {gi} in {c}"
                    seen[gi] = True
                    txn.put(f"latents_{gi}_data".encode(), lat.astype(np.float16).tobytes())
                    txn.put(f"prompts_{gi}_data".encode(), str(text).encode())
    env.close()
    assert seen.all(), f"missing indices, e.g. {np.where(~seen)[0][:5]}"
    print(f"wrote {total} samples to {args.lmdb_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "generate":
        run_generate(args)
    else:
        run_merge(args)
