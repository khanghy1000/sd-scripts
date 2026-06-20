"""Anima ControlNet-VACE inference.

This script reuses ``anima_minimal_inference`` (single / batch / interactive
modes, latent-decode mode, prompt-line override syntax, etc.) and adds:

  * ``--vace_weights``      ControlNet-VACE weights (.safetensors)
  * ``--control_image``     Control image path (single / global)
  * ``--vace_scale``        global VACE hint multiplier (default 1.0)
  * Prompt-line overrides ``--cn <path>`` and ``--vs <float>`` (per-prompt
    control image / multiplier in batch mode)

Implementation: monkey-patches ``parse_args``, ``parse_prompt_line``,
``load_dit_model`` and ``generate_body`` of ``anima_minimal_inference`` and
then delegates to ``anima_minimal_inference.main()``. All other behavior
(VAE loading, text encoding, save logic, batch/interactive flow, latent-only
decode mode) is inherited unchanged.

The VACE wrapper stages a VAE-encoded control latent on the wrapper via
``stage_control``; the patched ``dit.forward_mini_train_dit`` consumes the
staged latent on every denoising step. CFG runs the DiT twice per step
without re-staging. ``clear_staged_control`` is called after generation.

Usage examples:

  # single prompt
  python anima_minimal_inference_control_net_vace.py \\
    --dit ... --vae ... --text_encoder ... \\
    --vace_weights out/last.safetensors --control_image canny.png \\
    --prompt "a cat" --image_size 1024 1024 --save_path out/

  # batch
  python anima_minimal_inference_control_net_vace.py \\
    --dit ... --vae ... --text_encoder ... \\
    --vace_weights out/last.safetensors --control_image default.png \\
    --from_file prompts.txt --save_path out/
  # prompts.txt line:
  #   a cat sitting on a chair --w 1024 --h 1024 --d 42 --cn images/canny_a.png --vs 0.8
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

import anima_minimal_inference as ami
from networks.control_net_vace_anima import (
    ControlNetVACEAnima,
    AnimaControlNetVACEWrapper,
    attach_vace_to_dit,
    read_vace_metadata,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_control_image(
    path: str, height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Load and normalize a control image to a (1, 3, H, W) tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):  # PIL size is (W, H)
        img = img.resize((width, height), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    return t.to(device=device, dtype=dtype)


def _vae_encode_control(vae, control_image: torch.Tensor, target_hw, dtype) -> torch.Tensor:
    """VAE-encode the control image to match the base latent layout (B, C, 1, H/8, W/8)."""
    h, w = target_hw
    # Resize the image to the *pixel* dimensions if not already
    if control_image.shape[-2:] != (h, w):
        from torchvision import transforms

        control_image = transforms.functional.resize(
            control_image, [h, w], interpolation=transforms.InterpolationMode.BICUBIC
        )
    control_image = control_image.to(device=vae.device, dtype=vae.dtype)
    with torch.no_grad():
        latent = vae.encode_pixels_to_latents(control_image)  # (1, C, H/8, W/8) or (1, C, 1, H/8, W/8)
    if latent.dim() == 4:
        latent = latent.unsqueeze(2)  # -> (1, C, 1, H/8, W/8)
    return latent.to(dtype=dtype)


# ---------------------------------------------------------------------------
# parse_args (replaces ami.parse_args)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anima ControlNet-VACE inference")

    # --- mirror anima_minimal_inference.parse_args() ---
    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
    parser.add_argument("--vae_chunk_size", type=int, default=None)
    parser.add_argument("--vae_disable_cache", action="store_true")
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3 Text Encoder path")

    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None)
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None)

    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="height width")
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--flow_shift", type=float, default=5.0)

    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8_scaled", action="store_true")
    parser.add_argument("--text_encoder_cpu", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--attn_mode", type=str, default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],
    )
    parser.add_argument(
        "--output_type", type=str, default="images",
        choices=["images", "latent", "latent_images"],
    )
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None)
    parser.add_argument(
        "--lycoris", action="store_true",
        help=f"use lycoris{'' if ami.lycoris_available else ' (not available)'}",
    )

    parser.add_argument("--from_file", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")

    # --- VACE-specific ---
    parser.add_argument(
        "--vace_weights", type=str, default=None,
        help="ControlNet-VACE weights (.safetensors). Required unless --latent_path is given.",
    )
    parser.add_argument(
        "--control_image", type=str, default=None,
        help="Path to a control image. May be overridden per-prompt with --cn in --from_file mode.",
    )
    parser.add_argument(
        "--vace_scale", type=float, default=1.0,
        help="VACE hint multiplier (default 1.0). Per-prompt override: --vs <float>.",
    )
    parser.add_argument(
        "--vace_block_every_n", type=int, default=None,
        help="override block_every_n from weights metadata",
    )
    parser.add_argument(
        "--vace_condition_strategy", type=str, default=None,
        choices=["spaced", "first_n"],
        help="override condition_strategy from weights metadata",
    )
    parser.add_argument(
        "--vace_use_after_proj", type=str, default=None, choices=["true", "false"],
        help="override use_after_proj from weights metadata (true/false)",
    )

    args = parser.parse_args()

    # validation (mirrors ami.parse_args + VACE checks)
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    latents_mode = args.latent_path is not None and len(args.latent_path) > 0
    if not latents_mode:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError("Either --prompt, --from_file or --interactive must be specified")
        if args.vace_weights is None:
            raise ValueError("--vace_weights is required for inference (unless --latent_path is given)")
        if args.control_image is None and not args.from_file and not args.interactive:
            raise ValueError(
                "--control_image is required for single-prompt inference. "
                "In --from_file mode, you may instead specify --cn per prompt."
            )

    if args.lycoris and not ami.lycoris_available:
        raise ValueError("install lycoris: https://github.com/KohakuBlueleaf/LyCORIS")

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    return args


# ---------------------------------------------------------------------------
# parse_prompt_line (extends ami.parse_prompt_line with --cn / --vs)
# ---------------------------------------------------------------------------

def parse_prompt_line(line: str) -> Dict[str, Any]:
    parts = line.split(" --")
    prompt = parts[0].strip()
    overrides: Dict[str, Any] = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        if option == "w":
            overrides["image_size_width"] = int(value)
        elif option == "h":
            overrides["image_size_height"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option in ("g", "l"):
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "n":
            overrides["negative_prompt"] = value
        elif option == "cn":
            overrides["control_image"] = value
        elif option == "vs":
            overrides["vace_scale"] = float(value)

    return overrides


# ---------------------------------------------------------------------------
# load_dit_model (replaces ami.load_dit_model — attaches VACE)
# ---------------------------------------------------------------------------

_original_load_dit_model = ami.load_dit_model


def load_dit_model(args, device, dit_weight_dtype=None):
    dit = _original_load_dit_model(args, device, dit_weight_dtype)

    meta = read_vace_metadata(args.vace_weights)

    block_every_n = (
        args.vace_block_every_n
        if args.vace_block_every_n is not None
        else int(meta.get("vace.block_every_n", "2"))
    )
    condition_strategy = (
        args.vace_condition_strategy
        if args.vace_condition_strategy is not None
        else meta.get("vace.condition_strategy", "spaced")
    )
    if args.vace_use_after_proj is not None:
        use_after_proj = args.vace_use_after_proj == "true"
    else:
        use_after_proj = meta.get("vace.use_after_proj", "true").lower() == "true"
    version = meta.get("vace.version", "?")
    n_ctrl = meta.get("vace.num_control_blocks", "?")

    logger.info(
        f"VACE config (v{version}): block_every_n={block_every_n}, "
        f"strategy={condition_strategy}, use_after_proj={use_after_proj}, "
        f"num_control_blocks={n_ctrl}, scale={args.vace_scale}"
    )

    extra_kwargs = {}
    if args.vace_block_every_n is not None:
        extra_kwargs["vace_block_every_n"] = args.vace_block_every_n
    if args.vace_condition_strategy is not None:
        extra_kwargs["condition_strategy"] = args.vace_condition_strategy
    if args.vace_use_after_proj is not None:
        extra_kwargs["use_after_proj"] = args.vace_use_after_proj == "true"

    wrapper = attach_vace_to_dit(
        dit,
        args.vace_weights,
        control_context_scale=args.vace_scale,
        strict_load=False,
        extra_vace_kwargs=extra_kwargs if extra_kwargs else None,
    )
    # Move VACE to bf16 + device; dit is already there
    wrapper.vace.to(device=device, dtype=torch.bfloat16)
    wrapper.vace.eval().requires_grad_(False)

    # Attach onto dit so generate_body can reach the wrapper
    dit.vace_wrapper = wrapper
    return dit


# ---------------------------------------------------------------------------
# generate_body (replaces ami.generate_body — stages control latent)
# ---------------------------------------------------------------------------

_original_generate_body = ami.generate_body


def generate_body(
    args,
    anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    height, width = ami.check_inputs(args)

    ci_path = args.control_image
    if ci_path is None:
        raise ValueError(
            "control_image is not set. Specify --control_image globally, "
            "or --cn per prompt in --from_file mode."
        )
    cond_image = _load_control_image(ci_path, height, width, device, torch.bfloat16)
    logger.info(f"Loaded control image: {ci_path} -> {tuple(cond_image.shape)}")

    if not hasattr(anima, "vace_wrapper"):
        raise RuntimeError("DiT has no .vace_wrapper attribute; load_dit_model patch was not applied")
    wrapper: AnimaControlNetVACEWrapper = anima.vace_wrapper

    # Lazy-load a VAE just for control encoding. The original generate_body
    # owns its own VAE for decode, but it isn't passed in here, so we load a
    # fresh one. For batched runs consider pre-encoding controls offline.
    from library import qwen_image_autoencoder_kl

    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
    )
    vae.to(torch.bfloat16).eval()
    vae.to(device)

    try:
        control_latent = _vae_encode_control(vae, cond_image, (height, width), torch.bfloat16)
        logger.info(f"Control latent shape: {tuple(control_latent.shape)}")
    finally:
        vae.to("cpu")
        del vae

    # honor per-prompt override of scale
    wrapper.control_context_scale = float(args.vace_scale)
    wrapper.stage_control(control_latent)

    try:
        return _original_generate_body(args, anima, context, context_null, device, seed)
    finally:
        wrapper.clear_staged_control()


# ---------------------------------------------------------------------------
# install patches and run ami.main
# ---------------------------------------------------------------------------

ami.parse_args = parse_args
ami.parse_prompt_line = parse_prompt_line
ami.load_dit_model = load_dit_model
ami.generate_body = generate_body


if __name__ == "__main__":
    ami.main()
