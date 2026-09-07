import os
import glob
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import CLIPTokenizer, T5TokenizerFast

from library import flux_utils, train_util
from library.cache_utils import load_npz, save_npz
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy, variant_key

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


CLIP_L_TOKENIZER_ID = "openai/clip-vit-large-patch14"
T5_XXL_TOKENIZER_ID = "google/t5-v1_1-xxl"


class FluxTokenizeStrategy(TokenizeStrategy):
    def __init__(self, t5xxl_max_length: int = 512, tokenizer_cache_dir: Optional[str] = None) -> None:
        self.t5xxl_max_length = t5xxl_max_length
        self.clip_l = self._load_tokenizer(CLIPTokenizer, CLIP_L_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.t5xxl = self._load_tokenizer(T5TokenizerFast, T5_XXL_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        l_tokens = self.clip_l(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        t5_tokens = self.t5xxl(text, max_length=self.t5xxl_max_length, padding="max_length", truncation=True, return_tensors="pt")

        t5_attn_mask = t5_tokens["attention_mask"]
        l_tokens = l_tokens["input_ids"]
        t5_tokens = t5_tokens["input_ids"]

        return [l_tokens, t5_tokens, t5_attn_mask]


class FluxTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, apply_t5_attn_mask: Optional[bool] = None) -> None:
        """
        Args:
            apply_t5_attn_mask: Default value for apply_t5_attn_mask.
        """
        self.apply_t5_attn_mask = apply_t5_attn_mask

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
        apply_t5_attn_mask: Optional[bool] = None,
    ) -> List[torch.Tensor]:
        # supports single model inference

        if apply_t5_attn_mask is None:
            apply_t5_attn_mask = self.apply_t5_attn_mask

        clip_l, t5xxl = models if len(models) == 2 else (models[0], None)
        l_tokens, t5_tokens = tokens[:2]
        t5_attn_mask = tokens[2] if len(tokens) > 2 else None

        # clip_l is None when using T5 only
        if clip_l is not None and l_tokens is not None:
            l_pooled = clip_l(l_tokens.to(clip_l.device))["pooler_output"]
        else:
            l_pooled = None

        # t5xxl is None when using CLIP only
        if t5xxl is not None and t5_tokens is not None:
            # t5_out is [b, max length, 4096]
            attention_mask = None if not apply_t5_attn_mask else t5_attn_mask.to(t5xxl.device)
            t5_out, _ = t5xxl(t5_tokens.to(t5xxl.device), attention_mask, return_dict=False, output_hidden_states=True)
            # if zero_pad_t5_output:
            #     t5_out = t5_out * t5_attn_mask.to(t5_out.device).unsqueeze(-1)
            txt_ids = torch.zeros(t5_out.shape[0], t5_out.shape[1], 3, device=t5_out.device)
        else:
            t5_out = None
            txt_ids = None
            t5_attn_mask = None  # caption may be dropped/shuffled, so t5_attn_mask should not be used to make sure the mask is same as the cached one

        return [l_pooled, t5_out, txt_ids, t5_attn_mask]  # returns t5_attn_mask for attention mask in transformer


class FluxTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    FLUX_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_flux_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        apply_t5_attn_mask: bool = False,
        cache_dtype: str = "auto",
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial, cache_dtype=cache_dtype)
        self.apply_t5_attn_mask = apply_t5_attn_mask

        self.warn_fp8_weights = False

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + FluxTextEncoderOutputsCachingStrategy.FLUX_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str, num_caption_variants: int = 0, caption_aug_hash: Optional[str] = None):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = load_npz(npz_path)
            if "l_pooled" not in npz:
                return False
            if "t5_out" not in npz:
                return False
            if "txt_ids" not in npz:
                return False
            if "t5_attn_mask" not in npz:
                return False
            if "apply_t5_attn_mask" not in npz:
                return False
            npz_apply_t5_attn_mask = npz["apply_t5_attn_mask"]
            if npz_apply_t5_attn_mask != self.apply_t5_attn_mask:
                return False
            required_keys = {"l_pooled", "t5_out", "txt_ids", "t5_attn_mask"}
            if not self._check_cached_variant_keys(npz, required_keys, num_caption_variants, caption_aug_hash):
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str, variant: int = 0) -> List[np.ndarray]:
        data = load_npz(npz_path)
        l_pooled = self._npz_get(data, "l_pooled", variant)
        t5_out = self._npz_get(data, "t5_out", variant)
        txt_ids = self._npz_get(data, "txt_ids", variant)
        t5_attn_mask = self._npz_get(data, "t5_attn_mask", variant)
        # apply_t5_attn_mask should be same as self.apply_t5_attn_mask
        return [l_pooled, t5_out, txt_ids, t5_attn_mask]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        if not self.warn_fp8_weights:
            if flux_utils.get_t5xxl_actual_dtype(models[1]) == torch.float8_e4m3fn:
                logger.warning(
                    "T5 model is using fp8 weights for caching. This may affect the quality of the cached outputs."
                    " / T5モデルはfp8の重みを使用しています。これはキャッシュの品質に影響を与える可能性があります。"
                )
            self.warn_fp8_weights = True

        flux_text_encoding_strategy: FluxTextEncodingStrategy = text_encoding_strategy

        canonical_captions, variant_groups = self._get_variant_caption_groups(infos)

        def encode_captions(captions: List[str]) -> dict:
            tokens_and_masks = tokenize_strategy.tokenize(captions)
            with torch.no_grad():
                # attn_mask is applied in text_encoding_strategy.encode_tokens if apply_t5_attn_mask is True
                l_pooled, t5_out, txt_ids, _ = flux_text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens_and_masks)

            if l_pooled.dtype == torch.bfloat16:
                l_pooled = l_pooled.float()
            if t5_out.dtype == torch.bfloat16:
                t5_out = t5_out.float()
            if txt_ids.dtype == torch.bfloat16:
                txt_ids = txt_ids.float()

            return {
                "l_pooled": l_pooled.cpu().numpy(),
                "t5_out": t5_out.cpu().numpy(),
                "txt_ids": txt_ids.cpu().numpy(),
                "t5_attn_mask": tokens_and_masks[2].cpu().numpy(),
            }

        batched = encode_captions(canonical_captions)
        variant_batched = [(group, encode_captions([c for _, c in group])) for group in variant_groups]

        base_keys = ["l_pooled", "t5_out", "txt_ids", "t5_attn_mask"]

        for i, info in enumerate(infos):
            n_variants = len(info.caption_variants) if getattr(info, "caption_variants", None) else 0

            if self.cache_to_disk:
                save_kwargs = {key: batched[key][i] for key in base_keys}
                save_kwargs["apply_t5_attn_mask"] = self.apply_t5_attn_mask
                for k, (group, vb) in enumerate(variant_batched, start=1):
                    idxs = [idx for idx, _ in group]
                    if i not in idxs:
                        continue
                    j = idxs.index(i)
                    for key in base_keys:
                        save_kwargs[variant_key(key, k)] = vb[key][j]
                if n_variants > 1:
                    save_kwargs["caption_variants"] = np.array(n_variants)
                    save_kwargs["caption_aug_hash"] = np.array(getattr(info, "caption_aug_hash", None) or "")
                save_npz(info.text_encoder_outputs_npz, save_kwargs, cache_dtype=self.cache_dtype)
            else:
                # it's fine that attn mask is not None. it's overwritten before calling the model if necessary
                info.text_encoder_outputs = tuple(batched[key][i] for key in base_keys)
                if n_variants > 1:
                    variants_outputs = []
                    for k, (group, vb) in enumerate(variant_batched, start=1):
                        idxs = [idx for idx, _ in group]
                        if i not in idxs:
                            continue
                        j = idxs.index(i)
                        variants_outputs.append(tuple(vb[key][j] for key in base_keys))
                    info.text_encoder_outputs_variants = variants_outputs


class FluxLatentsCachingStrategy(LatentsCachingStrategy):
    FLUX_LATENTS_NPZ_SUFFIX = "_flux.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, cache_dtype: str = "auto") -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, cache_dtype=cache_dtype)

    @property
    def cache_suffix(self) -> str:
        return FluxLatentsCachingStrategy.FLUX_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + FluxLatentsCachingStrategy.FLUX_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        alpha_mask: bool,
        num_aug_variants: int = 0,
        aug_config_hash: Optional[str] = None,
    ):
        return self._default_is_disk_cached_latents_expected(
            8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True,
            num_aug_variants=num_aug_variants, aug_config_hash=aug_config_hash,
        )

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int], variant: int = 0
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray], Optional[bool]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso, variant=variant)  # support multi-resolution

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(
        self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool, random_crop_padding_percent: float = 0.05,
        num_aug_variants: int = 0, augmentor: Optional[Callable] = None, aug_config_hash: Optional[str] = None,
    ):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True,
            random_crop_padding_percent=random_crop_padding_percent,
            num_aug_variants=num_aug_variants, augmentor=augmentor, aug_config_hash=aug_config_hash,
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)


if __name__ == "__main__":
    # test code for FluxTokenizeStrategy
    # tokenizer = sd3_models.SD3Tokenizer()
    strategy = FluxTokenizeStrategy(256)
    text = "hello world"

    l_tokens, g_tokens, t5_tokens = strategy.tokenize(text)
    # print(l_tokens.shape)
    print(l_tokens)
    print(g_tokens)
    print(t5_tokens)

    texts = ["hello world", "the quick brown fox jumps over the lazy dog"]
    l_tokens_2 = strategy.clip_l(texts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
    g_tokens_2 = strategy.clip_g(texts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
    t5_tokens_2 = strategy.t5xxl(
        texts, max_length=strategy.t5xxl_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    print(l_tokens_2)
    print(g_tokens_2)
    print(t5_tokens_2)

    # compare
    print(torch.allclose(l_tokens, l_tokens_2["input_ids"][0]))
    print(torch.allclose(g_tokens, g_tokens_2["input_ids"][0]))
    print(torch.allclose(t5_tokens, t5_tokens_2["input_ids"][0]))

    text = ",".join(["hello world! this is long text"] * 50)
    l_tokens, g_tokens, t5_tokens = strategy.tokenize(text)
    print(l_tokens)
    print(g_tokens)
    print(t5_tokens)

    print(f"model max length l: {strategy.clip_l.model_max_length}")
    print(f"model max length g: {strategy.clip_g.model_max_length}")
    print(f"model max length t5: {strategy.t5xxl.model_max_length}")
