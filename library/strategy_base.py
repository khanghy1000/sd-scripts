# base class for platform strategies. this file defines the interface for strategies

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from library.offline_utils import safe_from_pretrained
from library.cache_utils import load_npz, normalize_cache_dtype, save_npz


# TODO remove circular import by moving ImageInfo to a separate file
# from library.train_util import ImageInfo

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Schema version for cached augmentation variants. Bump when the npz key layout changes.
AUG_VARIANT_SCHEMA_VERSION = 1


def variant_key(base: str, variant: int) -> str:
    """Return the npz key for a cached augmentation variant.

    Variant 0 is the canonical (un-augmented) sample and uses the legacy un-suffixed
    key, so existing caches remain valid and older code can read variant-0 data.
    Variants >= 1 use a ``_v{k}`` suffix, composed before any resolution suffix,
    e.g. ``latents_v3_32x64``.
    """
    if variant <= 0:
        return base
    return f"{base}_v{variant}"


def compute_aug_config_hash(config: Dict[str, Any]) -> str:
    """Compute a stable hash over an augmentation config dict.

    The variant counts (K) themselves must NOT be included in ``config``: validity
    compares the stored variant count separately (a superset cache stays valid when
    the requested K is lowered). Changing any augmentation parameter changes the
    hash and therefore invalidates previously written caches.
    """
    payload = {"schema": AUG_VARIANT_SCHEMA_VERSION, "config": config}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class TokenizeStrategy:
    _strategy = None  # strategy instance: actual strategy class

    _re_attention = re.compile(
        r"""\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
        re.X,
    )

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TokenizeStrategy"]:
        return cls._strategy

    def _load_tokenizer(
        self, model_class: Any, model_id: str, subfolder: Optional[str] = None, tokenizer_cache_dir: Optional[str] = None
    ) -> Any:
        tokenizer = None
        if tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(tokenizer_cache_dir, model_id.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = model_class.from_pretrained(local_tokenizer_path)  # same for v1 and v2

        if tokenizer is None:
            tokenizer = safe_from_pretrained(model_class, model_id, subfolder=subfolder)

        if tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        return tokenizer

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        raise NotImplementedError

    def tokenize_with_weights(self, text: Union[str, List[str]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        returns: [tokens1, tokens2, ...], [weights1, weights2, ...]
        """
        raise NotImplementedError

    def _get_weighted_input_ids(
        self, tokenizer: CLIPTokenizer, text: str, max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        max_length includes starting and ending tokens.
        """

        def parse_prompt_attention(text):
            """
            Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
            Accepted tokens are:
            (abc) - increases attention to abc by a multiplier of 1.1
            (abc:3.12) - increases attention to abc by a multiplier of 3.12
            [abc] - decreases attention to abc by a multiplier of 1.1
            \( - literal character '('
            \[ - literal character '['
            \) - literal character ')'
            \] - literal character ']'
            \\ - literal character '\'
            anything else - just text
            >>> parse_prompt_attention('normal text')
            [['normal text', 1.0]]
            >>> parse_prompt_attention('an (important) word')
            [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
            >>> parse_prompt_attention('(unbalanced')
            [['unbalanced', 1.1]]
            >>> parse_prompt_attention('\(literal\]')
            [['(literal]', 1.0]]
            >>> parse_prompt_attention('(unnecessary)(parens)')
            [['unnecessaryparens', 1.1]]
            >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
            [['a ', 1.0],
            ['house', 1.5730000000000004],
            [' ', 1.1],
            ['on', 1.0],
            [' a ', 1.1],
            ['hill', 0.55],
            [', sun, ', 1.1],
            ['sky', 1.4641000000000006],
            ['.', 1.1]]
            """

            res = []
            round_brackets = []
            square_brackets = []

            round_bracket_multiplier = 1.1
            square_bracket_multiplier = 1 / 1.1

            def multiply_range(start_position, multiplier):
                for p in range(start_position, len(res)):
                    res[p][1] *= multiplier

            for m in TokenizeStrategy._re_attention.finditer(text):
                text = m.group(0)
                weight = m.group(1)

                if text.startswith("\\"):
                    res.append([text[1:], 1.0])
                elif text == "(":
                    round_brackets.append(len(res))
                elif text == "[":
                    square_brackets.append(len(res))
                elif weight is not None and len(round_brackets) > 0:
                    multiply_range(round_brackets.pop(), float(weight))
                elif text == ")" and len(round_brackets) > 0:
                    multiply_range(round_brackets.pop(), round_bracket_multiplier)
                elif text == "]" and len(square_brackets) > 0:
                    multiply_range(square_brackets.pop(), square_bracket_multiplier)
                else:
                    res.append([text, 1.0])

            for pos in round_brackets:
                multiply_range(pos, round_bracket_multiplier)

            for pos in square_brackets:
                multiply_range(pos, square_bracket_multiplier)

            if len(res) == 0:
                res = [["", 1.0]]

            # merge runs of identical weights
            i = 0
            while i + 1 < len(res):
                if res[i][1] == res[i + 1][1]:
                    res[i][0] += res[i + 1][0]
                    res.pop(i + 1)
                else:
                    i += 1

            return res

        def get_prompts_with_weights(text: str, max_length: int):
            r"""
            Tokenize a list of prompts and return its tokens with weights of each token. max_length does not include starting and ending token.

            No padding, starting or ending token is included.
            """
            truncated = False

            texts_and_weights = parse_prompt_attention(text)
            tokens = []
            weights = []
            for word, weight in texts_and_weights:
                # tokenize and discard the starting and the ending token
                token = tokenizer(word).input_ids[1:-1]
                tokens += token
                # copy the weight by length of token
                weights += [weight] * len(token)
                # stop if the text is too long (longer than truncation limit)
                if len(tokens) > max_length:
                    truncated = True
                    break
            # truncate
            if len(tokens) > max_length:
                truncated = True
                tokens = tokens[:max_length]
                weights = weights[:max_length]
            if truncated:
                logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
            return tokens, weights

        def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad):
            r"""
            Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
            """
            tokens = [bos] + tokens + [eos] + [pad] * (max_length - 2 - len(tokens))
            weights = [1.0] + weights + [1.0] * (max_length - 1 - len(weights))
            return tokens, weights

        if max_length is None:
            max_length = tokenizer.model_max_length

        tokens, weights = get_prompts_with_weights(text, max_length - 2)
        tokens, weights = pad_tokens_and_weights(
            tokens, weights, max_length, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        )
        return torch.tensor(tokens).unsqueeze(0), torch.tensor(weights).unsqueeze(0)

    def _tokenize_tags(self, tokenizer, text_prompts: list, num_chunks: int = 3) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Advanced tokenizer with granular padding masks, specifically designed to output
        the shapes required by Kohya's sd-scripts data loading pipeline.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - final_input_ids: Shape [B * N, S] for direct use with the encoder.
                - final_cross_attention_mask: Shape [B, N, S] for later reshaping for the U-Net.
        """
        B = len(text_prompts)

        text_input = tokenizer(
            text=text_prompts,
            padding="max_length",
            max_length=77 * num_chunks,
            truncation=True,
            return_tensors="pt"
        )

        device = text_input['input_ids'].device
        bos_token_id = 49406
        eos_token_id = 49407
        comma_token_ids = torch.tensor([267, 2361], device=device)

        token_chunks = torch.zeros(B, num_chunks, 77, dtype=torch.long, device=device)
        token_cross_attention_mask = torch.zeros(B, num_chunks, 77, dtype=torch.bool, device=device)

        # --- Create the SAFE unconditional chunk and masks once to reuse ---
        uncond_chunk = torch.full((77,), eos_token_id, dtype=torch.long, device=device)
        uncond_chunk[0] = bos_token_id
        ones_mask_chunk = torch.ones(77, dtype=torch.bool, device=device)  # A mask of all True

        for j in range(B):
            prompt_tokens: torch.Tensor = text_input['input_ids'][j]
            start_index = 0
            is_comma = (prompt_tokens[:, None] == comma_token_ids).any(dim=1)
            comma_indices = torch.where(is_comma)[0]

            # --- Handle Caption Dropout ---
            is_dropout = prompt_tokens[1] == eos_token_id
            if is_dropout:
                # The first chunk is the "meaningful" unconditional concept.
                # It gets the standard empty prompt tokens and a MASK OF ALL ONES.
                token_chunks[j, 0] = uncond_chunk
                token_cross_attention_mask[j, 0] = ones_mask_chunk

                # Subsequent chunks are ignored padding.
                # They get the same tokens, but their mask remains all zeros.
                for i in range(1, num_chunks):
                    token_chunks[j, i] = uncond_chunk

                # Skip the rest of the loop for this prompt.
                continue
            # --- End of Dropout Logic ---

            for i in range(num_chunks):
                valid_comma_indices = comma_indices[comma_indices < start_index + 75]
                if len(valid_comma_indices) == 0:
                    eos_indices = torch.where(prompt_tokens == eos_token_id)[0]
                    split_point = eos_indices[0] if len(eos_indices) > 0 else (start_index + 75)
                else:
                    split_point = valid_comma_indices[-1]

                # If a split results in an empty chunk, it's an ignored padding chunk.
                if split_point <= start_index:
                    token_chunks[j, i] = uncond_chunk
                    continue  # Mask remains all zeros

                chunk = prompt_tokens[start_index + 1: split_point + 1]
                if len(chunk) == 1 and chunk[0] in comma_token_ids:
                    token_chunks[j, i] = uncond_chunk
                    continue  # Mask remains all zeros

                bos_tensor = torch.tensor([bos_token_id], device=device)
                chunk_with_bos = torch.cat([bos_tensor, chunk])

                padded_chunk = torch.full((77,), eos_token_id, dtype=torch.long, device=device)
                actual_len = min(len(chunk_with_bos), 77)
                padded_chunk[:actual_len] = chunk_with_bos[:actual_len]

                mask_chunk = torch.zeros(77, dtype=torch.bool, device=device)
                mask_chunk[:actual_len] = True

                token_chunks[j, i] = padded_chunk
                token_cross_attention_mask[j, i] = mask_chunk
                start_index = split_point

        final_input_ids = token_chunks.view(B * num_chunks, 77)
        final_cross_attention_mask = token_cross_attention_mask

        return final_input_ids, final_cross_attention_mask

    def _get_input_ids(
        self, tokenizer: CLIPTokenizer, text: str, max_length: Optional[int] = None, weighted: bool = False
    ) -> torch.Tensor:
        """
        for SD1.5/2.0/SDXL
        TODO support batch input
        """
        if max_length is None:
            max_length = tokenizer.model_max_length - 2

        if weighted:
            input_ids, weights = self._get_weighted_input_ids(tokenizer, text, max_length)
        else:
            input_ids = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids

        if max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
                # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
                for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
                for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                    # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77

            if weighted:
                weights = weights.squeeze(0)
                new_weights = torch.ones(input_ids.shape)
                for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    b = i // (tokenizer.model_max_length - 2)
                    new_weights[b, 1 : 1 + tokenizer.model_max_length - 2] = weights[i : i + tokenizer.model_max_length - 2]
                weights = new_weights

        if weighted:
            return input_ids, weights
        return input_ids


class TextEncodingStrategy:
    _strategy = None  # strategy instance: actual strategy class

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TextEncodingStrategy"]:
        return cls._strategy

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: list[torch.Tensor],
        attn_masks: Optional[list[Optional[torch.Tensor]]] = None,
    ) -> tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """
        Encode tokens into embeddings and outputs.
        :param tokens: list of token tensors for each TextModel
        :return: list of output embeddings for each architecture
        """
        raise NotImplementedError

    def encode_tokens_with_weights(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor], weights: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Encode tokens into embeddings and outputs.
        :param tokens: list of token tensors for each TextModel
        :param weights: list of weight tensors for each TextModel
        :return: list of output embeddings for each architecture
        """
        raise NotImplementedError


class TextEncoderOutputsCachingStrategy:
    _strategy = None  # strategy instance: actual strategy class

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: Optional[int],
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        is_weighted: bool = False,
        cache_dtype: str = "auto",
    ) -> None:
        self._cache_to_disk = cache_to_disk
        self._batch_size = batch_size
        self.skip_disk_cache_validity_check = skip_disk_cache_validity_check
        self._is_partial = is_partial
        self._is_weighted = is_weighted
        self.cache_dtype = normalize_cache_dtype(cache_dtype)

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TextEncoderOutputsCachingStrategy"]:
        return cls._strategy

    @property
    def cache_to_disk(self):
        return self._cache_to_disk

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def is_partial(self):
        return self._is_partial

    @property
    def is_weighted(self):
        return self._is_weighted

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        raise NotImplementedError

    def load_outputs_npz(self, npz_path: str, variant: int = 0) -> List[np.ndarray]:
        raise NotImplementedError

    def is_disk_cached_outputs_expected(self, npz_path: str, num_caption_variants: int = 0, caption_aug_hash: Optional[str] = None) -> bool:
        raise NotImplementedError

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, batch: List
    ):
        raise NotImplementedError

    def _get_variant_caption_groups(self, infos: List) -> Tuple[List[str], List[List[Tuple[int, str]]]]:
        """Split infos' captions into canonical captions and per-variant (info_index, caption) groups.

        Requires infos to optionally have ``caption_variants`` (index 0 = canonical). Infos
        without variants only appear in the canonical captions.

        Returns:
            Tuple[List[str], List[List[Tuple[int, str]]]]: canonical captions (one per info),
                and for each variant index k >= 1 a list of (info_index, caption).
        """
        canonical_captions = [(info.caption_variants[0] if getattr(info, "caption_variants", None) else info.caption) for info in infos]
        max_variant = 0
        for info in infos:
            if getattr(info, "caption_variants", None):
                max_variant = max(max_variant, len(info.caption_variants) - 1)
        variant_groups = []
        for k in range(1, max_variant + 1):
            group = [
                (i, info.caption_variants[k])
                for i, info in enumerate(infos)
                if getattr(info, "caption_variants", None) and len(info.caption_variants) > k
            ]
            variant_groups.append(group)
        return canonical_captions, variant_groups

    def _check_cached_variant_keys(self, npz, required_keys, num_caption_variants: int, caption_aug_hash: Optional[str]) -> bool:
        """Shared validity check for cached caption variants in *_te.npz files."""
        if num_caption_variants is None or num_caption_variants <= 1:
            return True
        if "caption_variants" not in npz or "caption_aug_hash" not in npz:
            return False
        if int(npz["caption_variants"]) < num_caption_variants:
            return False
        if caption_aug_hash is not None and str(npz["caption_aug_hash"].tolist()) != caption_aug_hash:
            return False
        for k in range(1, num_caption_variants):
            for key in required_keys:
                if variant_key(key, k) not in npz:
                    return False
        return True

    @staticmethod
    def _npz_get(npz, key: str, variant: int = 0):
        """Get a (possibly variant-suffixed) npz entry, falling back to the legacy key."""
        if variant > 0:
            vkey = variant_key(key, variant)
            if vkey in npz:
                return npz[vkey]
        return npz[key]


class LatentsCachingStrategy:
    # TODO commonize utillity functions to this class, such as npz handling etc.

    _strategy = None  # strategy instance: actual strategy class

    _warned_fallback_to_old_npz = False  # to avoid spamming logs about fallback

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        cache_dtype: str = "auto",
    ) -> None:
        self._cache_to_disk = cache_to_disk
        self._batch_size = batch_size
        self.skip_disk_cache_validity_check = skip_disk_cache_validity_check
        self.cache_dtype = normalize_cache_dtype(cache_dtype)

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["LatentsCachingStrategy"]:
        return cls._strategy

    @property
    def cache_to_disk(self):
        return self._cache_to_disk

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def cache_suffix(self):
        raise NotImplementedError

    def get_image_size_from_disk_cache_path(self, absolute_path: str, npz_path: str) -> Tuple[Optional[int], Optional[int]]:
        w, h = os.path.splitext(npz_path)[0].split("_")[-2].split("x")
        return int(w), int(h)

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        raise NotImplementedError

    def is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        alpha_mask: bool,
        num_aug_variants: int = 0,
        aug_config_hash: Optional[str] = None,
    ) -> bool:
        raise NotImplementedError

    def cache_batch_latents(
        self,
        model: Any,
        batch: List,
        flip_aug: bool,
        alpha_mask: bool,
        random_crop: bool,
        random_crop_padding_percent: float = 0.05,
        num_aug_variants: int = 0,
        augmentor: Optional[Callable] = None,
        aug_config_hash: Optional[str] = None,
    ):
        raise NotImplementedError

    def _default_is_disk_cached_latents_expected(
        self,
        latents_stride: int,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        apply_alpha_mask: bool,
        multi_resolution: bool = False,
        num_aug_variants: int = 0,
        aug_config_hash: Optional[str] = None,
    ) -> bool:
        """
        Args:
            latents_stride: stride of latents
            bucket_reso: resolution of the bucket
            npz_path: path to the npz file
            flip_aug: whether to flip images
            apply_alpha_mask: whether to apply alpha mask
            multi_resolution: whether to use multi-resolution latents
            num_aug_variants: number of augmentation variants expected (0 = legacy cache)
            aug_config_hash: expected augmentation config hash (None = do not check)

        Returns:
            bool
        """
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        expected_latents_size = (bucket_reso[1] // latents_stride, bucket_reso[0] // latents_stride)  # bucket_reso is (W, H)

        # e.g. "_32x64", HxW
        key_reso_suffix = f"_{expected_latents_size[0]}x{expected_latents_size[1]}" if multi_resolution else ""

        try:
            npz = load_npz(npz_path)

            # In old SD/SDXL npz files, if the actual latents shape does not match the expected shape, it doesn't raise an error as long as "latents" key exists (backward compatibility)
            # In non-SD/SDXL npz files (multi-resolution support), the latents key always has the resolution suffix, and no latents key without suffix exists, so it raises an error if the expected resolution suffix key is not found (this doesn't change the behavior for non-SD/SDXL npz files).
            if "latents" + key_reso_suffix not in npz and "latents" not in npz:
                return False
            if flip_aug and num_aug_variants <= 0 and ("latents_flipped" + key_reso_suffix not in npz and "latents_flipped" not in npz):
                return False
            if apply_alpha_mask and ("alpha_mask" + key_reso_suffix not in npz and "alpha_mask" not in npz):
                return False

            if num_aug_variants > 0:
                # variant caches must carry metadata, enough variants, a matching
                # augmentation config hash, and all per-variant keys
                if "aug_variants" not in npz or "aug_config_hash" not in npz:
                    return False
                if int(npz["aug_variants"]) < num_aug_variants:
                    return False
                if aug_config_hash is not None and str(npz["aug_config_hash"].tolist()) != aug_config_hash:
                    return False
                for k in range(1, num_aug_variants):
                    if variant_key("latents", k) + key_reso_suffix not in npz:
                        return False
                    if variant_key("crop_ltrb", k) + key_reso_suffix not in npz:
                        return False
                    if apply_alpha_mask and variant_key("alpha_mask", k) + key_reso_suffix not in npz:
                        return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    # TODO remove circular dependency for ImageInfo
    def _default_cache_batch_latents(
        self,
        encode_by_vae: Callable,
        vae_device: torch.device,
        vae_dtype: torch.dtype,
        image_infos: List,
        flip_aug: bool,
        apply_alpha_mask: bool,
        random_crop: bool,
        multi_resolution: bool = False,
        random_crop_padding_percent: float = 0.05,
        num_aug_variants: int = 0,
        augmentor: Optional[Callable] = None,
        aug_config_hash: Optional[str] = None,
    ):
        """
        Default implementation for cache_batch_latents. Image loading, VAE, flipping, alpha mask handling are common.

        Args:
            encode_by_vae: function to encode images by VAE
            vae_device: device to use for VAE
            vae_dtype: dtype to use for VAE
            image_infos: list of ImageInfo
            flip_aug: whether to flip images
            apply_alpha_mask: whether to apply alpha mask
            random_crop: whether to random crop images
            multi_resolution: whether to use multi-resolution latents
            num_aug_variants: number of augmentation variants to cache (>1 enables variant caching)
            augmentor: color/gamma augmentor (from AugHelper.get_augmentor), applied per variant in pixel space
            aug_config_hash: augmentation config hash stored in the npz for cache invalidation

        Returns:
            None
        """
        from library import train_util  # import here to avoid circular import

        if num_aug_variants is not None and num_aug_variants > 1:
            self._default_cache_batch_latents_with_variants(
                encode_by_vae,
                vae_device,
                vae_dtype,
                image_infos,
                flip_aug,
                apply_alpha_mask,
                random_crop,
                multi_resolution,
                random_crop_padding_percent,
                num_aug_variants,
                augmentor,
                aug_config_hash,
            )
            return

        img_tensor, alpha_masks, original_sizes, crop_ltrbs = train_util.load_images_and_masks_for_caching(
            image_infos, apply_alpha_mask, random_crop, random_crop_padding_percent=random_crop_padding_percent
        )
        img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            latents_tensors = encode_by_vae(img_tensor).to("cpu")
        if flip_aug:
            img_tensor = torch.flip(img_tensor, dims=[3])
            with torch.no_grad():
                flipped_latents = encode_by_vae(img_tensor).to("cpu")
        else:
            flipped_latents = [None] * len(latents_tensors)

        # for info, latents, flipped_latent, alpha_mask in zip(image_infos, latents_tensors, flipped_latents, alpha_masks):
        for i in range(len(image_infos)):
            info = image_infos[i]
            latents = latents_tensors[i]
            flipped_latent = flipped_latents[i]
            alpha_mask = alpha_masks[i]
            original_size = original_sizes[i]
            crop_ltrb = crop_ltrbs[i]

            latents_size = latents.shape[-2:]  # H, W (supports both 4D and 5D latents)
            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}" if multi_resolution else ""  # e.g. "_32x64", HxW

            if self.cache_to_disk:
                self.save_latents_to_disk(
                    info.latents_npz, latents, original_size, crop_ltrb, flipped_latent, alpha_mask, key_reso_suffix
                )
            else:
                info.latents_original_size = original_size
                info.latents_crop_ltrb = crop_ltrb
                info.latents = latents
                if flip_aug:
                    info.latents_flipped = flipped_latent
                info.alpha_mask = alpha_mask

    def _default_cache_batch_latents_with_variants(
        self,
        encode_by_vae: Callable,
        vae_device: torch.device,
        vae_dtype: torch.dtype,
        image_infos: List,
        flip_aug: bool,
        apply_alpha_mask: bool,
        random_crop: bool,
        multi_resolution: bool,
        random_crop_padding_percent: float,
        num_aug_variants: int,
        augmentor: Optional[Callable],
        aug_config_hash: Optional[str],
    ):
        """
        Cache K augmentation variants per image. All augmentations (random crop, flip,
        color/gamma) are applied in pixel space BEFORE VAE encoding, so every cached
        latent is exact (crop-then-encode, flip-then-encode). Variant 0 is canonical
        (center crop, unflipped, no color/gamma aug) and is stored under the legacy keys.

        Args:
            encode_by_vae: function to encode images by VAE
            vae_device: device to use for VAE
            vae_dtype: dtype to use for VAE
            image_infos: list of ImageInfo
            flip_aug: whether variants may be flipped (baked into variant pixels)
            apply_alpha_mask: whether to apply alpha mask
            random_crop: whether variants use random crop offsets
            multi_resolution: whether to use multi-resolution latents
            random_crop_padding_percent: padding percent for random crop resize
            num_aug_variants: total number of variants K (including canonical variant 0)
            augmentor: color/gamma augmentor, applied per variant in pixel space
            aug_config_hash: augmentation config hash stored in the npz for cache invalidation

        Returns:
            None
        """
        from library import train_util  # import here to avoid circular import

        images_per_variant, alpha_masks_per_variant, original_sizes, crop_ltrbs_per_variant, flippeds_per_variant = (
            train_util.load_image_variants_for_caching(
                image_infos, num_aug_variants, apply_alpha_mask, flip_aug, augmentor, random_crop, random_crop_padding_percent
            )
        )

        latents_per_variant: List[torch.Tensor] = []
        for k in range(num_aug_variants):
            img_tensor = images_per_variant[k].to(device=vae_device, dtype=vae_dtype)
            with torch.no_grad():
                latents_k = encode_by_vae(img_tensor).to("cpu")
            latents_per_variant.append(latents_k)
            del img_tensor

        for i in range(len(image_infos)):
            info = image_infos[i]
            latents = latents_per_variant[0][i]
            original_size = original_sizes[i]

            latents_size = latents.shape[-2:]  # H, W (supports both 4D and 5D latents)
            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}" if multi_resolution else ""  # e.g. "_32x64", HxW

            aug_variants = [
                {
                    "latents": latents_per_variant[k][i],
                    "crop_ltrb": crop_ltrbs_per_variant[k][i],
                    "flipped": flippeds_per_variant[k][i],
                    "alpha_mask": alpha_masks_per_variant[k][i],
                }
                for k in range(1, num_aug_variants)
            ]

            if self.cache_to_disk:
                self.save_latents_to_disk(
                    info.latents_npz,
                    latents,
                    original_size,
                    crop_ltrbs_per_variant[0][i],
                    None,  # no separate flipped latents: flip is baked into variants
                    alpha_masks_per_variant[0][i],
                    key_reso_suffix,
                    aug_variants=aug_variants,
                    aug_config_hash=aug_config_hash,
                )
            else:
                info.latents_original_size = original_size
                info.latents_crop_ltrb = crop_ltrbs_per_variant[0][i]
                info.latents = latents
                info.latents_flipped = None  # flip is baked into variants, not cached separately
                info.alpha_mask = alpha_masks_per_variant[0][i]
                info.latents_aug_variants = aug_variants

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int], variant: int = 0
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray], Optional[bool]]:
        """
        For single resolution architectures (currently no architecture is single resolution specific). Kept for reference.

        Args:
            npz_path (str): Path to the npz file.
            bucket_reso (Tuple[int, int]): The resolution of the bucket.
            variant (int): Augmentation variant index (0 = canonical/legacy).

        Returns:
            Tuple[
                Optional[np.ndarray],
                Optional[List[int]],
                Optional[List[int]],
                Optional[np.ndarray],
                Optional[np.ndarray],
                Optional[bool]
            ]: Latent np tensors, original size, crop (left top, right bottom), flipped latents, alpha mask,
                variant flipped flag (None for variant 0 / legacy caches)
        """
        return self._default_load_latents_from_disk(None, npz_path, bucket_reso, variant=variant)

    def _default_load_latents_from_disk(
        self, latents_stride: Optional[int], npz_path: str, bucket_reso: Tuple[int, int], variant: int = 0
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray], Optional[bool]]:
        """
        Args:
            latents_stride (Optional[int]): Stride for latents. If None, load all latents.
            npz_path (str): Path to the npz file.
            bucket_reso (Tuple[int, int]): The resolution of the bucket.
            variant (int): Augmentation variant index (0 = canonical/legacy).

        Returns:
            Tuple[
                Optional[np.ndarray],
                Optional[List[int]],
                Optional[List[int]],
                Optional[np.ndarray],
                Optional[np.ndarray],
                Optional[bool]
            ]: Latent np tensors, original size, crop (left top, right bottom), flipped latents, alpha mask,
                variant flipped flag (None for variant 0 / legacy caches, bool for variant >= 1)
        """
        if latents_stride is None:
            key_reso_suffix = ""
        else:
            expected_latents_size = (bucket_reso[1] // latents_stride, bucket_reso[0] // latents_stride)  # bucket_reso is (W, H)
            key_reso_suffix = f"_{expected_latents_size[0]}x{expected_latents_size[1]}"  # e.g. "_32x64", HxW

        npz = load_npz(npz_path)
        latents_key = variant_key("latents", variant) + key_reso_suffix
        if latents_key not in npz:
            if variant > 0:
                raise ValueError(f"{latents_key} not found in {npz_path} (augmentation variant {variant} is not cached)")
            # raise ValueError(f"latents{key_reso_suffix} not found in {npz_path}")
            # Fallback to old npz without resolution suffix
            if "latents" not in npz:
                raise ValueError(f"latents not found in {npz_path} (either with or without resolution suffix: {key_reso_suffix})")
            if not self._warned_fallback_to_old_npz:
                logger.warning(
                    f"latents{key_reso_suffix} not found in {npz_path}. Falling back to latents without resolution suffix (old npz). This warning will only be shown once. To avoid this warning, please re-cache the latents with the latest version."
                )
                self._warned_fallback_to_old_npz = True
            key_reso_suffix = ""
            latents_key = "latents"

        latents = npz[latents_key]
        original_size = npz[variant_key("original_size", variant) + key_reso_suffix].tolist()
        crop_ltrb = npz[variant_key("crop_ltrb", variant) + key_reso_suffix].tolist()

        variant_flipped: Optional[bool] = None
        if variant > 0:
            # flip is baked into variant latents (pixels are flipped before VAE encoding);
            # only the flag is needed for size conditioning
            flipped_key = variant_key("flipped", variant) + key_reso_suffix
            variant_flipped = bool(npz[flipped_key].tolist()) if flipped_key in npz else False
            flipped_latents = None
        else:
            if "aug_variants" in npz:
                # variant cache: flip is baked into the variants; the canonical variant is
                # unflipped and no separate flipped latents are stored for it
                variant_flipped = False
            flipped_latents = npz["latents_flipped" + key_reso_suffix] if "latents_flipped" + key_reso_suffix in npz else None

        alpha_mask_key = variant_key("alpha_mask", variant) + key_reso_suffix
        alpha_mask = npz[alpha_mask_key] if alpha_mask_key in npz else None
        return latents, original_size, crop_ltrb, flipped_latents, alpha_mask, variant_flipped

    def save_latents_to_disk(
        self,
        npz_path,
        latents_tensor,
        original_size,
        crop_ltrb,
        flipped_latents_tensor=None,
        alpha_mask=None,
        key_reso_suffix="",
        aug_variants: Optional[List[Dict[str, Any]]] = None,
        aug_config_hash: Optional[str] = None,
    ):
        """
        Args:
            npz_path (str): Path to the npz file.
            latents_tensor (torch.Tensor): Latent tensor (canonical / variant 0)
            original_size (List[int]): Original size of the image
            crop_ltrb (List[int]): Crop left top right bottom (canonical)
            flipped_latents_tensor (Optional[torch.Tensor]): Flipped latent tensor (legacy flip cache, variant 0 only)
            alpha_mask (Optional[torch.Tensor]): Alpha mask (canonical)
            key_reso_suffix (str): Key resolution suffix
            aug_variants (Optional[List[Dict[str, Any]]]): augmentation variants 1..K-1, each a dict with
                keys ``latents`` (tensor), ``crop_ltrb``, ``flipped`` (bool) and optional ``alpha_mask``.
                Flip is baked into the variant latents (pixels are flipped before encoding).
            aug_config_hash (Optional[str]): augmentation config hash for cache invalidation

        Returns:
            None
        """
        kwargs = {}

        if os.path.exists(npz_path):
            # load existing npz and update it
            kwargs.update(load_npz(npz_path))

        # TODO float() is needed if vae is in bfloat16. Remove it if vae is float16.
        kwargs["latents" + key_reso_suffix] = latents_tensor.float().cpu().numpy()
        kwargs["original_size" + key_reso_suffix] = np.array(original_size)
        kwargs["crop_ltrb" + key_reso_suffix] = np.array(crop_ltrb)
        if flipped_latents_tensor is not None:
            kwargs["latents_flipped" + key_reso_suffix] = flipped_latents_tensor.float().cpu().numpy()
        if alpha_mask is not None:
            kwargs["alpha_mask" + key_reso_suffix] = alpha_mask.float().cpu().numpy()

        if aug_variants:
            for k, var in enumerate(aug_variants, start=1):
                kwargs[variant_key("latents", k) + key_reso_suffix] = var["latents"].float().cpu().numpy()
                kwargs[variant_key("original_size", k) + key_reso_suffix] = np.array(original_size)
                kwargs[variant_key("crop_ltrb", k) + key_reso_suffix] = np.array(var["crop_ltrb"])
                kwargs[variant_key("flipped", k) + key_reso_suffix] = np.array(bool(var.get("flipped", False)))
                if var.get("alpha_mask") is not None:
                    kwargs[variant_key("alpha_mask", k) + key_reso_suffix] = var["alpha_mask"].float().cpu().numpy()
            kwargs["aug_variants"] = np.array(len(aug_variants) + 1)
        if aug_config_hash is not None:
            kwargs["aug_config_hash"] = np.array(aug_config_hash)

        save_npz(npz_path, kwargs, cache_dtype=self.cache_dtype)
