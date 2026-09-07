"""Smoke tests for the cache dtype CLI wiring.

These guard against:
- duplicate argparse option registration (crashes parser construction)
- the SD/SDXL text-encoder cache dtype flag being unreachable from the parser
"""

import argparse


def test_cache_latents_standalone_parser_registers_cache_dtype_once():
    """The standalone latent caching tool must not register --cache_latents_dtype twice."""
    from tools import cache_latents_standalone

    parser = cache_latents_standalone.setup_parser()
    actions = [a for a in parser._actions if a.dest == "cache_latents_dtype"]
    assert len(actions) == 1


def test_sdxl_parser_registers_te_outputs_dtype_with_default_auto():
    """SDXL parsers must expose --cache_text_encoder_outputs_dtype (default auto)."""
    from library import sdxl_train_util, train_util

    parser = argparse.ArgumentParser()
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    args = parser.parse_args([])
    assert args.cache_text_encoder_outputs_dtype == "auto"
    assert args.cache_latents_dtype == "auto"

    args = parser.parse_args(["--cache_text_encoder_outputs_dtype", "fp16", "--cache_latents_dtype", "bf16"])
    assert args.cache_text_encoder_outputs_dtype == "fp16"
    assert args.cache_latents_dtype == "bf16"


def test_sdxl_parser_without_te_caching_omits_te_dtype():
    """When text encoder caching is disabled, the TE dtype flag must not be registered."""
    from library import sdxl_train_util

    parser = argparse.ArgumentParser()
    sdxl_train_util.add_sdxl_training_arguments(parser, support_text_encoder_caching=False)
    dests = {a.dest for a in parser._actions}
    assert "cache_text_encoder_outputs_dtype" not in dests
    assert "cache_text_encoder_outputs_to_disk" not in dests
