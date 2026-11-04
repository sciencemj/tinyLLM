"""Llama model construction from a flat config mapping (a configs/model/*.yaml)."""

from collections.abc import Mapping

from omegaconf import DictConfig, OmegaConf
from transformers import LlamaConfig, LlamaForCausalLM


def build_config(cfg: Mapping) -> LlamaConfig:
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    return LlamaConfig(**cfg)


def build_model(cfg: Mapping) -> LlamaForCausalLM:
    return LlamaForCausalLM._from_config(build_config(cfg), attn_implementation="sdpa")
