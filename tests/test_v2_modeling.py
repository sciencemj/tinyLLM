from pathlib import Path

from omegaconf import OmegaConf

from tinyllm.v2.modeling import build_model

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "model"


def load(name):
    return OmegaConf.load(CONFIGS / f"{name}.yaml")


def test_compare_params_match_v1_within_2_percent():
    n = sum(p.numel() for p in build_model(load("compare")).parameters())
    assert abs(n - 29_577_728) / 29_577_728 < 0.02


def test_main_params_about_113m():
    n = sum(p.numel() for p in build_model(load("main")).parameters())
    assert 105_000_000 < n < 120_000_000


def test_gqa_shrinks_key_projection():
    m = build_model(load("tiny"))
    attn = m.model.layers[0].self_attn
    assert attn.k_proj.out_features < attn.q_proj.out_features


def test_embeddings_are_tied_and_no_position_table():
    m = build_model(load("tiny"))
    assert m.lm_head.weight is m.model.embed_tokens.weight
    assert not any("position" in n for n, _ in m.named_parameters())
