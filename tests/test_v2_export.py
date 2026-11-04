from pathlib import Path

import lightning as L
import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyllm.tokenizer import ASSISTANT, EOT, USER, train_tokenizer
from tinyllm.v2.chat import build_prompt, reply
from tinyllm.v2.export import export
from tinyllm.v2.module import LlamaLM

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "model"


@pytest.fixture(scope="module")
def tok_json(tmp_path_factory):
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_tokenizer(["Once upon a time Lily ran home. hi there z y x"] * 200, vocab_size=300, path=path)
    return path


@pytest.fixture(scope="module")
def tok(tok_json):
    from tinyllm.tokenizer import load_tokenizer

    return load_tokenizer(tok_json)


@pytest.fixture(scope="module")
def lm(tok):
    torch.manual_seed(0)
    cfg = OmegaConf.load(CONFIGS / "tiny.yaml")
    cfg.vocab_size = tok.get_vocab_size()
    cfg.max_position_embeddings = 128
    return LlamaLM(cfg, lr=1e-3, min_lr=1e-4, warmup=1, total_steps=4)


@pytest.fixture(scope="module")
def ckpt(lm, tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "model.ckpt"
    trainer = L.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.strategy.connect(lm)
    trainer.save_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def out(ckpt, tok_json, tmp_path_factory):
    path = tmp_path_factory.mktemp("export")
    export(ckpt, tok_json, path)
    return path


def test_export_loads_with_auto_classes_without_remote_code(out):
    m = AutoModelForCausalLM.from_pretrained(out)
    t = AutoTokenizer.from_pretrained(out)
    assert m.config.num_key_value_heads == 2
    assert t.eos_token_id == 3
    assert t.pad_token_id == 0
    assert t.bos_token is None
    assert {USER, ASSISTANT} <= set(t.extra_special_tokens)
    assert (out / "model.safetensors").exists()


def test_export_keeps_tying_and_float32(out):
    m = AutoModelForCausalLM.from_pretrained(out)
    assert m.lm_head.weight is m.model.embed_tokens.weight
    assert m.model.embed_tokens.weight.dtype == torch.float32


def test_exported_logits_match_checkpoint(out, lm):
    m = AutoModelForCausalLM.from_pretrained(out).eval()
    ids = torch.tensor([[0, 1, 5, 9, 2, 7]])
    lm.eval()
    with torch.no_grad():
        want = lm.model(input_ids=ids).logits
        got = m(input_ids=ids).logits
    torch.testing.assert_close(got, want)


def test_export_rejects_checkpoint_with_missing_weights(ckpt, tok_json, tmp_path):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    del state["state_dict"]["model.model.layers.0.mlp.up_proj.weight"]
    bad = tmp_path / "bad.ckpt"
    torch.save(state, bad)
    with pytest.raises(RuntimeError, match="up_proj"):
        export(bad, tok_json, tmp_path / "out")


def test_chat_template_matches_training_format(out):
    t = AutoTokenizer.from_pretrained(out)
    got = t.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                add_generation_prompt=True)
    assert got == "<|user|>hi<|eot|><|assistant|>"
    got = t.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
         {"role": "user", "content": "z"}], tokenize=False, add_generation_prompt=True)
    assert got == "<|user|>hi<|eot|><|assistant|>yo<|eot|><|user|>z<|eot|><|assistant|>"


def test_build_prompt_matches_training_render_for_short_history(tok):
    ids = build_prompt([("user", "hi"), ("assistant", "yo"), ("user", "z")], tok,
                       max_len=64, max_new_tokens=8)
    assert tok.decode(ids, skip_special_tokens=False) == (
        f"{USER}hi{EOT}{ASSISTANT}yo{EOT}{USER}z{EOT}{ASSISTANT}")
    assert ids[0] == 1  # no BOS prepended


def test_build_prompt_drops_oldest_turns_to_fit(tok):
    history = [("user", "x " * 500), ("assistant", "y"), ("user", "z")]
    ids = build_prompt(history, tok, max_len=32, max_new_tokens=8)
    assert len(ids) <= 24
    assert tok.decode(ids, skip_special_tokens=False) == f"{USER}z{EOT}{ASSISTANT}"


def test_build_prompt_never_cuts_a_turn_midway(tok):
    history = [("user", "a"), ("assistant", "b b b b b b"), ("user", "c"), ("assistant", "d"),
               ("user", "e")]
    full = len(tok.encode(f"{USER}a{EOT}{ASSISTANT}b b b b b b{EOT}{USER}c{EOT}{ASSISTANT}d{EOT}"
                          f"{USER}e{EOT}{ASSISTANT}").ids)
    ids = build_prompt(history, tok, max_len=full - 1 + 4, max_new_tokens=4)
    text = tok.decode(ids, skip_special_tokens=False)
    assert text.startswith(USER) and text.endswith(f"{USER}e{EOT}{ASSISTANT}")
    assert "a" not in text.replace("assistant", "")
    assert len(ids) <= full - 1


def test_build_prompt_keeps_tail_of_oversized_newest_turn(tok):
    ids = build_prompt([("user", "x " * 500)], tok, max_len=32, max_new_tokens=8)
    assert len(ids) == 24
    assert ids[-2:] == [3, 2]  # still ends with <|eot|><|assistant|>


def test_build_prompt_rejects_bad_history(tok):
    with pytest.raises(ValueError):
        build_prompt([("assistant", "hi")], tok, max_len=32, max_new_tokens=8)
    with pytest.raises(ValueError):
        build_prompt([], tok, max_len=32, max_new_tokens=8)


def test_reply_generates_from_exported_model(out):
    m = AutoModelForCausalLM.from_pretrained(out).eval()
    t = AutoTokenizer.from_pretrained(out)
    text = reply(m, t, [("user", "hi")], max_new_tokens=5)
    assert isinstance(text, str)
