from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from tinyllm.metrics import STORY_PROMPTS
from tinyllm.tokenizer import ASSISTANT, EOT, USER, train_tokenizer
from tinyllm.v2.module import LlamaLM
from tinyllm.v2.samples import (
    CHAT_PROMPTS, SampleCallback, generate_texts, prompts_for, render_chat,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "model"


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(["Once upon a time Lily ran home."] * 200, vocab_size=300)


@pytest.fixture
def tiny(tok):
    cfg = OmegaConf.load(CONFIGS / "tiny.yaml")
    cfg.vocab_size = tok.get_vocab_size()
    cfg.max_position_embeddings = 128
    return LlamaLM(cfg, lr=1e-3, min_lr=1e-4, warmup=1, total_steps=4)


class FakeLogger:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step=None):
        self.calls.append((step, metrics))


def fake_trainer(step, logger):
    return SimpleNamespace(global_step=step, logger=logger)


def test_generate_texts_returns_one_per_prompt(tiny, tok):
    assert len(generate_texts(tiny, tok, ["a", "b"], max_new_tokens=4)) == 2


def test_generate_restores_train_mode_and_cache_flag(tiny, tok):
    tiny.train()
    before = tiny.model.config.use_cache
    generate_texts(tiny, tok, ["a"], max_new_tokens=2)
    assert tiny.training is True
    assert tiny.model.config.use_cache == before
    tiny.eval()
    generate_texts(tiny, tok, ["a"], max_new_tokens=2)
    assert tiny.training is False


def test_generate_keeps_special_tokens_only_when_asked(tiny, tok):
    # Untie the head and make it emit only token 3 (EOT) so generation stops at once.
    head = torch.nn.Linear(tiny.model.config.hidden_size, tok.get_vocab_size())
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[3] = 10.0
    tiny.model.lm_head = head
    raw = generate_texts(tiny, tok, ["a"], max_new_tokens=5, temperature=0.01, skip_special_tokens=False)
    clean = generate_texts(tiny, tok, ["a"], max_new_tokens=5, temperature=0.01)
    assert EOT in raw[0]
    assert EOT not in clean[0]


def test_render_chat_single_and_multi_turn():
    assert render_chat(["hi"]) == f"{USER}hi{EOT}{ASSISTANT}"
    assert render_chat(["hi", "yo", "bye"]) == (
        f"{USER}hi{EOT}{ASSISTANT}yo{EOT}{USER}bye{EOT}{ASSISTANT}"
    )


def test_chat_prompts_are_valid_conversations():
    assert len(CHAT_PROMPTS) >= 8
    assert any(len(t) > 1 for t in CHAT_PROMPTS)
    assert all(len(t) % 2 == 1 for t in CHAT_PROMPTS)  # ends on a user turn


def test_prompts_for_names():
    assert prompts_for("stories") == (STORY_PROMPTS, False)
    texts, chat = prompts_for("chat")
    assert chat and texts[0] == render_chat(CHAT_PROMPTS[0])
    with pytest.raises(ValueError):
        prompts_for("nope")


def test_callback_logs_metrics_at_interval(tiny, tok):
    cb = SampleCallback(["a", "b"], every_n_steps=2, tokenizer=tok, max_new_tokens=4, chat=True)
    logger = FakeLogger()
    for step in range(1, 6):
        cb.on_train_batch_end(fake_trainer(step, logger), tiny, None, None, 0)
    assert [s for s, _ in logger.calls] == [2, 4]
    keys = set(logger.calls[0][1])
    assert {"samples/repetition_rate", "samples/distinct_2", "samples/eot_compliance"} <= keys
    assert tiny.training is True


def test_callback_skips_eot_metric_for_non_chat(tiny, tok):
    cb = SampleCallback(["a"], every_n_steps=1, tokenizer=tok, max_new_tokens=3)
    logger = FakeLogger()
    cb.on_train_batch_end(fake_trainer(1, logger), tiny, None, None, 0)
    assert "samples/eot_compliance" not in logger.calls[0][1]


@pytest.mark.parametrize("mode", ["disabled", "offline"])
def test_callback_logs_table_with_wandb_logger(tiny, tok, tmp_path, mode):
    logger = WandbLogger(project="t", save_dir=str(tmp_path), mode=mode)
    cb = SampleCallback(["a", "b"], every_n_steps=1, tokenizer=tok, max_new_tokens=3)
    cb.on_train_batch_end(fake_trainer(1, logger), tiny, None, None, 0)
    logger.experiment.finish()
