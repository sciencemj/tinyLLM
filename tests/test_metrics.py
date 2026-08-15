import pytest

from tinyllm.metrics import (
    STORY_PROMPTS, distinct_n, eot_compliance, repetition_rate,
    utterance_lengths, write_samples,
)
from tinyllm.model import TinyLM
from tinyllm.tokenizer import ASSISTANT, EOT, USER, train_tokenizer

CORPUS = ["Once upon a time Lily ran home."] * 200


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(CORPUS, vocab_size=300)


@pytest.fixture
def model(tok):
    return TinyLM(vocab_size=tok.get_vocab_size(), d_model=32, n_head=4,
                  n_layer=2, d_ff=64, block_size=64, device_type="cpu")


def test_repetition_rate_is_zero_for_all_unique_trigrams():
    assert repetition_rate([1, 2, 3, 4, 5, 6], n=3) == pytest.approx(0.0)


def test_repetition_rate_catches_a_looping_sequence():
    # [1,2,3] 이 네 번 -> trigram 10 개 중 유일한 것은 3 개.
    assert repetition_rate([1, 2, 3] * 4, n=3) > 0.6


def test_repetition_rate_is_zero_when_too_short():
    assert repetition_rate([1, 2], n=3) == pytest.approx(0.0)


def test_distinct_n_is_one_for_all_unique():
    assert distinct_n([1, 2, 3, 4], n=2) == pytest.approx(1.0)


def test_distinct_n_drops_with_repetition():
    assert distinct_n([1, 2] * 5, n=2) < 0.3


def test_eot_compliance_counts_generations_that_terminate():
    texts = [f"{ASSISTANT}hello{EOT}", f"{ASSISTANT}no end here"]
    assert eot_compliance(texts) == pytest.approx(0.5)


def test_eot_compliance_of_an_empty_list_is_zero():
    assert eot_compliance([]) == pytest.approx(0.0)


def test_utterance_lengths_counts_words_between_markers():
    text = f"{USER}hi there{EOT}{ASSISTANT}how are you today{EOT}"
    assert utterance_lengths([text]) == [2, 4]


def test_story_prompts_are_fifteen_distinct_strings():
    assert len(STORY_PROMPTS) == 15
    assert len(set(STORY_PROMPTS)) == 15


def test_write_samples_returns_one_generation_per_prompt(tmp_path, tok, model):
    path = tmp_path / "samples.txt"
    out = write_samples(model, tok, ["Once", "Lily"], path, label="chunk-1",
                        max_new_tokens=5)
    assert len(out) == 2
    assert all(isinstance(t, str) for t in out)
    assert "chunk-1" in path.read_text()


def test_write_samples_accumulates_across_calls(tmp_path, tok, model):
    path = tmp_path / "samples.txt"
    write_samples(model, tok, ["Once"], path, label="a", max_new_tokens=3)
    first = len(path.read_text())
    write_samples(model, tok, ["Once"], path, label="b", max_new_tokens=3)
    assert len(path.read_text()) > first
    assert "a" in path.read_text() and "b" in path.read_text()
