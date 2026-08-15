import pytest

from tinyllm.tokenizer import (
    ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, EOT, EOT_ID, SPECIAL_TOKENS,
    USER, USER_ID, load_tokenizer, train_tokenizer,
)

CORPUS = (
    ["Once upon a time there was a little girl named Lily."] * 200
    + ["Tom and Sara went to the park and played all day."] * 200
    + [f"{USER}Hi there{EOT}{ASSISTANT}Hello, how are you?{EOT}"] * 50
)


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(CORPUS, vocab_size=500)


def test_special_tokens_get_the_reserved_ids(tok):
    assert tok.token_to_id(BOS) == BOS_ID == 0
    assert tok.token_to_id(USER) == USER_ID == 1
    assert tok.token_to_id(ASSISTANT) == ASSISTANT_ID == 2
    assert tok.token_to_id(EOT) == EOT_ID == 3


def test_special_tokens_stay_single_tokens(tok):
    for name in SPECIAL_TOKENS:
        assert len(tok.encode(name).ids) == 1


def test_special_tokens_survive_inside_text(tok):
    ids = tok.encode(f"{USER}Hi there{EOT}").ids
    assert ids[0] == USER_ID
    assert ids[-1] == EOT_ID


def test_roundtrip_is_lossless(tok):
    text = "Once upon a time, Lily said: hello! 123"
    assert tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text


def test_roundtrip_keeps_special_tokens(tok):
    text = f"{USER}Hi{EOT}{ASSISTANT}Bye{EOT}"
    assert tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text


def test_save_and_load(tmp_path, tok):
    path = tmp_path / "tok.json"
    train_tokenizer(CORPUS, vocab_size=500, path=path)
    reloaded = load_tokenizer(path)
    text = f"{USER}Hi there{EOT}"
    assert reloaded.encode(text).ids == tok.encode(text).ids
