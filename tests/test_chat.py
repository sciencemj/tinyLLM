import pytest

from tinyllm.chat import build_prompt, reply
from tinyllm.model import TinyLM
from tinyllm.tokenizer import ASSISTANT_ID, EOT_ID, USER_ID, train_tokenizer

CORPUS = ["Once upon a time there was a little girl named Lily."] * 200


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(CORPUS, vocab_size=300)


@pytest.fixture
def model(tok):
    return TinyLM(vocab_size=tok.get_vocab_size(), d_model=32, n_head=4,
                  n_layer=2, d_ff=64, block_size=64, device_type="cpu")


def test_build_prompt_ends_with_the_assistant_marker(tok):
    ids = build_prompt([("user", "hi")], tok)
    assert ids[0, -1].item() == ASSISTANT_ID


def test_build_prompt_marks_each_turn(tok):
    ids = build_prompt([("user", "hi"), ("assistant", "hello")], tok)[0].tolist()
    assert ids.count(USER_ID) == 1
    assert ids.count(ASSISTANT_ID) == 2   # 지난 답 하나 + 이번 차례 표시
    assert ids.count(EOT_ID) == 2


def test_reply_returns_a_string(tok, model):
    out = reply(model, tok, [("user", "hi")], max_new_tokens=8)
    assert isinstance(out, str)


def test_reply_strips_the_eot_token(tok, model):
    out = reply(model, tok, [("user", "hi")], max_new_tokens=8)
    assert "<|eot|>" not in out


def test_render_prompt_ends_with_the_assistant_marker():
    """이 표시가 없으면 모델은 대화 대신 이야기를 이어쓴다."""
    from tinyllm.chat import render_prompt
    from tinyllm.tokenizer import ASSISTANT, EOT, USER

    out = render_prompt([("user", "hi"), ("assistant", "hello")])
    assert out == f"{USER}hi{EOT}{ASSISTANT}hello{EOT}{ASSISTANT}"
