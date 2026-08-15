import numpy as np
import pytest
import torch

from tinyllm.data import (
    PretrainData, SFTData, detokenize, encode_split, render_dialog, token_count,
)
from tinyllm.tokenizer import ASSISTANT, BOS_ID, EOT, USER, train_tokenizer

CORPUS = ["Once upon a time there was a little girl named Lily."] * 200


def test_encode_split_writes_uint16_and_returns_the_count(tmp_path):
    tok = train_tokenizer(CORPUS, vocab_size=300)
    path = tmp_path / "train.bin"
    n = encode_split(["hello world", "goodbye world"], tok, path, BOS_ID)
    arr = np.fromfile(path, dtype=np.uint16)
    assert arr.size == n
    assert arr.dtype == np.uint16


def test_encode_split_puts_a_separator_after_every_document(tmp_path):
    tok = train_tokenizer(CORPUS, vocab_size=300)
    path = tmp_path / "train.bin"
    encode_split(["hello", "world"], tok, path, BOS_ID)
    arr = np.fromfile(path, dtype=np.uint16)
    assert int(arr[-1]) == BOS_ID
    assert (arr == BOS_ID).sum() == 2


def test_encode_split_survives_more_documents_than_one_batch(tmp_path):
    tok = train_tokenizer(CORPUS, vocab_size=300)
    path = tmp_path / "train.bin"
    n = encode_split(["hello"] * 25, tok, path, BOS_ID, batch_size=10)
    assert token_count(path) == n
    assert (np.fromfile(path, dtype=np.uint16) == BOS_ID).sum() == 25


def test_token_count_matches_file_size(tmp_path):
    path = tmp_path / "x.bin"
    np.arange(7, dtype=np.uint16).tofile(path)
    assert token_count(path) == 7


# --- detokenize -------------------------------------------------------------

def test_detokenize_removes_space_before_punctuation():
    raw = "Say , Jim , how about going for a few beers after dinner ?"
    assert detokenize(raw) == "Say, Jim, how about going for a few beers after dinner?"


def test_detokenize_joins_spaced_contractions():
    assert detokenize("Good.Let ' s go now .") == "Good. Let's go now."
    assert detokenize("That ’ s bogus .") == "That's bogus."
    assert detokenize("I ’ m fine .") == "I'm fine."


def test_detokenize_leaves_correct_contractions_alone():
    assert detokenize("I don't feel like sitting at home .") == \
        "I don't feel like sitting at home."


def test_detokenize_adds_the_missing_space_after_a_sentence():
    assert detokenize("I guess you are right.But what shall we do ?") == \
        "I guess you are right. But what shall we do?"


def test_detokenize_does_not_split_a_domain_name():
    """소문자+구두점+소문자라 규칙에 걸리지 않아야 한다."""
    assert detokenize("After Stars.com starts paying us .") == \
        "After Stars.com starts paying us."


def test_detokenize_normalises_the_cjk_full_stop():
    assert detokenize("He is an acrobat 。") == "He is an acrobat."


def test_detokenize_collapses_runs_of_spaces_and_strips():
    assert detokenize("  hello   there  ") == "hello there"


def test_detokenize_leaves_hyphenated_words_alone():
    assert detokenize("I can do 30 push-ups a minute !") == \
        "I can do 30 push-ups a minute!"


# --- PretrainData -----------------------------------------------------------

@pytest.fixture
def bins(tmp_path):
    rng = np.random.default_rng(0)
    train = tmp_path / "train.bin"
    val = tmp_path / "val.bin"
    rng.integers(0, 8000, 5000, dtype=np.uint16).tofile(train)
    rng.integers(0, 8000, 5000, dtype=np.uint16).tofile(val)
    return train, val


def _data(bins, **kwargs):
    train, val = bins
    opts = dict(block_size=16, batch_size=4, steps_per_epoch=10,
                val_batches=3, num_workers=0)
    opts.update(kwargs)
    return PretrainData(train_bin=train, val_bin=val, **opts)


def test_train_loader_has_exactly_steps_per_epoch_batches(bins):
    assert len(_data(bins).train_dataloader()) == 10


def test_val_loader_has_exactly_val_batches(bins):
    assert len(_data(bins).val_dataloader()) == 3


def test_batch_shapes_and_dtype(bins):
    x, y = next(iter(_data(bins).train_dataloader()))
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert x.dtype == torch.int64


def test_targets_are_inputs_shifted_by_one(bins):
    x, y = next(iter(_data(bins).train_dataloader()))
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_validation_windows_are_deterministic(bins):
    """val 윈도우가 매번 바뀌면 val loss 가 흔들려 best 체크포인트가 오염된다."""
    data = _data(bins)
    first = next(iter(data.val_dataloader()))[0]
    second = next(iter(data.val_dataloader()))[0]
    assert torch.equal(first, second)


def test_training_windows_are_not_deterministic(bins):
    torch.manual_seed(0)
    data = _data(bins)
    first = next(iter(data.train_dataloader()))[0]
    second = next(iter(data.train_dataloader()))[0]
    assert not torch.equal(first, second)


# --- SFTData ----------------------------------------------------------------

def test_render_dialog_alternates_roles():
    out = render_dialog(["Hi there", "Hello!", "How are you?"])
    assert out == (f"{USER}Hi there{EOT}"
                   f"{ASSISTANT}Hello!{EOT}"
                   f"{USER}How are you?{EOT}")


def test_render_dialog_strips_whitespace():
    assert render_dialog(["  Hi  "]) == f"{USER}Hi{EOT}"


def test_render_dialog_detokenizes_each_utterance():
    """원본이 토큰화된 형태라 렌더링 단계에서 산문 간격으로 되돌려야 한다."""
    assert render_dialog(["Say , Jim , how about a beer ?"]) == \
        f"{USER}Say, Jim, how about a beer?{EOT}"


def test_render_dialog_of_empty_is_empty():
    assert render_dialog([]) == ""


def test_sft_epoch_covers_the_whole_train_bin(bins):
    train, val = bins  # 5000 토큰짜리 bin
    data = SFTData(train_bin=train, val_bin=val, block_size=16, batch_size=4,
                   val_batches=2, num_workers=0)
    # 5000 // (16*4) = 78 배치
    assert len(data.train_dataloader()) == 78


def test_sft_respects_an_explicit_steps_per_epoch(bins):
    train, val = bins
    data = SFTData(train_bin=train, val_bin=val, block_size=16, batch_size=4,
                   steps_per_epoch=7, val_batches=2, num_workers=0)
    assert len(data.train_dataloader()) == 7
