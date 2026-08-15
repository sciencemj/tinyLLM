import numpy as np

from tinyllm.data import detokenize, encode_split, token_count
from tinyllm.tokenizer import BOS_ID, train_tokenizer

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
