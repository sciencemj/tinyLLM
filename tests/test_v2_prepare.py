import json

import numpy as np
import pytest

from tinyllm.tokenizer import train_tokenizer
from tinyllm.v2.prepare import encode_until, render_messages, render_turns, take_chars


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(["hello world and more words"] * 100, vocab_size=300)


def test_render_messages_drops_system_and_keeps_turns():
    out = render_messages([{"role": "system", "content": "s"},
                           {"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "yo"}])
    assert out == "<|user|>hi<|eot|><|assistant|>yo<|eot|>"


def test_render_messages_skips_without_assistant():
    assert render_messages([{"role": "system", "content": "s"}]) is None


def test_render_turns_alternates_and_skips_empty():
    assert render_turns(["a", "b"]) == "<|user|>a<|eot|><|assistant|>b<|eot|>"
    assert render_turns([]) is None
    assert render_turns(["a", ""]) is None


def test_encode_until_stops_near_limit_and_writes_meta(tmp_path, tok):
    meta = encode_until(iter(["hello world"] * 1000), tok, tmp_path / "x.bin", 0, 100, batch_size=10)
    assert 100 <= meta["tokens"] < 200
    assert json.loads((tmp_path / "x.meta.json").read_text())["chars"] == meta["chars"]
    assert (tmp_path / "x.bin").stat().st_size == meta["tokens"] * 2
    assert not (tmp_path / "x.bin.tmp").exists()


def test_encode_until_skips_empty_documents(tmp_path, tok):
    meta = encode_until(iter(["", "hello"]), tok, tmp_path / "y.bin", 0, 100)
    assert meta["tokens"] > 0
    assert meta["chars"] == 5


def test_encode_until_appends_sep_after_each_document(tmp_path, tok):
    meta = encode_until(iter(["hello", "hello"]), tok, tmp_path / "z.bin", 7, 10**9)
    arr = np.fromfile(tmp_path / "z.bin", dtype=np.uint16)
    assert arr.size == meta["tokens"]
    assert list(arr).count(7) == 2 and arr[-1] == 7


def test_encode_until_does_not_leave_finished_looking_files_on_failure(tmp_path, tok):
    def boom():
        yield "hello"
        raise RuntimeError("stream died")

    with pytest.raises(RuntimeError):
        encode_until(boom(), tok, tmp_path / "w.bin", 0, 10**9, batch_size=1)
    assert not (tmp_path / "w.bin").exists()
    assert not (tmp_path / "w.meta.json").exists()


def test_take_chars_leaves_the_rest_of_the_stream_untouched():
    it = iter(["aaaa", "bbbb", "cccc", "dddd"])
    head = list(take_chars(it, 5))
    assert head == ["aaaa", "bbbb"]
    assert list(it) == ["cccc", "dddd"]
