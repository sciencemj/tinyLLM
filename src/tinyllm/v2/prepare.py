"""32k 토크나이저와 웹/대화 코퍼스 bin 을 만든다.

    python -m tinyllm.v2.prepare --out data/v2 --max-tokens 30000000000
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np

from tinyllm.data import _chunks
from tinyllm.tokenizer import ASSISTANT, BOS_ID, EOT, USER, load_tokenizer, train_tokenizer

VOCAB_SIZE = 32000
FINEWEB_VAL_CHARS = 100_000_000
TOKENIZER_FINEWEB_CHARS = 2_000_000_000


def render_turns(utterances):
    """번갈아 말하는 발화 목록을 user/assistant 턴으로 렌더링한다."""
    utterances = [u.strip() for u in utterances]  # build_prompt strips too: same text at train and chat
    if len(utterances) < 2 or any(not u for u in utterances):
        return None
    parts = []
    for i, u in enumerate(utterances):
        parts.append(f"{USER if i % 2 == 0 else ASSISTANT}{u}{EOT}")
    return "".join(parts)


def render_messages(messages):
    """system 턴은 버리고 user/assistant 턴만 렌더링한다. assistant 가 없으면 None."""
    parts = []
    has_assistant = False
    for m in messages:
        content = m["content"].strip()  # build_prompt strips too: same text at train and chat
        if m["role"] not in ("user", "assistant") or not content:
            continue
        has_assistant |= m["role"] == "assistant"
        tag = USER if m["role"] == "user" else ASSISTANT
        parts.append(f"{tag}{content}{EOT}")
    return "".join(parts) if has_assistant else None


def take_chars(it, limit):
    """`it` 에서 누적 글자 수가 `limit` 이상이 될 때까지만 꺼낸다. 나머지는 `it` 에 남는다."""
    total = 0
    while total < limit:
        try:
            text = next(it)
        except StopIteration:
            return
        total += len(text)
        yield text


def _meta_path(path):
    return Path(path).with_suffix(".meta.json")


def is_done(path):
    return Path(path).exists() and _meta_path(path).exists()


def encode_until(texts, tokenizer, path, sep_id, max_tokens, batch_size=1000):
    """`texts` 를 uint16 bin 으로 쓴다. 토큰이 `max_tokens` 이상이 된 첫 배치 경계에서 멈춘다.

    임시 이름으로 쓰고 끝나면 rename 하며 meta.json 은 마지막에 쓴다.
    중간에 죽은 파일이 끝난 것처럼 보이지 않게 하려는 것이다.
    `max_tokens=None` 이면 끝까지 쓴다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tokens = chars = 0
    with open(tmp, "wb") as f:
        for batch in _chunks((t for t in texts if t), batch_size):
            buf = []
            for text, enc in zip(batch, tokenizer.encode_batch(batch)):
                buf.extend(enc.ids)
                buf.append(sep_id)
                chars += len(text)
            np.asarray(buf, dtype=np.uint16).tofile(f)
            tokens += len(buf)
            if max_tokens is not None and tokens >= max_tokens:
                break
    os.replace(tmp, path)
    meta = {"tokens": tokens, "chars": chars}
    meta_tmp = _meta_path(path).with_name(_meta_path(path).name + ".tmp")
    meta_tmp.write_text(json.dumps(meta))
    os.replace(meta_tmp, _meta_path(path))
    return meta


# 아래는 네트워크를 쓰므로 테스트에서 부르지 않는다.

def fineweb_texts():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-100BT", split="train", streaming=True)
    return (row["text"] for row in ds)


def smoltalk_texts(split):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
    return filter(None, (render_messages(row["messages"]) for row in ds))


def soda_texts(split):
    from datasets import load_dataset
    ds = load_dataset("allenai/soda", split=split)
    return filter(None, (render_turns(row["dialogue"]) for row in ds))


def chat_texts(smol_split, soda_split):
    return itertools.chain(smoltalk_texts(smol_split), soda_texts(soda_split))


def tokenizer_corpus():
    yield from take_chars(iter(fineweb_texts()), TOKENIZER_FINEWEB_CHARS)
    yield from chat_texts("train", "train")
    yield from chat_texts("test", "validation")


def prepare(out, max_tokens):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = out / "tokenizer.json"
    if tok_path.exists():
        tok = load_tokenizer(tok_path)
    else:
        tok = train_tokenizer(tokenizer_corpus(), vocab_size=VOCAB_SIZE, path=tok_path)

    stream = iter(fineweb_texts())
    val_texts = take_chars(stream, FINEWEB_VAL_CHARS)
    if is_done(out / "fineweb_val.bin"):
        for _ in val_texts:  # train 이 val 뒤에서 이어지도록 val 구간을 소비만 한다
            pass
    else:
        encode_until(val_texts, tok, out / "fineweb_val.bin", BOS_ID, None)
    if not is_done(out / "fineweb_train.bin"):
        encode_until(stream, tok, out / "fineweb_train.bin", BOS_ID, max_tokens)

    for name, smol, soda in (("train", "train", "train"), ("val", "test", "validation")):
        path = out / f"chat_{name}.bin"
        if not is_done(path):
            encode_until(chat_texts(smol, soda), tok, path, BOS_ID, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/v2")
    p.add_argument("--max-tokens", type=int, default=30_000_000_000)
    args = p.parse_args()
    prepare(args.out, args.max_tokens)


if __name__ == "__main__":
    main()
