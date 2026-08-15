"""코퍼스 다운로드 -> 토크나이저 학습 -> .bin 생성.

    uv run python -m tinyllm.prepare --out data
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from tinyllm.data import detokenize, encode_split, token_count
from tinyllm.tokenizer import BOS_ID, load_tokenizer, train_tokenizer


def _stories(split):
    ds = load_dataset("roneneldan/TinyStories", split=split)
    return (row["text"] for row in ds)


def _dialog_texts(split):
    # li2017dailydialog/daily_dialog 는 로딩 스크립트라 datasets 5.x 에서 열리지 않는다.
    # roskoN/dailydialog 는 같은 데이터의 parquet 변환본이고 필드명이 utterances 다.
    ds = load_dataset("roskoN/dailydialog", "full", split=split)
    return (" ".join(detokenize(u) for u in row["utterances"]) for row in ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--vocab-size", type=int, default=8000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok_path = args.out / "tokenizer.json"
    if tok_path.exists():
        print(f"reusing {tok_path}")
        tok = load_tokenizer(tok_path)
    else:
        # 두 코퍼스 합집합에서 학습해야 2 단계에서 분포 shift 가 줄어든다.
        def corpus():
            yield from _stories("train")
            yield from _dialog_texts("train")

        print("training tokenizer...")
        tok = train_tokenizer(corpus(), vocab_size=args.vocab_size, path=tok_path)

    for split, name in (("train", "train"), ("validation", "val")):
        path = args.out / f"stories_{name}.bin"
        if path.exists():
            print(f"reusing {path} ({token_count(path):,} tokens)")
            continue
        n = encode_split(_stories(split), tok, path, BOS_ID)
        print(f"{path}: {n:,} tokens")


if __name__ == "__main__":
    main()
