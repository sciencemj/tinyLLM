"""코퍼스 다운로드 -> 토크나이저 학습 -> .bin 생성.

    uv run python -m tinyllm.prepare --out data
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from tinyllm.data import detokenize, encode_split, render_dialog, token_count
from tinyllm.tokenizer import BOS_ID, load_tokenizer, train_tokenizer

# 허브가 자동 변환해 둔 parquet. load_dataset("li2017dailydialog/daily_dialog") 는
# datasets 5.x 에서 열리지 않는다 -- 저장소가 로딩 스크립트 기반인데 스크립트 지원이
# 제거됐고 trust_remote_code 도 없어졌다. parquet 를 URL 로 직접 읽어 그 경로를
# 통째로 우회한다.
_DAILYDIALOG = (
    "https://huggingface.co/datasets/li2017dailydialog/daily_dialog/"
    "resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
)


def _stories(split):
    ds = load_dataset("roneneldan/TinyStories", split=split)
    return (row["text"] for row in ds)


def _dialogs(split):
    """발화 리스트의 이터러블. 원본이 토큰화된 형태라 소비하는 쪽이 detokenize 한다."""
    ds = load_dataset(
        "parquet",
        data_files={split: _DAILYDIALOG.format(split=split)},
        split=split,
    )
    return (row["dialog"] for row in ds)


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
            for dialog in _dialogs("train"):
                yield " ".join(detokenize(u) for u in dialog)

        print("training tokenizer...")
        tok = train_tokenizer(corpus(), vocab_size=args.vocab_size, path=tok_path)

    for split, name in (("train", "train"), ("validation", "val")):
        path = args.out / f"stories_{name}.bin"
        if path.exists():
            print(f"reusing {path} ({token_count(path):,} tokens)")
        else:
            n = encode_split(_stories(split), tok, path, BOS_ID)
            print(f"{path}: {n:,} tokens")

        path = args.out / f"dialog_{name}.bin"
        if path.exists():
            print(f"reusing {path} ({token_count(path):,} tokens)")
        else:
            texts = (render_dialog(d) for d in _dialogs(split))
            n = encode_split(texts, tok, path, BOS_ID)
            print(f"{path}: {n:,} tokens")


if __name__ == "__main__":
    main()
