import re
from itertools import islice
from pathlib import Path

import numpy as np


def _chunks(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch


def encode_split(texts, tokenizer, path, sep_id, batch_size=1000):
    """`texts` 를 인코딩해 uint16 로 `path` 에 이어 쓴다. 쓴 토큰 수를 돌려준다.

    문서마다 끝에 `sep_id` 를 붙인다. 전체를 메모리에 모으지 않고 배치마다
    파일로 흘려보낸다 -- 500M 토큰을 파이썬 리스트로 들면 10GB 를 넘는다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(path, "wb") as f:
        for batch in _chunks(texts, batch_size):
            buf = []
            for encoding in tokenizer.encode_batch(batch):
                buf.extend(encoding.ids)
                buf.append(sep_id)
            arr = np.asarray(buf, dtype=np.uint16)
            arr.tofile(f)
            total += arr.size
    return total


def token_count(path):
    """`.bin` 파일이 담은 토큰 수. uint16 이므로 바이트 수의 절반."""
    return Path(path).stat().st_size // 2


# DailyDialog 원본은 토큰화된 형태다. TinyStories 는 깨끗한 산문이라 그대로 두면
# SFT 가 사전학습에서 배운 구두점 분포와 싸운다.
_CONTRACTION = re.compile(r"\s*'\s*(?=(?:s|t|m|d|re|ve|ll)\b)", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_MISSING_SPACE_AFTER = re.compile(r"([a-z])([.?!,])([A-Z])")
_RUNS_OF_SPACE = re.compile(r"\s{2,}")


def detokenize(text):
    """토큰화된 대화 문장을 보통의 산문 간격으로 되돌린다.

    아포스트로피 결합을 축약 접미사로 한정하는 이유는 원본의 축약형 대부분이 이미
    정상이고(`don't`), 공백이 낀 것만 소수이기 때문이다. 무차별로 붙이면 인용부호가
    깨진다. 구두점 뒤 공백은 소문자+구두점+대문자일 때만 넣는다 -- 그러지 않으면
    `Stars.com` 같은 도메인이 쪼개진다.
    """
    text = text.replace("’", "'").replace("。", ".").replace("，", ",")
    text = _CONTRACTION.sub("'", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MISSING_SPACE_AFTER.sub(r"\1\2 \3", text)
    return _RUNS_OF_SPACE.sub(" ", text).strip()
