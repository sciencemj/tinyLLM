import re
from itertools import islice
from pathlib import Path

import deeptool as dt
import numpy as np
import torch
from torch.utils import data as torch_data

from tinyllm.tokenizer import ASSISTANT, EOT, USER


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


class _WindowDataset(torch_data.Dataset):
    """uint16 memmap 에서 길이 `block_size` 짜리 창을 뽑는다.

    `seed` 가 None 이면 호출마다 무작위 위치를 고른다 (학습용). 값이 있으면
    인덱스마다 고정된 위치를 준다 (검증용) -- val loss 가 흔들리면
    best 체크포인트 선택이 오염되기 때문이다.
    """

    def __init__(self, path, block_size, length, seed=None):
        self.path = str(path)
        self.block_size = block_size
        self.length = length
        self.seed = seed
        self.n_tokens = token_count(self.path)
        if self.n_tokens <= block_size + 1:
            raise ValueError(
                f"{self.path} has {self.n_tokens} tokens, "
                f"too few for block_size={block_size}"
            )
        self._data = None

    def _array(self):
        # memmap 은 워커 프로세스마다 따로 열어야 한다. 그래서 지연 오픈.
        if self._data is None:
            self._data = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self._data

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        high = self.n_tokens - self.block_size - 1
        if self.seed is None:
            start = int(torch.randint(high, (1,)).item())
        else:
            generator = torch.Generator().manual_seed(self.seed + i)
            start = int(torch.randint(high, (1,), generator=generator).item())
        window = self._array()[start : start + self.block_size + 1]
        window = torch.from_numpy(window.astype(np.int64))
        return window[:-1], window[1:]


class PretrainData(dt.DataModule):
    """토큰 bin 위의 학습/검증 로더.

    train 로더의 길이를 `steps_per_epoch * batch_size` 로 맞추는 것이 핵심이다.
    deeptool 의 Trainer 는 epoch 마다 eval·체크포인트·early stopping 을 하는데,
    진짜 epoch 은 두 시간대라 너무 굵다. 여기서 epoch 을 100 step 청크로
    재정의해 원하는 주기를 만든다.
    """

    def __init__(self, train_bin, val_bin, block_size=512, batch_size=32,
                 steps_per_epoch=100, val_batches=20, num_workers=2, seed=1234):
        super().__init__()
        self.save_hyperparameters()

    def get_dataloader(self, train):
        if train:
            dataset = _WindowDataset(
                self.train_bin, self.block_size,
                self.steps_per_epoch * self.batch_size,
            )
        else:
            dataset = _WindowDataset(
                self.val_bin, self.block_size,
                self.val_batches * self.batch_size, seed=self.seed,
            )
        return torch_data.DataLoader(
            dataset, self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )


def render_dialog(utterances):
    """발화 리스트를 하나의 학습 문자열로 편다.

    DailyDialog 에는 화자 이름이 없고 두 사람이 번갈아 말한다. 홀/짝 번째를
    user/assistant 로 고정 매핑한다. 원본이 토큰화된 형태라 detokenize 를 거친다.
    """
    parts = []
    for i, utterance in enumerate(utterances):
        role = USER if i % 2 == 0 else ASSISTANT
        parts.append(f"{role}{detokenize(utterance)}{EOT}")
    return "".join(parts)


class SFTData(dt.DataModule):
    """DailyDialog 토큰 bin 위의 로더.

    사전학습과 달리 코퍼스가 작아서 한 epoch 이 수십 초다. 그래서 청크로
    쪼갤 이유가 없고, epoch 이 곧 코퍼스 1 회 통과다.
    """

    def __init__(self, train_bin, val_bin, block_size=512, batch_size=32,
                 steps_per_epoch=None, val_batches=10, num_workers=2,
                 seed=1234):
        super().__init__()
        self.save_hyperparameters()

    def _steps(self):
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch
        return token_count(self.train_bin) // (self.block_size * self.batch_size)

    def get_dataloader(self, train):
        if train:
            dataset = _WindowDataset(
                self.train_bin, self.block_size, self._steps() * self.batch_size,
            )
        else:
            dataset = _WindowDataset(
                self.val_bin, self.block_size,
                self.val_batches * self.batch_size, seed=self.seed,
            )
        return torch_data.DataLoader(
            dataset, self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )
