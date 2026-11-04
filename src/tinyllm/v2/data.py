import json
import math
from pathlib import Path

import lightning as L
from torch.utils import data as torch_data

from tinyllm.data import _WindowDataset


class BinData(L.LightningDataModule):
    """토큰 bin 위의 학습/검증 로더.

    train 은 매번 무작위 창을 뽑고 길이는 `steps_per_chunk * batch_size` 다.
    val 은 v1 과 같은 시드·창 수·drop_last 를 써서 v1 의 기록과 같은 창을 읽는다.
    """

    def __init__(self, train_bin, val_bin, block_size, batch_size,
                 steps_per_chunk, val_batches, num_workers, seed=1234):
        super().__init__()
        self.train_bin = train_bin
        self.val_bin = val_bin
        self.block_size = block_size
        self.batch_size = batch_size
        self.steps_per_chunk = steps_per_chunk
        self.val_batches = val_batches
        self.num_workers = num_workers
        self.seed = seed

    def _loader(self, dataset):
        return torch_data.DataLoader(
            dataset, self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def train_dataloader(self):
        return self._loader(_WindowDataset(
            self.train_bin, self.block_size, self.steps_per_chunk * self.batch_size,
        ))

    def val_dataloader(self):
        return self._loader(_WindowDataset(
            self.val_bin, self.block_size, self.val_batches * self.batch_size,
            seed=self.seed,
        ))


def bits_per_char(val_loss_nats, chars_per_token):
    return val_loss_nats / math.log(2) / chars_per_token


def chars_per_token(path):
    """`x.bin` 옆의 `x.meta.json` 에서 토큰당 글자 수를 읽는다."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    return meta["chars"] / meta["tokens"]
