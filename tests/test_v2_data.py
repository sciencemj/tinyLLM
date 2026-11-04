import numpy as np
import pytest
import torch

from tinyllm.data import PretrainData
from tinyllm.v2.data import BinData, bits_per_char, chars_per_token


def write_bins(tmp_path):
    rng = np.random.default_rng(0)
    train, val = tmp_path / "train.bin", tmp_path / "val.bin"
    rng.integers(0, 300, 5000).astype(np.uint16).tofile(train)
    rng.integers(0, 300, 2000).astype(np.uint16).tofile(val)
    return train, val


def test_val_batches_identical_to_v1(tmp_path):
    train, val = write_bins(tmp_path)
    v1 = PretrainData(train, val, block_size=16, batch_size=4, val_batches=2, num_workers=0)
    v2 = BinData(train, val, 16, 4, 100, 2, 0)
    a = next(iter(v1.get_dataloader(False)))
    b = next(iter(v2.val_dataloader()))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_val_loader_covers_val_batches_and_drops_last(tmp_path):
    train, val = write_bins(tmp_path)
    loader = BinData(train, val, 16, 4, 100, 3, 0).val_dataloader()
    assert len(loader) == 3
    assert all(x.shape == (4, 16) for x, _ in loader)


def test_train_length_is_steps_per_chunk_batches(tmp_path):
    train, val = write_bins(tmp_path)
    loader = BinData(train, val, 16, 4, 7, 2, 0).train_dataloader()
    assert len(loader) == 7
    x, y = next(iter(loader))
    assert x.shape == y.shape == (4, 16)


def test_bits_per_char_reproduces_v1_record():
    assert bits_per_char(1.3202, 4.088) == pytest.approx(0.4659, abs=1e-4)


def test_chars_per_token_reads_meta(tmp_path):
    (tmp_path / "x.meta.json").write_text('{"tokens": 100, "chars": 412}')
    assert chars_per_token(tmp_path / "x.bin") == pytest.approx(4.12)
