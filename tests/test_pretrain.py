import numpy as np
import pytest
import torch

from tinyllm.data import PretrainData
from tinyllm.model import TinyLM
from tinyllm.pretrain import fit


@pytest.fixture
def bins(tmp_path):
    rng = np.random.default_rng(0)
    train = tmp_path / "train.bin"
    val = tmp_path / "val.bin"
    rng.integers(0, 64, 4000, dtype=np.uint16).tofile(train)
    rng.integers(0, 64, 4000, dtype=np.uint16).tofile(val)
    return train, val


def test_fit_runs_and_returns_history(bins, tmp_path):
    train, val = bins
    model = TinyLM(vocab_size=64, d_model=32, n_head=4, n_layer=2, d_ff=64,
                   block_size=16, device_type="cpu", warmup=2, total_steps=20)
    data = PretrainData(train_bin=train, val_bin=val, block_size=16,
                        batch_size=4, steps_per_epoch=5, val_batches=2,
                        num_workers=0)
    history, trainer = fit(model, data, max_epochs=4, device="cpu",
                           best_path=tmp_path / "best.pt")
    assert len(history["train_loss"]) == 4
    assert len(history["val_loss"]) == 4
    assert trainer.best_val_loss is not None


def test_learning_rate_follows_the_per_step_schedule(bins, tmp_path):
    """warmup 이 끝나기 전 LR 이 base 보다 낮아야 한다 -- scheduler 없이도 걸린다."""
    train, val = bins
    model = TinyLM(vocab_size=64, d_model=32, n_head=4, n_layer=2, d_ff=64,
                   block_size=16, device_type="cpu", lr=1e-2, min_lr=1e-4,
                   warmup=1000, total_steps=10_000)
    data = PretrainData(train_bin=train, val_bin=val, block_size=16,
                        batch_size=4, steps_per_epoch=3, val_batches=1,
                        num_workers=0)
    _, trainer = fit(model, data, max_epochs=1, device="cpu", best_path=None)
    assert trainer.optim.param_groups[0]["lr"] < 1e-2


def test_best_checkpoint_is_written(bins, tmp_path):
    train, val = bins
    path = tmp_path / "best.pt"
    model = TinyLM(vocab_size=64, d_model=32, n_head=4, n_layer=2, d_ff=64,
                   block_size=16, device_type="cpu", warmup=2, total_steps=20)
    data = PretrainData(train_bin=train, val_bin=val, block_size=16,
                        batch_size=4, steps_per_epoch=5, val_batches=2,
                        num_workers=0)
    fit(model, data, max_epochs=2, device="cpu", best_path=path)
    assert path.exists()
    assert "model" in torch.load(path, map_location="cpu", weights_only=False)
