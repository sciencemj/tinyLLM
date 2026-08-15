import pytest
import torch

from tinyllm.model import TinyLM
from tinyllm.sft import load_pretrained

TINY = dict(vocab_size=64, d_model=32, n_head=4, n_layer=2, d_ff=64,
            block_size=16, device_type="cpu")


@pytest.fixture
def checkpoint(tmp_path):
    source = TinyLM(**TINY)
    path = tmp_path / "best.pt"
    torch.save({"model": source.state_dict(), "epoch": 3, "val_loss": 1.0}, path)
    return source, path


def test_load_pretrained_restores_the_weights(checkpoint):
    source, path = checkpoint
    loaded = load_pretrained(path, **TINY)
    for (name, a), (_, b) in zip(source.named_parameters(),
                                 loaded.named_parameters()):
        assert torch.equal(a, b), name


def test_load_pretrained_keeps_weight_tying(checkpoint):
    _, path = checkpoint
    loaded = load_pretrained(path, **TINY)
    assert loaded.head.weight is loaded.tok.weight


def test_load_pretrained_applies_overrides(checkpoint):
    _, path = checkpoint
    loaded = load_pretrained(path, dropout=0.1, lr=1e-4, **TINY)
    assert loaded.dropout == 0.1
    assert loaded.lr == 1e-4
