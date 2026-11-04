from pathlib import Path

import lightning as L
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from tinyllm.schedule import lr_at
from tinyllm.v2.modeling import build_model
from tinyllm.v2.module import LlamaLM, param_groups

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "model"


def make_tiny(lr=6e-4, min_lr=6e-5, warmup=10, total_steps=100):
    cfg = OmegaConf.load(CONFIGS / "tiny.yaml")
    return LlamaLM(cfg, lr=lr, min_lr=min_lr, warmup=warmup, total_steps=total_steps)


def test_loss_uses_given_targets_without_second_shift():
    lm = make_tiny()
    x = torch.randint(0, 64, (2, 8))
    y = torch.roll(x, -1, 1)
    logits = lm.model(input_ids=x, use_cache=False).logits
    expected = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    assert lm.compute_loss(x, y).item() == pytest.approx(expected.item())


def test_no_decay_on_embeddings_and_norms():
    lm = make_tiny()
    decay, no_decay = param_groups(lm)
    embed = lm.model.model.embed_tokens.weight
    assert all(p.ndim >= 2 and p is not embed for p in decay["params"])
    assert any(p is embed for p in no_decay["params"])
    assert all(p.ndim == 1 for p in no_decay["params"] if p is not embed)
    ids = [id(p) for p in decay["params"] + no_decay["params"]]
    assert len(ids) == len(set(ids)) == len(list(lm.parameters()))
    assert decay["weight_decay"] > 0 and no_decay["weight_decay"] == 0


def test_lr_schedule_matches_v1():
    lm = make_tiny(lr=6e-4, min_lr=6e-5, warmup=10, total_steps=100)
    sched = lm.configure_optimizers()["lr_scheduler"]["scheduler"]
    opt = sched.optimizer
    for step in range(30):
        assert opt.param_groups[0]["lr"] == pytest.approx(lr_at(step, 100, 6e-4, 6e-5, 10))
        opt.step()
        sched.step()


def test_hyperparameters_carry_plain_model_cfg_that_rebuilds_the_model():
    lm = make_tiny()
    cfg = lm.hparams["model_cfg"]
    assert type(cfg) is dict
    assert cfg["hidden_size"] == 32
    rebuilt = build_model(cfg)
    assert [p.shape for p in rebuilt.parameters()] == [p.shape for p in lm.model.parameters()]


def test_fit_logs_expected_keys_and_checkpoint_rebuilds(tmp_path):
    lm = make_tiny(total_steps=2, warmup=1)
    x = torch.randint(0, 64, (4, 8))
    data = torch.utils.data.TensorDataset(x, torch.roll(x, -1, 1))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    trainer = L.Trainer(
        max_steps=2, accelerator="cpu", precision="32-true", default_root_dir=tmp_path,
        logger=False, enable_progress_bar=False, log_every_n_steps=1,
    )
    trainer.fit(lm, loader, loader)
    assert {"train/loss", "val/loss", "train/lr"} <= set(trainer.callback_metrics)
    trainer.save_checkpoint(tmp_path / "c.ckpt")
    ckpt = torch.load(tmp_path / "c.ckpt", weights_only=False)
    assert type(ckpt["hyper_parameters"]["model_cfg"]) is dict
