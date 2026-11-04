import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tinyllm.schedule import lr_at
from tinyllm.v2.module import LlamaLM
from tinyllm.v2.samples import SampleCallback
from tinyllm.v2.train import (
    build_run, compose_cfg, main, make_logger, resolve_resume, train,
)

ROOT = Path(__file__).resolve().parent.parent


def write_bins(tmp_path):
    rng = np.random.default_rng(0)
    train_bin, val_bin = tmp_path / "train.bin", tmp_path / "val.bin"
    rng.integers(0, 64, 2000).astype(np.uint16).tofile(train_bin)
    rng.integers(0, 64, 500).astype(np.uint16).tofile(val_bin)
    return train_bin, val_bin


def overrides(tmp_path, ckpt_dir, **extra):
    train_bin, val_bin = write_bins(tmp_path)
    base = {
        "model": "tiny", "run": "smoke", "tag": "t", "wandb.mode": "disabled",
        "data.train_bin": train_bin, "data.val_bin": val_bin, "ckpt_dir": ckpt_dir,
    }
    return [f"{k}={v}" for k, v in {**base, **extra}.items()]


def read_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def make_ckpt(path):
    torch.save({"state_dict": {}, "global_step": 0}, path)
    return path


def cfg(tag):
    return compose_cfg(["model=tiny", "run=smoke", f"tag={tag}"])


def test_smoke_run_writes_last_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "ck"
    main(overrides(tmp_path, ckpt_dir))
    assert (ckpt_dir / "last.ckpt").exists()
    assert read_ckpt(ckpt_dir / "last.ckpt")["global_step"] == 4


def test_cli_runs_from_repo_root_without_changing_cwd(tmp_path):
    ckpt_dir = tmp_path / "ck"
    before = {p.name for p in ROOT.iterdir()}
    subprocess.run(
        [sys.executable, "-m", "tinyllm.v2.train", *overrides(tmp_path, ckpt_dir)],
        cwd=ROOT, check=True, capture_output=True,
    )
    assert (ckpt_dir / "last.ckpt").exists()
    assert {p.name for p in ROOT.iterdir()} == before  # no outputs/ or .hydra/ litter


def test_time_based_checkpoints_keep_two_most_recent(tmp_path):
    ckpt_dir = tmp_path / "ck"
    main(overrides(tmp_path, ckpt_dir, **{"run.total_steps": 12, "run.save_every_minutes": 1e-6}))
    steps = sorted(ckpt_dir.glob("step-*.ckpt"))
    assert len(steps) == 2
    assert (ckpt_dir / "best.ckpt").exists()


def test_resume_continues_step_and_lr(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ck"
    train(compose_cfg(overrides(tmp_path, ckpt_dir, **{"run.total_steps": 4})))

    calls = []
    original = LlamaLM.training_step

    def counting(self, batch, batch_idx):
        calls.append(1)
        return original(self, batch, batch_idx)

    monkeypatch.setattr(LlamaLM, "training_step", counting)
    run_cfg = compose_cfg(overrides(tmp_path, ckpt_dir, **{"run.total_steps": 8}))
    trainer = train(run_cfg)

    assert len(calls) == 4  # resumed at step 4, did not restart from 0
    ckpt = read_ckpt(ckpt_dir / "last.ckpt")
    assert ckpt["global_step"] == 8
    r = run_cfg.run
    expected = lr_at(8, 8, r.lr, r.min_lr, r.warmup)
    assert ckpt["optimizer_states"][0]["param_groups"][0]["lr"] == pytest.approx(expected)
    last_logged = lr_at(7, 8, r.lr, r.min_lr, r.warmup)
    assert trainer.callback_metrics["train/lr"].item() == pytest.approx(last_logged)


def test_resolve_resume_skips_truncated_last(tmp_path):
    good = make_ckpt(tmp_path / "step-4.ckpt")
    (tmp_path / "last.ckpt").write_bytes(b"xx")
    assert resolve_resume(tmp_path) == good


def test_resolve_resume_prefers_newest_loadable(tmp_path):
    make_ckpt(tmp_path / "step-4.ckpt")
    newest = make_ckpt(tmp_path / "last.ckpt")
    os.utime(tmp_path / "step-4.ckpt", (1, 1))
    assert resolve_resume(tmp_path) == newest


def test_resolve_resume_none_when_empty(tmp_path):
    assert resolve_resume(tmp_path) is None
    assert resolve_resume(tmp_path / "missing") is None


def test_resolve_resume_accepts_a_real_lightning_checkpoint(tmp_path):
    # The load check and Lightning's own load must agree on weights_only.
    ckpt_dir = tmp_path / "ck"
    train(compose_cfg(overrides(tmp_path, ckpt_dir)))
    assert resolve_resume(ckpt_dir) is not None


def test_wandb_run_id_stable_for_tag():
    assert make_logger(cfg("a"))._wandb_init["id"] == make_logger(cfg("a"))._wandb_init["id"]
    assert make_logger(cfg("a"))._wandb_init["id"] != make_logger(cfg("b"))._wandb_init["id"]


def test_wandb_falls_back_offline_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert make_logger(cfg("a"))._wandb_init["mode"] == "offline"


def test_wandb_online_with_key(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    assert make_logger(cfg("a"))._wandb_init["mode"] == "online"


def test_init_from_loads_weights_not_optimizer(tmp_path):
    src_dir = tmp_path / "src"
    train(compose_cfg(overrides(tmp_path, src_dir)))
    saved = read_ckpt(src_dir / "last.ckpt")
    assert saved["global_step"] == 4

    dst = compose_cfg(overrides(
        tmp_path, tmp_path / "dst", **{"run.init_from": src_dir / "last.ckpt"},
    ))
    lm, trainer, _, ckpt_path = build_run(dst)
    assert ckpt_path is None
    assert trainer.global_step == 0
    for name, tensor in saved["state_dict"].items():
        assert torch.equal(lm.state_dict()[name], tensor)


def test_resume_wins_over_init_from(tmp_path):
    ckpt_dir = tmp_path / "ck"
    train(compose_cfg(overrides(tmp_path, ckpt_dir)))
    other = tmp_path / "other.ckpt"
    other.write_bytes((ckpt_dir / "last.ckpt").read_bytes())
    run_cfg = compose_cfg(overrides(tmp_path, ckpt_dir, **{"run.init_from": other}))
    _, _, _, ckpt_path = build_run(run_cfg)
    assert ckpt_path is not None and ckpt_path.parent == ckpt_dir


def test_trainer_receives_grad_clip(tmp_path):
    run_cfg = compose_cfg(overrides(tmp_path, tmp_path / "ck"))
    _, trainer, _, _ = build_run(run_cfg)
    assert run_cfg.run.grad_clip == 1.0
    assert trainer.gradient_clip_val == 1.0


def test_resolve_resume_warns_on_truncated_file(tmp_path):
    good = make_ckpt(tmp_path / "step-4.ckpt")
    os.utime(good, (1, 1))
    bad = tmp_path / "last.ckpt"
    bad.write_bytes(b"xx")
    with pytest.warns(UserWarning, match="last.ckpt"):
        assert resolve_resume(tmp_path) == good


def test_resolve_resume_propagates_oserror(tmp_path, monkeypatch):
    make_ckpt(tmp_path / "last.ckpt")

    def flaky(*a, **k):
        raise OSError("fuse mount hiccup")

    monkeypatch.setattr(torch, "load", flaky)
    with pytest.raises(OSError):
        resolve_resume(tmp_path)


def test_resolve_resume_reports_decision(tmp_path, capsys):
    resolve_resume(tmp_path)
    assert "fresh start" in capsys.readouterr().out
    ckpt = make_ckpt(tmp_path / "last.ckpt")
    resolve_resume(tmp_path)
    assert f"resuming from {ckpt}" in capsys.readouterr().out


def test_wandb_uses_wandb_dir_not_ckpt_dir(tmp_path):
    run_cfg = compose_cfg(
        ["model=tiny", "run=smoke", "tag=a", f"ckpt_dir={tmp_path / 'ck'}", f"wandb.dir={tmp_path / 'w'}"]
    )
    init = make_logger(run_cfg)._wandb_init
    assert init["dir"] == str(tmp_path / "w")
    assert make_logger(cfg("a"))._wandb_init["dir"] == "out/wandb"


def test_smoke_config_disables_sampling(tmp_path):
    _, trainer, _, _ = build_run(compose_cfg(overrides(tmp_path, tmp_path / "ck")))
    assert not any(isinstance(c, SampleCallback) for c in trainer.callbacks)


def test_run_config_enables_sampling_with_data_tokenizer(tmp_path):
    ov = overrides(tmp_path, tmp_path / "ck") + [
        "run.samples.prompts=chat", "run.samples.every_n_steps=3",
        f"data.tokenizer={ROOT / 'data' / 'tokenizer.json'}",
    ]
    _, trainer, _, _ = build_run(compose_cfg(ov))
    cb = [c for c in trainer.callbacks if isinstance(c, SampleCallback)]
    assert len(cb) == 1 and cb[0].every_n_steps == 3 and cb[0].chat


def test_training_run_with_sampling_enabled_completes(tmp_path):
    from tinyllm.tokenizer import train_tokenizer

    tok_path = tmp_path / "tok.json"
    train_tokenizer(["Once upon a time Lily ran home."] * 50, vocab_size=64).save(str(tok_path))
    ov = overrides(tmp_path, tmp_path / "ck", **{"wandb.mode": "offline"}) + [
        "run.samples.prompts=chat", "run.samples.every_n_steps=2", f"data.tokenizer={tok_path}",
        f"wandb.dir={tmp_path / 'wb'}",
    ]
    main(ov)
    assert (tmp_path / "ck" / "last.ckpt").exists()


@pytest.mark.parametrize("data,path", [
    ("fineweb", "data/v2/tokenizer.json"), ("chat", "data/v2/tokenizer.json"),
    ("stories", "data/tokenizer.json"),
])
def test_data_config_tokenizer_path(data, path):
    cfg = compose_cfg(["model=tiny", "run=smoke", "tag=t", f"data={data}"])
    assert cfg.data.tokenizer == path


def test_resolve_resume_raises_when_listing_fails(tmp_path, monkeypatch):
    # An unreadable checkpoint directory must not look like "no checkpoint": restart from 0.
    def boom(path):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "scandir", boom)
    with pytest.raises(OSError):
        resolve_resume(tmp_path)


def test_train_exits_nonzero_when_fit_is_cut_short(tmp_path, monkeypatch):
    # Lightning turns SIGTERM mid-fit into SystemExit(0); that must not look like "finished".
    import lightning as L

    def cut_short(self, *a, **k):
        raise SystemExit(0)

    monkeypatch.setattr(L.Trainer, "fit", cut_short)
    with pytest.raises(SystemExit) as e:
        train(compose_cfg(overrides(tmp_path, tmp_path / "ck")))
    assert e.value.code not in (0, None)


def test_train_exits_nonzero_when_fit_returns_early(tmp_path, monkeypatch):
    import lightning as L

    monkeypatch.setattr(L.Trainer, "fit", lambda self, *a, **k: None)
    with pytest.raises(SystemExit) as e:
        train(compose_cfg(overrides(tmp_path, tmp_path / "ck")))
    assert e.value.code not in (0, None)
