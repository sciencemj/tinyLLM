"""Hydra training entry point: `python -m tinyllm.v2.train model=... data=... run=... tag=...`."""

import hashlib
import os
import sys
from datetime import timedelta
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from tinyllm.v2.data import BinData
from tinyllm.v2.module import LlamaLM

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

# resolve_resume's load check, init_from and Trainer.fit(ckpt_path) all use this, so a file
# that passes the check is one Lightning can load. These are our own checkpoints, and a
# Lightning checkpoint holds more than tensors, so the restricted loader does not apply.
WEIGHTS_ONLY = False


def compose_cfg(overrides=()) -> DictConfig:
    with initialize_config_dir(str(CONFIG_DIR), version_base="1.3"):
        return compose("config", overrides=[str(o) for o in overrides])


def resolve_resume(ckpt_dir: Path) -> Path | None:
    """Newest checkpoint in `ckpt_dir` that loads. A write cut off by preemption does not."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    for path in sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            torch.load(path, map_location="cpu", weights_only=WEIGHTS_ONLY)
        except Exception:
            continue
        return path
    return None


def make_logger(cfg) -> WandbLogger:
    mode = cfg.wandb.mode
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        mode = "offline"
    # The id is a function of the tag, so a restarted job continues the same W&B run.
    run_id = hashlib.sha1(str(cfg.tag).encode()).hexdigest()[:16]
    return WandbLogger(
        project=cfg.wandb.project, name=str(cfg.tag), id=run_id, save_dir=str(cfg.ckpt_dir),
        resume="allow", mode=mode,
    )


def build_run(cfg):
    """Return (module, trainer, datamodule, ckpt_path). ckpt_path is None for a fresh start."""
    run, ckpt_dir = cfg.run, Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = resolve_resume(ckpt_dir)
    if ckpt_path is None:
        L.seed_everything(cfg.seed)

    lm = LlamaLM(
        cfg.model, lr=run.lr, min_lr=run.min_lr, warmup=run.warmup,
        total_steps=run.total_steps, weight_decay=run.weight_decay,
    )
    if ckpt_path is None and run.init_from:
        state = torch.load(run.init_from, map_location="cpu", weights_only=WEIGHTS_ONLY)
        lm.load_state_dict(state["state_dict"])

    data = BinData(
        str(cfg.data.train_bin), str(cfg.data.val_bin), run.block_size, run.batch_size,
        run.steps_per_chunk, run.val_batches, run.num_workers, seed=cfg.seed,
    )
    callbacks = [
        # Time-based saves; "step" is always a monitor candidate, so top-2 means the two latest.
        ModelCheckpoint(
            dirpath=ckpt_dir, filename="step-{step}", monitor="step", mode="max", save_top_k=2,
            save_last=True, train_time_interval=timedelta(minutes=run.save_every_minutes),
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir, filename="best", monitor="val/loss", mode="min", save_top_k=1,
        ),
    ]
    trainer = L.Trainer(
        max_steps=run.total_steps, accelerator="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=make_logger(cfg), callbacks=callbacks, log_every_n_steps=run.log_every_n_steps,
        default_root_dir=ckpt_dir, enable_progress_bar=sys.stderr.isatty(),
    )
    return lm, trainer, data, ckpt_path


def train(cfg):
    lm, trainer, data, ckpt_path = build_run(cfg)
    trainer.fit(lm, datamodule=data, ckpt_path=ckpt_path, weights_only=WEIGHTS_ONLY)
    # The time-based callback may never have fired in a short run, so always leave `last`.
    last = Path(cfg.ckpt_dir) / "last.ckpt"
    tmp = last.with_suffix(".tmp")
    trainer.save_checkpoint(tmp, weights_only=False)
    os.replace(tmp, last)
    return trainer


def main(overrides=()):
    train(compose_cfg(overrides))


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def _cli(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    _cli()
