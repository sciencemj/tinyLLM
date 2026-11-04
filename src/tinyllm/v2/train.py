"""Hydra training entry point: `python -m tinyllm.v2.train model=... data=... run=... tag=...`."""

import hashlib
import os
import pickle
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from tokenizers import Tokenizer

from tinyllm.v2.data import BinData, chars_per_token
from tinyllm.v2.module import LlamaLM
from tinyllm.v2.samples import SampleCallback, prompts_for

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
        print(f"no checkpoint directory {ckpt_dir}: fresh start")
        return None
    # os.scandir raises on a listing error; Path.glob would swallow it and report "empty".
    with os.scandir(ckpt_dir) as it:
        found = [Path(e.path) for e in it if e.name.endswith(".ckpt") and e.is_file()]
    for path in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            torch.load(path, map_location="cpu", weights_only=WEIGHTS_ONLY)
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            # Corruption only. OSError (a flaky mount) and MemoryError must propagate, or a
            # transient failure would look like "no checkpoint" and restart from step 0.
            warnings.warn(f"skipping unreadable checkpoint {path}: {e!r}")
            continue
        print(f"resuming from {path}")
        return path
    print(f"no loadable checkpoint in {ckpt_dir}: fresh start")
    return None


def make_logger(cfg) -> WandbLogger:
    mode = cfg.wandb.mode
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        mode = "offline"
    # The id is a function of the tag, so a restarted job continues the same W&B run.
    run_id = hashlib.sha1(str(cfg.tag).encode()).hexdigest()[:16]
    return WandbLogger(
        project=cfg.wandb.project, name=str(cfg.tag), id=run_id, save_dir=str(cfg.wandb.dir),
        resume="allow", mode=mode,
    )


def build_run(cfg):
    """Return (module, trainer, datamodule, ckpt_path). ckpt_path is None for a fresh start."""
    run, ckpt_dir = cfg.run, Path(cfg.ckpt_dir)
    tok_path = cfg.data.get("tokenizer")
    if tok_path:
        vocab = Tokenizer.from_file(str(tok_path)).get_vocab_size()
        if vocab > cfg.model.vocab_size:
            raise ValueError(f"tokenizer {tok_path} has {vocab} tokens but model.vocab_size is "
                             f"{cfg.model.vocab_size}: ids would index outside the embedding")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = resolve_resume(ckpt_dir)
    if ckpt_path is None:
        L.seed_everything(cfg.seed)

    lm = LlamaLM(
        cfg.model, lr=run.lr, min_lr=run.min_lr, warmup=run.warmup,
        total_steps=run.total_steps, weight_decay=run.weight_decay,
    )
    cpt = cfg.data.get("chars_per_token")  # v1 data has no meta.json: its number comes from config
    if cpt is None and Path(cfg.data.val_bin).with_suffix(".meta.json").exists():
        cpt = chars_per_token(cfg.data.val_bin)
    lm.chars_per_token = cpt
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
    if run.samples.prompts != "none":
        prompts, chat = prompts_for(run.samples.prompts)
        callbacks.append(SampleCallback(
            prompts, run.samples.every_n_steps, Tokenizer.from_file(str(cfg.data.tokenizer)),
            chat=chat,
        ))
    trainer = L.Trainer(
        max_steps=run.total_steps, accelerator="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=make_logger(cfg), callbacks=callbacks, log_every_n_steps=run.log_every_n_steps,
        gradient_clip_val=run.grad_clip,
        default_root_dir=ckpt_dir, enable_progress_bar=sys.stderr.isatty(),
    )
    return lm, trainer, data, ckpt_path


def train(cfg):
    lm, trainer, data, ckpt_path = build_run(cfg)
    total = cfg.run.total_steps
    try:
        trainer.fit(lm, datamodule=data, ckpt_path=ckpt_path, weights_only=WEIGHTS_ONLY)
    except SystemExit:
        # Lightning turns SIGTERM mid-fit into SystemExit(0). A run that stopped before
        # total_steps is not finished, whatever the status says: the GCP wrapper records DONE
        # on status 0 and DONE blocks every later boot.
        if trainer.global_step < total:
            raise SystemExit(1)
        raise
    if trainer.global_step < total:
        print(f"stopped at step {trainer.global_step} of {total}", file=sys.stderr)
        raise SystemExit(1)
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
