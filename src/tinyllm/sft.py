"""2 단계: DailyDialog SFT.

    uv run python -m tinyllm.sft --data data --out out --lr 1e-4 --epochs 6
"""

import argparse
from pathlib import Path

import torch

from tinyllm.data import SFTData
from tinyllm.model import TinyLM
from tinyllm.pretrain import fit


def load_pretrained(path, **overrides):
    """사전학습 체크포인트에서 TinyLM 을 세운다. `overrides` 로 dropout·lr 등을 바꾼다."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyLM(**overrides)
    model.load_state_dict(checkpoint["model"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--tag", default="sft",
                    help="체크포인트와 런 로그를 구분할 이름")
    args = ap.parse_args()

    checkpoint = args.checkpoint or args.out / "pretrain_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = SFTData(train_bin=args.data / "dialog_train.bin",
                   val_bin=args.data / "dialog_val.bin",
                   block_size=args.block_size, batch_size=args.batch_size,
                   num_workers=args.num_workers)
    steps_per_epoch = len(data.train_dataloader())
    total_steps = steps_per_epoch * args.epochs
    print(f"{steps_per_epoch} steps/epoch | {total_steps} total steps")

    model = load_pretrained(
        checkpoint, block_size=args.block_size, dropout=args.dropout,
        lr=args.lr, min_lr=args.min_lr, warmup=args.warmup,
        total_steps=total_steps, device_type=device,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    history, trainer = fit(model, data, args.epochs, device,
                           best_path=args.out / f"{args.tag}_best.pt",
                           log_dir=args.out / "runs" / args.tag,
                           patience=args.patience)
    restored = trainer.restore_best()
    print(f"best val {trainer.best_val_loss:.4f} at epoch {restored}")
    return history


if __name__ == "__main__":
    main()
