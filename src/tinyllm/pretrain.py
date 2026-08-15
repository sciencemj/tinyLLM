"""1 단계: TinyStories 사전학습.

    uv run python -m tinyllm.pretrain --data data --out out
"""

import argparse
from pathlib import Path

import deeptool as dt
import torch

from tinyllm.data import PretrainData, token_count
from tinyllm.model import TinyLM


def fit(model, data, max_epochs, device, best_path, log_dir=None,
        gradient_clip_val=1.0, patience=None):
    """Trainer 를 세우고 학습을 돌린다. (history, trainer) 를 돌려준다."""
    trainer = dt.Trainer(
        max_epochs=max_epochs,
        device=device,
        gradient_clip_val=gradient_clip_val,
        plot=False,
        snapshot_best=True,
        best_path=best_path,
        log_dir=log_dir,
        patience=patience,
    )
    history = trainer.fit(model, data)
    return history, trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 이면 토큰 수에서 2 epoch 분량으로 계산한다")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--tag", default="pretrain",
                    help="체크포인트와 런 로그를 구분할 이름")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epoch 만 돌려 배선을 확인한다")
    args = ap.parse_args()

    train_bin = args.data / "stories_train.bin"
    val_bin = args.data / "stories_val.bin"
    tokens_per_step = args.block_size * args.batch_size

    if args.epochs:
        max_epochs = args.epochs
    else:
        passes = 2
        total = passes * token_count(train_bin) // tokens_per_step
        max_epochs = max(1, total // args.steps_per_epoch)
    if args.smoke:
        max_epochs = 2
    total_steps = max_epochs * args.steps_per_epoch

    print(f"train {token_count(train_bin):,} tokens | "
          f"{tokens_per_step:,} tok/step | {total_steps:,} steps | "
          f"{max_epochs:,} chunks of {args.steps_per_epoch}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyLM(block_size=args.block_size, dropout=args.dropout,
                   lr=args.lr, min_lr=args.min_lr, warmup=args.warmup,
                   total_steps=total_steps, device_type=device)
    print(f"params {sum(p.numel() for p in model.parameters()):,}")

    data = PretrainData(train_bin=train_bin, val_bin=val_bin,
                        block_size=args.block_size, batch_size=args.batch_size,
                        steps_per_epoch=args.steps_per_epoch,
                        val_batches=args.val_batches,
                        num_workers=args.num_workers)

    args.out.mkdir(parents=True, exist_ok=True)
    # log_dir 은 런마다 따로 줘야 한다. 공유하면 RunRecorder 가 한 파일에 이어 쓰고,
    # load_runs 는 root/<런>/history.jsonl 만 찾으므로 아무것도 못 읽는다.
    history, trainer = fit(model, data, max_epochs, device,
                           best_path=args.out / f"{args.tag}_best.pt",
                           log_dir=args.out / "runs" / args.tag)
    print(f"best val {trainer.best_val_loss:.4f} at chunk {trainer.best_epoch}")
    return history


if __name__ == "__main__":
    main()
