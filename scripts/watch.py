"""원격 학습 진행 상황을 한 화면으로 본다.

    uv run python scripts/watch.py            # 한 번 찍고 끝
    uv run python scripts/watch.py --follow   # 30 초마다 갱신
"""

import argparse
import json
import subprocess
import time

REMOTE = "mlbox-wsl"
RUN_DIR = "~/tinyLLM/out/runs"


def _read(host, path):
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=15", host, f"cat {path}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"cannot read {path}")
    return out.stdout


def fetch(host, run_dir):
    meta = json.loads(_read(host, f"{run_dir}/meta.json"))
    rows = [json.loads(line) for line in
            _read(host, f"{run_dir}/history.jsonl").splitlines() if line.strip()]
    return meta, rows


def render(meta, rows):
    if not rows:
        return "아직 첫 청크가 끝나지 않았다."

    total = meta["trainer_hparams"]["max_epochs"]
    done = len(rows)
    per_chunk = sum(r["sec"] for r in rows[-10:]) / min(10, done)
    remaining = (total - done) * per_chunk

    best = min(rows, key=lambda r: r["val_loss"])
    last = rows[-1]
    steps = meta["model_hparams"]["total_steps"]

    width = 40
    filled = int(width * done / total)
    bar = "█" * filled + "·" * (width - filled)

    lines = [
        f"[{bar}] {done}/{total} chunks  ({100 * done / total:.1f}%)",
        f"  train {last['train_loss']:.4f}   val {last['val_loss']:.4f}   "
        f"lr {last['lr']:.2e}",
        f"  best  val {best['val_loss']:.4f} at chunk {best['epoch']}",
        f"  {per_chunk:.1f} s/chunk   경과 {sum(r['sec'] for r in rows) / 3600:.2f} h"
        f"   남음 {remaining / 3600:.2f} h",
        f"  {steps:,} steps total | {meta['model_hparams']['d_model']}d "
        f"× {meta['model_hparams']['n_layer']}L",
    ]

    # 최근 20 청크의 val loss 를 스파크라인으로.
    tail = [r["val_loss"] for r in rows[-20:]]
    if len(tail) > 1:
        blocks = "▁▂▃▄▅▆▇█"
        lo, hi = min(tail), max(tail)
        span = (hi - lo) or 1.0
        spark = "".join(blocks[int((v - lo) / span * (len(blocks) - 1))]
                        for v in tail)
        # 스파크라인은 시간순이므로 lo/hi 를 양끝에 두면 축을 오해하게 된다.
        lines.append(f"  val 최근 {len(tail)} 청크 {spark}  범위 {lo:.3f}~{hi:.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=REMOTE)
    ap.add_argument("--run-dir", default=RUN_DIR)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--every", type=int, default=30)
    args = ap.parse_args()

    while True:
        try:
            print(render(*fetch(args.host, args.run_dir)))
        except RuntimeError as exc:
            print(f"읽기 실패: {exc}")
        if not args.follow:
            return
        print()
        time.sleep(args.every)


if __name__ == "__main__":
    main()
