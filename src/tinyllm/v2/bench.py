"""Throughput sweep: tokens per second and peak VRAM per (block, batch), on the real data loader."""

import gc
import time

import torch

from tinyllm.v2.data import BinData
from tinyllm.v2.module import LlamaLM

SKIP_STEPS = 10  # the first steps pay for worker start-up and a cold page cache


def measure(cfg, block_size, batch_size, steps):
    """Run `steps` real training steps at this size. Returns one result dict."""
    run = cfg.run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = LlamaLM(
        cfg.model, lr=run.lr, min_lr=run.min_lr, warmup=run.warmup,
        total_steps=run.total_steps, weight_decay=run.weight_decay,
    ).to(device)
    opt = lm.configure_optimizers()["optimizer"]
    data = BinData(
        str(cfg.data.train_bin), str(cfg.data.val_bin), block_size, batch_size,
        SKIP_STEPS + steps, run.val_batches, run.num_workers, seed=cfg.seed,
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = None
    for i, (x, y) in enumerate(data.train_dataloader()):
        if i == SKIP_STEPS:
            start = time.perf_counter()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = lm.compute_loss(x.to(device), y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), run.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss.item()  # wait for the GPU, so the clock measures finished work
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    return {
        "block_size": block_size, "batch_size": batch_size,
        "tok_s": steps * batch_size * block_size / seconds,
        "step_s": seconds / steps, "peak_gb": peak,
    }


def sweep(cfg):
    """Measure each `run.bench_sizes` [block, batch] pair. An out-of-memory size is reported and skipped."""
    results = []
    for block_size, batch_size in cfg.run.bench_sizes:
        result = None
        try:
            result = measure(cfg, block_size, batch_size, cfg.run.bench_steps)
        except torch.cuda.OutOfMemoryError:
            print(f"bench block={block_size} batch={batch_size} out of memory", flush=True)
        # Outside the except: the exception's traceback keeps the failed model alive until here.
        gc.collect()
        torch.cuda.empty_cache()
        if result:
            print("bench block={block_size} batch={batch_size} tok_s={tok_s:.0f} step_s={step_s:.3f} "
                  "peak_gb={peak_gb:.1f}".format(**result), flush=True)
            results.append(result)
    return results
