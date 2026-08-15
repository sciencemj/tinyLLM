import math


def lr_at(step, total_steps, base_lr, min_lr, warmup):
    """Linear warmup for `warmup` steps, then cosine decay to `min_lr`.

    `step` is 0-based. Past `total_steps` the value is clamped at `min_lr`, so
    a run that overshoots its planned length does not climb back up.
    """
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
