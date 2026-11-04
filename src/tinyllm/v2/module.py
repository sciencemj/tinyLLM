"""LightningModule wrapping LlamaForCausalLM."""

from collections.abc import Mapping

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from tinyllm.schedule import lr_at
from tinyllm.v2.data import bits_per_char
from tinyllm.v2.modeling import build_model


def param_groups(lm, weight_decay=0.1):
    """Split params into (decay, no_decay). 1-D params and the embedding get no decay."""
    embed = lm.model.model.embed_tokens.weight
    decay, no_decay = [], []
    for p in lm.parameters():  # parameters() yields a tied tensor once
        (no_decay if p.ndim < 2 or p is embed else decay).append(p)
    return (
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    )


class LlamaLM(L.LightningModule):
    def __init__(self, model_cfg: Mapping, lr, min_lr, warmup, total_steps, weight_decay=0.1):
        super().__init__()
        if isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        # A plain dict, so a checkpoint alone is enough to rebuild the model.
        self.save_hyperparameters({
            "model_cfg": dict(model_cfg), "lr": lr, "min_lr": min_lr,
            "warmup": warmup, "total_steps": total_steps, "weight_decay": weight_decay,
        })
        self.model = build_model(model_cfg)
        self.lr, self.min_lr, self.warmup = lr, min_lr, warmup
        self.total_steps, self.weight_decay = total_steps, weight_decay
        self.chars_per_token = None  # set by train.build_run; enables val/bits_per_char

    def compute_loss(self, x, y):
        # Targets arrive pre-shifted, so take cross entropy on logits directly (no `labels=`).
        logits = self.model(input_ids=x, use_cache=False).logits
        return F.cross_entropy(logits.flatten(0, 1), y.flatten())

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(*batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(*batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        if self.chars_per_token:
            # Linear in the loss, so the epoch mean of this equals bits/char of the mean loss.
            self.log("val/bits_per_char", bits_per_char(loss, self.chars_per_token), sync_dist=True)

    def configure_optimizers(self):
        decay, no_decay = param_groups(self, self.weight_decay)
        opt = torch.optim.AdamW([decay, no_decay], lr=self.lr, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: lr_at(s, self.total_steps, self.lr, self.min_lr, self.warmup) / self.lr
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
