"""Periodic sample generation and hygiene metrics during training."""

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger

from tinyllm.metrics import STORY_PROMPTS, distinct_n, eot_compliance, repetition_rate
from tinyllm.tokenizer import ASSISTANT, EOT, USER

# Fixed multi-turn set: turns alternate user/assistant and end on a user turn.
CHAT_PROMPTS = [
    ["Hi!"],
    ["What is your name?"],
    ["Can you tell me a short story about a cat?"],
    ["Why is the sky blue?"],
    ["Give me three tips for sleeping better."],
    ["Explain what a computer does in simple words."],
    ["Write a short poem about the rain."],
    ["What is two plus two?"],
    ["Hello", "Hello! How can I help you today?", "Tell me a fun fact."],
    ["I feel sad today.", "I'm sorry to hear that. Do you want to talk about it?", "Yes, please."],
]


def render_chat(turns):
    """Training format: `<|user|>a<|eot|><|assistant|>b<|eot|>...<|user|>z<|eot|><|assistant|>`."""
    parts = []
    for i, text in enumerate(turns):
        parts.append(f"{USER if i % 2 == 0 else ASSISTANT}{text}{EOT}")
    return "".join(parts) + ASSISTANT


def prompts_for(name):
    """(prompt strings, is_chat) for a run config's `samples.prompts` name."""
    if name == "stories":
        return STORY_PROMPTS, False
    if name == "chat":
        return [render_chat(t) for t in CHAT_PROMPTS], True
    raise ValueError(f"unknown prompt set {name!r}; expected 'stories' or 'chat'")


@torch.no_grad()
def generate_texts(model, tokenizer, prompts, max_new_tokens=100, temperature=0.8, top_k=40,
                   skip_special_tokens=True):
    """One continuation per prompt (prompt not included). `model` is a LlamaLM.

    `tokenizer` is a `tokenizers.Tokenizer`. Runs on the model's current device, then restores
    training mode and the config's `use_cache` flag.
    """
    hf = model.model
    was_training, was_cache = model.training, hf.config.use_cache
    device = next(model.parameters()).device
    model.eval()
    hf.config.use_cache = True
    try:
        texts = []
        for prompt in prompts:
            ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
            out = hf.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens,
                use_cache=True, eos_token_id=3, pad_token_id=0,
            )
            new_ids = out[0, ids.size(1):].tolist()
            texts.append(tokenizer.decode(new_ids, skip_special_tokens=skip_special_tokens))
        return texts
    finally:
        hf.config.use_cache = was_cache
        model.train(was_training)


class SampleCallback(L.Callback):
    """Every `every_n_steps` optimizer steps: sample from fixed prompts and log hygiene metrics.

    Chat runs decode with special tokens kept so `eot_compliance` can see the end token.
    """

    def __init__(self, prompts, every_n_steps, tokenizer, max_new_tokens=100, chat=False):
        self.prompts, self.every_n_steps = list(prompts), every_n_steps
        self.tokenizer, self.max_new_tokens, self.chat = tokenizer, max_new_tokens, chat
        self._last_step = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps or step == self._last_step:
            return
        self._last_step = step
        logger = trainer.logger
        if logger is None:
            return
        texts = generate_texts(
            pl_module, self.tokenizer, self.prompts, self.max_new_tokens,
            skip_special_tokens=not self.chat,
        )
        id_lists = [self.tokenizer.encode(t).ids for t in texts]
        n = len(id_lists)
        metrics = {
            "samples/repetition_rate": sum(repetition_rate(i) for i in id_lists) / n,
            "samples/distinct_2": sum(distinct_n(i, 2) for i in id_lists) / n,
        }
        if self.chat:
            metrics["samples/eot_compliance"] = eot_compliance(texts)
        if isinstance(logger, WandbLogger):
            import wandb

            metrics["samples/table"] = wandb.Table(
                columns=["step", "prompt", "sample"],
                data=[[step, p, t] for p, t in zip(self.prompts, texts)],
            )
        logger.log_metrics(metrics, step=step)
