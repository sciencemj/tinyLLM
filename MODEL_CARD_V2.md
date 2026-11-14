---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: sciencemj/tinyllm-113m-base
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/smol-smoltalk
- allenai/soda
tags:
- llama
- small-language-model
- chat
---

# tinyllm-113m-chat

A 112,678,656-parameter English chat model built on the stock `LlamaForCausalLM`
architecture. It was pretrained from scratch on 15B tokens of FineWeb-Edu, then
fine-tuned on one epoch of smol-smoltalk and SODA conversations. Everything was trained
on a single NVIDIA L4.

It is a learning project: an end-to-end run of a modern LLM recipe at small scale
(own tokenizer, pretraining, SFT, export). It holds simple multi-turn small talk in
natural English. It is not a source of facts and cannot reliably do math or code.

Base model (before SFT): [`sciencemj/tinyllm-113m-base`](https://huggingface.co/sciencemj/tinyllm-113m-base).
Earlier version: [`sciencemj/tinyllm-29m-chat`](https://huggingface.co/sciencemj/tinyllm-29m-chat).

## Usage

Loads with plain `transformers`, no `trust_remote_code`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "sciencemj/tinyllm-113m-chat"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo).eval()

messages = [{"role": "user", "content": "Hi! What do you like to do on weekends?"}]
enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                              return_tensors="pt", return_dict=True)
out = model.generate(**enc, max_new_tokens=128, do_sample=True, temperature=0.8, top_k=40)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

Chat format: `<|user|>...<|eot|><|assistant|>...<|eot|>`. `<|eot|>` (id 3) is the EOS
token; generation stops there. The context is 1,024 tokens: drop old turns from long chats.

## Architecture

| | |
|---|---|
| Class | `LlamaForCausalLM` (transformers 5.22.1) |
| Parameters | 112,678,656 |
| Layers / hidden / FFN | 14 / 768 / 2,048 (SwiGLU) |
| Attention | 12 query heads, 4 KV heads (GQA), head dim 64, RoPE θ=10,000 |
| Norm | RMSNorm, eps 1e-6 |
| Vocab | 32,000 (byte-level BPE trained on FineWeb-Edu + chat data) |
| Context | 1,024 tokens |
| Embeddings | tied input/output |

## Training

| | Pretraining | SFT |
|---|---|---|
| Data | FineWeb-Edu `sample-100BT` | smol-smoltalk + SODA |
| Tokens | 15.0B (915,528 steps × 16,384); about 0.75 of a 20.0B-token sample | 600.4M (36,644 steps, 1 epoch) |
| Batch | 16 × 1,024 tokens | 16 × 1,024 tokens |
| LR | 6e-4 → 6e-5 cosine, 2,000 warmup | 1e-4 → 1e-5 cosine, 200 warmup |
| Time on one L4 | about 201 hours | about 9 hours |

Shared: AdamW (β 0.9/0.95), weight decay 0.1 (none on embeddings and norms), grad clip
1.0, bf16 mixed precision, SDPA attention. Built with Lightning, Hydra and W&B. SFT
starts from the pretraining checkpoint with the best validation loss (step 879,999);
the loss covers every token, user turns included. The SFT learning rate was picked
from two 2,000-step probes (val loss 1.689 at 1e-4 vs 1.793 at 3e-5).

## Results

| | Value |
|---|---|
| Pretraining val loss (FineWeb-Edu, 22.2M tokens) | 3.1298 nats/token |
| Pretraining val bits/char | 1.0021 |
| SFT val loss (smol-smoltalk test + SODA validation) | 1.4069 |
| SFT train loss (last 500 steps) | 1.4952 |

Per-token losses depend on the tokenizer and do not compare across models. The 29M v1
model's 0.4659 bits/char was measured on TinyStories, a much easier corpus, so it is not
comparable either.

Samples from the final SFT checkpoint (temperature 0.8, top-k 40):

> **User:** Why is the sky blue?
> **Assistant:** The sky is blue because of the way sunlight interacts with the atmosphere. The blue light is scattered more than other colors, like red, and it has to travel through more of the atmosphere to reach us. That's why the sky is blue.

> **User:** I feel sad today. / **Assistant:** I'm sorry to hear that. Do you want to talk about it? / **User:** Yes, please.
> **Assistant:** Okay. Here, let me ask you something. What's happened this week that's making you sad?

> **User:** What is two plus two?
> **Assistant:** Two plus two is a simple addition problem.

## Limitations

- **Facts are unreliable.** It produces fluent text with confident errors.
- **Math, code and reasoning:** it often dodges the question or answers wrongly.
- **Long outputs** such as stories or poems drift off topic; it sometimes agrees to a
  request and then ends the turn without doing it.
- **English only.** No safety tuning: it can repeat biases present in web text.

Do not use it for advice, decisions, or anything where wrong answers cause harm.

## Data and licenses

| Dataset | License | Notes |
|---|---|---|
| [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ODC-By 1.0 | Filtered Common Crawl web text; also subject to the Common Crawl terms of use |
| [`HuggingFaceTB/smol-smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | Apache-2.0 | Synthetic conversations generated by LLMs; see its dataset card for the generators |
| [`allenai/soda`](https://huggingface.co/datasets/allenai/soda) | CC-BY-4.0 | Synthetic social dialogues generated by an LLM; see its dataset card and paper |

The weights are released under Apache-2.0. Attribution for the datasets above is given here.
