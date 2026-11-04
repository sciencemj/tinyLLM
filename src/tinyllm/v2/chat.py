"""CLI chat with an exported v2 model: `tinyllm-chat-v2 --model <path|repo>`."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyllm.v2.samples import render_chat


def build_prompt(history, tokenizer, max_len, max_new_tokens):
    """Token ids for `history` ((role, text) pairs, alternating, starting and ending on user).

    Fits in `max_len - max_new_tokens` tokens. The oldest user/assistant exchange is dropped
    whole until it fits, so no turn is cut. If the newest user turn alone is too long, its
    tail is kept. `tokenizer` is a `tokenizers.Tokenizer`; no BOS is added.
    """
    expected = ["user" if i % 2 == 0 else "assistant" for i in range(len(history))]
    if not history or [r for r, _ in history] != expected or len(history) % 2 == 0:
        raise ValueError("history must alternate user/assistant, start and end with user")
    budget = max_len - max_new_tokens
    if budget <= 0:
        raise ValueError("max_new_tokens leaves no room for a prompt")
    texts = [t.strip() for _, t in history]
    while True:
        ids = tokenizer.encode(render_chat(texts)).ids
        if len(ids) <= budget or len(texts) == 1:
            return ids[-budget:]
        texts = texts[2:]


def reply(model, tokenizer, history, max_new_tokens=128, temperature=0.8, top_k=40):
    """The model's next assistant turn as text. `tokenizer` is a Hugging Face fast tokenizer."""
    ids = build_prompt(history, tokenizer.backend_tokenizer,
                       model.config.max_position_embeddings, max_new_tokens)
    device = next(model.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids=x, attention_mask=torch.ones_like(x), do_sample=True,
            temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens,
            use_cache=True, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0, x.size(1):], skip_special_tokens=True).strip()


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="exported directory or Hugging Face repo id")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    print("Empty line or Ctrl-D to quit.\n")
    history = []
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            break
        history.append(("user", text))
        answer = reply(model, tokenizer, history, args.max_new_tokens,
                       args.temperature, args.top_k)
        print(f"bot> {answer}\n")
        history.append(("assistant", answer))


if __name__ == "__main__":
    main()
