"""학습된 모델과 직접 대화한다.

    uv run tinyllm-chat --checkpoint out/sft_best.pt --tokenizer data/tokenizer.json
"""

import argparse
from pathlib import Path

import torch

from tinyllm.sft import load_pretrained
from tinyllm.tokenizer import ASSISTANT, EOT, EOT_ID, USER, load_tokenizer


def build_prompt(history, tokenizer):
    """history: [(role, text), ...] -> (1, T) int64. 끝은 항상 assistant 차례 표시."""
    parts = []
    for role, text in history:
        marker = USER if role == "user" else ASSISTANT
        parts.append(f"{marker}{text.strip()}{EOT}")
    parts.append(ASSISTANT)
    ids = tokenizer.encode("".join(parts)).ids
    return torch.tensor([ids], dtype=torch.long)


def reply(model, tokenizer, history, max_new_tokens=80, temperature=0.8,
          top_k=40):
    """모델의 다음 발화를 문자열로 돌려준다. 종료 토큰은 잘라낸다."""
    prompt = build_prompt(history, tokenizer)
    device = next(model.parameters()).device
    out = model.generate(prompt.to(device), max_new_tokens,
                         temperature=temperature, top_k=top_k, stop_id=EOT_ID)
    new_ids = out[0, prompt.size(1):].tolist()
    if EOT_ID in new_ids:
        new_ids = new_ids[: new_ids.index(EOT_ID)]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("out/sft_best.pt"))
    ap.add_argument("--tokenizer", type=Path,
                    default=Path("data/tokenizer.json"))
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--turns", type=int, default=6,
                    help="프롬프트에 남길 최근 발화 수")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_pretrained(args.checkpoint, block_size=args.block_size,
                            device_type=device).to(device).eval()

    print("빈 줄이나 Ctrl-D 로 종료.\n")
    history = []
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            break
        history.append(("user", text))
        answer = reply(model, tokenizer, history[-args.turns:],
                       max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, top_k=args.top_k)
        print(f"bot> {answer}\n")
        history.append(("assistant", answer))


if __name__ == "__main__":
    main()
