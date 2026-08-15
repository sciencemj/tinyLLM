"""생성물의 퇴화를 잡는 값싼 지표들과 고정 프롬프트 샘플링.

val loss 는 낮은데 같은 말만 반복하거나 종료 토큰을 안 내는 모드를 loss 만으로는
알 수 없다. 여기 있는 것들이 그걸 숫자로 만든다.
"""

import re
from pathlib import Path

import torch

from tinyllm.tokenizer import ASSISTANT, EOT, USER


def _ngrams(ids, n):
    return [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]


def repetition_rate(ids, n=3):
    """중복된 n-gram 의 비율. 0 이면 전부 유일, 1 에 가까우면 루프."""
    grams = _ngrams(list(ids), n)
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def distinct_n(ids, n=2):
    """유일한 n-gram 수 / 전체 n-gram 수."""
    grams = _ngrams(list(ids), n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def eot_compliance(texts):
    """생성물 중 종료 토큰을 낸 것의 비율."""
    if not texts:
        return 0.0
    return sum(EOT in t for t in texts) / len(texts)


def utterance_lengths(texts):
    """발화별 단어 수. 역할 토큰과 종료 토큰 사이를 한 발화로 본다."""
    pattern = re.compile(
        f"{re.escape(USER)}|{re.escape(ASSISTANT)}|{re.escape(EOT)}"
    )
    lengths = []
    for text in texts:
        for piece in pattern.split(text):
            words = piece.split()
            if words:
                lengths.append(len(words))
    return lengths


# 체크포인트마다 같은 프롬프트로 생성해야 시간에 따른 변화를 볼 수 있다.
# TinyStories 어휘 안에 머무는 문장들로 골랐다.
STORY_PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "Tom and Sara found a big red box in the garden.",
    "One day, a small dog saw a cat sitting on the wall.",
    "Lily wanted to bake a cake for her mom.",
    "The little boy was very sad because he lost his toy.",
    "It was raining, so Ben stayed inside and read a book.",
    "Anna had a bright blue balloon that floated away.",
    "The old man in the park gave the children some bread.",
    "Sam was scared of the dark room under the stairs.",
    "Mia and her brother built a tall tower with blocks.",
    "The cat was hungry and looked for something to eat.",
    "Jack ran as fast as he could to catch the bus.",
    "Emma found a shiny stone near the river.",
    "The bird could not fly because its wing was hurt.",
    "Every morning, the baker made warm bread for the town.",
]


def write_samples(model, tokenizer, prompts, path, label, max_new_tokens=100,
                  temperature=0.8, top_k=40, stop_id=None):
    """각 프롬프트로 생성해 `path` 에 이어 쓰고 생성 문자열 리스트를 돌려준다.

    파일을 덮어쓰지 않고 이어 쓰는 이유는 체크포인트별 생성물을 한 파일에
    쌓아 시간에 따른 변화를 그대로 읽기 위해서다. `label` 은 그 구분자다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    outputs = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        generated = model.generate(ids.to(device), max_new_tokens,
                                   temperature=temperature, top_k=top_k,
                                   stop_id=stop_id)
        new_ids = generated[0, ids.size(1):].tolist()
        outputs.append(tokenizer.decode(new_ids, skip_special_tokens=True))

    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {label} =====\n")
        for prompt, text in zip(prompts, outputs):
            f.write(f"\n--- {prompt}\n{text}\n")
    return outputs
