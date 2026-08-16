"""체크포인트를 Hugging Face 업로드 폴더로 변환한다.

    uv run python scripts/export_hf.py

체크포인트 두 개를 서로 다른 폴더로 낸다. 한 저장소에 같이 올리면 저장소 전체가
가장 엄격한 조건(비상업)에 묶이는데, 사전학습 가중치는 그럴 이유가 없기 때문이다.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

# head 수는 state_dict 모양으로 복원할 수 없다. attention 가중치가 (3d, d) 라
# head 를 몇 개로 쪼갰는지 흔적이 남지 않는다. 그래서 명시적으로 받는다.
N_HEAD = 8

RELEASES = [
    {
        "dir": "tinyllm-29m-tinystories",
        "checkpoint": "pretrain_best.pt",
        "license": "cdla-sharing-1.0",
        "title": "tinyLLM 29M — TinyStories",
        "stage": "1단계 사전학습만 거친 가중치. 대화는 못 하고 이야기를 이어쓴다.",
        "datasets": ["roneneldan/TinyStories"],
        "val": "val loss 1.3202 nats/token (perplexity 3.744, bits/char 0.4659)",
        "terms": (
            "**이용 제약이 없다.** TinyStories 는 CDLA-Sharing-1.0 이고 §3.5 가 명시한다 — "
            "*\"This Agreement imposes no obligations or restrictions on Your Use or "
            "Publication of Results.\"* §1.11 에서 Results 는 데이터의 Computational Use 로 "
            "얻은 산출물이며, 조건은 데이터의 de minimis 분량 이상을 포함하지 않는 것이다. "
            "이 모델은 train/val 격차가 0.03 이라 코퍼스를 외우고 있지 않다."
        ),
    },
    {
        "dir": "tinyllm-29m-chat",
        "checkpoint": "sft_lr3e-4_e24_best.pt",
        "license": "cc-by-nc-sa-4.0",
        "title": "tinyLLM 29M — Chat",
        "stage": "TinyStories 사전학습 후 DailyDialog 로 SFT 한 가중치. 짧은 대화를 주고받는다.",
        "datasets": ["roneneldan/TinyStories", "li2017dailydialog/daily_dialog"],
        "val": "val loss 2.3975 (DailyDialog), 종료 토큰 준수율 100%",
        "terms": (
            "**비상업 전용이다.** DailyDialog 가 CC BY-NC-SA 4.0 이고 데이터셋 카드에 "
            "\"Dataset provided for research purposes only\" 라고 적혀 있다. 유료 서비스, "
            "광고가 붙은 데모, 사내 제품 어디에도 쓸 수 없다. 재배포할 때는 출처를 표시하고 "
            "동일 조건으로 공개해야 한다.\n\n"
            "상업적 이용이 필요하면 제약이 없는 사전학습 가중치"
            "([tinyllm-29m-tinystories](https://huggingface.co/sciencemj/tinyllm-29m-tinystories))"
            "에서 시작해 2 단계 데이터만 허용적 라이선스로 교체하면 된다. SFT 는 2.4 분이다."
        ),
    },
]

CARD = """---
license: {license}
language:
- en
library_name: pytorch
pipeline_tag: text-generation
datasets:
{dataset_lines}
tags:
- tinystories
- small-language-model
---

# {title}

{stage}

파라미터 **29,577,728** 개. RTX 3060 Ti 한 대에서 사전학습 3.06 시간 + SFT 2.4 분.
학습 코드와 설계 근거: https://github.com/sciencemj/tinyLLM

{val}

## 이용 조건

{terms}

## 쓰는 법

`transformers` 를 쓰지 않는다. 이 저장소의 `modeling_tinyllm.py` 하나면 된다.

```python
import torch
from tokenizers import Tokenizer
from modeling_tinyllm import TinyLM

model = TinyLM.from_pretrained(".")
tok = Tokenizer.from_file("tokenizer.json")

ids = torch.tensor([tok.encode("{prompt}").ids])
out = model.generate(ids, 60, temperature=0.6, top_k=20{stop})
print(tok.decode(out[0].tolist(), skip_special_tokens=True))
```

{usage_note}

## 구조

```
ids (B, 512)
  → nn.Embedding(8000, 512)  +  nn.Embedding(512, 512)
  → nn.TransformerEncoder(
        nn.TransformerEncoderLayer(512, nhead=8, dim_feedforward=2048,
                                   activation="gelu", norm_first=True,
                                   batch_first=True),
        num_layers=8, norm=nn.RMSNorm(512))
  → nn.Linear(512, 8000, bias=False)     # token embedding 과 tying
```

decoder-only 를 `TransformerEncoderLayer` 로 만든다. `TransformerDecoderLayer` 는
cross-attention 용 `memory` 를 필수로 요구해서 맞지 않는다.

토크나이저는 TinyStories 와 DailyDialog 합집합에서 학습한 자체 8k byte-level BPE 다.
**같이 받은 `tokenizer.json` 을 반드시 써야 한다.** 다른 토크나이저로는 동작하지 않는다.

## 한계

**된다** — 문법, 구두점, 따옴표 대화 형식, 문단 나누기, 인물 이름 유지, 인과 연결.

**안 된다** — 턴 간 기억, 질문에 대한 직접 답변, 사실성, 문장 안 반복, 논리 일관성.
영어만 안다. 사실 정보를 얻는 용도로 쓰면 안 된다.

자세한 것은 [MODEL_CARD.md](https://github.com/sciencemj/tinyLLM/blob/main/MODEL_CARD.md).

## 인용

```bibtex
@article{{eldan2023tinystories,
  title={{TinyStories: How Small Can Language Models Be and Still Speak Coherent English?}},
  author={{Eldan, Ronen and Li, Yuanzhi}},
  journal={{arXiv preprint arXiv:2305.07759}},
  year={{2023}}
}}
{dailydialog_bib}```
"""

DAILYDIALOG_BIB = """
@InProceedings{li2017dailydialog,
  author = {Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi},
  title = {DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
  booktitle = {Proceedings of The 8th International Joint Conference on Natural Language Processing (IJCNLP 2017)},
  year = {2017}
}
"""

CHAT_NOTE = """대화는 턴 표시가 필요하다. 모델이 실제로 읽는 것은 끊기지 않는 한 줄이다:

```
<|user|>Hi, how are you today?<|eot|><|assistant|>
```

끝의 `<|assistant|>` 가 있어야 모델이 이야기를 이어쓰는 대신 자기 차례로 답한다.
특수 토큰 id 는 `<|endoftext|>`=0, `<|user|>`=1, `<|assistant|>`=2, `<|eot|>`=3 이다."""

STORY_NOTE = """이 가중치는 대화를 못 한다. 질문을 주면 이야기의 첫 문장으로 받아
계속 써 내려간다. 대화가 필요하면
[tinyllm-29m-chat](https://huggingface.co/sciencemj/tinyllm-29m-chat) 을 쓴다."""


def export(release, out_root, ckpt_dir, tokenizer_path, modeling_path,
           n_head=N_HEAD):
    checkpoint = ckpt_dir / release["checkpoint"]
    target = out_root / release["dir"]
    target.mkdir(parents=True, exist_ok=True)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    # head.weight 는 tok.weight 와 저장소를 공유한다(weight tying). safetensors 는
    # 공유 저장소를 거부하므로 뺀다. modeling_tinyllm 의 생성자가 다시 묶는다.
    shared = state["head.weight"].data_ptr() == state["tok.weight"].data_ptr()
    if not shared:
        raise ValueError("head.weight 가 tok.weight 와 묶여 있지 않다")
    state = {k: v.contiguous() for k, v in state.items() if k != "head.weight"}
    save_file(state, str(target / "model.safetensors"))

    d_model = state["tok.weight"].shape[1]
    config = {
        "model_type": "tinyllm",
        "vocab_size": state["tok.weight"].shape[0],
        "d_model": d_model,
        "n_head": n_head,
        "n_layer": sum(1 for k in state if k.endswith("self_attn.in_proj_weight")),
        "d_ff": state["blocks.layers.0.linear1.weight"].shape[0],
        "block_size": state["pos.weight"].shape[0],
        "tie_word_embeddings": True,
    }
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    shutil.copy(tokenizer_path, target / "tokenizer.json")
    shutil.copy(modeling_path, target / "modeling_tinyllm.py")

    is_chat = "chat" in release["dir"]
    card = CARD.format(
        license=release["license"],
        dataset_lines="\n".join(f"- {d}" for d in release["datasets"]),
        title=release["title"],
        stage=release["stage"],
        val=release["val"],
        terms=release["terms"],
        prompt="<|user|>Hi, how are you today?<|eot|><|assistant|>" if is_chat
               else "Once upon a time, there was a little girl named Lily.",
        stop=", stop_id=3" if is_chat else "",
        usage_note=CHAT_NOTE if is_chat else STORY_NOTE,
        dailydialog_bib=DAILYDIALOG_BIB if is_chat else "",
    )
    (target / "README.md").write_text(card)
    return target, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("dist"))
    ap.add_argument("--checkpoints", type=Path, default=Path("out"))
    ap.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    args = ap.parse_args()

    modeling = Path(__file__).resolve().parents[1] / "src" / "tinyllm" / "modeling.py"
    for release in RELEASES:
        target, config = export(release, args.out, args.checkpoints,
                                args.tokenizer, modeling)
        size = (target / "model.safetensors").stat().st_size / 1e6
        print(f"{target}  ({size:.1f} MB, {release['license']})")
        for f in sorted(target.iterdir()):
            print(f"    {f.name}")


if __name__ == "__main__":
    main()
