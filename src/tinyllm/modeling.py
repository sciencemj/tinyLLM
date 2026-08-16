"""배포용 독립 모델 정의.

이 파일 하나와 `config.json`, `model.safetensors`, `tokenizer.json` 만 있으면
가중치를 열 수 있다. 학습 하네스(deeptool)는 필요 없다 -- 가중치 하나 열자고
학습용 라이브러리를 설치하게 만들 이유가 없다.

`src/tinyllm/model.py` 의 TinyLM 과 모듈 이름이 같아야 state_dict 키가 맞는다.
그 동일성은 tests/test_modeling.py 가 실제 체크포인트로 검증한다.

    from modeling_tinyllm import TinyLM
    model = TinyLM.from_pretrained(".")
"""

import json
from pathlib import Path

import torch
from torch import nn


class TinyLM(nn.Module):
    """GPT-2 형태의 decoder-only LM, torch 내장 레이어로 조립.

    decoder-only 를 `TransformerEncoderLayer` 로 만드는 이유는
    `TransformerDecoderLayer` 가 cross-attention 용 `memory` 를 필수로 요구하기
    때문이다. causal mask 를 넘기면 encoder layer 가 곧 GPT 블록이다.
    """

    def __init__(self, vocab_size=8000, d_model=512, n_head=8, n_layer=8,
                 d_ff=2048, block_size=512, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_ff, dropout,
            activation="gelu", norm_first=True, batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, n_layer, norm=nn.RMSNorm(d_model), enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight   # weight tying

    def forward(self, ids):
        """ids: (B, T) int64 -> logits (B, T, vocab_size)"""
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device)
        h = self.drop(self.tok(ids) + self.pos(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        return self.head(self.blocks(h, mask=mask, is_causal=True))

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=40,
                 stop_id=None):
        """ids: (B, T) 프롬프트 -> (B, T + n). stop_id 가 전 배치에 나오면 조기 종료.

        KV 캐시는 쓰지 않는다. 29M · 512 토큰이면 forward 가 수 ms 라 캐시가 없어도
        100 토큰 생성이 1 초 미만이다.
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            window = ids[:, -self.block_size:]
            logits = self(window)[:, -1] / temperature
            if top_k:
                kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)
            if stop_id is not None and bool((nxt == stop_id).all()):
                break
        self.train(was_training)
        return ids

    @classmethod
    def from_pretrained(cls, path, device="cpu"):
        """`config.json` 과 `model.safetensors` 가 있는 폴더에서 모델을 세운다.

        저장본에는 `head.weight` 가 없다. tying 때문에 `tok.weight` 와 저장소를
        공유하는데 safetensors 는 공유 저장소를 거부하기 때문이다. 생성자가 다시
        묶으므로 `strict=False` 로 넣어도 head 가 비지 않는다.
        """
        from safetensors.torch import load_file

        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        model = cls(**{k: config[k] for k in
                       ("vocab_size", "d_model", "n_head", "n_layer",
                        "d_ff", "block_size")})
        state = load_file(path / "model.safetensors")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"저장본에 모르는 텐서가 있다: {unexpected}")
        if missing != ["head.weight"]:
            raise ValueError(f"빠진 텐서가 head.weight 만이 아니다: {missing}")
        return model.to(device).eval()
