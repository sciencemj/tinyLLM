import math

import deeptool as dt
import torch
from torch import nn

from tinyllm.schedule import lr_at


class TinyLM(dt.Module):
    """GPT-2 형태의 decoder-only LM, torch 내장 레이어로 조립.

    decoder-only 스택을 `TransformerEncoderLayer` 로 만드는 이유는
    `TransformerDecoderLayer` 가 cross-attention 용 `memory` 를 필수로
    요구하기 때문이다. causal mask 를 넘기면 encoder layer 가 곧 GPT 블록이다.
    """

    def __init__(self, vocab_size=8000, d_model=512, n_head=8, n_layer=8,
                 d_ff=2048, block_size=512, dropout=0.0,
                 lr=6e-4, min_lr=6e-5, warmup=200, total_steps=61_000,
                 weight_decay=0.1, device_type="cuda"):
        super().__init__()
        self.save_hyperparameters()

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

        self.apply(self._init_weights)
        # 잔차 경로로 더해지는 투영은 층 수에 맞춰 더 작게 시작한다 (GPT-2 sec 2.3).
        # 그러지 않으면 8층을 지나며 잔차 분산이 누적된다.
        residual_std = 0.02 / math.sqrt(2 * n_layer)
        for name, p in self.named_parameters():
            if name.endswith("linear2.weight") or name.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)
        # tying 은 초기화 뒤에 연결한다. 먼저 묶으면 한쪽 초기화가 다른 쪽을 덮어쓴다.
        self.head.weight = self.tok.weight

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.MultiheadAttention):
            # qkv 는 nn.Linear 가 아니라 하나의 Parameter 라 apply 가 못 잡는다.
            nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.02)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)

    def forward(self, ids):
        """ids: (B, T) int64 -> logits (B, T, vocab_size)"""
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device)
        h = self.drop(self.tok(ids) + self.pos(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        return self.head(self.blocks(h, mask=mask, is_causal=True))

    def loss(self, logits, y):
        return nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr,
                                 betas=(0.9, 0.95), weight_decay=self.weight_decay)

    def _autocast(self):
        # bf16 은 CUDA 에서만 쓴다. CPU/MPS 개발 중에는 그냥 통과시킨다.
        return torch.autocast(self.device_type, dtype=torch.bfloat16,
                              enabled=self.device_type == "cuda")

    def training_step(self, batch):
        # deeptool 의 scheduler 는 epoch 단위라 warmup 을 표현할 수 없다.
        # LR 은 여기서 step 단위로 직접 건다. 라이브러리에 손대는 유일한 지점.
        if self.trainer is not None:
            value = lr_at(self.trainer.train_batch_idx, self.total_steps,
                          self.lr, self.min_lr, self.warmup)
            for group in self.trainer.optim.param_groups:
                group["lr"] = value
        with self._autocast():
            loss = self.loss(self(batch[0]), batch[1])
        self.plot("loss", loss, train=True)
        return loss

    def validation_step(self, batch):
        with self._autocast():
            loss = self.loss(self(batch[0]), batch[1])
        self.plot("loss", loss, train=False)
        return loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=40,
                 stop_id=None):
        """ids: (B, T) 프롬프트 -> (B, T + n). stop_id 가 전 배치에 나오면 조기 종료.

        KV 캐시는 쓰지 않는다. 29M · 512 토큰이면 forward 가 수 ms 라
        캐시가 없어도 100 토큰 생성이 1초 미만이다.
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
