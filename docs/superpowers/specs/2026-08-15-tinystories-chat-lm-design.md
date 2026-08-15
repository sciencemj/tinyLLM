# TinyStories 기반 소형 대화 모델 — 설계

작성일: 2026-08-15
저장소: `tinyLLM`

## 1. 목표

TinyStories로 사전학습하고 DailyDialog로 SFT한 **29.6M 파라미터 영어 대화 모델**을 만든다.
CLI로 직접 대화해서 그럴듯한 짧은 대화가 오가는 것이 성공 기준이다.

명시적으로 아닌 것: RAG 아님. 기성 사전학습 모델 파인튜닝 아님. 위키 사전학습 아님.

**"직접 만든다"의 범위**: 스택 구성(트랜스포머 층 수, 리니어 레이어 수와 폭, 임베딩 배치,
head 구조)을 직접 정한다. attention 수식이나 임베딩 연산을 손으로 구현하지는 않는다.
레이어는 `torch.nn` 내장을 쓰고, 학습 하네스는 `deeptool`을 쓴다.

## 2. 라이브러리 경계

| 영역 | 담당 | 비고 |
|---|---|---|
| 트랜스포머 레이어 | `torch.nn` 내장 | `TransformerEncoderLayer` / `TransformerEncoder` / `MultiheadAttention` |
| 스택 구성 | 직접 | 층 수·폭·head 수·FFN 비율·tying — 이 프로젝트의 산출물 |
| 학습 루프 | `deeptool` 0.3.0 | `Module` / `DataModule` / `Trainer` |
| 토크나이저 | `tokenizers` (HF) | BPE 학습 |
| 데이터셋 다운로드 | `datasets` (HF) | 이후 uint16 memmap으로 고정 |
| LR 스케줄·지표·CLI | 직접 | 100줄 미만 |

## 3. 모델

### 3.1 스택

```
ids (B, 512)
  → nn.Embedding(8000, 512)  +  nn.Embedding(512, 512)     # token + learned position
  → nn.Dropout
  → nn.TransformerEncoder(
        nn.TransformerEncoderLayer(512, nhead=8, dim_feedforward=2048,
                                   activation="gelu", norm_first=True,
                                   batch_first=True),
        num_layers=8,
        norm=nn.RMSNorm(512),
        enable_nested_tensor=False)
  → nn.Linear(512, 8000, bias=False)                        # token embedding과 weight tying
```

decoder-only는 `TransformerDecoderLayer`가 아니라 **`TransformerEncoderLayer` + causal mask**로
만든다. `TransformerDecoderLayer`는 cross-attention용 `memory`를 필수로 요구하므로 맞지 않는다.

### 3.2 파라미터 (torch 2.13에서 조립해 실측)

| 부품 | 파라미터 |
|---|---|
| token embedding (= lm_head, tied) | 4,096,000 |
| position embedding (512 × 512) | 262,144 |
| `TransformerEncoderLayer` × 8 (층당 3,152,384) | 25,219,072 |
| 최종 `RMSNorm` | 512 |
| **총계** | **29,577,728 (29.58M)** |

### 3.3 구성 선택의 근거

| 선택 | 값 | 근거 |
|---|---|---|
| 트랜스포머 층 수 | 8 | 이야기 일관성(인물 유지·인과)은 폭보다 깊이에서 나온다. 6층은 문법은 되지만 일관성이 흔들린다. 12층은 29M 예산에서 d를 384로 낮춰야 해 head_dim이 32로 떨어진다 |
| d_model / head 수 | 512 / 8 | head_dim = 64 고정이 목표. flash 커널이 64·128에 최적화되어 있고 64 미만은 head가 표현력을 잃는다 |
| FFN 폭 | 2048 (= 4·d) | 원 논문 비율. 층당 파라미터의 2/3를 차지하므로 여기를 줄이면 층을 늘릴 수 있으나 검증된 조합에서 벗어난다 |
| 리니어 head | 1개, bias 없음, tied | 별도 MLP head를 얹을 이유가 없다. tying으로 4.1M(총계의 14%) 절약 |
| 위치 임베딩 | 학습형 `nn.Embedding(512, 512)` | 내장 레이어만 쓰기로 한 이상 RoPE는 불가 |
| block_size | 512 | 아래 3.5 참조 |

층 수(6 / 8 / 10)는 같은 파라미터 예산에서 짧은 런으로 비교할 여지가 있다. 필수는 아니다.

### 3.4 내장 레이어를 쓰기로 한 결과

`nn.TransformerEncoderLayer`의 실제 구성: `self_attn(MultiheadAttention)`, `linear1`,
`linear2`, `norm1(LayerNorm)`, `norm2(LayerNorm)`.

가능: Pre-LN(`norm_first=True`), GELU(`activation='gelu'`), bias 제거(`bias=False`),
최종 RMSNorm(`TransformerEncoder(norm=nn.RMSNorm(512))`), SDPA/flash(내부가
`F.scaled_dot_product_attention` 사용).

이 중 **bias 제거는 쓰지 않는다.** `bias=True`(torch 기본값)를 유지하며, §3.2의 파라미터
실측치가 그 기준이다. 끄면 층당 약 4,600 줄어들 뿐이고 GPT-2도 bias를 유지한다.

불가: **층 내부 RMSNorm**(`norm1`/`norm2`가 `nn.LayerNorm`으로 하드코딩), **SwiGLU**
(`activation`은 `linear1` 출력에 걸리는 콜백 1개이고 게이팅은 투영 2갈래가 필요),
**RoPE**·**QK-norm**(`MultiheadAttention`에 훅 없음).

따라서 아키텍처는 **GPT-2 형태**로 확정된다. TinyStories 논문이 쓴 것이 GPT-Neo로 같은
계열이며, 논문이 검증한 것이 정확히 이 구조·이 크기다.

### 3.5 block_size를 512로 두는 이유

256도 검토했으나 이 규모에서 attention은 비용의 주범이 아니다. 층당 토큰당 FLOP은
FFN 4.2M, attention 투영 2.1M, score·AV 0.5M이다. block을 반으로 줄여도 총 4% 절약에
그친다. flash 커널을 쓰므로 메모리 차이도 없다. 512로 두고 여유를 확보한다.

### 3.6 초기화

`nn.Embedding`의 torch 기본값은 `N(0, 1)`로 너무 크다. GPT-2식으로 바꾼다.

- 모든 `Linear`·`Embedding`: `N(0, 0.02)`
- 잔차 투영(`linear2`, `self_attn.out_proj`): `N(0, 0.02 / sqrt(2 · n_layer))`
- weight tying은 초기화 **후에** 연결한다 (한쪽 초기화가 다른 쪽을 덮어쓰지 않도록)

## 4. 학습 루프

### 4.1 deeptool의 epoch 루프에 올리는 방법

deeptool은 epoch 단위로 eval·체크포인트·early stopping을 수행하고, scheduler를 epoch당
1회만 밟는다(`Trainer` docstring: *"Batch-stepped schedulers are not supported"*).
사전학습은 1 epoch이 2시간대라 그대로는 맞지 않는다.

**해결: epoch을 100 step 청크로 재정의한다.** `DataModule`이 정확히 `100 × batch_size`
길이의 map-style dataset을 주면 epoch 경계가 곧 eval·체크포인트 주기가 된다.
61,000 step이 610 epoch이 된다. **라이브러리는 수정하지 않는다.**

**LR 스케줄은 scheduler를 쓰지 않고 `training_step` 안에서 전부 per-step으로 처리한다.**
이렇게 하면 청크 굵기가 LR에 영향을 주지 않으므로, 청크는 eval·체크포인트 주기만 담당한다.

```python
def training_step(self, batch):
    step = self.trainer.train_batch_idx
    for g in self.trainer.optim.param_groups:
        g["lr"] = lr_at(step, self.total_steps, self.lr, self.min_lr, self.warmup)
    with torch.autocast(self.device_type, dtype=torch.bfloat16):
        loss = self.loss(self(batch[0]), batch[1])
    self.plot("loss", loss, train=True)
    return loss
```

`configure_optimizers`는 AdamW만 반환한다(scheduler 없음). `backward()`는 deeptool의
`fit_epoch`에서 autocast 밖에서 실행되며 이것이 올바른 사용법이다. bf16이므로
`GradScaler`는 불필요하다.

`total_steps` / `min_lr` / `warmup` / `device_type`은 `TinyLM` 생성자 인자로 받아
`save_hyperparameters()`가 잡는다. 따라서 `RunRecorder`가 남기는 `model_hparams`에
LR 스케줄 전체가 기록된다.

**deeptool에 손대는 지점은 `self.trainer.optim.param_groups` 한 곳뿐이다.** 나머지는
모두 라이브러리 계약 안에서 동작한다.

### 4.2 1단계 사전학습 설정

| 항목 | 값 |
|---|---|
| 데이터 | TinyStories (`roneneldan/TinyStories`) |
| block × batch | 512 × 32 = 16,384 tok/step |
| epoch(청크) | 100 step |
| 총 epoch | 610 (≈ 61,000 step ≈ 1B 토큰 = 500M × 2회) |
| optimizer | AdamW, betas (0.9, 0.95), weight_decay 0.1 |
| LR | warmup 200 step → cosine, 6e-4 → 6e-5 |
| grad clip | 1.0 (`Trainer(gradient_clip_val=1.0)`) |
| dropout | 0.0 |
| 정밀도 | bf16 autocast |
| best 체크포인트 | `Trainer(best_path=...)` — `atomic_save`로 크래시 안전 |
| 예상 시간 | 2.3 ~ 4.6h (§8의 미검증 항목에 의존) |

체크포인트 I/O: 29.58M × 4B = 118MB를 개선 시마다 쓴다. 초반에는 거의 매 eval마다
개선되므로 200~400회 × 118MB. 전체 런의 약 0.3% 오버헤드로, 3~5시간 런에서는
RAM 스냅샷보다 크래시 안전을 택한다.

### 4.3 2단계 SFT 설정

| 항목 | 값 |
|---|---|
| 데이터 | DailyDialog (`li2017dailydialog/daily_dialog`), 약 150만 토큰 |
| 초기 가중치 | 1단계 best 체크포인트 |
| epoch | 실제 epoch (≈ 92 step ≈ 수십 초) |
| LR | 사전학습보다 낮게 (1e-4 부근에서 시작, 스윕 대상) |
| dropout | 0.1 (150만 토큰이라 반복 노출이 크다) |
| 조기 종료 | `Trainer(patience=...)` + `restore_best()` |
| 런 비교 | `RunRecorder` + `plot_runs` |

SFT는 싸므로 형식·에폭·LR을 여러 번 실험한다. deeptool의 런 기록·비교 도구가
설계상 정확히 이 용도다.

## 5. 데이터 파이프라인

### 5.1 토크나이저

`tokenizers`로 BPE를 학습한다. **TinyStories train + DailyDialog train 합집합**에서
학습해야 2단계에서 분포 shift가 줄어든다.

- vocab 8,000
- 특수 토큰 4개 예약: `<|user|>`, `<|assistant|>`, `<|eot|>`, `<|endoftext|>`
- 특수 토큰은 BPE 병합 대상에서 제외되어 항상 단일 토큰으로 남아야 한다

TinyStories의 어휘가 약 3,000단어이므로 8k BPE에는 통단어 토큰이 많이 들어가 압축률이 좋다.

### 5.2 사전 토큰화 → uint16 memmap

vocab 8,000 < 65,536이므로 uint16으로 충분하다. 500M 토큰이면 1GB.

```
datasets로 다운로드
  → BPE 학습
  → 전체 인코딩, 이야기 사이에 <|endoftext|> 삽입
  → train.bin / val.bin (uint16 memmap)
```

### 5.3 `dt.DataModule` 두 개

**`PretrainData`**
- `train_dataloader`: map-style dataset, 길이 = `100 × batch_size`.
  deeptool의 `len(loader)` 요구를 충족하는 **동시에** 청크 경계를 만든다.
  `__getitem__`이 무작위 offset에서 `block_size + 1` 슬라이스를 떠 `(x, y)`로 쪼갠다.
  인덱스를 무시하고 매번 무작위이므로 epoch마다 새 윈도우가 나온다.
- `val_dataloader`: **결정적 고정 윈도우 20배치.** val 윈도우가 매번 무작위면 val loss가
  흔들려 best 체크포인트 선택이 오염된다. 이 결정성은 테스트로 못 박는다.

val 비용: 610 evals × 20 batch = 12,200 forward ≈ 4,000 step 상당 = 전체의 약 6.5%.

**`SFTData`** — DailyDialog를 5.4의 포맷으로 렌더링해 동일한 memmap 경로를 태운다.

**memmap 함정**: `num_workers > 0`이면 워커마다 열어야 한다. `__getitem__`에서 지연 오픈한다.

### 5.4 대화 턴 포맷

DailyDialog는 화자 이름 없이 발화 리스트만 있고 A/B가 번갈아 나온다. 홀/짝 번째 발화를
user/assistant로 고정 매핑한다.

```
<|user|>Say, Jim, how about going for a few beers after dinner?<|eot|>
<|assistant|>You know that is tempting but is really not good for our fitness.<|eot|>
```

**패딩 없이 block에 packing한다.** 패딩 + 마스크를 쓰면 flash 커널을 놓친다. 대화가
block 경계에서 잘려도 `<|eot|>`가 있어 학습에 문제가 없다.

**손실 마스킹은 하지 않는다.** 관행은 assistant 토큰만 학습시키는 것이지만 DailyDialog는
역할 비대칭이 없는 일상 대화라 전체 학습이 신호가 더 많고 구현도 단순하다. 결과가 나쁘면
마스킹으로 전환한다.

## 6. 평가와 데모

- **val perplexity** — deeptool이 제공. 런 줄세우기의 1차 수단
- **자동 위생지표** — 반복률, distinct-2/3, `<|eot|>` 준수율, 발화 길이 분포.
  loss는 낮은데 같은 말만 반복하거나 종료 토큰을 안 내는 퇴화 모드를 잡는다
- **고정 프롬프트 15개** — 체크포인트마다 생성해 파일로 축적, 시간에 따른 변화를 추적
- **대화 CLI** — `chat.py`

LLM judge는 넣지 않는다. 런 줄세우기는 val loss로 이미 되고, 절대 품질 주장이 필요해지면
나중에 얹는다.

**KV 캐시는 불필요하다.** `nn.TransformerEncoder`에는 캐시가 없어 매 토큰마다 전체 윈도우를
재계산하지만, 29M · 512 토큰 이하면 forward가 수 ms다. 100 토큰을 생성해도 1초 미만이다.

## 7. 구조와 테스트

```
src/tinyllm/
  model.py       TinyLM(dt.Module)
  tokenizer.py   BPE 학습·로드, 특수 토큰
  data.py        prepare_bins(), PretrainData, SFTData
  schedule.py    lr_at()
  metrics.py     위생지표
  pretrain.py    1단계 진입점
  sft.py         2단계 진입점
  chat.py        대화 CLI
tests/
```

성질 기반 테스트. 직전 프로젝트에서 효과가 좋았던 방식을 따른다.

| 테스트 | 잡는 것 |
|---|---|
| causal 누수 | 미래 토큰을 바꿔도 앞 위치 logits 불변 |
| weight tying | `head.weight is tok.weight`, 저장·복원 후에도 공유 유지 |
| 고정 배치 과적합 | 작은 설정이 한 배치를 loss ≈ 0으로 외운다 |
| grad 도달성 | 모든 파라미터에 grad가 도달 |
| 토크나이저 왕복 | encode → decode 무손실, 특수 토큰이 쪼개지지 않음 |
| val 결정성 | `val_dataloader()` 두 번 호출이 같은 윈도우를 준다 |
| x/y shift | `y[i] == x[i+1]` |
| LR 스케줄 | warmup 끝점·cosine 끝점 값 |
| 체크포인트 왕복 | 저장 → 로드 후 동일 출력 |

## 8. 미검증 — 착수 직후 확인할 것

이 설계의 숫자 중 아래 4개는 추정이며, mlbox-wsl에서 15~20분이면 1~3이 확정된다.

1. **`mask=` + `is_causal=True`가 실제로 flash 커널을 타는가.** Mac MPS로는 판정 불가.
   안 타면 명시적 mask를 빼고 `is_causal=True`만 넘기는 형태로 바꾼다
2. **bf16 + SDPA의 실제 속도 배수.** 2.3~4.6h 추정의 근거. fp32 실측 82k tok/s에서
   2~4배로 잡았다
3. **TinyStories의 실제 토큰 수.** 500M은 8k BPE 기준 추정치. 총 step 수가 여기 의존한다
4. **TinyStories val loss의 공개 기준선 존재 여부.** 없으면 자기 자신과의 비교만 가능하다

## 9. 하드웨어

**mlbox-wsl** (주 학습 머신, `ssh mlbox-wsl`)
- RTX 3060 Ti 8GB (sm_86), WSL2, Ryzen 5 7500F 12스레드, 디스크 926G 여유
- `nvidia-smi`가 PATH에 없다. `/usr/lib/wsl/lib/nvidia-smi`에 있다
- uv는 `~/.local/bin/uv` — `export PATH="$HOME/.local/bin:$PATH"` 필요
- torch 2.13.0+cu130 동작 확인됨

**로컬** M-series Mac (MPS). 개발·디버깅용. 본 학습은 mlbox-wsl에서.

메모리 추정 (block 512 / batch 32 / bf16 + SDPA): 파라미터+grad+AdamW 0.47GB,
activation 약 2.1GB, logits 0.52GB → peak 약 3.5GB / 8GB.
**logit 텐서가 단일 최대 activation이며 모델 크기와 무관하다.** batch나 block, vocab을
키우면 여기가 먼저 터진다.

## 10. 검증된 데이터 사실

**TinyStories** (`roneneldan/TinyStories`) — 2,141,709행, 영어 전용.
train 2.12M편 / validation 22k편. 어휘 약 3,000단어. GPT-3.5/GPT-4로 합성 생성.
논문 arXiv 2305.07759. 논문 기준 단일 GPU로 하루 미만에 학습 완료.
논문의 최대 모델이 33M이며 그것이 문법·일관성·추론이 되는 이야기를 생성한다.

**DailyDialog** (`li2017dailydialog/daily_dialog`) — 13,118 대화 / 102,979 발화 /
평균 7.9턴 / 발화당 평균 약 14.6토큰. train 11,118 / val 1,000 / test 1,000.
사람이 직접 쓴 일상 대화. 총 약 150만 토큰. 논문 arXiv 1710.03957.

**사용하지 않음**: `roneneldan/TinyStories-Instruct` — 제어된 생성이지 대화가 아니다.
