# tinyLLM

TinyStories로 사전학습하고 DailyDialog로 SFT한 **29.6M 파라미터 영어 대화 모델**.
RTX 3060 Ti 한 대에서 사전학습 3.06시간 + SFT 7분.

```
you> Can you help me find a restaurant?
bot> Yes, of course. We have a good restaurant near the Golden Square.
you> Italian food would be great.
bot> You can find a restaurant there.
you> What time do they open?
bot> They open at 9:00.
```

## 이 프로젝트가 직접 만든 것

**스택 구성을 직접 정한다.** 트랜스포머를 몇 층 쌓을지, 각 층을 얼마나 넓게 할지,
head를 몇 개로 나눌지, 리니어 head를 어떻게 붙일지, 임베딩을 묶을지.

**레이어 자체는 직접 구현하지 않는다.** attention 수식이나 정규화는 `torch.nn` 내장을
쓰고, 학습 루프는 [`deeptool`](https://github.com/sciencemj/deeptool)에 맡긴다.

| 영역 | 담당 |
|---|---|
| 트랜스포머 레이어 | `torch.nn` 내장 (`TransformerEncoderLayer` 등) |
| **스택 구성** | **직접** — 층 수·폭·head 수·FFN 비율·tying·초기화 |
| 학습 루프 | `deeptool` 0.3.0 |
| 토크나이저 | HF `tokenizers` (BPE 직접 학습) |
| 데이터셋 | HF `datasets` |
| LR 스케줄·지표·CLI | 직접 |

## 모델

```
ids (B, 512)
  → nn.Embedding(8000, 512)  +  nn.Embedding(512, 512)     # token + learned position
  → nn.Dropout
  → nn.TransformerEncoder(
        nn.TransformerEncoderLayer(512, nhead=8, dim_feedforward=2048,
                                   activation="gelu", norm_first=True,
                                   batch_first=True),
        num_layers=8, norm=nn.RMSNorm(512), enable_nested_tensor=False)
  → nn.Linear(512, 8000, bias=False)                        # token embedding과 tying
```

| 부품 | 파라미터 |
|---|---|
| token embedding (= lm_head, tied) | 4,096,000 |
| position embedding | 262,144 |
| `TransformerEncoderLayer` × 8 | 25,219,072 |
| 최종 `RMSNorm` | 512 |
| **총계** | **29,577,728** |

decoder-only를 `TransformerDecoderLayer`가 아니라 **`TransformerEncoderLayer` +
causal mask**로 만든다. `TransformerDecoderLayer`는 cross-attention용 `memory`를
필수로 요구하므로 맞지 않는다.

**구성을 이렇게 고른 이유**

- **8층** — 이야기 일관성(인물 유지·인과)은 폭보다 깊이에서 나온다. 12층으로 가려면
  같은 예산에서 `d_model`을 384로 낮춰야 하고 그러면 head_dim이 32로 떨어진다.
- **512 / head 8** — head_dim을 64로 고정하기 위한 조합. flash 커널이 64·128에
  최적화되어 있고 64 미만은 head가 표현력을 잃는다.
- **weight tying** — 4.1M(총계의 14%)을 절약한다. 이 규모에서는 거의 공짜다.
- **block 512** — 256도 검토했으나 이 규모에서 attention은 비용의 주범이 아니다.
  층당 토큰당 FLOP이 FFN 4.2M, attention 투영 2.1M, score·AV 0.5M이라 block을 반으로
  줄여도 총 4% 절약에 그친다.

`nn.TransformerEncoderLayer`를 쓰기로 한 결과 RoPE·SwiGLU·QK-norm·층 내부 RMSNorm은
넣을 수 없다(내부가 `nn.LayerNorm` 하드코딩이고 `activation`은 콜백 하나뿐이다).
따라서 아키텍처는 GPT-2 형태로 확정된다. TinyStories 논문이 쓴 것이 GPT-Neo로 같은
계열이며, 논문이 검증한 것이 정확히 이 구조·이 크기다.

## 결과

### 1단계 — TinyStories 사전학습

| | |
|---|---|
| **best val loss** | **1.3202** nats/token (perplexity 3.744) |
| **bits/char** | **0.4659** |
| 학습량 | 57,000 step × 16,384 tok = 934M 토큰 (467M × 2회) |
| 소요 | **3.06시간** (RTX 3060 Ti, bf16) |

> **Once upon a time, there was a little girl named Lily.**
> She loved to play outside with her friends. One day, they found a big box and
> decided to play with it. [...] After a while, it started to rain. Lily and her
> friends ran under a big tree to stay dry. They waited and waited, but the rain
> didn't stop.

문법·구두점·따옴표 대화 형식·문단 나누기·인물 유지·인과 연결이 된다. 문장 안 반복과
논리 미끄러짐은 남는다. 자세한 것은 [`docs/pretrain-run.md`](docs/pretrain-run.md).

**per-token val loss는 토크나이저가 다르면 비교할 수 없다.** 토큰이 굵으면 개당
예측이 어려워 loss가 오르고 잘면 내려간다. 이 저장소는 자체 8k BPE(4.088 chars/token)를
쓰므로 다른 토크나이저를 쓴 공개 수치와 나란히 놓을 수 없다. 비교가 필요하면
토크나이저 무관 지표인 **bits/char 0.4659**를 쓴다.

### 2단계 — DailyDialog SFT

| | |
|---|---|
| **best val loss** | **2.3975** (LR 3e-4, 24 epoch) |
| **eot 준수율** | **100%** |
| 소요 | 런당 2.4~7분 (사전학습의 약 1/80) |

LR 7개 설정을 스윕했다. **epoch이 LR보다 영향이 크고, epoch을 늘리면 LR 순위가
뒤집힌다.** 8 epoch에서는 1e-3이 최고였는데 24 epoch에서는 3e-4가 이겼다. val loss가
거의 같아도 train/val 격차를 봐야 한다 — 6e-4/24ep는 val이 2.4103으로 비슷하지만
격차가 0.93으로 3e-4의 0.42보다 두 배 이상 크다(외우고 있다는 뜻).
자세한 것은 [`docs/sft-runs.md`](docs/sft-runs.md).

**된다**: 대화 형식 유지, 종료 토큰, 직전 발화에 대한 반응.
**안 된다**: 턴 간 기억, 질문에 대한 직접 답변, 사실성.

## 대화 턴 포맷

```
<|user|>Say, Jim, how about going for a few beers after dinner?<|eot|>
<|assistant|>You know that is tempting but is really not good for our fitness.<|eot|>
```

특수 토큰 4개(`<|endoftext|>`=0, `<|user|>`=1, `<|assistant|>`=2, `<|eot|>`=3)를 BPE
vocab에 예약한다. DailyDialog는 화자 이름이 없고 두 사람이 번갈아 말하므로 홀/짝
번째를 user/assistant로 고정 매핑한다. 패딩 없이 block에 packing한다 — 패딩+마스크를
쓰면 flash 커널을 놓친다.

## 실행

```bash
uv sync

# 코퍼스 다운로드 -> BPE 학습 -> uint16 memmap 생성
uv run python -m tinyllm.prepare --out data

# 1단계: TinyStories 사전학습
uv run python -m tinyllm.pretrain

# 2단계: DailyDialog SFT
uv run python -m tinyllm.sft --lr 3e-4 --epochs 24 --tag sft_lr3e-4_e24

# 대화
uv run tinyllm-chat --checkpoint out/sft_lr3e-4_e24_best.pt \
                    --tokenizer data/tokenizer.json

# 원격 학습 진행 상황
uv run python scripts/watch.py --follow
```

## 학습 루프를 deeptool에 올리는 법

deeptool은 epoch 단위로 eval·체크포인트·early stopping을 하고 scheduler를 epoch당
1회만 밟는다. 사전학습은 1 epoch이 시간 단위라 그대로는 맞지 않는다.

**epoch을 100 step 청크로 재정의한다.** `DataModule`이 정확히 `100 × batch_size`
길이의 dataset을 주면 epoch 경계가 곧 eval·체크포인트 주기가 된다. **라이브러리는
수정하지 않는다.**

**LR 스케줄은 `training_step` 안에서 per-step으로 처리한다.** scheduler를 쓰지 않으므로
청크 굵기가 LR에 영향을 주지 않는다. deeptool에 손대는 지점은
`self.trainer.optim.param_groups` 한 곳뿐이다.

`log_dir`은 **런마다 따로 줘야 한다**(`out/runs/<tag>`). 공유하면 `RunRecorder`가 한
파일에 이어 써서 런들이 한 곡선으로 섞이고, `load_runs`는 `root/<런>/history.jsonl`만
찾으므로 root 바로 밑의 파일은 아예 못 읽는다.

## 구조

```
src/tinyllm/
  model.py       TinyLM(dt.Module)
  tokenizer.py   BPE 학습·로드, 특수 토큰
  data.py        detokenize, encode_split, PretrainData, SFTData, render_dialog
  schedule.py    lr_at()
  metrics.py     위생지표, 고정 프롬프트 샘플링
  prepare.py     코퍼스 -> .bin
  pretrain.py    1단계
  sft.py         2단계
  chat.py        대화 CLI
scripts/
  bench.py       throughput·메모리·커널 실측
  watch.py       원격 학습 진행 상황
tests/           성질 기반 테스트 71개
```

## 테스트

shape만 보지 않고 성질을 본다.

| 테스트 | 잡는 것 |
|---|---|
| causal 누수 | 미래 토큰을 바꿔도 앞 위치 logits 불변 |
| weight tying | 저장·복원 후에도 저장소 공유 유지 |
| 고정 배치 과적합 | 작은 설정이 한 배치를 loss≈0으로 외운다 |
| grad 도달성 | 모든 파라미터에 grad가 도달 |
| val 결정성 | `val_dataloader()` 두 번 호출이 같은 윈도우 |
| 런 로그 격리 | 두 런이 `load_runs`에서 따로 읽힌다 |
| 초기 loss | 초기화가 정상이면 `ln(vocab)` 근처 |

`test_initial_loss_is_near_uniform`은 타깃을 **입력과 독립인 난수**로 둔다. 입력 자신에
채점하면 weight tying 때문에 잔차 스트림에 남은 입력 임베딩이 현재 토큰의 logit을
밀어 올려, 초기화가 정상이어도 loss가 `ln(vocab)`보다 뚜렷하게 낮게 나온다
(8개 seed 실측: 자기 채점 |diff| 0.403~0.437, 독립 타깃 0.0013~0.0264).

```bash
uv run pytest
```

## 하드웨어

RTX 3060 Ti 8GB, bf16 + flash. 실측:

| batch | 정밀도 | ms/step | tok/s | peak |
|---|---|---|---|---|
| 32 | fp32 | 531.4 | 30.8k | 6.58 GB |
| 32 | **bf16** | 174.0 | **94.2k** | **4.22 GB** |

bf16 배수 **2.46배**. `mask=` + `is_causal=True` 조합이 flash 커널을 탄다.
