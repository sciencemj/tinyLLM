# tinyLLM

TinyStories로 사전학습하고 DailyDialog로 SFT한 **29.6M 파라미터 영어 대화 모델**.
RTX 3060 Ti 한 대에서 사전학습 3.06시간 + SFT 7분.

가중치: [**tinyllm-29m-chat**](https://huggingface.co/sciencemj/tinyllm-29m-chat) (대화) ·
[**tinyllm-29m-tinystories**](https://huggingface.co/sciencemj/tinyllm-29m-tinystories) (사전학습만)

> 코드는 MIT. **SFT 체크포인트는 비상업 전용**입니다 — DailyDialog가 CC BY-NC-SA 4.0.
> 자세한 것은 [라이선스](#라이선스)와 [`MODEL_CARD.md`](MODEL_CARD.md).

```
you> Can you help me find a restaurant?
bot> Yes, of course. We have a good restaurant near the Golden Square.
you> Italian food would be great.
bot> You can find a restaurant there.
you> What time do they open?
bot> They open at 9:00.
```

## v2 — `LlamaForCausalLM` 113M (2026-11)

v1 다음 단계로 모델을 약 3.8배 키우고 구조를 현대 LLM에 맞췄다. FineWeb-Edu 150억 토큰으로
사전학습하고 smol-smoltalk + SODA로 SFT했다. GCP L4 한 대에서 사전학습 약 201시간, SFT 약 9시간.

가중치: [**tinyllm-113m-chat**](https://huggingface.co/sciencemj/tinyllm-113m-chat) (대화) ·
[**tinyllm-113m-base**](https://huggingface.co/sciencemj/tinyllm-113m-base) (사전학습만). 둘 다
Apache-2.0이고 `transformers`만으로 로드된다.

```
you> Hello! How are you?
bot> I'm good, thank you. I saw your name when you came over and I thought I'd come over and say hi.
you> What should I cook for dinner tonight?
bot> Well, since it's just dinner, I thought I'd make something special. Do you have any recommendations?
```

| | v1 | v2 |
|---|---|---|
| 구조 | GPT-2형, `torch.nn` 레이어 | `LlamaForCausalLM` (RoPE, SwiGLU, RMSNorm, GQA) |
| 파라미터 | 29.6M | 112.7M |
| 사전학습 | TinyStories 934M 토큰 | FineWeb-Edu 15.0B 토큰 |
| SFT | DailyDialog | smol-smoltalk + SODA 600M 토큰 |
| 도구 | `deeptool` | Lightning, Hydra, W&B |
| 하드웨어 | RTX 3060 Ti | GCP L4 (온디맨드) |

코드는 `src/tinyllm/v2/`, 설정은 `configs/`, GCP 실행은 [`docs/gcp-runbook.md`](docs/gcp-runbook.md).
결과는 [`docs/v2-pretrain.md`](docs/v2-pretrain.md)와 [`docs/v2-sft.md`](docs/v2-sft.md), 모델 카드는
[`MODEL_CARD_V2.md`](MODEL_CARD_V2.md).

```bash
uv run tinyllm-chat-v2 --model sciencemj/tinyllm-113m-chat   # 또는 내보낸 로컬 디렉터리
```

아래부터는 v1 설명이다.

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

# 대화 (터미널)
uv run tinyllm-chat --checkpoint out/sft_lr3e-4_e24_best.pt \
                    --tokenizer data/tokenizer.json

# 대화 (브라우저)
uv run streamlit run src/tinyllm/app.py

# 원격 학습 진행 상황
uv run python scripts/watch.py --follow
```

## 화면

```
uv run streamlit run src/tinyllm/app.py
```

사이드바에서 체크포인트를 고르고(사전학습 체크포인트를 고르면 말차례를 배우기 전
상태와 비교할 수 있다), 무작위성·후보 토큰 수를 조절한다.

화면 아래에 **모델이 실제로 읽는 글자**가 상시 노출된다. 채팅 말풍선은 우리가 만든
허구이고, 모델이 보는 것은 끊기지 않는 한 줄이다:

```
<|user|>Hi, how are you today?<|eot|><|assistant|>Fine, thanks.<|eot|><|assistant|>
```

끝에 `<|assistant|>`가 붙는 순간이 보이면 왜 모델이 이야기를 이어쓰지 않고 대답하는지
알 수 있다. 이것이 2단계 SFT가 가르친 전부다.

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
  app.py         대화 화면 (Streamlit)
  modeling.py    배포용 독립 정의 (deeptool 비의존)
scripts/
  bench.py       throughput·메모리·커널 실측
  watch.py       원격 학습 진행 상황
  export_hf.py   체크포인트 -> HF 업로드 폴더
tests/           성질 기반 테스트 83개
LICENSE          코드 (MIT)
MODEL_CARD.md    모델 카드 — 체크포인트별 이용 조건, 한계, 인용
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

## 가중치 공개

```bash
uv run python scripts/export_hf.py     # dist/ 에 업로드 폴더 두 개 생성
```

체크포인트마다 이용 조건이 다르므로 **저장소를 나눈다.** 한 저장소에 같이 올리면
저장소 전체가 가장 엄격한 조건(비상업)에 묶이는데, 사전학습 가중치는 그럴 이유가 없다.

| 폴더 | 체크포인트 | HF license 태그 |
|---|---|---|
| `dist/tinyllm-29m-tinystories` | 사전학습 | `cdla-sharing-1.0` |
| `dist/tinyllm-29m-chat` | SFT | `cc-by-nc-sa-4.0` |

각 폴더에 `model.safetensors`, `config.json`, `tokenizer.json`,
`modeling_tinyllm.py`, `README.md`(모델 카드)가 들어간다.

**`.pt` 가 아니라 safetensors 로 낸다.** `torch.save` 는 pickle 이라 로드할 때 임의
코드가 실행될 수 있고, 공개 배포에서 남이 내 pickle 을 믿어야 하는 상황을 만들 이유가
없다. 변환할 때 `head.weight` 를 뺀다 — tying 때문에 `tok.weight` 와 저장소를 공유하는데
safetensors 는 공유 저장소를 거부한다. `modeling_tinyllm.py` 의 생성자가 다시 묶는다.

**`modeling_tinyllm.py` 는 deeptool 을 요구하지 않는다.** 가중치 하나 열자고 학습용
라이브러리를 설치하게 만들 이유가 없다. 학습용 정의와 같은 모델인지는
`tests/test_modeling.py` 가 실제 체크포인트로 검증한다 — 변환 후 logits 가 원본과
비트 단위로 같아야 통과한다.

업로드:

```bash
uv run hf auth login
uv run hf upload sciencemj/tinyllm-29m-tinystories dist/tinyllm-29m-tinystories . --repo-type model
uv run hf upload sciencemj/tinyllm-29m-chat        dist/tinyllm-29m-chat        . --repo-type model
```

## 라이선스

**코드는 MIT** ([`LICENSE`](LICENSE)). 모델 가중치와 학습 데이터는 별개 조건이다.

| 대상 | 조건 |
|---|---|
| 이 저장소의 코드 | MIT |
| `pretrain_best.pt` | 제약 없음 — CDLA-Sharing-1.0 §3.5 |
| `sft_*_best.pt` | **비상업 + 출처표시 + 동일조건** — CC BY-NC-SA 4.0 |
| TinyStories 데이터 | CDLA-Sharing-1.0 |
| DailyDialog 데이터 | CC BY-NC-SA 4.0, "research purposes only" |
| v2 가중치 (`tinyllm-113m-chat`, `-base`) | Apache-2.0 |
| FineWeb-Edu 데이터 (v2) | ODC-By 1.0 + Common Crawl 이용약관 |
| smol-smoltalk / SODA 데이터 (v2) | Apache-2.0 / CC-BY-4.0 |

**SFT 체크포인트를 상업적으로 쓸 수 없다.** 유료 서비스, 광고가 붙은 데모, 사내 제품
어디에도 해당한다. 상업적 이용이 필요하면 사전학습 체크포인트를 그대로 두고 2단계
데이터만 허용적 라이선스로 교체하면 된다 — 사전학습 3시간은 재사용되고 SFT는 2.4분이라
교체 비용이 거의 없다.

사전학습 체크포인트가 자유로운 근거는 CDLA-Sharing-1.0 §3.5다: *"This Agreement imposes
no obligations or restrictions on Your Use or Publication of Results."* §1.11에서 Results는
데이터의 Computational Use로 얻은 산출물이고, 조건은 데이터의 de minimis 분량 이상을
포함하지 않는 것이다. 이 모델은 train/val 격차가 0.03이라 코퍼스를 외우고 있지 않다.

모델 가중치가 법적으로 학습 데이터의 2차적 저작물인지는 아직 정리되지 않은 쟁점이다.
위 구분은 각 라이선스의 문언과 데이터 제공자가 밝힌 의도를 따른 것이다.

의존성은 모두 허용적이다 — PyTorch·`tokenizers`·`datasets`·Streamlit은 Apache-2.0,
NumPy는 BSD-3 계열, [`deeptool`](https://github.com/sciencemj/deeptool)은 MIT.
화면이 쓰는 Lexend와 JetBrains Mono는 Google Fonts에서 런타임에 불러오므로 번들되지 않는다.

### 인용

```bibtex
@article{eldan2023tinystories,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}

@InProceedings{li2017dailydialog,
  author = {Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi},
  title = {DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
  booktitle = {Proceedings of The 8th International Joint Conference on Natural Language Processing (IJCNLP 2017)},
  year = {2017}
}
```
