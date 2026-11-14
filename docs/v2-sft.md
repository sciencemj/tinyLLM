# v2 SFT 결과 (2026-11-14)

사전학습 `best.ckpt`(step 879,999)에서 가중치만 가져와 시작했다. 옵티마이저와 step은 0부터
다시 시작한다. 데이터는 smol-smoltalk + SODA(`data=chat`)이고, 사용자 턴을 포함한 모든
토큰에 손실을 준다. GCP 온디맨드 L4, 3런(계획 상한).

## LR 탐색 (각 2,000 step, 32.8M 토큰)

| 런 | LR | val loss (step 1999) | train loss (마지막 200 step 평균) |
|---|---|---|---|
| `sft-lr-low-1` | 3e-5 | 1.7931 | 2.0547 |
| `sft-lr-high-1` | **1e-4** | **1.6894** | 1.9837 |

두 런 모두 val이 train보다 낮아 과적합 징후가 없었다. 1e-4를 골랐다(커밋 `1ce7241`).
1e-4 런은 이미 대화 형식을 지키고 `<|eot|>`에서 멈췄다(repetition_rate 0.064).

## 최종 런 `sft-main-1`

| 항목 | 값 |
|---|---|
| 학습량 | 36,644 step × 16,384 tok = 600.4M 토큰 (chat train 1 epoch) |
| LR | 1e-4 → 1e-5 cosine, warmup 200, weight decay 0.1 |
| **best val loss** | **1.4069** (step 36,499, 마지막 검증이 최저) |
| train loss (마지막 500 step 평균) | 1.4952 |
| 샘플 repetition_rate / distinct_2 | 0.008 / 0.968 |
| 소요 | 11-14 01:59 → 11:03 KST, 약 9시간 |
| 배정 대기 | 00:43부터 a/b/c 존 L4 재고 부족, 두 바퀴째인 01:59에 a에서 배정 (대기 중 비용 없음) |

val loss: 1.689(2k) → 1.458(24.5k) → 1.416(32.5k) → 1.407(36.5k). 끝까지 내려가고 있었다.
`tok_s`가 두 번 3~5k로 찍힌 건 체크포인트 저장과 겹친 순간이고, 바로 다음 기록에서 21.5k로
돌아왔다.

## 샘플 (최종, temperature 0.8, top-k 40)

| 프롬프트 | 답 |
|---|---|
| Hi! | Hey, how are you doing? |
| Why is the sky blue? | 빛의 산란으로 맞게 설명 |
| Give me three tips for sleeping better. | 번호를 매긴 3개 |
| I feel sad today. → Yes, please. | "What's happened this week that's making you sad?" (앞 턴을 이어받음) |
| What is two plus two? | "Two plus two is a simple addition problem." (답을 안 함) |
| Can you tell me a short story about a cat? | "Sure, I can try." 하고 턴을 끝냄 |
| Write a short poem about the rain. | 시 형식은 맞고 내용은 엉뚱함 |

여러 턴 잡담은 자연스럽게 이어진다. 계산, 사실 질문, 긴 창작은 약하다. 설계의 비목표
그대로다.

## 공개

- [`sciencemj/tinyllm-113m-chat`](https://huggingface.co/sciencemj/tinyllm-113m-chat): 이 런의 best.
- [`sciencemj/tinyllm-113m-base`](https://huggingface.co/sciencemj/tinyllm-113m-base): 사전학습 best.
- 둘 다 `AutoModelForCausalLM`·`AutoTokenizer`로 `trust_remote_code` 없이 로드되고, 허브에서
  새로 받아 여러 턴 생성까지 확인했다. 모델 카드는 [`MODEL_CARD_V2.md`](../MODEL_CARD_V2.md).
- 가중치 라이선스는 Apache-2.0이다. 데이터는 모두 허용적 라이선스다(ODC-By, Apache-2.0,
  CC-BY-4.0).

## 메모

- 로컬 다운로드 중 `gcloud storage cp`가 분할 다운로드의 crc32c 검증 단계에서 5시간 멈췄다.
  `CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0`으로 분할을 끄자 2분 만에 끝났다.
- transformers 5에서 `apply_chat_template(..., return_tensors="pt")`는 텐서가 아니라
  `BatchEncoding`을 돌려준다. `return_dict=True`로 받아 `generate(**enc)`에 넘긴다.
  `tinyllm-chat-v2`는 직접 토큰화해서 영향이 없다.
