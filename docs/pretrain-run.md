# 1단계 사전학습 결과 (2026-08-16)

## 결과

| 항목 | 값 |
|---|---|
| **best val loss** | **1.3202** nats/token (청크 567 / 570) |
| perplexity | 3.744 |
| **bits/char** | **0.4659** — 토크나이저 무관 지표 |
| 총 소요 | **3.06시간** (570 청크 × 19.0초) |
| 총 학습량 | 57,000 step × 16,384 tok = 934M 토큰 (467M × 2회) |
| 최종 train loss | 1.2898 |
| 최종 LR | 6.00e-05 (= min_lr, cosine 완주) |

체크포인트 `out/pretrain_best.pt` (118MB), 히스토리 `out/runs/pretrain/`.

## 예측 대비

| | 추정 | 실측 | 차이 |
|---|---|---|---|
| 청크당 시간 | 19.5초 (벤치 174ms/step × 100 + val) | 19.0초 | -2.6% |
| 총 시간 | 2.95시간 | 3.06시간 | +3.7% |
| 총 step | 61,000 (500M 토큰 추정) | 57,000 (467M 실측) | -6.6% |

시간 추정이 4% 안에 들었다. 오차의 대부분은 토큰 수 추정에서 왔고, 처리량 추정은
거의 정확했다.

## 학습 곡선

**best가 570 청크 중 567번째.** 마지막 20청크의 val loss가 1.320~1.323으로 평평해졌을
뿐, 올라가지 않았다. 즉 **과적합 구간에 들어가지 않았다.** 직전 프로젝트
(tiny shakespeare, char-level)는 step 3500이 변곡점이었고 이후 val loss가 올라갔는데,
여기서는 그런 반전이 없다. 코퍼스가 467M 토큰으로 충분히 크고 2회만 통과했기 때문이다.

예산을 늘리면 더 내려갈 여지가 있다는 뜻이기도 하다. 다만 마지막 구간의 기울기가
거의 0이므로, 같은 설정으로 3회차를 돌려서 얻는 이득은 작을 것이다. 더 내리려면
모델을 키우거나 코퍼스를 넓혀야 한다.

## 생성물 (온도 0.8, top-k 40, 120 토큰)

> **Once upon a time, there was a little girl named Lily.**
> She loved to play outside with her friends. One day, they found a big box and
> decided to play with it. They took turns to pick up things and put them in the
> box. Lily got a red apple, her friend got a blue one, and her friend got a blue
> one. They laughed and played some more.
>
> After a while, it started to rain. Lily and her friends ran under a big tree to
> stay dry. They waited and waited, but the rain didn't stop. Finally, they had an
> idea. They decided to make a rainbow with all the colors in

> **Tom and Sara found a big red box in the garden.**
> They opened it and saw a lot of shiny things inside. There were rings, necklaces,
> bracelets, and necklaces. They put on the jewelry and felt very fancy.
>
> "Wow, we look like rich too!" Sara said.
>
> "Let's keep some of these," Tom said.
>
> They took the jewelry out of the box and put them on. They felt very happy and
> silly. They decided to take some more and show their mom and dad.
>
> But then, they heard a noise behind them. It was their mom and dad. They looked
> very angry and worried

**되는 것**: 문법, 구두점, 따옴표 대화 형식, 문단 나누기, 인물 이름 유지,
인과 연결("The cat got scared and ran away. The dog was sad because he wanted to
play with the cat").

**안 되는 것**: 문장 안 반복("her friend got a blue one, and her friend got a blue
one", "rings, necklaces, bracelets, and necklaces"), 논리 미끄러짐("the cat was not
far away" 직후 "he got more lost", 이미 구운 케이크를 다시 "put it in the oven").
TinyStories 논문이 이 규모에서 보고한 실패 양상과 같다.

## 자동 위생지표

| 지표 | 값 | 판정 |
|---|---|---|
| repetition_rate (3-gram) | 0.110 | 정상 (퇴화 기준 0.5 한참 아래) |
| distinct_2 | 0.673 | 정상 |

생성물 전문은 `out/samples.txt`.

## 공개 기준선 (spec §8의 4)

**직접 비교 가능한 기준선을 찾지 못했다.** per-token val loss는 토크나이저가 다르면
비교할 수 없기 때문이다. 토큰이 굵으면 개당 예측이 어려워 loss가 올라가고, 잘면
내려간다. 우리는 자체 8k BPE(4.088 chars/token)를 쓰고, 공개 수치들은 GPT-Neo 10k 등
다른 토크나이저를 쓴다.

참고로 찾은 수치:
- RTS Labs, nanoGPT 계열 10.8M 파라미터, TinyStories val loss 1.30
  (https://rtslabs.com/experiment-we-built-a-tiny-llm-from-scratch)
- 일부 논문이 1.078~1.113 nats/token 보고 (설정 상이)

우리 1.3202를 이 숫자들 옆에 나란히 놓는 것은 의미가 없다. 그래서 토크나이저 무관
지표인 **bits/char = 0.4659**를 함께 기록한다. 이후 비교는 이 값으로 해야 한다.

내부 비교로는 층 수(6 / 8 / 10)를 같은 파라미터 예산에서 짧은 런으로 붙여볼 수 있다.
필수는 아니다.

## 운영 중 발견한 결함

**`log_dir`을 런마다 나누지 않으면 히스토리가 섞인다.** 스모크 런과 본 런이 같은
`out/runs`를 써서 `history.jsonl` 앞에 스모크 2줄이 남았다. 더 심각한 것은
`load_runs(root)`가 `root/<런>/history.jsonl`만 찾는다는 점 — 파일이 root 바로 밑에
있으면 아무것도 못 읽는다. 2단계 SFT 스윕의 런 비교가 여기에 달려 있었다.

`pretrain.py`/`sft.py`가 `out/runs/<tag>`를 쓰도록 고치고 테스트로 못 박았다
(`test_each_run_gets_its_own_log_directory`). 이번 런의 기록은 사후에
`out/runs/pretrain/`으로 옮기고 스모크 2줄을 제거했다.
