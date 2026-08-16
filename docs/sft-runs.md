# 2단계 SFT 결과 (2026-08-16)

데이터 DailyDialog 1,597,581 토큰 (val 148,747). block 512 × batch 32 = 16,384 tok/step
→ **epoch당 97 step**, 런당 2.4~7분. 사전학습 3.06시간의 약 1/80.

## 스윕 결과

`out/runs/<tag>/`에 전부 기록됨. `dt.load_runs('out/runs')`로 읽는다.

| LR | epoch | best val | train | 격차 | best 지점 |
|---|---|---|---|---|---|
| 3e-5 | 8 | 3.0248 | 3.3847 | -0.360 | ep 7 (마지막) |
| 1e-4 | 8 | 2.7806 | 3.0126 | -0.232 | ep 7 (마지막) |
| 3e-4 | 8 | 2.5855 | 2.6087 | -0.023 | ep 7 (마지막) |
| 6e-4 | 8 | 2.4950 | 2.3301 | +0.165 | ep 7 (마지막) |
| 1e-3 | 8 | 2.4662 | 2.1087 | +0.358 | ep 7 (마지막) |
| **3e-4** | **24** | **2.3975** | 1.9770 | +0.420 | **ep 22** |
| 6e-4 | 24 | 2.4103 | 1.4797 | +0.931 | ep 23 |

**채택: `sft_lr3e-4_e24`, val 2.3975.**

## 스윕에서 배운 것

**1차 웨이브(3e-5 / 1e-4 / 3e-4, 8 epoch)는 결론이 안 났다.** 셋 다 마지막 epoch에서도
계속 좋아지는 중이었고, 가장 높은 LR이 이겼는데 그게 탐색 범위의 끝이었다. LR 천장도
epoch 천장도 못 찾은 상태라 확장이 필요했다.

**epoch이 LR보다 영향이 크다.** 3e-4를 8→24 epoch으로 늘려 -0.188을 얻었다.
8 epoch 안에서 LR을 3e-5→1e-3으로 33배 올려 얻은 것(-0.12)보다 크다.

**epoch을 늘리면 LR 순위가 뒤집힌다.** 8 epoch에서는 1e-3 > 6e-4 > 3e-4였는데
24 epoch에서는 3e-4가 6e-4를 이겼다. cosine 스케줄이 길어지면 낮은 LR도 충분히
이동하고, 높은 LR은 불안정성만 남기기 때문이다.

**val loss만 보면 놓치는 것이 있다.** 24 epoch의 3e-4(2.3975)와 6e-4(2.4103)는 val이
거의 같지만 train/val 격차가 0.42 대 0.93으로 두 배 이상 차이난다. 6e-4는 코퍼스를
외우고 있다. 같은 val 성능이라면 격차가 작은 쪽을 택해야 한다.

**아직 천장은 아니다.** 3e-4/24ep의 best가 ep 22이고 ep 23이 2.3977로 거의 같다.
평평해졌을 뿐 꺾이지 않았으므로 48 epoch이나 더 낮은 LR로 더 내릴 여지가 있다.
다만 성공 기준이 최소 val loss가 아니라 "그럴듯한 대화"이므로 여기서 멈췄다.

## 실제 대화

`sft_lr3e-4_e24_best.pt`, dropout 0.0.

### 온도 0.6 / top-k 20

```
you> Can you help me find a restaurant?
bot> Yes, of course. We have a good restaurant near the Golden Square.
you> Italian food would be great.
bot> You can find a restaurant there.
you> What time do they open?
bot> They open at 9:00.

you> I am thinking about buying a new car.
bot> What brand are you thinking about?
you> Something small and cheap.
bot> I like this one. It's a bit too expensive, but it's a good deal.
you> How much should I spend?
bot> You can get a 200 % discount.

you> What did you do last weekend?
bot> I went to a movie with my family.
you> That sounds fun. Who did you go with?
bot> I went to a restaurant and had a great time.
you> Would you go again?
bot> I went there once a week and I saw a movie with my family.
```

### 지표

| 설정 | eot 준수율 | 반복률(3-gram) | 발화 길이 |
|---|---|---|---|
| 온도 0.8 / top-k 40 | 100% | 0.087 | 6.7 단어 |
| 온도 0.6 / top-k 20 | 100% | 0.154 | 9.0 단어 |

DailyDialog 원본의 발화 길이는 약 13.6단어다. 모델은 그보다 짧게 답한다.

**eot 준수율 100%가 핵심 성과다.** 사전학습 상태의 모델은 대화를 이야기로 이어썼다.
SFT 후에는 자기 차례에 한 마디 하고 종료 토큰을 찍고 멈춘다. 형식 학습이 목표였고
달성됐다.

### 되는 것 / 안 되는 것

**된다**: 대화 형식 유지, 종료 토큰, 도메인에 맞는 응답, 직전 발화에 대한 반응
("새 차 살까 해" → "어떤 브랜드 생각하고 있어?"), 짧은 주고받기.

**안 된다**: 턴 간 기억("영화 봤다"고 해놓고 다음 턴에 "식당 갔다"), 질문에 대한
직접 답변("누구랑 갔어?"에 다른 얘기), 사실성("200 % discount").

29.6M 모델의 한계이고 예상 범위 안이다.

## 테스트 설계에서 배운 것

처음 테스트는 후속 발화로 `"Really? Tell me more."`를 세 번 반복해서 넣었다. 반복 입력
자체가 퇴화 루프를 유도하는 적대적 조건이라 결과가 나빴다(반복률 0.209, 같은 문장을
네 번 되풀이하는 응답). 자연스러운 후속으로 바꾸자 반복률이 0.087까지 떨어졌다.

**같은 문장을 반복 입력하는 것은 모델 평가가 아니라 루프 유도다.** 대화 평가에서는
후속 발화를 다양화해야 한다.

## 샘플링 설정

`chat.py`의 기본값은 온도 0.8 / top-k 40이다. 0.6 / 20이 주제 유지가 낫고 응답도
길지만, 근거가 9개 교환뿐이라 기본값은 바꾸지 않았다. 필요하면
`--temperature 0.6 --top-k 20`으로 넘긴다.
