# 모델 카드 — tinyLLM

## 요약

29,577,728 파라미터 영어 대화 모델. TinyStories로 사전학습하고 DailyDialog로 SFT했다.
RTX 3060 Ti 한 대에서 사전학습 3.06시간 + SFT 2.4분.

`torch.nn` 내장 트랜스포머 레이어로 조립한 GPT-2 형태의 decoder-only 스택이다.
구조 근거와 실행법은 [`README.md`](README.md)에 있다.

## ⚠️ 체크포인트마다 이용 조건이 다르다

| 체크포인트 | 학습 데이터 | 이용 조건 |
|---|---|---|
| `pretrain_best.pt` | TinyStories only | 제약 없음 (CDLA-Sharing-1.0 §3.5) |
| `sft_*_best.pt` | + DailyDialog | **비상업 전용** (CC BY-NC-SA 4.0) |

**SFT 체크포인트는 상업적으로 쓸 수 없다.** DailyDialog가 CC BY-NC-SA 4.0이고
데이터셋 카드에 "Dataset provided for research purposes only"라고 적혀 있다. 유료
서비스, 광고가 붙은 데모, 사내 제품 어디에도 쓰면 안 된다. 재배포할 때는 출처를
표시하고 동일 조건으로 공개해야 한다.

사전학습 체크포인트는 다르다. CDLA-Sharing-1.0 §3.5가 명시한다 — *"This Agreement
imposes no obligations or restrictions on Your Use or Publication of Results."*
§1.11에서 Results는 데이터의 Computational Use로 얻은 산출물이며 조건은 데이터의
de minimis 분량 이상을 포함하지 않는 것이다. 이 모델은 train/val 격차가 0.03이라
코퍼스를 외우고 있지 않으므로 조건을 충족한다.

모델 가중치가 법적으로 학습 데이터의 2차적 저작물인지는 아직 정리되지 않은 쟁점이다.
위 구분은 각 라이선스의 문언과 데이터 제공자가 밝힌 의도를 따른 것이다.

**상업적 이용이 필요하면** 사전학습 체크포인트를 그대로 두고 2단계 데이터만 허용적
라이선스로 교체하면 된다. 사전학습은 재사용되고 SFT는 2.4분이므로 교체 비용이 거의 없다.

## 의도된 용도

교육과 연구. 소형 언어모델이 어디까지 되고 어디서 무너지는지 직접 확인하는 것.
2단계 학습(사전학습 → SFT)이 무엇을 바꾸는지 관찰하는 것.

`app.py`의 화면에서 사전학습 체크포인트와 SFT 체크포인트를 바꿔가며 같은 질문을
던지면 SFT가 가르친 것이 무엇인지 직접 보인다.

## 하면 안 되는 것

- **상업적 배포** (SFT 체크포인트)
- **사실 정보를 얻는 용도.** 아래 한계 참조 — 사실성이 없다
- **영어 이외의 언어.** 영어 코퍼스만 봤다
- **조언·판단이 필요한 대화.** 일상 잡담 코퍼스 11,118개가 본 전부다

## 학습 데이터

**TinyStories** ([`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories),
CDLA-Sharing-1.0) — 어휘를 약 3,000단어로 제한한 합성 동화. GPT-3.5와 GPT-4가
생성했다. train 2,119,719편 / validation 21,990편. 자체 8k BPE로 467,003,476 토큰.

**DailyDialog** ([`li2017dailydialog/daily_dialog`](https://huggingface.co/datasets/li2017dailydialog/daily_dialog),
CC BY-NC-SA 4.0) — 사람이 직접 쓴 일상 대화 13,118개 / 발화 102,979개.
train 11,118 / validation 1,000 / test 1,000. 1,597,581 토큰.

토크나이저는 두 코퍼스 합집합에서 학습했다(vocab 8,000). 2단계에서 분포 shift를
줄이기 위해서다.

## 결과

| | 사전학습 | SFT |
|---|---|---|
| best val loss | 1.3202 nats/token | 2.3975 |
| perplexity | 3.744 | — |
| bits/char | 0.4659 | — |
| eot 준수율 | — | 100% |
| 소요 | 3.06시간 | 2.4분 |

**per-token val loss는 토크나이저가 다르면 비교할 수 없다.** 토큰이 굵으면 개당
예측이 어려워 loss가 오르고 잘면 내려간다. 다른 모델과 비교하려면 토크나이저 무관
지표인 bits/char를 쓴다.

## 한계

이 규모에서 예상되는 그대로다.

**된다** — 문법, 구두점, 따옴표 대화 형식, 문단 나누기, 인물 이름 유지, 인과 연결,
대화 형식 유지, 종료 토큰, 직전 발화에 대한 반응.

**안 된다** —
- **턴 간 기억.** "영화 봤다"고 해놓고 다음 턴에 "식당 갔다"고 한다
- **질문에 대한 직접 답변.** "누구랑 갔어?"에 다른 얘기를 한다
- **사실성.** "You can get a 200 % discount." 같은 문장을 만든다
- **문장 안 반복.** "rings, necklaces, bracelets, and necklaces"
- **논리 일관성.** 이미 구운 케이크를 다시 오븐에 넣는다

생성 온도 0.6 / top-k 20이 주제 유지가 낫고, 1.0을 넘기면 헛소리가 늘어난다.

## 편향과 위험

DailyDialog는 사람이 쓴 일상 대화이므로 그 안의 사회적 편향을 그대로 학습했을 수
있다. 별도로 측정하지 않았다. TinyStories는 아동용으로 생성된 합성 데이터라 어휘와
소재가 좁다.

모델이 생성하는 사실 진술은 신뢰할 수 없다. 위 한계의 예시가 그것이다.

## 인용

TinyStories:

```bibtex
@article{eldan2023tinystories,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}
```

DailyDialog:

```bibtex
@InProceedings{li2017dailydialog,
  author = {Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi},
  title = {DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
  booktitle = {Proceedings of The 8th International Joint Conference on Natural Language Processing (IJCNLP 2017)},
  year = {2017}
}
```
