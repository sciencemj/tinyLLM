"""대화 화면.

    uv run streamlit run src/tinyllm/app.py
"""

import html
import re
from pathlib import Path

import streamlit as st
import torch

from tinyllm.chat import render_prompt, reply
from tinyllm.sft import load_pretrained
from tinyllm.tokenizer import ASSISTANT, EOT, SPECIAL_TOKENS, USER, load_tokenizer

OUT = Path("out")
TOKENIZER = Path("data/tokenizer.json")

PALETTE = {
    "ink": "#1A1D29",
    "paper": "#E8EAE6",
    "rule": "#C3CBC0",
    "mark": "#C1462F",
    "dim": "#6B7169",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
  --ink: %(ink)s; --paper: %(paper)s; --rule: %(rule)s;
  --mark: %(mark)s; --dim: %(dim)s;
}

.stApp, section[data-testid="stSidebar"] { background: var(--paper); }

/* 테이프를 항상 바닥에 붙인다. sticky 만으로는 내용이 짧을 때 화면 중간에 떠서
   아래가 텅 빈다. flex 는 실제 자식이 있는 stVerticalBlock 에 걸어야 한다 --
   stMainBlockContainer 에 걸면 자식이 그 블록 하나뿐이라 아무 일도 안 일어난다. */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
  display: flex; flex-direction: column; min-height: calc(100vh - 11rem);
}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :last-child {
  margin-top: auto;
}
/* 상속으로 내려보낸다. [class*="css"] 같은 광범위 선택자를 쓰면 Streamlit 의
   Material 아이콘 폰트까지 덮어써서 아바타가 "face" 라는 글자로 나온다. */
.stApp, .stApp input, .stApp textarea, .stApp button {
  font-family: 'Lexend', system-ui, sans-serif;
  color: var(--ink);
}
[data-testid="stIconMaterial"], .material-icons, [class*="material-symbols"],
[data-testid^="stChatMessageAvatar"] span {
  font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}

/* 머리말 */
.masthead { padding: 0.5rem 0 1.25rem; }
.masthead h1 {
  font-family: 'Lexend', sans-serif; font-weight: 600; font-size: 2.1rem;
  letter-spacing: -0.03em; margin: 0; color: var(--ink);
}
.masthead p {
  margin: 0.35rem 0 0; font-weight: 300; font-size: 0.92rem;
  color: var(--dim); max-width: 46rem; line-height: 1.5;
}

/* 대화 */
[data-testid="stChatMessage"] {
  background: transparent; border-bottom: 1px solid var(--rule);
  border-radius: 0; padding: 0.9rem 0.2rem;
}
[data-testid="stChatMessage"] p { margin: 0; line-height: 1.55; }

/* Streamlit 기본 아바타는 선명한 빨강·주황이라 "빨강은 특수 토큰 전용" 규칙을
   침범한다. 조용한 글자 라벨로 바꾼다. */
[data-testid^="stChatMessageAvatar"] { display: none !important; }
[data-testid="stChatMessage"]::before {
  font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
  color: var(--dim); min-width: 5rem; flex: none; padding-top: 0.28rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])::before {
  content: "you";
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])::before {
  content: "tinyLLM";
}

/* 빈 화면 */
.invite {
  border-left: 2px solid var(--rule); padding: 0.1rem 0 0.1rem 1rem;
  margin: 1.5rem 0; color: var(--dim); font-weight: 300; line-height: 1.6;
  max-width: 34rem;
}

/* 시그니처: 테이프. 모델이 실제로 읽는 한 줄. */
.tape-dock {
  margin-top: auto; position: sticky; bottom: 0; z-index: 5;
  background: var(--paper); padding: 1.2rem 0 0.6rem;
}
.tape-label {
  font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
  letter-spacing: 0.08em; color: var(--dim); margin: 0 0 0.4rem;
}
.tape {
  font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
  line-height: 1.7rem; color: var(--ink);
  /* 합자를 끈다. <| 와 |> 가 다른 기호로 합쳐지면 "모델이 읽는 글자 그대로"가
     아니게 되고, 테이프의 존재 이유가 사라진다. */
  font-variant-ligatures: none; font-feature-settings: "liga" 0, "calt" 0;
  word-break: break-all; white-space: pre-wrap;
  padding: 0.85rem 1rem; max-height: 11rem; overflow-y: auto;
  /* 글씨 연습장 괘선. 이 화면에서 괘선을 쓰는 곳은 여기뿐이다. */
  background:
    repeating-linear-gradient(
      to bottom, transparent, transparent 1.65rem,
      #AEB8AB 1.65rem, #AEB8AB calc(1.65rem + 1px), transparent calc(1.65rem + 1px), transparent 1.7rem
    );
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
}
.tape .mark { color: var(--mark); font-weight: 700; }
.tape-count {
  font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
  color: var(--dim); text-align: right; margin-top: 0.35rem;
}

/* 사이드바를 조용하게 */
section[data-testid="stSidebar"] { border-right: 1px solid var(--rule); }
section[data-testid="stSidebar"] label {
  font-size: 0.78rem !important; color: var(--dim) !important;
  letter-spacing: 0.02em;
}
.side-head {
  font-family: 'Lexend', sans-serif; font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.02em; color: var(--dim); margin: 1.1rem 0 0.3rem;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--ink); }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
""" % PALETTE

_MARKERS = re.compile("|".join(re.escape(t) for t in SPECIAL_TOKENS))


def checkpoints(out_dir=OUT):
    """`out/` 의 체크포인트. SFT 를 먼저, 그 다음 사전학습."""
    found = sorted(p for p in Path(out_dir).glob("*.pt"))
    return sorted(found, key=lambda p: (not p.name.startswith("sft"), p.name))


def tape_html(prompt):
    """특수 토큰만 빨갛게. HTML 이스케이프를 먼저 해야 <|user|> 가 태그로 먹히지 않는다."""
    pieces = []
    cursor = 0
    for match in _MARKERS.finditer(prompt):
        pieces.append(html.escape(prompt[cursor:match.start()]))
        pieces.append(f'<span class="mark">{html.escape(match.group())}</span>')
        cursor = match.end()
    pieces.append(html.escape(prompt[cursor:]))
    return "".join(pieces)


@st.cache_resource(show_spinner="체크포인트 여는 중")
def load(checkpoint, tokenizer_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained(str(checkpoint), dropout=0.0,
                            device_type=device).to(device).eval()
    return model, load_tokenizer(tokenizer_path)


def main():
    st.set_page_config(page_title="tinyLLM", page_icon="✒", layout="wide")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

    available = checkpoints()
    if not available or not TOKENIZER.exists():
        st.markdown('<div class="masthead"><h1>tinyLLM</h1></div>',
                    unsafe_allow_html=True)
        st.error(
            f"체크포인트나 토크나이저가 없습니다. "
            f"`{OUT}/*.pt` 와 `{TOKENIZER}` 를 먼저 만드세요:\n\n"
            "```\nuv run python -m tinyllm.prepare --out data\n"
            "uv run python -m tinyllm.pretrain\n"
            "uv run python -m tinyllm.sft --lr 3e-4 --epochs 24 "
            "--tag sft_lr3e-4_e24\n```"
        )
        return

    with st.sidebar:
        st.markdown('<div class="side-head">체크포인트</div>',
                    unsafe_allow_html=True)
        checkpoint = st.selectbox("체크포인트", available, index=0,
                                  format_func=lambda p: p.stem,
                                  label_visibility="collapsed")
        st.caption("사전학습 체크포인트를 고르면 말차례를 배우기 전 상태와 비교할 수 "
                   "있습니다.")

        st.markdown('<div class="side-head">생성</div>', unsafe_allow_html=True)
        temperature = st.slider("무작위성", 0.1, 1.5, 0.6, 0.05)
        top_k = st.slider("후보 토큰 수", 1, 100, 20, 1)
        max_new = st.slider("최대 길이", 10, 200, 60, 10)
        turns = st.slider("기억할 발화 수", 2, 20, 6, 2)

        if st.button("대화 지우기", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    model, tokenizer = load(checkpoint, str(TOKENIZER))
    st.session_state.setdefault("history", [])
    history = st.session_state.history

    st.markdown(
        '<div class="masthead"><h1>tinyLLM</h1>'
        '<p>파라미터 29,577,728개. 동화 210만 편으로 영어를 배우고, '
        '일상 대화 11,118개로 말차례를 배웠습니다.</p></div>',
        unsafe_allow_html=True,
    )

    if not history:
        st.markdown(
            '<div class="invite">말을 걸어보세요. 짧고 일상적인 문장으로 답합니다 — '
            '그것 말고는 배운 적이 없습니다.<br>아래 테이프에 모델이 실제로 읽는 '
            '글자가 그대로 나타납니다.</div>',
            unsafe_allow_html=True,
        )

    for role, text in history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.write(text)

    if said := st.chat_input("메시지를 입력하세요"):
        history.append(("user", said))
        with st.chat_message("user"):
            st.write(said)
        with st.chat_message("assistant"):
            with st.spinner(""):
                answer = reply(model, tokenizer, history[-turns:],
                               max_new_tokens=max_new, temperature=temperature,
                               top_k=top_k)
            st.write(answer or "…")
        history.append(("assistant", answer))

    prompt = render_prompt(history[-turns:]) if history else f"{USER}…{EOT}{ASSISTANT}"
    st.markdown(
        '<div class="tape-dock">'
        '<div class="tape-label">모델이 읽는 글자</div>'
        f'<div class="tape">{tape_html(prompt)}</div>'
        f'<div class="tape-count">{len(tokenizer.encode(prompt).ids)} 토큰 · '
        f'최대 {model.block_size}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


main()
