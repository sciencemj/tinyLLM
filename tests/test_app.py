from tinyllm.app import checkpoints, tape_html
from tinyllm.tokenizer import ASSISTANT, EOT, USER


def test_checkpoints_lists_sft_before_pretrain(tmp_path):
    for name in ("pretrain_best.pt", "sft_lr3e-4_e24_best.pt", "sft_lr1e-4_best.pt"):
        (tmp_path / name).touch()
    assert [p.name for p in checkpoints(tmp_path)] == [
        "sft_lr1e-4_best.pt", "sft_lr3e-4_e24_best.pt", "pretrain_best.pt",
    ]


def test_checkpoints_of_an_empty_directory_is_empty(tmp_path):
    assert checkpoints(tmp_path) == []


def test_tape_html_escapes_the_special_tokens():
    """이스케이프를 안 하면 <|user|> 가 태그로 먹혀 화면에서 사라진다."""
    out = tape_html(f"{USER}hi{EOT}")
    assert "&lt;|user|&gt;" in out
    assert "<|user|>" not in out


def test_tape_html_marks_only_the_special_tokens():
    out = tape_html(f"{USER}hi{EOT}{ASSISTANT}")
    assert out.count('<span class="mark">') == 3
    assert "</span>hi<span" in out   # 사람이 쓴 말은 마크 밖의 평문으로 남는다


def test_tape_html_escapes_user_text_too():
    out = tape_html(f"{USER}<script>alert(1)</script>{EOT}")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
