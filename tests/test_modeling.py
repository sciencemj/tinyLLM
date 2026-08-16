"""배포용 독립 정의가 학습용 정의와 같은 모델인지 검증한다.

두 파일에 아키텍처가 중복돼 있다. 배포본이 deeptool 을 요구하지 않게 하려는
의도적 중복이고, 드리프트는 여기서 잡는다.
"""

import sys

import pytest
import torch

from tinyllm.model import TinyLM as TrainingTinyLM
from tinyllm.modeling import TinyLM as ReleaseTinyLM

TINY = dict(vocab_size=64, d_model=32, n_head=4, n_layer=2, d_ff=64,
            block_size=16)


@pytest.fixture
def exported(tmp_path):
    """학습용 모델을 배포 폴더로 내보내고 (원본, 폴더) 를 준다."""
    sys.path.insert(0, "scripts")
    from export_hf import export

    torch.manual_seed(0)
    source = TrainingTinyLM(**TINY, dropout=0.0, device_type="cpu").eval()
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    torch.save({"model": source.state_dict(), "epoch": 1, "val_loss": 1.0},
               ckpt_dir / "pretrain_best.pt")

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    modeling = tmp_path / "modeling.py"
    modeling.write_text("# stub\n")

    release = dict(dir="rel", checkpoint="pretrain_best.pt",
                   license="cdla-sharing-1.0", title="t", stage="s",
                   datasets=["d"], val="v", terms="x")
    target, _ = export(release, tmp_path / "dist", ckpt_dir, tokenizer,
                       modeling, n_head=TINY["n_head"])
    return source, target


def test_exported_weights_produce_identical_logits(exported):
    """변환이 무손실인지. 여기가 깨지면 조용히 다른 모델을 배포하게 된다."""
    source, target = exported
    loaded = ReleaseTinyLM.from_pretrained(target)
    ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        assert torch.equal(source(ids), loaded(ids))


def test_exported_model_keeps_weight_tying(exported):
    """safetensors 가 head.weight 를 못 담으므로 생성자가 다시 묶어야 한다."""
    _, target = exported
    loaded = ReleaseTinyLM.from_pretrained(target)
    assert loaded.head.weight is loaded.tok.weight


def test_safetensors_file_drops_only_the_tied_duplicate(exported):
    from safetensors.torch import load_file

    source, target = exported
    saved = load_file(target / "model.safetensors")
    assert "head.weight" not in saved
    assert set(saved) == set(source.state_dict()) - {"head.weight"}


def test_config_matches_the_architecture(exported):
    import json

    _, target = exported
    config = json.loads((target / "config.json").read_text())
    for key, want in TINY.items():
        assert config[key] == want, key


def test_release_folder_has_everything_needed_to_load(exported):
    _, target = exported
    names = {f.name for f in target.iterdir()}
    assert names == {"model.safetensors", "config.json", "tokenizer.json",
                     "modeling_tinyllm.py", "README.md"}


def test_card_declares_the_license_in_frontmatter(exported):
    _, target = exported
    card = (target / "README.md").read_text()
    assert card.startswith("---\nlicense: cdla-sharing-1.0\n")
