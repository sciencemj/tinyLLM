import math

import torch
from torch import nn

from tinyllm.model import TinyLM

TINY = dict(vocab_size=64, d_model=32, n_head=4, n_layer=2, d_ff=64,
            block_size=16, dropout=0.0, device_type="cpu")


def test_output_shape():
    model = TinyLM(**TINY)
    ids = torch.randint(0, 64, (2, 16))
    assert model(ids).shape == (2, 16, 64)


def test_no_causal_leak():
    """마지막 토큰을 바꿔도 그 앞 위치의 logits 는 변하지 않아야 한다."""
    model = TinyLM(**TINY).eval()
    ids = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        before = model(ids)
        ids2 = ids.clone()
        ids2[0, -1] = (ids2[0, -1] + 1) % 64
        after = model(ids2)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-5)


def test_weight_tying_shares_storage():
    model = TinyLM(**TINY)
    assert model.head.weight is model.tok.weight


def test_weight_tying_survives_state_dict_roundtrip():
    model = TinyLM(**TINY)
    reloaded = TinyLM(**TINY)
    reloaded.load_state_dict(model.state_dict())
    assert reloaded.head.weight is reloaded.tok.weight


def test_every_parameter_gets_a_gradient():
    model = TinyLM(**TINY)
    ids = torch.randint(0, 64, (2, 16))
    model.loss(model(ids), ids).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_initial_loss_is_near_uniform():
    """초기화가 제대로면 첫 loss 가 ln(vocab_size) 근처여야 한다.

    타깃을 입력과 독립인 난수로 둔다. ids 자기 자신에 채점하면 weight tying 때문에
    잔차 스트림에 남은 입력 임베딩이 현재 토큰의 logit 을 밀어 올려, 초기화가
    정상이어도 loss 가 ln(vocab) 보다 뚜렷하게 낮게 나온다.
    """
    torch.manual_seed(0)
    model = TinyLM(**TINY).eval()
    ids = torch.randint(0, 64, (8, 16))
    targets = torch.randint(0, 64, (8, 16))
    with torch.no_grad():
        loss = model.loss(model(ids), targets).item()
    assert abs(loss - math.log(64)) < 0.1


def test_overfits_a_fixed_batch():
    torch.manual_seed(0)
    model = TinyLM(**TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ids = torch.randint(0, 64, (4, 16))
    for _ in range(300):
        loss = model.loss(model(ids), ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_residual_projections_are_scaled_down():
    """잔차 투영은 0.02 가 아니라 0.02/sqrt(2*n_layer) 로 초기화된다."""
    model = TinyLM(**TINY)
    expected = 0.02 / math.sqrt(2 * TINY["n_layer"])
    for name, p in model.named_parameters():
        if name.endswith("linear2.weight") or name.endswith("out_proj.weight"):
            assert p.std().item() < 0.02 * 0.9
            assert abs(p.std().item() - expected) < expected * 0.5


def test_parameter_count_at_full_size():
    model = TinyLM(device_type="cpu")
    assert sum(p.numel() for p in model.parameters()) == 29_577_728


def test_generate_extends_the_sequence():
    model = TinyLM(**TINY)
    ids = torch.randint(0, 64, (2, 4))
    assert model.generate(ids, 5, top_k=8).shape == (2, 9)


def test_generate_runs_to_length_without_a_stop_id():
    model = TinyLM(**TINY)
    ids = torch.randint(0, 64, (1, 4))
    out = model.generate(ids, 20, top_k=1, stop_id=None)
    assert out.shape == (1, 24)


def test_generate_stops_early_at_stop_id():
    """head 를 갈아 끼워 토큰 7 만 나오게 만든다 -- 첫 스텝에 끊겨야 한다.

    tying 된 head 의 가중치를 직접 건드리면 임베딩까지 같이 망가지므로,
    묶이지 않은 새 Linear 로 교체하고 bias 로만 승부를 낸다.
    """
    model = TinyLM(**TINY).eval()
    model.head = nn.Linear(TINY["d_model"], TINY["vocab_size"], bias=True)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[7] = 1e3
    ids = torch.randint(0, 64, (1, 4))
    out = model.generate(ids, 20, top_k=1, stop_id=7)
    assert out.shape == (1, 5)
