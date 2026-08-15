"""사전학습 설정의 throughput·메모리·커널 선택을 실측한다.

mlbox-wsl 에서 실행:
    ssh mlbox-wsl 'cd ~/tinyLLM && ~/.local/bin/uv run python scripts/bench.py'
"""

import argparse
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tinyllm.model import TinyLM


def run(model, batch_size, block_size, device, steps, amp):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, model.vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, model.vocab_size, (batch_size, block_size), device=device)

    for i in range(steps + 5):
        if i == 5:  # 워밍업 5 step 은 계측에서 뺀다
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = model.loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / steps, batch_size * block_size * steps / elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    device = "cuda"
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__)

    for amp in (False, True):
        torch.cuda.reset_peak_memory_stats()
        model = TinyLM(block_size=args.block_size, device_type=device).to(device)
        ms, toks = run(model, args.batch_size, args.block_size, device,
                       args.steps, amp)
        peak = torch.cuda.max_memory_allocated() / 1e9
        label = "bf16" if amp else "fp32"
        print(f"{label}: {ms * 1000:6.1f} ms/step  {toks / 1e3:7.1f}k tok/s  "
              f"peak {peak:.2f} GB")
        del model
        torch.cuda.empty_cache()

    # mask= + is_causal=True 조합이 flash 커널을 타는지 확인한다.
    # flash 만 허용한 상태에서 forward 가 통과하면 탄 것이다.
    model = TinyLM(block_size=args.block_size, device_type=device).to(device)
    x = torch.randint(0, model.vocab_size, (2, args.block_size), device=device)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION), \
             torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
        print("flash kernel: OK")
    except RuntimeError as exc:
        print(f"flash kernel: NOT USED -- {exc}")


if __name__ == "__main__":
    main()
