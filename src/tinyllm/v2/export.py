"""Export a training checkpoint as a plain Hugging Face model + tokenizer directory."""

import argparse
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from tinyllm.tokenizer import ASSISTANT, BOS, EOT, USER
from tinyllm.v2.modeling import build_model
from tinyllm.v2.train import WEIGHTS_ONLY

# Same layout as samples.render_chat and prepare.render_messages: no BOS, text stripped, system
# (and any other non user/assistant) messages dropped, the generation prompt is the assistant marker.
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    f"{{% if m['role'] == 'user' %}}{USER}{{{{ m['content'] | trim }}}}{EOT}"
    f"{{% elif m['role'] == 'assistant' %}}{ASSISTANT}{{{{ m['content'] | trim }}}}{EOT}{{% endif %}}"
    "{% endfor %}"
    f"{{% if add_generation_prompt %}}{ASSISTANT}{{% endif %}}"
)


def export(ckpt: Path, tokenizer_json: Path, out: Path) -> None:
    state = torch.load(ckpt, map_location="cpu", weights_only=WEIGHTS_ONLY)
    model = build_model(state["hyper_parameters"]["model_cfg"])

    prefix = "model."
    weights = {k[len(prefix):]: v for k, v in state["state_dict"].items() if k.startswith(prefix)}
    leftover = [k for k in state["state_dict"] if not k.startswith(prefix)]
    if leftover:
        raise RuntimeError(f"checkpoint has weights outside the model: {leftover}")
    # strict: a missing or unexpected key is an error, never a silently dropped weight.
    model.load_state_dict(weights, strict=True)
    model.float()

    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json), eos_token=EOT, pad_token=BOS,
        additional_special_tokens=[USER, ASSISTANT], chat_template=CHAT_TEMPLATE,
    )
    if len(tok) > model.config.vocab_size:
        raise ValueError(f"tokenizer has {len(tok)} tokens but the model vocab is "
                         f"{model.config.vocab_size}")
    if (tok.pad_token_id, tok.eos_token_id) != (0, 3):
        raise ValueError("tokenizer special token ids do not match the model (pad 0, eos 3)")

    out = Path(out)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export a training checkpoint as a Hugging Face model.")
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True, type=Path, help="tokenizer.json")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    export(args.ckpt, args.tokenizer, args.out)


if __name__ == "__main__":
    main()
