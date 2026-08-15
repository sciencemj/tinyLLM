from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

BOS = "<|endoftext|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
EOT = "<|eot|>"

# 순서가 곧 id 다. BpeTrainer 가 special_tokens 를 이 순서로 0 번부터 넣는다.
SPECIAL_TOKENS = [BOS, USER, ASSISTANT, EOT]
BOS_ID, USER_ID, ASSISTANT_ID, EOT_ID = 0, 1, 2, 3


def train_tokenizer(texts, vocab_size=8000, path=None):
    """byte-level BPE 를 학습한다. `texts` 는 문자열 이터러블.

    unk 토큰을 두지 않는 이유는 byte-level 이라 표현 못 하는 입력이 없기 때문이다.
    """
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(path))
    return tok


def load_tokenizer(path):
    return Tokenizer.from_file(str(path))
