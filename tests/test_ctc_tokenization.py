from types import SimpleNamespace

import pytest

from prosody_adaptation.asr_data import tokenize_ctc_target


class RecordingTokenizer:
    unk_token_id = 3

    def __init__(self, input_ids):
        self.input_ids = input_ids
        self.observed = None

    def __call__(self, text):
        self.observed = text
        return SimpleNamespace(input_ids=self.input_ids)


def test_ctc_targets_are_uppercased_for_pinned_tokenizer():
    tokenizer = RecordingTokenizer([10, 27, 17, 4, 13, 5, 7, 14, 22])
    assert tokenize_ctc_target(tokenizer, "i'm ready") == tokenizer.input_ids
    assert tokenizer.observed == "I'M READY"


def test_ctc_targets_reject_unknown_symbols():
    tokenizer = RecordingTokenizer([11, 3])
    with pytest.raises(ValueError, match="outside the CTC vocabulary"):
        tokenize_ctc_target(tokenizer, "h2")
