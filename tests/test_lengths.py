import numpy as np
import torch

from prosody_adaptation.lengths import hubert_output_lengths, pad_waveforms
from prosody_adaptation.prosody import PackedBiGRU, phase1_losses


def test_true_per_utterance_ctc_lengths():
    batch = pad_waveforms([np.ones(16000), np.ones(8000)])
    assert batch.lengths.tolist() == [16000, 8000]
    assert batch.attention_mask.sum(1).tolist() == [16000, 8000]
    output = hubert_output_lengths(torch.tensor(batch.lengths))
    assert output[0] > output[1]


def test_packed_recurrent_padding_invariance():
    torch.manual_seed(3)
    model = PackedBiGRU(4, 3).eval()
    valid = torch.randn(1, 5, 4)
    padded = torch.cat((valid, torch.randn(1, 4, 4)), dim=1)
    short = model(valid, torch.tensor([5]))
    long = model(padded, torch.tensor([5]))
    torch.testing.assert_close(short[:, :5], long[:, :5])
    assert torch.count_nonzero(long[:, 5:]) == 0


def test_phase1_padding_excluded_from_every_loss():
    keys = ("log_f0", "delta_f0", "energy", "tilt")
    pred = {key: torch.zeros(1, 4) for key in keys}
    pred["voicing_logits"] = torch.zeros(1, 4)
    target = {key: torch.ones(1, 4) for key in keys}
    target["voicing"] = torch.ones(1, 4)
    valid = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    voiced = valid.clone()
    transitions = torch.tensor([[0, 1, 0, 0]], dtype=torch.bool)
    first = phase1_losses(pred, target, valid, voiced, transitions)
    for value in pred.values():
        value[:, 2:] = 1000
    second = phase1_losses(pred, target, valid, voiced, transitions)
    for key in first:
        torch.testing.assert_close(first[key], second[key])

