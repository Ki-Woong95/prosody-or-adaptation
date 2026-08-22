from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PackedBiGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, values, lengths):
        packed = pack_padded_sequence(
            values, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=values.shape[1]
        )
        return output


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def phase1_losses(pred, target, valid_mask, voiced_mask, voiced_transition_mask):
    valid = valid_mask.bool()
    voiced = valid & voiced_mask.bool()
    transitions = valid & voiced_transition_mask.bool()
    losses = {
        "f0": masked_mean((pred["log_f0"] - target["log_f0"]) ** 2, voiced),
        "voicing": masked_mean(
            F.binary_cross_entropy_with_logits(
                pred["voicing_logits"], target["voicing"].float(), reduction="none"
            ),
            valid,
        ),
        "delta_f0": masked_mean(
            (pred["delta_f0"] - target["delta_f0"]) ** 2, transitions
        ),
        "energy": masked_mean((pred["energy"] - target["energy"]) ** 2, valid),
        "tilt": masked_mean((pred["tilt"] - target["tilt"]) ** 2, valid),
    }
    losses["total"] = (
        losses["f0"] + losses["voicing"] + 0.5 * losses["delta_f0"]
        + 0.5 * losses["energy"] + 0.5 * losses["tilt"]
    )
    return losses

