from __future__ import annotations

import torch
from torch import nn

from .lengths import hubert_output_lengths


class LengthAwareCTC(nn.Module):
    def __init__(self, blank_id: int = 0):
        super().__init__()
        self.loss = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

    def forward(self, logits, targets, waveform_lengths, target_lengths):
        frame_lengths = hubert_output_lengths(waveform_lengths).to(dtype=torch.long)
        if torch.any(frame_lengths > logits.shape[1]):
            raise ValueError("Computed CTC length exceeds logit sequence")
        log_probs = logits.float().log_softmax(-1).transpose(0, 1)
        return self.loss(log_probs, targets, frame_lengths, target_lengths), frame_lengths


def greedy_decode(logits, frame_lengths, blank_id: int = 0):
    tokens = logits.argmax(-1)
    output = []
    for row, length in zip(tokens, frame_lengths):
        decoded, previous = [], None
        for token in row[: int(length)].tolist():
            if token != blank_id and token != previous:
                decoded.append(token)
            previous = token
        output.append(decoded)
    return output

