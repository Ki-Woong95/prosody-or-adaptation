from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PaddedWaveforms:
    values: np.ndarray
    attention_mask: np.ndarray
    lengths: np.ndarray


def pad_waveforms(waveforms: Sequence[np.ndarray]) -> PaddedWaveforms:
    if not waveforms:
        raise ValueError("Cannot collate an empty batch")
    lengths = np.asarray([len(w) for w in waveforms], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("Empty waveform")
    width = int(lengths.max())
    values = np.zeros((len(waveforms), width), dtype=np.float32)
    mask = np.zeros((len(waveforms), width), dtype=np.int64)
    for index, waveform in enumerate(waveforms):
        values[index, : lengths[index]] = np.asarray(waveform, dtype=np.float32)
        mask[index, : lengths[index]] = 1
    return PaddedWaveforms(values, mask, lengths)


PROSODY_HOP_LENGTH = 320
PROSODY_WINDOW_LENGTH = 800


def prosody_frame_count(waveform_length: int) -> int:
    """Number of non-centered 50 ms / 20 ms Phase-1 frames."""
    length = int(waveform_length)
    if length < PROSODY_WINDOW_LENGTH:
        raise ValueError(
            f"Waveform of {length} samples is shorter than the {PROSODY_WINDOW_LENGTH}-sample "
            "analysis window; no Phase-1 frame is defined"
        )
    return (length - PROSODY_WINDOW_LENGTH) // PROSODY_HOP_LENGTH + 1


def prosody_frame_centre_samples(frames: int) -> np.ndarray:
    """Sample index of the centre of each canonical Phase-1 frame."""
    return np.arange(frames, dtype=np.float64) * PROSODY_HOP_LENGTH + PROSODY_WINDOW_LENGTH / 2


def hubert_output_lengths(lengths):
    """Exact convolution lengths for HuBERT-base: kernels 10,3,3,3,3,2,2; strides 5,2..."""
    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        lengths = (lengths - kernel) // stride + 1
    return lengths


def frame_mask(lengths, max_frames: int | None = None):
    import torch

    lengths = torch.as_tensor(lengths, dtype=torch.long)
    width = int(lengths.max()) if max_frames is None else max_frames
    return torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]
