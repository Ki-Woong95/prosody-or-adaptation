from __future__ import annotations

import io
import wave

import numpy as np
import torch
from torch.utils.data import Dataset

from .lengths import (
    PROSODY_HOP_LENGTH,
    PROSODY_WINDOW_LENGTH,
    frame_mask,
    prosody_frame_count,
)

TARGET_COLUMNS = {
    "log_f0": "teacher_log_f0",
    "voiced_mask": "teacher_crepe_conf",
    "delta_f0": "teacher_delta_log_f0",
    "energy": "teacher_log_energy",
    "tilt": "teacher_spectral_tilt",
}


class CasperProsodyDataset(Dataset):
    def __init__(self, path):
        try:
            from datasets import Audio, load_from_disk
        except ImportError as error:
            raise ImportError("Install prosody-adaptation[train] to load CASPER datasets") from error
        self.dataset = load_from_disk(path)
        missing = {"audio", *TARGET_COLUMNS.values()} - set(self.dataset.column_names)
        if missing:
            raise ValueError(f"Missing CASPER Phase 1 columns: {sorted(missing)}")
        self.dataset = self.dataset.cast_column("audio", Audio(decode=False))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        audio = item["audio"]
        source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
        with wave.open(source, "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ValueError("CASPER cache must contain 16-bit PCM WAV")
            channels = wav.getnchannels()
            waveform = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
            if channels > 1:
                waveform = waveform.reshape(-1, channels).mean(axis=1)
            waveform /= 32768.0
        return {
            "waveform": waveform,
            **{name: np.asarray(item[column], dtype=np.float32) for name, column in TARGET_COLUMNS.items()},
        }


def _waveform_length(item) -> int:
    """Sample count of an item, whether or not its audio was decoded."""
    if "waveform" in item:
        return len(item["waveform"])
    return int(item["waveform_length"])


def _validated_frame_count(item) -> int:
    """Validate teacher-target widths against the canonical frame grid."""
    samples = _waveform_length(item)
    expected = prosody_frame_count(samples)
    observed = {name: len(item[name]) for name in TARGET_COLUMNS}
    wrong = {name: length for name, length in observed.items() if length != expected}
    if wrong:
        raise ValueError(f"canonical Phase-1 grid mismatch: expected {expected}, got {wrong}")
    return expected


def collate_phase1(items):
    if not items:
        raise ValueError("Empty Phase 1 batch")
    waveform_lengths = torch.tensor([_waveform_length(item) for item in items], dtype=torch.long)
    decoded = all("waveform" in item for item in items)
    waveforms = torch.zeros(len(items), int(waveform_lengths.max()) if decoded else 0)
    if decoded:
        for index, item in enumerate(items):
            waveforms[index, : waveform_lengths[index]] = torch.from_numpy(item["waveform"])
    target_lengths = torch.tensor(
        [_validated_frame_count(item) for item in items], dtype=torch.long
    )
    max_frames = int(target_lengths.max())
    batch = {"waveforms": waveforms, "waveform_lengths": waveform_lengths}
    for name in TARGET_COLUMNS:
        values = torch.zeros(len(items), max_frames)
        for index, item in enumerate(items):
            length = len(item[name])
            values[index, :length] = torch.from_numpy(item[name])
        batch[name] = values
    batch["valid_mask"] = frame_mask(target_lengths, max_frames)
    batch["voiced_mask"] = batch["voiced_mask"] > 0.5
    transitions = torch.zeros_like(batch["voiced_mask"])
    transitions[:, 1:] = batch["voiced_mask"][:, 1:] & batch["voiced_mask"][:, :-1]
    batch["voiced_transition_mask"] = transitions & batch["valid_mask"]
    return batch


def iter_target_batches(path, batch_size=128):
    """Stream teacher arrays without decoding CASPER audio for train-only statistics."""
    from datasets import load_from_disk

    dataset = load_from_disk(path).select_columns(list(TARGET_COLUMNS.values()))
    for start in range(0, len(dataset), batch_size):
        raw = dataset[start : min(start + batch_size, len(dataset))]
        items = []
        size = len(raw[next(iter(TARGET_COLUMNS.values()))])
        for index in range(size):
            item = {
                name: np.asarray(raw[column][index], dtype=np.float32)
                for name, column in TARGET_COLUMNS.items()
            }
            # Recover the waveform length implied by the canonical frame grid.
            frames = len(item["energy"])
            item["waveform_length"] = (
                (frames - 1) * PROSODY_HOP_LENGTH + PROSODY_WINDOW_LENGTH
            )
            items.append(item)
        yield collate_phase1(items)
