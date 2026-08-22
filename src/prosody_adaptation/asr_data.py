from __future__ import annotations

import io
import json
import re
import wave
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .buckeye import load_manifest
from .cache import TarCacheReader, _clip_wav, _read_nested_wav
from .config import file_sha256

TRANSCRIPT_NORMALIZATION_VERSION = "conservative_apostrophe_v1"


def normalize_transcript_v1(text):
    """Corpus-agnostic character mapping; no lexical or grammatical rewrites."""
    text = text.lower().replace("’", "'").replace("`", "'").replace("´", "'")
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text).split())


conservative_apostrophe = normalize_transcript_v1


def normalize_arrow_audio(waveform, sample_rate, location):
    """Accept mono 16 kHz audio and collapse only exactly duplicated stereo."""
    if sample_rate != 16000:
        raise ValueError(f"Expected mono 16 kHz audio: {location}")
    if waveform.ndim == 1:
        return waveform
    if (
        waveform.ndim == 2
        and waveform.shape[1] == 2
        and np.array_equal(waveform[:, 0], waveform[:, 1])
    ):
        return waveform[:, 0]
    raise ValueError(f"Expected mono 16 kHz audio: {location}")


def tokenize_ctc_target(tokenizer, text):
    """Encode normalized text in the uppercase alphabet of the pinned CTC tokenizer."""
    encoded = tokenizer(text.upper()).input_ids
    unk_token_id = tokenizer.unk_token_id
    if unk_token_id is not None and unk_token_id in encoded:
        raise ValueError(f"Transcript contains a symbol outside the CTC vocabulary: {text!r}")
    return encoded


def decode_wav(payload):
    with wave.open(io.BytesIO(payload), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("Expected mono 16-bit PCM")
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
        return samples.astype(np.float32) / 32768.0, source.getframerate()


class ManifestAudioDataset(Dataset):
    def __init__(self, manifest, split, cache=None, normalization="pending_approval"):
        if normalization != TRANSCRIPT_NORMALIZATION_VERSION:
            raise ValueError(
                f"Expected frozen transcript policy {TRANSCRIPT_NORMALIZATION_VERSION}"
            )
        self.rows = [row for row in load_manifest(manifest) if row.split == split]
        self.cache = TarCacheReader(manifest, cache) if cache else None
        self.source_cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.cache:
            payload = self.cache.read(row.segment_id)
        else:
            if row.source_file not in self.source_cache:
                self.source_cache = {row.source_file: _read_nested_wav(row.source_file)}
            payload = _clip_wav(self.source_cache[row.source_file], row.start_time, row.end_time)
        waveform, sample_rate = decode_wav(payload)
        if len(waveform) != row.sample_count:
            raise ValueError(f"Sample count mismatch: {row.segment_id}")
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "text": normalize_transcript_v1(row.text),
            "segment_id": row.segment_id,
            "speaker_id": row.speaker_id,
        }


class ArrowDiskAudioDataset(Dataset):
    """Read a frozen on-disk Arrow split without network access or audio copies."""

    def __init__(
        self,
        path,
        corpus,
        split,
        normalization="pending_approval",
        expected_rows=None,
        expected_state_sha256=None,
        excluded_rows=(),
        expected_usable_rows=None,
    ):
        if normalization != TRANSCRIPT_NORMALIZATION_VERSION:
            raise ValueError(
                f"Expected frozen transcript policy {TRANSCRIPT_NORMALIZATION_VERSION}"
            )
        from datasets import Audio, load_from_disk

        self.corpus, self.split = corpus, split
        self.dataset = load_from_disk(str(path), keep_in_memory=False).cast_column(
            "audio", Audio(decode=False)
        )
        if expected_rows is not None and len(self.dataset) != expected_rows:
            raise ValueError(f"Frozen Arrow row-count mismatch: {corpus}/{split}")
        state_path = Path(path) / "state.json"
        if expected_state_sha256 and file_sha256(state_path) != expected_state_sha256:
            raise ValueError(f"Frozen Arrow state hash mismatch: {corpus}/{split}")
        excluded_indices = set()
        for exclusion in excluded_rows:
            index = int(exclusion["index"])
            if index < 0 or index >= len(self.dataset):
                raise ValueError(f"Excluded Arrow row is out of range: {corpus}/{split}/{index}")
            observed_id = self.dataset[index].get("audio_id")
            if observed_id != exclusion["audio_id"]:
                raise ValueError(f"Excluded Arrow row ID mismatch: {corpus}/{split}/{index}")
            excluded_indices.add(index)
        self.indices = [
            index for index in range(len(self.dataset)) if index not in excluded_indices
        ]
        if expected_usable_rows is not None and len(self.indices) != expected_usable_rows:
            raise ValueError(f"Frozen usable row-count mismatch: {corpus}/{split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        import soundfile as sf

        row = self.dataset[source_index]
        audio = row["audio"]
        waveform, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        waveform = normalize_arrow_audio(
            waveform, sample_rate, f"{self.corpus}/{self.split}/{source_index}"
        )
        segment_id = (
            row.get("segment_id")
            or row.get("audio_id")
            or f"{self.corpus}-{self.split}-{index:07d}"
        )
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "text": normalize_transcript_v1(row["text"]),
            "segment_id": segment_id,
            "speaker_id": row.get("speaker_id") or "speaker-metadata-unavailable",
        }


class ASRCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, items):
        waveforms = [item["waveform"] for item in items]
        waveform_lengths = torch.tensor([len(item) for item in waveforms], dtype=torch.long)
        encoded = self.processor(
            waveforms,
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tokenized = [tokenize_ctc_target(self.processor.tokenizer, item["text"]) for item in items]
        target_lengths = torch.tensor([len(item) for item in tokenized], dtype=torch.long)
        targets = torch.zeros(len(items), int(target_lengths.max()), dtype=torch.long)
        for index, item in enumerate(tokenized):
            targets[index, : len(item)] = torch.tensor(item)
        return {
            "input_values": encoded.input_values,
            "attention_mask": encoded.attention_mask,
            "waveform_lengths": waveform_lengths,
            "targets": targets,
            "target_lengths": target_lengths,
            "text": [item["text"] for item in items],
            "segment_id": [item["segment_id"] for item in items],
            "speaker_id": [item["speaker_id"] for item in items],
        }


def load_asr_splits(experiment):
    """Build the train, validation, and test datasets from an experiment config."""
    manifest = Path(experiment["data_manifest"])
    normalization = experiment["transcript_normalization"]
    if experiment.get("data_backend", "manifest") != "arrow_disk":
        return tuple(
            ManifestAudioDataset(manifest, split, experiment.get("audio_cache"), normalization)
            for split in ("train", "validation", "test")
        )

    descriptor = json.loads(manifest.read_text())
    datasets = []
    for split in ("train", "validation", "test"):
        spec = descriptor["splits"][split]
        datasets.append(
            ArrowDiskAudioDataset(
                manifest.parent / spec["path"],
                descriptor["corpus"],
                split,
                normalization,
                spec["rows"],
                spec["state_sha256"],
                spec.get("excluded_rows", ()),
                spec.get("usable_rows"),
            )
        )
    return tuple(datasets)


def make_asr_loaders(experiment, processor, use_cuda):
    train, validation, test = load_asr_splits(experiment)
    options = {
        "batch_size": experiment["batch_size"],
        "num_workers": int(experiment.get("num_workers", 0)),
        "pin_memory": bool(experiment.get("pin_memory", use_cuda)),
        "collate_fn": ASRCollator(processor),
    }
    return (
        DataLoader(train, shuffle=True, **options),
        DataLoader(validation, shuffle=False, **options),
        DataLoader(test, shuffle=False, **options),
    )
