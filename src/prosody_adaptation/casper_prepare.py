from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .casper import dataset_hash
from .config import file_sha256
from .teacher import (
    CREPE_FRAME_OFFSET,
    HOP_LENGTH,
    SAMPLE_RATE,
    WINDOW_LENGTH,
    TeacherExtractor,
)

CHUNK_SAMPLES = 15 * SAMPLE_RATE
MIN_TAIL_SAMPLES = SAMPLE_RATE


def split_waveform(waveform: np.ndarray) -> list[np.ndarray]:
    chunks = [
        waveform[start : start + CHUNK_SAMPLES]
        for start in range(0, len(waveform) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES)
    ]
    tail_start = len(chunks) * CHUNK_SAMPLES
    if len(waveform) - tail_start >= MIN_TAIL_SAMPLES:
        chunks.append(waveform[tail_start:])
    return chunks


def _source_inventory(recordings: Path, paths: list[Path]) -> tuple[list[dict], str]:
    rows = [
        {
            "path": str(path.relative_to(recordings)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return rows, hashlib.sha256(payload).hexdigest()


def prepare_casper(recordings, output, device="cuda", version="v1"):
    import torch
    import torchaudio
    from datasets import Audio, Dataset, Features

    recordings = Path(recordings).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    paths = sorted(recordings.rglob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"No WAV files found under {recordings}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    inventory, inventory_hash = _source_inventory(recordings, paths)

    def examples():
        for path in paths:
            waveform, sample_rate = torchaudio.load(path)
            waveform = waveform.mean(0)
            if sample_rate != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
            for chunk in split_waveform(waveform.numpy().astype(np.float32)):
                yield {"audio": {"array": chunk, "sampling_rate": SAMPLE_RATE}}

    dataset = Dataset.from_generator(
        examples,
        features=Features({"audio": Audio(sampling_rate=SAMPLE_RATE)}),
    )
    extractor = TeacherExtractor(crepe_model="tiny", device=device)
    dataset = dataset.map(
        lambda row: extractor.extract(
            np.asarray(row["audio"]["array"], dtype=np.float32)
        ).as_columns(),
        load_from_cache_file=False,
        desc="Extracting teacher targets",
    )
    split = dataset.train_test_split(test_size=0.20, seed=42)
    output.mkdir(parents=True)
    train_path = output / f"train_prosody_{version}"
    validation_path = output / f"val_prosody_{version}"
    split["train"].save_to_disk(str(train_path))
    split["test"].save_to_disk(str(validation_path))
    train_hash, _ = dataset_hash(train_path)
    validation_hash, _ = dataset_hash(validation_path)
    metadata = {
        "version": version,
        "frame_grid": {
            "hop_length": HOP_LENGTH,
            "window_length": WINDOW_LENGTH,
            "mel_center": False,
            "crepe_pad": True,
            "crepe_frame_offset": CREPE_FRAME_OFFSET,
            "frame_centre_samples": "j * hop + window / 2",
            "note": (
                "Canonical student grid: teacher mel framing is bitwise identical to "
                "LogMelFrontend and CREPE is resampled onto it by timestamp."
            ),
        },
        "source_root": str(recordings),
        "source_file_count": len(paths),
        "source_inventory_sha256": inventory_hash,
        "source_files": inventory,
        "total_segments": len(dataset),
        "train_segments": len(split["train"]),
        "validation_segments": len(split["test"]),
        "train_dataset_hash": train_hash,
        "validation_dataset_hash": validation_hash,
        "split_seed": 42,
        "validation_fraction": 0.20,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
