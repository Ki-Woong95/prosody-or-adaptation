from __future__ import annotations

import io
import json
import random
import re
import wave
import zipfile
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from .config import load_yaml, stable_hash

SPEAKER_RE = re.compile(r"^s\d{2}$")
WORD_RE = re.compile(r"^\s*([0-9.]+)\s+\d+\s+([^;]+);")
LEXICAL_SPECIAL = ("<EXT-", "<HES-", "<LAUGH-")
BOUNDARY_MARKERS = {"<IVER>", "<SIL>", "<NOISE>", "<VOCNOISE>", "{B_TRANS}"}


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    token: str


@dataclass(frozen=True)
class ManifestRow:
    speaker_id: str
    recording_id: str
    source_file: str
    split: str
    segment_id: str
    start_time: float
    end_time: float
    duration: float
    text: str
    sample_rate: int
    sample_count: int


def deterministic_splits(speakers: list[str], counts: dict[str, int], seed: int):
    ordered = sorted(speakers)
    random.Random(seed).shuffle(ordered)
    if sum(counts.values()) != len(ordered):
        raise ValueError("split_counts must sum to the number of speaker archives")
    result: dict[str, list[str]] = {}
    cursor = 0
    for split in ("train", "validation", "test"):
        count = int(counts[split])
        result[split] = sorted(ordered[cursor : cursor + count])
        cursor += count
    return result


def _normalized_token(token: str) -> str | None:
    token = token.strip()
    if token.startswith(LEXICAL_SPECIAL) and token.endswith(">"):
        token = token[token.index("-") + 1 : -1]
    if token.startswith(("<", "{")):
        return None
    token = token.replace("_", " ").strip().lower()
    return token or None


def parse_words(payload: bytes) -> tuple[list[Word], list[float]]:
    previous = 0.0
    words: list[Word] = []
    native_boundaries: list[float] = []
    after_header = False
    for line in payload.decode("latin-1").splitlines():
        if line.strip() == "#":
            after_header = True
            continue
        if not after_header:
            continue
        match = WORD_RE.match(line)
        if not match:
            continue
        end = float(match.group(1))
        raw = match.group(2).strip()
        if raw in BOUNDARY_MARKERS and (raw == "<IVER>" or end - previous >= 0.35):
            native_boundaries.append(end)
        token = _normalized_token(raw)
        if token:
            words.append(Word(previous, end, token))
        previous = end
    return words, native_boundaries


def _wav_info(payload: bytes) -> tuple[int, int, np.ndarray]:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Buckeye input must be mono 16-bit PCM")
        rate, count = wav.getframerate(), wav.getnframes()
        samples = np.frombuffer(wav.readframes(count), dtype="<i2").astype(np.float32)
    return rate, count, samples


def _quiet_boundary(samples, rate, target, left, right, window_seconds, quantile):
    window = max(1, int(rate * window_seconds))
    start = max(int(left * rate), window)
    stop = min(int(right * rate), len(samples) - window)
    if stop <= start:
        return target
    centers = np.arange(start, stop, window)
    rms = np.asarray(
        [np.sqrt(np.mean(samples[c - window : c + window] ** 2) + 1e-9) for c in centers]
    )
    threshold = np.quantile(rms, quantile)
    quiet = centers[rms <= threshold]
    if not len(quiet):
        quiet = centers
    return float(quiet[np.argmin(np.abs(quiet / rate - target))] / rate)


def acoustic_segments(samples, rate, duration, native_boundaries, config):
    maximum = float(config["max_duration_seconds"])
    search = float(config["silence_search_seconds"])
    candidates = sorted({0.0, duration, *(x for x in native_boundaries if 0 < x < duration)})
    output: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > maximum:
        limit = start + maximum
        native = [x for x in candidates if start + 0.25 < x <= limit]
        use_native = config.get("boundary_strategy", "annotation_then_silence") == "annotation_then_silence"
        if use_native and native and limit - native[-1] <= search:
            boundary = native[-1]
        else:
            boundary = _quiet_boundary(
                samples,
                rate,
                limit,
                max(start + 0.25, limit - search),
                min(duration, limit + search),
                float(config["silence_window_seconds"]),
                float(config["silence_quantile"]),
            )
        if boundary <= start:
            boundary = limit
        output.append((start, min(boundary, duration)))
        start = min(boundary, duration)
    if duration - start >= float(config["min_duration_seconds"]):
        output.append((start, duration))
    return output


def _text_for_interval(words: list[Word], start: float, end: float) -> str:
    return " ".join(word.token for word in words if word.end > start and word.end <= end).strip()


def _recordings(outer_path: Path):
    with zipfile.ZipFile(outer_path) as outer:
        for member in sorted(outer.namelist()):
            base = Path(member).name
            if not base.endswith(".zip") or base == f"{outer_path.stem}.zip":
                continue
            nested_bytes = outer.read(member)
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                recording = Path(base).stem
                wav_name, words_name = f"{recording}.wav", f"{recording}.words"
                names = set(nested.namelist())
                if wav_name not in names or words_name not in names:
                    raise ValueError(f"Missing WAV or words file in {outer_path.name}::{member}")
                yield recording, member, nested.read(wav_name), nested.read(words_name)


def prepare_buckeye(raw_archive: str | Path, output: str | Path, config_path: str | Path):
    raw_path, output_path = Path(raw_archive), Path(output)
    config = load_yaml(config_path)
    archives = sorted(raw_path.glob("s??.zip")) if raw_path.is_dir() else [raw_path]
    speakers = [path.stem for path in archives if SPEAKER_RE.match(path.stem)]
    if len(speakers) != len(archives):
        raise ValueError("Expected speaker archives named sNN.zip")
    splits = deterministic_splits(speakers, config["split_counts"], int(config["seed"]))
    speaker_to_split = {speaker: split for split, values in splits.items() for speaker in values}
    rows: list[ManifestRow] = []
    for archive in archives:
        speaker, split = archive.stem, speaker_to_split[archive.stem]
        for recording, nested, wav_bytes, words_bytes in _recordings(archive):
            rate, sample_count, samples = _wav_info(wav_bytes)
            if rate != int(config["expected_sample_rate"]):
                raise ValueError(f"Unexpected sample rate {rate}: {recording}")
            duration = sample_count / rate
            words, native = parse_words(words_bytes)
            if config.get("boundary_strategy") == "acoustic_silence_only":
                native = []
            intervals = acoustic_segments(samples, rate, duration, native, config)
            for index, (start, end) in enumerate(intervals):
                text = _text_for_interval(words, start, end)
                if not text and not config.get("allow_empty_transcript_markers", False):
                    continue
                start_sample, end_sample = round(start * rate), round(end * rate)
                rows.append(
                    ManifestRow(
                        speaker, recording, f"{archive.resolve()}::{nested}", split,
                        f"{recording}-{index:04d}", start_sample / rate, end_sample / rate,
                        (end_sample - start_sample) / rate, text, rate, end_sample - start_sample,
                    )
                )
    rows.sort(key=lambda row: (row.split, row.speaker_id, row.recording_id, row.start_time))
    validate_rows(rows)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = output_path / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    metadata = {
        "format_version": 1,
        "config": config,
        "splits": splits,
        "speaker_count": len(speakers),
        "segment_count": len(rows),
        "hours_by_split": {
            split: sum(row.duration for row in rows if row.split == split) / 3600
            for split in splits
        },
        "manifest_content_hash": stable_hash([asdict(row) for row in rows]),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_manifest(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        return [ManifestRow(**json.loads(line)) for line in handle if line.strip()]


def validate_rows(rows: list[ManifestRow]):
    if not rows:
        raise ValueError("Manifest is empty")
    speakers: dict[str, set[str]] = {}
    ids: set[str] = set()
    intervals: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        speakers.setdefault(row.split, set()).add(row.speaker_id)
        if row.segment_id in ids:
            raise ValueError(f"Duplicate segment ID: {row.segment_id}")
        ids.add(row.segment_id)
        if not 0 <= row.start_time < row.end_time or row.duration <= 0:
            raise ValueError(f"Invalid timestamps: {row.segment_id}")
        if not row.text.strip():
            raise ValueError(f"Unexpected empty transcript: {row.segment_id}")
        expected = round(row.duration * row.sample_rate)
        if abs(expected - row.sample_count) > 1:
            raise ValueError(f"Inconsistent sample count: {row.segment_id}")
        intervals.setdefault(row.recording_id, []).append((row.start_time, row.end_time))
    split_names = sorted(speakers)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = speakers[left] & speakers[right]
            if overlap:
                raise ValueError(f"Speaker overlap {left}/{right}: {sorted(overlap)}")
    for recording, spans in intervals.items():
        spans.sort()
        for previous, current in pairwise(spans):
            if current[0] < previous[1] - 1e-6:
                raise ValueError(f"Duplicate/overlapping intervals in {recording}")
    return {
        "segments": len(rows),
        "speakers": {split: len(values) for split, values in speakers.items()},
    }
