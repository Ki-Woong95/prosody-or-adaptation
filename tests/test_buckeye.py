
import pytest

from prosody_adaptation.buckeye import ManifestRow, deterministic_splits, validate_rows


def row(speaker, split, segment, start=0.0, end=1.0):
    return ManifestRow(
        speaker, f"{speaker}01", "outer.zip::inner.zip", split, segment,
        start, end, end - start, "hello", 16000, round((end - start) * 16000),
    )


def test_deterministic_speaker_disjoint_splits():
    speakers = [f"s{i:02d}" for i in range(1, 41)]
    counts = {"train": 30, "validation": 5, "test": 5}
    first = deterministic_splits(speakers, counts, 7)
    second = deterministic_splits(list(reversed(speakers)), counts, 7)
    assert first == second
    assert not (set(first["train"]) & set(first["test"]))


def test_validation_fails_on_speaker_overlap():
    with pytest.raises(ValueError, match="Speaker overlap"):
        validate_rows([row("s01", "train", "a"), row("s01", "test", "b")])


def test_validation_fails_on_overlapping_intervals():
    with pytest.raises(ValueError, match="overlapping"):
        validate_rows([
            row("s01", "train", "a", 0, 1),
            row("s01", "train", "b", 0.5, 1.5),
        ])

