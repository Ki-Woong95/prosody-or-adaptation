import io
import json
import wave
import zipfile

import numpy as np
import pytest

from prosody_adaptation.cache import TarCacheReader, materialize_tar_cache, validate_cache


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.arange(16000, dtype="<i2").tobytes())
    return output.getvalue()


def fixture(tmp_path):
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("s0101a.wav", wav_bytes())
    outer = tmp_path / "s01.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("s01/s0101a.zip", nested.getvalue())
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "speaker_id": "s01", "recording_id": "s0101a",
        "source_file": f"{outer.resolve()}::s01/s0101a.zip", "split": "train",
        "segment_id": "s0101a-0000", "start_time": 0.0, "end_time": 0.5,
        "duration": 0.5, "text": "hello", "sample_rate": 16000, "sample_count": 8000,
    }
    manifest.write_text(json.dumps(row) + "\n")
    return manifest


def test_deterministic_tar_cache_and_hash_validation(tmp_path):
    manifest = fixture(tmp_path)
    first, second = tmp_path / "cache1", tmp_path / "cache2"
    one = materialize_tar_cache(manifest, first, 1)
    two = materialize_tar_cache(manifest, second, 1)
    assert one["shard_sha256"] == two["shard_sha256"]
    assert validate_cache(manifest, first, verify_audio=True)["segments"] == 1
    assert len(TarCacheReader(manifest, first).read("s0101a-0000")) > 8000
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest hash"):
        validate_cache(manifest, first)
