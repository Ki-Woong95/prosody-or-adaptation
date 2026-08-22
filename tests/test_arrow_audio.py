import numpy as np
import pytest

from prosody_adaptation.asr_data import normalize_arrow_audio


def test_arrow_audio_accepts_mono_16khz_without_copying():
    waveform = np.arange(8, dtype=np.float32)
    observed = normalize_arrow_audio(waveform, 16000, "ami/train/0")
    assert observed is waveform


def test_arrow_audio_collapses_exactly_duplicated_stereo():
    mono = np.arange(8, dtype=np.float32)
    stereo = np.column_stack((mono, mono))
    observed = normalize_arrow_audio(stereo, 16000, "ami/train/1")
    np.testing.assert_array_equal(observed, mono)
    assert observed.ndim == 1


def test_arrow_audio_rejects_nonidentical_stereo():
    stereo = np.zeros((8, 2), dtype=np.float32)
    stereo[3, 1] = 1e-7
    with pytest.raises(ValueError, match="ami/train/2"):
        normalize_arrow_audio(stereo, 16000, "ami/train/2")


def test_arrow_audio_rejects_non_16khz():
    with pytest.raises(ValueError, match="ami/train/3"):
        normalize_arrow_audio(np.zeros(8, dtype=np.float32), 8000, "ami/train/3")
