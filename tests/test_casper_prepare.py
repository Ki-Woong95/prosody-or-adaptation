import numpy as np

from prosody_adaptation.casper_prepare import CHUNK_SAMPLES, MIN_TAIL_SAMPLES, split_waveform


def test_casper_chunking_keeps_only_tails_of_at_least_one_second():
    waveform = np.arange(CHUNK_SAMPLES + MIN_TAIL_SAMPLES, dtype=np.float32)
    chunks = split_waveform(waveform)
    assert [len(chunk) for chunk in chunks] == [CHUNK_SAMPLES, MIN_TAIL_SAMPLES]

    short_tail = waveform[:-1]
    assert [len(chunk) for chunk in split_waveform(short_tail)] == [CHUNK_SAMPLES]
