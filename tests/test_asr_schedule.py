import pytest

from prosody_adaptation.asr_train import accumulation_schedule


def test_accumulation_flushes_partial_final_group():
    schedule = [accumulation_schedule(index, 615, 4) for index in range(615)]
    assert sum(closes for _, closes in schedule) == 154
    assert schedule[-3:] == [(3, False), (3, False), (3, True)]


def test_accumulation_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="positive"):
        accumulation_schedule(0, 1, 0)
    with pytest.raises(ValueError, match="identify"):
        accumulation_schedule(1, 1, 4)
