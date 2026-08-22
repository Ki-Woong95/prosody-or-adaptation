import numpy as np
import pytest

from prosody_adaptation.inference import holm_adjust, paired_poisson_bootstrap


def test_paired_bootstrap_preserves_direction_and_reports_hierarchy():
    differences = np.full((3, 20), -1.0)
    references = np.full(20, 10.0)
    speakers = np.asarray(["s1"] * 10 + ["s2"] * 10)

    result = paired_poisson_bootstrap(
        differences, references, speakers, samples=500, seed=7, chunk_size=25
    )

    assert result["difference_wer_points"] == -10.0
    assert result["ci95_wer_points"][1] < 0
    assert result["speaker_count"] == 2
    assert result["p_value_two_sided"] < 0.01


def test_paired_bootstrap_falls_back_when_speakers_are_missing():
    differences = np.asarray([[-1, 1], [-1, 1], [-1, 1]], dtype=float)
    result = paired_poisson_bootstrap(
        differences,
        np.asarray([10, 10]),
        np.asarray(["speaker-metadata-unavailable"] * 2),
        samples=100,
        seed=3,
    )
    assert result["speaker_count"] is None
    assert result["method"] == "paired seed-utterance Poisson bootstrap"


def test_holm_adjustment_is_monotone_in_rank():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.2})
