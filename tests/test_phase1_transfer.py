import math

import pytest

from prosody_adaptation.phase1_transfer import aggregate_batch_metrics


def test_transfer_metrics_match_phase1_batch_aggregation():
    records = [
        {"metrics": {"loss_total": 2.0, "f0_rmse": 3.0, "counts_valid_total_frames": 5}},
        {"metrics": {"loss_total": 4.0, "f0_rmse": float("nan"), "counts_valid_total_frames": 7}},
    ]
    result = aggregate_batch_metrics(records)
    assert result["loss_total"] == 3.0
    assert result["f0_rmse"] == 3.0
    assert result["counts_valid_total_frames"] == 12.0


def test_transfer_metrics_reject_empty_or_inconsistent_records():
    with pytest.raises(ValueError, match="empty"):
        aggregate_batch_metrics([])
    with pytest.raises(ValueError, match="keys changed"):
        aggregate_batch_metrics([
            {"metrics": {"loss": 1.0}},
            {"metrics": {"other": 1.0}},
        ])


def test_transfer_metrics_preserve_all_nan_metric():
    result = aggregate_batch_metrics([
        {"metrics": {"correlation": float("nan")}},
        {"metrics": {"correlation": float("nan")}},
    ])
    assert math.isnan(result["correlation"])
