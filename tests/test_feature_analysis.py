import torch
import pytest

from prosody_adaptation.feature_analysis import (
    ShiftedDataset,
    _accumulate_layer_analysis,
    _finish_layer_analysis,
)


def test_shifted_dataset_is_deterministic_and_has_no_fixed_points():
    dataset = list(range(6))
    shifted = ShiftedDataset(dataset)
    assert len(shifted) == len(dataset)
    assert [shifted[index] for index in range(len(shifted))] == [3, 4, 5, 0, 1, 2]


def test_residual_statistics_exclude_padding_and_normalize_by_hidden_norm():
    totals = {
        "gate": None, "residual": None, "hidden": None,
        "relative_residual": None, "counts": None,
    }
    hidden = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])
    contribution = torch.tensor([[[0.0, 1.0], [0.0, 9.0]]])
    analysis = [{"gate": torch.tensor([[[0.25], [0.75]]]),
                 "residual_contribution": contribution}]
    _accumulate_layer_analysis(
        totals, analysis, (torch.zeros_like(hidden), hidden), torch.tensor([[True, False]])
    )
    result = _finish_layer_analysis(totals)
    assert result["valid_frames"] == 1
    assert result["layer_gate_mean"] == [0.25]
    assert result["layer_residual_norm_mean"] == [1.0]
    assert result["layer_hidden_norm_mean"] == [5.0]
    assert result["layer_relative_residual_mean"] == pytest.approx([0.2])
