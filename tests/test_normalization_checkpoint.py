import random

import numpy as np
import pytest
import torch

from prosody_adaptation.checkpointing import restore_training_checkpoint, save_training_checkpoint
from prosody_adaptation.normalization import compute_training_statistics


def batch(offset=0.0):
    return {
        "valid_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        "voiced_mask": torch.tensor([[1, 0, 0]], dtype=torch.bool),
        "voiced_transition_mask": torch.tensor([[0, 1, 0]], dtype=torch.bool),
        "log_f0": torch.tensor([[1.0 + offset, 9.0, 99.0]]),
        "delta_f0": torch.tensor([[9.0, 2.0 + offset, 99.0]]),
        "energy": torch.tensor([[3.0 + offset, 5.0 + offset, 99.0]]),
        "tilt": torch.tensor([[4.0 + offset, 6.0 + offset, 99.0]]),
    }


def test_statistics_train_only_and_padding_masked():
    stats = compute_training_statistics([batch()], "train")
    assert stats["targets"]["log_f0"]["mean"] == 1.0
    assert stats["targets"]["energy"]["mean"] == 4.0
    with pytest.raises(ValueError, match="training"):
        compute_training_statistics([batch(1000)], "validation")


def test_atomic_checkpoint_restores_optimizer_scheduler_scaler_rng(tmp_path):
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    value = model(torch.ones(1, 2)).sum()
    value.backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, model, optimizer, scheduler, None, {"epoch": 3, "step": 7}, {"x": 1})
    expected_random = (random.random(), np.random.rand(), torch.rand(1))
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, 1)
    state, stats = restore_training_checkpoint(path, restored, restored_optimizer, restored_scheduler)
    observed_random = (random.random(), np.random.rand(), torch.rand(1))
    assert state == {"epoch": 3, "step": 7} and stats == {"x": 1}
    assert expected_random[0] == observed_random[0]
    assert expected_random[1] == observed_random[1]
    torch.testing.assert_close(expected_random[2], observed_random[2])
    for expected, observed in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, observed)
