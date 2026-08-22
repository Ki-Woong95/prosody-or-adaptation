from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

TARGETS = ("log_f0", "delta_f0", "energy", "tilt")


class RunningMoments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_squared = 0.0

    def update(self, values):
        values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        self.count += values.numel()
        self.total += values.sum().item()
        self.total_squared += (values * values).sum().item()

    def result(self):
        if not self.count:
            raise ValueError("Cannot normalize from zero frames")
        mean = self.total / self.count
        variance = max(self.total_squared / self.count - mean * mean, 1e-12)
        return {"mean": mean, "std": variance**0.5, "count": self.count}


def compute_training_statistics(batches, split_name: str = "train"):
    if split_name != "train":
        raise ValueError("Normalization statistics may only be computed from training data")
    moments = {name: RunningMoments() for name in TARGETS}
    for batch in batches:
        valid = batch["valid_mask"].bool()
        voiced = valid & batch["voiced_mask"].bool()
        transition = valid & batch["voiced_transition_mask"].bool()
        moments["log_f0"].update(batch["log_f0"][voiced])
        moments["delta_f0"].update(batch["delta_f0"][transition])
        moments["energy"].update(batch["energy"][valid])
        moments["tilt"].update(batch["tilt"][valid])
    return {"version": 1, "source_split": "train", "targets": {k: v.result() for k, v in moments.items()}}


def save_statistics(stats, path):
    payload = json.dumps(stats, indent=2, sort_keys=True) + "\n"
    Path(path).write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_targets(batch, stats):
    output = dict(batch)
    for name in TARGETS:
        item = stats["targets"][name]
        output[name] = (batch[name] - item["mean"]) / item["std"]
    return output


def denormalize(values, name, stats):
    item = stats["targets"][name]
    return values * item["std"] + item["mean"]

