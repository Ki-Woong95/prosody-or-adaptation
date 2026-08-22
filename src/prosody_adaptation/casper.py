from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import file_sha256
from .normalization import compute_training_statistics, save_statistics
from .phase1_data import iter_target_batches


def dataset_hash(path):
    path = Path(path)
    files = []
    for item in sorted(path.glob("*")):
        if item.is_file():
            files.append({"name": item.name, "size": item.stat().st_size, "sha256": file_sha256(item)})
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest(), files


def freeze_casper_metadata(train_path, validation_path, output_dir, batch_size=256):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_hash, train_files = dataset_hash(train_path)
    validation_hash, validation_files = dataset_hash(validation_path)
    stats = compute_training_statistics(iter_target_batches(train_path, batch_size), "train")
    normalization_hash = save_statistics(stats, output_dir / "normalization.json")
    metadata = {
        "phase1_corpus": "CASPER only",
        "train_path": Path(train_path).name,
        "validation_path": Path(validation_path).name,
        "train_dataset_hash": train_hash,
        "validation_dataset_hash": validation_hash,
        "normalization_sha256": normalization_hash,
        "normalization_source": "valid CASPER training frames only; F0 uses voiced frames",
        "train_files": train_files,
        "validation_files": validation_files,
    }
    (output_dir / "splits_and_hashes.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
