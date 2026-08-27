from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .asr_data import load_asr_splits
from .config import file_sha256
from .phase1_data import collate_phase1
from .phase1_train import evaluate
from .prosody_model import ProsodyEncoderWithHeads
from .teacher import TeacherExtractor


CORPORA = ("buckeye", "switchboard", "ami_ihm")


def _teacher_item(item, extractor):
    waveform = np.asarray(item["waveform"], dtype=np.float32)
    targets = extractor.extract(waveform)
    return {
        "waveform": waveform,
        "log_f0": targets.log_f0,
        "voiced_mask": targets.voiced,
        "delta_f0": targets.delta_log_f0,
        "energy": targets.log_energy,
        "tilt": targets.spectral_tilt,
    }


def aggregate_batch_metrics(records):
    if not records:
        raise ValueError("Cannot aggregate an empty Phase 1 evaluation")
    keys = records[0]["metrics"]
    if any(record["metrics"].keys() != keys.keys() for record in records):
        raise ValueError("Phase 1 metric keys changed between batches")
    output = {}
    for key in keys:
        values = [record["metrics"][key] for record in records]
        finite = [value for value in values if np.isfinite(value)]
        output[key] = (
            float(np.sum(finite)) if key.startswith("counts_")
            else float(np.mean(finite)) if finite else float("nan")
        )
    return output


def _load_progress(path):
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[(record["corpus"], record["batch_index"])] = record
    return records


def _append_progress(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def evaluate_phase1_transfer(
    registry_path,
    checkpoint_path,
    output_path,
    device="cuda",
    batch_size=32,
    corpus_filter=None,
    max_items=None,
):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use --device cpu only for a smoke test")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(registry_path).resolve().parents[2]
    registry = yaml.safe_load(Path(registry_path).read_text())
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    phase1_config = yaml.safe_load(
        (checkpoint_path.parent / "config.resolved.yaml").read_text()
    )
    model = ProsodyEncoderWithHeads(**phase1_config["model"])
    model.load_state_dict(payload["model"])
    torch_device = torch.device(device)
    model.to(torch_device).eval()
    stats = payload["normalization"]
    extractor = TeacherExtractor(crepe_model="tiny", device=device)
    output_path = Path(output_path)
    progress_path = output_path.with_suffix(".progress.jsonl")
    progress = _load_progress(progress_path)
    started = time.time()
    report = {
        "split": "validation",
        "teacher": "CREPE tiny; corrected canonical Phase-1 grid",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "normalization_sha256": payload["training_state"]["normalization_sha256"],
        "batch_size": batch_size,
        "max_items": max_items,
        "corpora": {},
    }
    selected = (corpus_filter,) if corpus_filter else CORPORA
    for corpus in selected:
        if corpus not in registry["corpora"]:
            raise ValueError(f"Corpus is absent from the paper registry: {corpus}")
        run = root / registry["corpora"][corpus]["full"].format(seed=1)
        resolved = yaml.safe_load((run / "config.resolved.yaml").read_text())
        experiment = resolved["experiment"]
        validation = load_asr_splits(experiment)[1]
        item_count = min(len(validation), max_items) if max_items else len(validation)
        batch_records = []
        for batch_index, start in enumerate(range(0, item_count, batch_size)):
            cached = progress.get((corpus, batch_index))
            expected_end = min(start + batch_size, item_count)
            if cached and cached["start"] == start and cached["end"] == expected_end:
                batch_records.append(cached)
                continue
            items = []
            segment_ids = []
            for index in range(start, expected_end):
                item = validation[index]
                try:
                    items.append(_teacher_item(item, extractor))
                except Exception as error:
                    raise RuntimeError(
                        f"Teacher extraction failed for {corpus} validation item "
                        f"{index} ({item['segment_id']})"
                    ) from error
                segment_ids.append(item["segment_id"])
            batch = collate_phase1(items)
            metrics = evaluate(
                model,
                [batch],
                stats,
                torch_device,
                mixed_precision=torch_device.type == "cuda",
            )
            record = {
                "corpus": corpus,
                "batch_index": batch_index,
                "start": start,
                "end": expected_end,
                "first_segment_id": segment_ids[0],
                "last_segment_id": segment_ids[-1],
                "metrics": metrics,
            }
            _append_progress(progress_path, record)
            progress[(corpus, batch_index)] = record
            batch_records.append(record)
        report["corpora"][corpus] = {
            "manifest": experiment["data_manifest"],
            "manifest_sha256": file_sha256(experiment["data_manifest"]),
            "validation_utterances": item_count,
            "batches": len(batch_records),
            "metrics": aggregate_batch_metrics(batch_records),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["elapsed_seconds"] = time.time() - started
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
