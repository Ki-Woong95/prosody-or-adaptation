from __future__ import annotations

import gzip
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from .artifacts import append_jsonl, create_run_directory, initialize_run
from .asr_data import (
    TRANSCRIPT_NORMALIZATION_VERSION,
    make_asr_loaders,
)
from .asr_model import ProductionASR
from .checkpointing import restore_training_checkpoint, save_training_checkpoint
from .config import file_sha256, load_yaml
from .runtime import runtime_metadata


def _edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(
                previous[j - 1]
                if ref == hyp
                else 1 + min(previous[j], current[-1], previous[j - 1])
            )
        previous = current
    return previous[-1]


def _load_configs(experiment_path):
    experiment = load_yaml(experiment_path)
    model = load_yaml(experiment["model_config"])
    base = load_yaml(model["base_config"])
    return experiment, model, base


def _move(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def accumulation_schedule(batch_index, total_batches, accumulation):
    """Return the current group size and whether this microbatch closes the group."""
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if total_batches < 1 or not 0 <= batch_index < total_batches:
        raise ValueError("batch_index must identify a batch in the epoch")
    group_start = (batch_index // accumulation) * accumulation
    group_size = min(accumulation, total_batches - group_start)
    return group_size, batch_index + 1 == group_start + group_size


@torch.no_grad()
def evaluate(model, loader, processor, device):
    model.eval()
    predictions, edits, words, losses = [], 0, 0, []
    for batch in loader:
        batch = _move(batch, device)
        output = model(
            batch["input_values"],
            batch["attention_mask"],
            batch["waveform_lengths"],
            batch["targets"],
            batch["target_lengths"],
        )
        losses.append(float(output["loss"]))
        decoded = model.decode(output)
        for index, token_ids in enumerate(decoded):
            hypothesis = (
                processor.tokenizer.decode(token_ids, skip_special_tokens=True).lower().strip()
            )
            reference = batch["text"][index]
            error = _edit_distance(reference.split(), hypothesis.split())
            edits, words = edits + error, words + max(len(reference.split()), 1)
            predictions.append(
                {
                    "segment_id": batch["segment_id"][index],
                    "speaker_id": batch["speaker_id"][index],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "word_errors": error,
                    "reference_words": max(len(reference.split()), 1),
                }
            )
    return {
        "wer": edits / max(words, 1),
        "loss": float(np.mean(losses)),
        "word_errors": edits,
        "reference_words": words,
    }, predictions


def _speaker_metrics(predictions):
    totals = {}
    for row in predictions:
        item = totals.setdefault(row["speaker_id"], {"word_errors": 0, "reference_words": 0})
        item["word_errors"] += row["word_errors"]
        item["reference_words"] += row["reference_words"]
    return {
        speaker: {**counts, "wer": counts["word_errors"] / max(counts["reference_words"], 1)}
        for speaker, counts in sorted(totals.items())
    }


def train_asr(config_path):
    experiment, model_config, base = _load_configs(config_path)
    if experiment.get("manifest_status", "frozen") != "frozen":
        raise ValueError("Experiment requires a frozen data manifest")
    if experiment["transcript_normalization"] != TRANSCRIPT_NORMALIZATION_VERSION:
        raise ValueError(f"Paper runs require frozen policy {TRANSCRIPT_NORMALIZATION_VERSION}")
    if file_sha256(experiment["data_manifest"]) != experiment["data_manifest_sha256"]:
        raise ValueError("Data manifest SHA-256 mismatch")
    seed = int(experiment["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    from transformers import Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(
        base["processor_model"], revision=base["processor_revision"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, test_loader = make_asr_loaders(
        experiment, processor, device.type == "cuda"
    )
    model = ProductionASR(
        base,
        processor.tokenizer.vocab_size,
        model_config["condition"],
        experiment.get("prosody_checkpoint"),
        processor.tokenizer.pad_token_id,
    ).to(device)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=experiment["learning_rate"],
        weight_decay=experiment.get("weight_decay", 0.01),
        betas=tuple(experiment.get("adam_betas", (0.9, 0.98))),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(experiment.get("mixed_precision", False)) and device.type == "cuda"
    )
    run_dir = create_run_directory(experiment["output_root"], experiment["experiment"])
    initialize_run(
        run_dir,
        {"experiment": experiment, "model": model_config, "base": base},
        experiment["data_manifest"],
    )
    (run_dir / "parameter_counts.json").write_text(
        json.dumps(model.parameter_counts(), indent=2) + "\n"
    )
    state = {"epoch": 0, "step": 0, "best_validation_wer": float("inf"), "patience": 0}
    if experiment.get("resume_checkpoint"):
        state, normalization = restore_training_checkpoint(
            experiment["resume_checkpoint"], model, optimizer, None, scaler
        )
        if normalization is not None:
            raise ValueError("ASR checkpoints must not contain Phase 1 normalization state")
        previous_best = Path(experiment["resume_checkpoint"]).with_name("checkpoint_best.pt")
        if not previous_best.exists():
            raise FileNotFoundError("checkpoint_best.pt not found for resumed run")
        shutil.copy2(previous_best, run_dir / "checkpoint_best.pt")
    log_path = run_dir / "training_log.jsonl"
    max_batches = experiment.get("max_train_batches")
    accumulation = int(experiment.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    state.setdefault("processed_utterances", 0)
    state.setdefault("processed_frames", 0)
    state.setdefault("processed_audio_seconds", 0.0)
    runtime = runtime_metadata(
        experiment["data_manifest"],
        experiment.get("audio_cache"),
        experiment.get("prosody_checkpoint"),
    )
    (run_dir / "runtime_metadata.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n"
    )
    log_every = int(experiment.get("log_every_steps", 200))
    started = time.monotonic()
    starting_step = int(state["step"])
    batches_per_epoch = (
        min(len(train_loader), max_batches) if max_batches is not None else len(train_loader)
    )
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_planned_steps = experiment["epochs"] * optimizer_steps_per_epoch
    gate_analysis = bool(experiment.get("analysis", {}).get("gates", False))
    for epoch in range(state["epoch"] + 1, experiment["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum, epoch_optimizer_steps = 0.0, 0
        for batch_index, batch in enumerate(train_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                output = model(
                    batch["input_values"],
                    batch["attention_mask"],
                    batch["waveform_lengths"],
                    batch["targets"],
                    batch["target_lengths"],
                    return_analysis=gate_analysis,
                )
                if not torch.isfinite(output["loss"]):
                    raise FloatingPointError(
                        "Non-finite CTC loss at "
                        f"epoch={epoch}, batch={batch_index}, global_step={state['step']}"
                    )
                group_size, closes_group = accumulation_schedule(
                    batch_index, batches_per_epoch, accumulation
                )
                scaled_loss = output["loss"] / group_size
            scaler.scale(scaled_loss).backward()
            state["processed_utterances"] += int(batch["waveform_lengths"].numel())
            state["processed_frames"] += int(output["frame_lengths"].sum())
            state["processed_audio_seconds"] += float(batch["waveform_lengths"].sum()) / 16000
            if not closes_group:
                continue
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), experiment.get("gradient_clip", 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            epoch_loss_sum += float(output["loss"])
            epoch_optimizer_steps += 1
            if state["step"] % log_every == 0:
                elapsed = time.monotonic() - started
                completed_this_run = max(state["step"] - starting_step, 1)
                remaining_steps = max(total_planned_steps - state["step"], 0)
                record = {
                    "event": "train",
                    "epoch": epoch,
                    "step": state["step"],
                    "global_step": state["step"],
                    "ctc_loss": float(output["loss"]),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(gradient_norm),
                    "processed_utterances": state["processed_utterances"],
                    "processed_frames": state["processed_frames"],
                    "processed_audio_hours": state["processed_audio_seconds"] / 3600,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
                    if device.type == "cuda"
                    else 0,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": elapsed / completed_this_run * remaining_steps,
                }
                if "mean_gate_activation" in output:
                    record["mean_gate_activation"] = float(output["mean_gate_activation"])
                    record["mean_residual_norm"] = float(output["mean_residual_norm"])
                append_jsonl(log_path, record)
            evaluation_interval = int(experiment.get("eval_every_steps", 0) or 0)
            if evaluation_interval and state["step"] % evaluation_interval == 0:
                interval_metrics, _ = evaluate(model, validation_loader, processor, device)
                interval_record = {
                    "event": "validation_interval",
                    "epoch": epoch,
                    "step": state["step"],
                    "global_step": state["step"],
                    **interval_metrics,
                }
                append_jsonl(log_path, interval_record)
                model.train()
        metrics, predictions = evaluate(model, validation_loader, processor, device)
        state["epoch"] = epoch
        improved = metrics["wer"] < state["best_validation_wer"]
        if improved:
            state["best_validation_wer"], state["patience"] = metrics["wer"], 0
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            with (run_dir / "predictions.jsonl").open("w") as handle:
                for prediction in predictions:
                    handle.write(json.dumps(prediction, sort_keys=True) + "\n")
        else:
            state["patience"] += 1
        validation_record = {
            "event": "validation",
            "epoch": epoch,
            "step": state["step"],
            "global_step": state["step"],
            "train_loss": epoch_loss_sum / max(epoch_optimizer_steps, 1),
            "early_stopping_counter": state["patience"],
            **metrics,
        }
        append_jsonl(log_path, validation_record)
        if improved:
            save_training_checkpoint(
                run_dir / "checkpoint_best.pt", model, optimizer, None, scaler, state, None
            )
        save_training_checkpoint(
            run_dir / "checkpoint_latest.pt", model, optimizer, None, scaler, state, None
        )
        if epoch >= int(experiment.get("early_stopping_min_epochs", 0)) and state[
            "patience"
        ] >= experiment.get("early_stopping_patience", 10):
            break
    best = torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics, test_predictions = evaluate(model, test_loader, processor, device)
    validation_metrics, _ = evaluate(model, validation_loader, processor, device)
    final_metrics = {"validation": validation_metrics, "test": test_metrics}
    (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    with (run_dir / "predictions.jsonl").open("w") as handle:
        for prediction in test_predictions:
            handle.write(json.dumps(prediction, sort_keys=True) + "\n")
    (run_dir / "speaker_metrics.json").write_text(
        json.dumps(_speaker_metrics(test_predictions), indent=2, sort_keys=True) + "\n"
    )
    with (
        (run_dir / "predictions.jsonl").open("rb") as source,
        gzip.GzipFile(filename=run_dir / "predictions.jsonl.gz", mode="wb", mtime=0) as target,
    ):
        shutil.copyfileobj(source, target)
    (run_dir / "checkpoint_best_sha256.txt").write_text(
        file_sha256(run_dir / "checkpoint_best.pt") + "\n"
    )
    return run_dir
