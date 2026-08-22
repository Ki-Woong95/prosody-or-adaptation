from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .artifacts import append_jsonl, create_run_directory, initialize_run
from .casper import dataset_hash
from .checkpointing import restore_training_checkpoint, save_training_checkpoint
from .config import file_sha256, load_yaml
from .lengths import frame_mask
from .normalization import compute_training_statistics, normalize_targets, save_statistics
from .phase1_data import CasperProsodyDataset, collate_phase1, iter_target_batches
from .prosody import phase1_losses
from .prosody_metrics import validation_metrics
from .prosody_model import ProsodyEncoderWithHeads
from .runtime import runtime_metadata


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _targets(batch):
    return {name: batch[name] for name in ("log_f0", "delta_f0", "energy", "tilt")} | {"voicing": batch["voiced_mask"]}


def _align(batch, pred):
    """Validate and pair student predictions with teacher targets frame by frame."""
    width = pred["representation"].shape[1]
    if width != batch["valid_mask"].shape[1]:
        raise ValueError(
            f"Student produced {width} frames but the teacher batch has "
            f"{batch['valid_mask'].shape[1]}; the Phase-1 frame grids disagree"
        )
    target_lengths = batch["valid_mask"].sum(dim=1)
    if not torch.equal(pred["frame_lengths"].to(target_lengths.device), target_lengths):
        raise ValueError(
            "Per-item student frame lengths do not match the teacher target lengths"
        )
    valid = batch["valid_mask"] & frame_mask(pred["frame_lengths"], width).to(
        batch["valid_mask"].device
    )
    return valid, _targets(batch), batch["voiced_mask"], batch["voiced_transition_mask"]


@torch.no_grad()
def evaluate(model, loader, stats, device, mixed_precision=False):
    model.eval()
    loss_sums, batches = {}, 0
    metric_inputs = []
    for batch in loader:
        batch = _move(normalize_targets(batch, stats), device)
        with torch.autocast(device_type=device.type, enabled=mixed_precision):
            pred = model(batch["waveforms"], batch["waveform_lengths"])
            valid, target, voiced, transitions = _align(batch, pred)
            losses = phase1_losses(pred, target, valid, voiced, transitions)
        for key, value in losses.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value)
        metric_inputs.append(validation_metrics(pred, target, valid, voiced, transitions, stats))
        batches += 1
    metrics = {f"loss_{key}": value / max(batches, 1) for key, value in loss_sums.items()}
    if metric_inputs:
        for group in metric_inputs[0]:
            for key in metric_inputs[0][group]:
                values = [item[group][key] for item in metric_inputs if np.isfinite(item[group][key])]
                metrics[f"{group}_{key}"] = (
                    float(np.sum(values)) if group == "counts"
                    else float(np.mean(values)) if values else float("nan")
                )
    return metrics


def train_phase1(config_path):
    config = load_yaml(config_path)
    if config.get("phase1_corpus") != "casper" or "CASPER" not in config["train_data"] or "CASPER" not in config["validation_data"]:
        raise ValueError("Primary Phase 1 configuration must use CASPER-only train and validation data")
    observed_train_hash, _ = dataset_hash(config["train_data"])
    observed_validation_hash, _ = dataset_hash(config["validation_data"])
    if observed_train_hash != config["train_dataset_hash"]:
        raise ValueError("CASPER training dataset hash mismatch")
    if observed_validation_hash != config["validation_dataset_hash"]:
        raise ValueError("CASPER validation dataset hash mismatch")
    _seed_everything(int(config["seed"]))
    run_dir = create_run_directory(config["output_root"], config["experiment"])
    initialize_run(run_dir, config, config["train_data"] + "/state.json")
    train_dataset, validation_dataset = CasperProsodyDataset(config["train_data"]), CasperProsodyDataset(config["validation_data"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], collate_fn=collate_phase1)
    validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], collate_fn=collate_phase1)
    stats = compute_training_statistics(
        iter_target_batches(config["train_data"], config.get("normalization_batch_size", 256)),
        "train",
    )
    normalization_hash = save_statistics(stats, run_dir / "normalization.json")
    if normalization_hash != config["normalization_sha256"]:
        raise ValueError("CASPER training normalization hash mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProsodyEncoderWithHeads(**config["model"]).to(device)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = len(train_loader) * config["epochs"]
    warmup = round(total_steps * config["warmup_ratio"])
    def schedule(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=config["mixed_precision"] and device.type == "cuda")
    state = {"epoch": 0, "step": 0, "best_validation_loss": float("inf"), "patience": 0, "normalization_sha256": normalization_hash}
    if config.get("resume_checkpoint"):
        state, restored_stats = restore_training_checkpoint(config["resume_checkpoint"], model, optimizer, scheduler, scaler)
        if restored_stats != stats:
            raise ValueError("Resume checkpoint normalization differs from CASPER training statistics")
        previous_best = Path(config["resume_checkpoint"]).with_name("checkpoint_best.pt")
        if not previous_best.exists():
            raise FileNotFoundError("Resume requires checkpoint_best.pt beside checkpoint_latest.pt")
        shutil.copy2(previous_best, run_dir / "checkpoint_best.pt")
        previous_metrics = previous_best.with_name("metrics.json")
        if previous_metrics.exists():
            shutil.copy2(previous_metrics, run_dir / "metrics.json")
    log_path = run_dir / "training_log.jsonl"
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    runtime = runtime_metadata(config["train_data"] + "/state.json")
    runtime["parameter_counts"] = parameter_counts
    (run_dir / "runtime_metadata.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n"
    )
    log_every = int(config.get("log_every_steps", 50))
    for epoch in range(state["epoch"] + 1, config["epochs"] + 1):
        model.train()
        for batch in train_loader:
            batch = _move(normalize_targets(batch, stats), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                pred = model(batch["waveforms"], batch["waveform_lengths"])
                valid, target, voiced, transitions = _align(batch, pred)
                losses = phase1_losses(pred, target, valid, voiced, transitions)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            state["step"] += 1
            if state["step"] % log_every == 0:
                train_record = {
                    "event": "train", "epoch": epoch, "step": state["step"],
                    "global_step": state["step"],
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm": float(gradient_norm),
                    "valid_total_frames": int(valid.sum()),
                    "valid_voiced_frames": int((valid & voiced).sum()),
                    **{f"loss_{key}": float(value) for key, value in losses.items()},
                }
                append_jsonl(log_path, train_record)
        metrics = evaluate(model, validation_loader, stats, device, scaler.is_enabled())
        state["epoch"] = epoch
        improved = metrics["loss_total"] < state["best_validation_loss"]
        state["patience"] = 0 if improved else state["patience"] + 1
        if improved:
            state["best_validation_loss"] = metrics["loss_total"]
        validation_record = {
            "event": "validation", "epoch": epoch, "step": state["step"],
            "global_step": state["step"], **metrics,
        }
        append_jsonl(log_path, validation_record)
        save_training_checkpoint(run_dir / "checkpoint_latest.pt", model, optimizer, scheduler, scaler, state, stats)
        if improved:
            save_training_checkpoint(run_dir / "checkpoint_best.pt", model, optimizer, scheduler, scaler, state, stats)
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        if state["patience"] >= config["early_stopping_patience"]:
            break
    (run_dir / "checkpoint_best_sha256.txt").write_text(
        file_sha256(run_dir / "checkpoint_best.pt") + "\n"
    )
    return run_dir
