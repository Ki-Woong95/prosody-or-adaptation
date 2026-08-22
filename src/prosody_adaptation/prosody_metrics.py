from __future__ import annotations

import math

from .normalization import denormalize


def _masked_values(prediction, target, mask):
    mask = mask.bool()
    return prediction[mask].float(), target[mask].float()


def _regression(prediction, target, mask):
    prediction, target = _masked_values(prediction, target, mask)
    if not prediction.numel():
        return {"rmse": float("nan"), "mae": float("nan"), "correlation": float("nan")}
    error = prediction - target
    centered_p, centered_t = prediction - prediction.mean(), target - target.mean()
    denominator = centered_p.square().sum().sqrt() * centered_t.square().sum().sqrt()
    correlation = centered_p.mul(centered_t).sum() / denominator.clamp_min(1e-12)
    return {"rmse": float(error.square().mean().sqrt()), "mae": float(error.abs().mean()), "correlation": float(correlation)}


def validation_metrics(pred, target, valid, voiced, transitions, stats):
    raw_pred = {name: denormalize(pred[name], name, stats) for name in ("log_f0", "delta_f0", "energy", "tilt")}
    raw_target = {name: denormalize(target[name], name, stats) for name in ("log_f0", "delta_f0", "energy", "tilt")}
    result = {
        "f0": _regression(raw_pred["log_f0"], raw_target["log_f0"], valid & voiced),
        "delta_f0": _regression(raw_pred["delta_f0"], raw_target["delta_f0"], valid & transitions),
        "energy": _regression(raw_pred["energy"], raw_target["energy"], valid),
        "tilt": _regression(raw_pred["tilt"], raw_target["tilt"], valid),
    }
    p, t = _masked_values(raw_pred["log_f0"], raw_target["log_f0"], valid & voiced)
    result["f0"]["cents_mae"] = float((1200 / math.log(2) * (p - t).abs()).mean()) if p.numel() else float("nan")
    predicted_voiced = pred["voicing_logits"].sigmoid() >= 0.5
    truth = voiced.bool()
    mask = valid.bool()
    tp = int((predicted_voiced & truth & mask).sum())
    fp = int((predicted_voiced & ~truth & mask).sum())
    fn = int((~predicted_voiced & truth & mask).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    result["voicing"] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}
    result["counts"] = {
        "valid_total_frames": int(mask.sum()),
        "valid_voiced_frames": int((truth & mask).sum()),
        "valid_transition_frames": int((transitions.bool() & mask).sum()),
    }
    return result
