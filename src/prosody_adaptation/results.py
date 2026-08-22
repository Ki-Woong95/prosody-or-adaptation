from __future__ import annotations

import json
import statistics
from pathlib import Path

import yaml


CONDITIONS = ("ab1", "ab3", "full")
SEEDS = (1, 2, 3)
RUN_FILES = (
    "checkpoint_best.pt",
    "checkpoint_latest.pt",
    "config.resolved.yaml",
    "dataset_manifest_hash.txt",
    "git_commit.txt",
    "metrics.json",
    "parameter_counts.json",
    "predictions.jsonl",
    "speaker_metrics.json",
    "training_log.jsonl",
)


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_run(path, seed):
    path = Path(path)
    missing = [name for name in RUN_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete run {path}: missing {', '.join(missing)}")
    metrics = _read_json(path / "metrics.json")
    return {
        "seed": seed,
        "path": str(path),
        "validation": metrics["validation"],
        "test": metrics["test"],
        "parameter_counts": _read_json(path / "parameter_counts.json"),
        "git_commit": (path / "git_commit.txt").read_text().strip(),
        "manifest_sha256": (path / "dataset_manifest_hash.txt").read_text().strip(),
    }


def _execution_config(path):
    config = yaml.safe_load((Path(path) / "config.resolved.yaml").read_text())
    config = json.loads(json.dumps(config))
    experiment = config.get("experiment", config)
    for key in ("experiment", "seed", "resume_checkpoint"):
        experiment.pop(key, None)
    config.get("model", {}).pop("description", None)
    return config


def _condition_summary(runs):
    validation = [run["validation"]["wer"] for run in runs]
    test = [run["test"]["wer"] for run in runs]
    parameter_counts = {json.dumps(run["parameter_counts"], sort_keys=True) for run in runs}
    if len(parameter_counts) != 1:
        raise ValueError("Parameter counts differ across seeds")
    execution_configs = [_execution_config(run["path"]) for run in runs]
    if not all(config == execution_configs[0] for config in execution_configs[1:]):
        raise ValueError("Execution configuration differs across seeds")
    return {
        "runs": runs,
        "validation_wer_mean": statistics.mean(validation),
        "validation_wer_sd": statistics.stdev(validation),
        "test_wer_mean": statistics.mean(test),
        "test_wer_sd": statistics.stdev(test),
        "parameter_counts": runs[0]["parameter_counts"],
        "execution_config_consistent": True,
    }


def _prediction_keys(path):
    keys = {}
    speakers = set()
    with (Path(path) / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            segment_id = row["segment_id"]
            if segment_id in keys:
                raise ValueError(f"Duplicate prediction ID: {segment_id}")
            keys[segment_id] = (row["reference"], row["reference_words"], row["speaker_id"])
            speakers.add(row["speaker_id"])
    return keys, speakers


def _validate_prediction_pairing(conditions):
    baseline, speakers = _prediction_keys(conditions["ab1"]["runs"][0]["path"])
    for condition in CONDITIONS:
        for run in conditions[condition]["runs"]:
            observed, _ = _prediction_keys(run["path"])
            if observed != baseline:
                raise ValueError(f"Predictions are not paired: {run['path']}")
    return {
        "utterance_count": len(baseline),
        "speaker_count": len(speakers),
        "speaker_metadata_available": speakers != {"speaker-metadata-unavailable"},
    }


def _paired_difference(left, right):
    differences = [
        (right_run["test"]["wer"] - left_run["test"]["wer"]) * 100
        for left_run, right_run in zip(left["runs"], right["runs"], strict=True)
    ]
    return {
        "test_wer_point_difference_by_seed": differences,
        "test_wer_point_difference_mean": statistics.mean(differences),
        "test_wer_point_difference_sd": statistics.stdev(differences),
    }


def summarize_results(registry_path, root=None):
    registry_path = Path(registry_path).resolve()
    root = Path(root).resolve() if root else registry_path.parents[2]
    registry = yaml.safe_load(registry_path.read_text())
    phase1_path = root / registry["phase1"]
    phase1 = {
        "path": str(phase1_path),
        "metrics": _read_json(phase1_path / "metrics.json"),
        "git_commit": (phase1_path / "git_commit.txt").read_text().strip(),
        "checkpoint_sha256": (phase1_path / "checkpoint_best_sha256.txt").read_text().strip(),
    }

    corpora = {}
    all_commits = set()
    for corpus, patterns in registry["corpora"].items():
        conditions = {}
        manifest_hashes = set()
        corpus_commits = set()
        for condition in CONDITIONS:
            runs = [_read_run(root / patterns[condition].format(seed=seed), seed) for seed in SEEDS]
            conditions[condition] = _condition_summary(runs)
            all_commits.update(run["git_commit"] for run in runs)
            corpus_commits.update(run["git_commit"] for run in runs)
            manifest_hashes.update(run["manifest_sha256"] for run in runs)
        if len(manifest_hashes) != 1:
            raise ValueError(f"Manifest hashes differ within {corpus}")
        corpora[corpus] = {
            "manifest_sha256": manifest_hashes.pop(),
            "git_commits": sorted(corpus_commits),
            "prediction_pairing": _validate_prediction_pairing(conditions),
            "conditions": conditions,
            "comparisons": {
                "ab3_minus_ab1": _paired_difference(conditions["ab1"], conditions["ab3"]),
                "full_minus_ab1": _paired_difference(conditions["ab1"], conditions["full"]),
                "full_minus_ab3": _paired_difference(conditions["ab3"], conditions["full"]),
            },
        }
    phase1["path"] = str(phase1_path.relative_to(root))
    for corpus in corpora.values():
        for condition in corpus["conditions"].values():
            for run in condition["runs"]:
                run["path"] = str(Path(run["path"]).relative_to(root))
    return {"phase1": phase1, "phase2_git_commits": sorted(all_commits), "corpora": corpora}


def markdown_report(summary):
    lines = [
        "# Paper experiment results",
        "",
        "WER values are percentages. Mean and sample standard deviation are computed over seeds.",
        "Negative paired differences favor the condition named first in the comparison label.",
        "",
        "## Phase 1",
        "",
    ]
    metrics = summary["phase1"]["metrics"]
    lines.extend(
        [
            f"- Validation loss: {metrics['loss_total']:.6f}",
            f"- F0 cents MAE: {metrics['f0_cents_mae']:.3f}",
            f"- Voicing F1: {metrics['voicing_f1']:.4f}",
            f"- Delta-F0 RMSE: {metrics['delta_f0_rmse']:.6f}",
            f"- Energy RMSE / correlation: {metrics['energy_rmse']:.6f} / {metrics['energy_correlation']:.4f}",
            f"- Tilt RMSE / correlation: {metrics['tilt_rmse']:.6f} / {metrics['tilt_correlation']:.4f}",
            "",
            "## Phase 2 test WER",
            "",
            "| Corpus | Condition | Validation mean | Test seed 1 | Test seed 2 | Test seed 3 | Test mean ± SD |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "ab1": "Baseline (AB1)",
        "ab3": "Null-feature (AB3)",
        "full": "Learned-feature (Full)",
    }
    for corpus, corpus_data in summary["corpora"].items():
        for condition in CONDITIONS:
            item = corpus_data["conditions"][condition]
            values = [run["test"]["wer"] * 100 for run in item["runs"]]
            lines.append(
                f"| {corpus} | {labels[condition]} | {item['validation_wer_mean'] * 100:.3f} | "
                f"{values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | "
                f"{item['test_wer_mean'] * 100:.3f} ± "
                f"{item['test_wer_sd'] * 100:.3f} |"
            )
    first_corpus = next(iter(summary["corpora"].values()))
    lines.extend(
        [
            "",
            "## Parameter counts",
            "",
            "| Condition | Trainable | Total |",
            "|---|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        counts = first_corpus["conditions"][condition]["parameter_counts"]
        lines.append(f"| {labels[condition]} | {counts['trainable']:,} | {counts['total']:,} |")
    lines.extend(["", "## Paired descriptive differences", ""])
    for corpus, corpus_data in summary["corpora"].items():
        comparisons = corpus_data["comparisons"]
        lines.append(
            f"- **{corpus}**: AB3−AB1 {comparisons['ab3_minus_ab1']['test_wer_point_difference_mean']:+.3f}; "
            f"Full−AB1 {comparisons['full_minus_ab1']['test_wer_point_difference_mean']:+.3f}; "
            f"Full−AB3 {comparisons['full_minus_ab3']['test_wer_point_difference_mean']:+.3f} WER points."
        )
    lines.extend(["", "## Prediction pairing", ""])
    for corpus, corpus_data in summary["corpora"].items():
        pairing = corpus_data["prediction_pairing"]
        speaker_status = "available" if pairing["speaker_metadata_available"] else "unavailable"
        lines.append(
            f"- **{corpus}**: {pairing['utterance_count']:,} paired test utterances; "
            f"{pairing['speaker_count']:,} speaker labels ({speaker_status})."
        )
    lines.extend(
        [
            "",
            "These are descriptive seed-level summaries, not confidence intervals or significance tests.",
            "Use the paired hierarchical analysis on utterance/speaker predictions for inference.",
            "",
        ]
    )
    return "\n".join(lines)
