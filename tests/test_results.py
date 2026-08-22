import json

import pytest
import yaml

from prosody_adaptation.results import RUN_FILES, markdown_report, summarize_results


def _run(root, corpus, condition, seed, wer):
    path = root / f"{corpus}_{condition}_{seed}"
    path.mkdir()
    metrics = {
        "validation": {"wer": wer + 0.01, "loss": 1.0, "word_errors": 2, "reference_words": 10},
        "test": {"wer": wer, "loss": 1.0, "word_errors": 2, "reference_words": 10},
    }
    payloads = {
        "metrics.json": json.dumps(metrics),
        "parameter_counts.json": json.dumps({"total": 10, "trainable": 5}),
        "git_commit.txt": "abc123\n",
        "dataset_manifest_hash.txt": f"{corpus}-hash\n",
        "config.resolved.yaml": yaml.safe_dump(
            {
                "experiment": {
                    "experiment": f"run-{seed}",
                    "seed": seed,
                    "batch_size": 8,
                },
                "model": {"condition": condition},
            }
        ),
        "predictions.jsonl": json.dumps(
            {
                "segment_id": "utterance-1",
                "speaker_id": "speaker-1",
                "reference": "hello",
                "reference_words": 1,
                "hypothesis": "hello",
                "word_errors": 0,
            }
        )
        + "\n",
    }
    for name in RUN_FILES:
        (path / name).write_text(payloads.get(name, "placeholder\n"))
    return str(path.relative_to(root))


def test_registered_results_are_aggregated_without_globbing(tmp_path):
    phase1 = tmp_path / "phase1"
    phase1.mkdir()
    (phase1 / "metrics.json").write_text(
        json.dumps(
            {
                "loss_total": 1.0,
                "f0_cents_mae": 2.0,
                "voicing_f1": 0.8,
                "delta_f0_rmse": 0.1,
                "energy_rmse": 0.2,
                "energy_correlation": 0.9,
                "tilt_rmse": 0.3,
                "tilt_correlation": 0.8,
            }
        )
    )
    (phase1 / "git_commit.txt").write_text("abc123\n")
    (phase1 / "checkpoint_best_sha256.txt").write_text("checkpoint-hash\n")
    registry = {"phase1": "phase1", "corpora": {}}
    for corpus in ("buckeye", "switchboard", "ami_ihm"):
        registry["corpora"][corpus] = {}
        for offset, condition in enumerate(("ab1", "ab3", "full")):
            pattern = f"{corpus}_{condition}_{{seed}}"
            registry["corpora"][corpus][condition] = pattern
            for seed in (1, 2, 3):
                _run(tmp_path, corpus, condition, seed, 0.4 - offset * 0.01 + seed * 0.001)
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry))

    summary = summarize_results(registry_path, tmp_path)

    assert summary["phase2_git_commits"] == ["abc123"]
    assert summary["phase1"]["path"] == "phase1"
    assert summary["corpora"]["buckeye"]["conditions"]["ab1"]["runs"][0][
        "path"
    ] == "buckeye_ab1_1"
    assert summary["corpora"]["buckeye"]["prediction_pairing"]["utterance_count"] == 1
    assert summary["corpora"]["buckeye"]["comparisons"]["full_minus_ab3"][
        "test_wer_point_difference_mean"
    ] == pytest.approx(-1.0)
    assert "Paper experiment results" in markdown_report(summary)
