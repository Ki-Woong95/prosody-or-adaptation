"""Tests for the exploratory feature-intervention bootstrap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prosody_adaptation.intervention_inference import (
    CONTRASTS,
    SEEDS,
    _paired_arrays,
    markdown_report,
    run_intervention_inference,
)

MODES = ("true", "zero", "time_shuffle", "utterance_shuffle")


def _write(directory, corpus, seed, mode, rows):
    path = directory / f"{corpus}_seed{seed}_{mode}.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def _rows(errors, speakers=("spk1", "spk2", "spk1", "spk2")):
    return [
        {
            "segment_id": f"utt{index}",
            "speaker_id": speakers[index % len(speakers)],
            "word_errors": int(value),
            "reference_words": 10,
            "reference": "a b c",
        }
        for index, value in enumerate(errors)
    ]


def _build(directory, corpus="buckeye", advantage=1):
    """`true` beats every other mode by `advantage` errors on each utterance."""
    base = np.array([4, 3, 5, 2, 6, 3, 4, 5], dtype=int)
    for seed in SEEDS:
        _write(directory, corpus, seed, "true", _rows(base))
        for mode in ("zero", "time_shuffle", "utterance_shuffle"):
            _write(directory, corpus, seed, mode, _rows(base + advantage))


def test_paired_arrays_align_on_segment_id(tmp_path):
    _build(tmp_path)
    first = [tmp_path / f"buckeye_seed{s}_true.jsonl" for s in SEEDS]
    second = [tmp_path / f"buckeye_seed{s}_zero.jsonl" for s in SEEDS]
    differences, references, speakers, segment_ids = _paired_arrays(first, second)

    assert differences.shape == (3, 8)
    assert np.all(differences == -1)  # true has one fewer error everywhere
    assert np.all(references == 10)
    assert len(segment_ids) == 8 and segment_ids == sorted(segment_ids)
    assert set(speakers) == {"spk1", "spk2"}


def test_paired_arrays_reject_unpaired_records(tmp_path):
    _build(tmp_path)
    short = _rows(np.array([1, 2, 3], dtype=int))
    _write(tmp_path, "buckeye", 2, "zero", short)
    first = [tmp_path / f"buckeye_seed{s}_true.jsonl" for s in SEEDS]
    second = [tmp_path / f"buckeye_seed{s}_zero.jsonl" for s in SEEDS]
    with pytest.raises(ValueError, match="not paired"):
        _paired_arrays(first, second)


def test_paired_arrays_reject_inconsistent_reference_metadata(tmp_path):
    _build(tmp_path)
    rows = _rows(np.array([4, 3, 5, 2, 6, 3, 4, 5], dtype=int))
    rows[0]["reference_words"] = 99
    _write(tmp_path, "buckeye", 1, "zero", rows)
    first = [tmp_path / f"buckeye_seed{s}_true.jsonl" for s in SEEDS]
    second = [tmp_path / f"buckeye_seed{s}_zero.jsonl" for s in SEEDS]
    with pytest.raises(ValueError, match="Reference metadata differs"):
        _paired_arrays(first, second)


def _analysis(tmp_path, corpora):
    directory = tmp_path / "interventions"
    directory.mkdir()
    for corpus in corpora:
        _build(directory, corpus)
    path = tmp_path / "features.json"
    path.write_text(json.dumps({
        "utterance_record_directory": str(directory),
        "corpora": dict.fromkeys(corpora, {}),
    }))
    return path


def test_intervention_inference_reports_effect_and_interval(tmp_path):
    analysis = _analysis(tmp_path, ["buckeye"])
    results = run_intervention_inference(
        analysis, tmp_path / "out.json", samples=2000, seed=42
    )
    item = results["corpora"]["buckeye"]["true_minus_zero"]
    # -1 error per utterance on 10 reference words = -10 WER points.
    assert item["difference_wer_points"] == pytest.approx(-10.0, abs=1e-9)
    low, high = item["ci95_wer_points"]
    assert low <= item["difference_wer_points"] <= high
    assert item["first_condition"] == "true" and item["second_condition"] == "zero"
    assert set(results["corpora"]["buckeye"]) == set(CONTRASTS)


def test_holm_family_spans_all_corpora_and_contrasts(tmp_path):
    corpora = ["buckeye", "switchboard", "ami_ihm"]
    analysis = _analysis(tmp_path, corpora)
    results = run_intervention_inference(
        analysis, tmp_path / "out.json", samples=1000, seed=42
    )
    assert results["family_size"] == 9
    assert results["family"] == "exploratory"
    for corpus in corpora:
        for name in CONTRASTS:
            item = results["corpora"][corpus][name]
            adjusted = item["p_value_holm_exploratory"]
            assert item["p_value_two_sided"] <= adjusted + 1e-12
            assert 0.0 <= adjusted <= 1.0


def test_zero_effect_data_gives_an_interval_spanning_zero(tmp_path):
    directory = tmp_path / "interventions"
    directory.mkdir()
    _build(directory, "buckeye", advantage=0)   # identical error counts
    path = tmp_path / "features.json"
    path.write_text(json.dumps({
        "utterance_record_directory": str(directory), "corpora": {"buckeye": {}},
    }))
    results = run_intervention_inference(path, tmp_path / "out.json", samples=2000, seed=42)
    item = results["corpora"]["buckeye"]["true_minus_zero"]
    assert item["difference_wer_points"] == pytest.approx(0.0, abs=1e-9)
    low, high = item["ci95_wer_points"]
    assert low <= 0.0 <= high
    assert item["p_value_two_sided"] > 0.05


def test_markdown_report_lists_every_contrast(tmp_path):
    analysis = _analysis(tmp_path, ["buckeye", "ami_ihm"])
    results = run_intervention_inference(
        analysis, tmp_path / "out.json", samples=500, seed=42
    )
    text = markdown_report(results)
    for corpus in ("buckeye", "ami_ihm"):
        for first, second in CONTRASTS.values():
            assert f"| {corpus} | {first}−{second} |" in text
    assert "not evidence that an intervention has no effect" in text


def test_missing_records_raise_rather_than_silently_skipping(tmp_path):
    analysis = _analysis(tmp_path, ["buckeye"])
    (tmp_path / "interventions" / "buckeye_seed2_zero.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="Missing intervention records"):
        run_intervention_inference(analysis, tmp_path / "out.json", samples=100)


def test_record_path_handles_relative_output_under_absolute_root(tmp_path, monkeypatch):
    """`--output results/x.json` is relative while root is resolved; the pair
    must still produce a repo-relative record path rather than raising."""
    from prosody_adaptation.feature_analysis import record_path

    root = tmp_path / "repo"
    (root / "results" / "interventions").mkdir(parents=True)
    monkeypatch.chdir(root)

    relative = Path("results/interventions/buckeye_seed1_true.jsonl")
    relative.write_text("{}\n")
    assert record_path(relative, root) == "results/interventions/buckeye_seed1_true.jsonl"
    # Absolute inputs under root behave identically.
    assert record_path(relative.resolve(), root) == (
        "results/interventions/buckeye_seed1_true.jsonl"
    )


def test_record_path_falls_back_to_absolute_outside_root(tmp_path):
    from prosody_adaptation.feature_analysis import record_path

    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "x.jsonl"
    outside.parent.mkdir()
    outside.write_text("{}\n")
    assert record_path(outside, root) == str(outside.resolve())


def test_repo_relative_record_directory_resolves_from_any_cwd(tmp_path, monkeypatch):
    """The analysis JSON stores a repo-relative directory; inference must find it
    even when invoked from somewhere else."""
    repo = tmp_path / "repo"
    directory = repo / "results" / "interventions"
    directory.mkdir(parents=True)
    _build(directory, "buckeye")
    analysis = repo / "results" / "features.json"
    analysis.write_text(json.dumps({
        "utterance_record_directory": "results/interventions",
        "corpora": {"buckeye": {}},
    }))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    results = run_intervention_inference(
        analysis, tmp_path / "out.json", samples=500, seed=42
    )
    assert results["corpora"]["buckeye"]["true_minus_zero"][
        "difference_wer_points"
    ] == pytest.approx(-10.0, abs=1e-9)
