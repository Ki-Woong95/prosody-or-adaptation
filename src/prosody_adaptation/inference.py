from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


COMPARISONS = {
    "full_minus_ab3": ("full", "ab3", "primary"),
    "ab3_minus_ab1": ("ab3", "ab1", "secondary"),
}
SEEDS = (1, 2, 3)
MISSING_SPEAKER = "speaker-metadata-unavailable"


def _load_predictions(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            segment_id = row["segment_id"]
            if segment_id in rows:
                raise ValueError(f"Duplicate segment ID in {path}: {segment_id}")
            rows[segment_id] = (
                int(row["word_errors"]),
                int(row["reference_words"]),
                row["reference"],
                row["speaker_id"],
            )
    return rows


def paired_error_arrays(first_paths, second_paths):
    first = [_load_predictions(path) for path in first_paths]
    second = [_load_predictions(path) for path in second_paths]
    segment_ids = sorted(first[0])
    expected_ids = set(segment_ids)
    for rows in (*first, *second):
        if set(rows) != expected_ids:
            raise ValueError("Prediction IDs are not paired across conditions and seeds")

    differences = np.empty((len(SEEDS), len(segment_ids)), dtype=np.float64)
    references = np.empty(len(segment_ids), dtype=np.float64)
    speakers = []
    for column, segment_id in enumerate(segment_ids):
        reference_metadata = {
            (rows[segment_id][1], rows[segment_id][2], rows[segment_id][3])
            for rows in (*first, *second)
        }
        if len(reference_metadata) != 1:
            raise ValueError(f"Reference metadata differs for {segment_id}")
        references[column], _, speaker = reference_metadata.pop()
        speakers.append(speaker)
        for seed_index in range(len(SEEDS)):
            differences[seed_index, column] = (
                first[seed_index][segment_id][0] - second[seed_index][segment_id][0]
            )
    return differences, references, np.asarray(speakers), segment_ids


def paired_poisson_bootstrap(
    differences,
    references,
    speakers,
    samples=10_000,
    seed=42,
    chunk_size=32,
):
    """Paired hierarchical Poisson bootstrap over seeds and available speaker clusters."""
    differences = np.asarray(differences, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    speakers = np.asarray(speakers)
    if differences.ndim != 2 or differences.shape[1] != len(references):
        raise ValueError("Expected a seed-by-utterance difference matrix")
    if differences.shape[0] < 2 or len(references) == 0 or np.any(references <= 0):
        raise ValueError("Bootstrap requires paired seeds and non-empty references")

    seed_count = differences.shape[0]
    unique_speakers = np.unique(speakers)
    has_speakers = not (len(unique_speakers) == 1 and unique_speakers[0] == MISSING_SPEAKER)
    groups = (
        [np.flatnonzero(speakers == speaker) for speaker in unique_speakers]
        if has_speakers
        else [np.arange(len(references))]
    )

    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    seed_probabilities = np.full(seed_count, 1.0 / seed_count)
    speaker_probabilities = np.full(len(groups), 1.0 / len(groups))
    for start in range(0, samples, chunk_size):
        width = min(chunk_size, samples - start)
        seed_weights = rng.multinomial(seed_count, seed_probabilities, size=width)
        combined_differences = seed_weights @ differences
        if has_speakers:
            speaker_weights = rng.multinomial(len(groups), speaker_probabilities, size=width)
        else:
            speaker_weights = np.ones((width, 1), dtype=np.int64)
        numerator = np.zeros(width, dtype=np.float64)
        denominator = np.zeros(width, dtype=np.float64)
        for group_index, indices in enumerate(groups):
            observation_weights = rng.poisson(
                speaker_weights[:, group_index, None], size=(width, len(indices))
            )
            numerator += np.sum(observation_weights * combined_differences[:, indices], axis=1)
            denominator += seed_count * np.sum(observation_weights * references[indices], axis=1)
        empty = np.flatnonzero(denominator == 0)
        if len(empty):
            fallback = rng.integers(0, len(references), size=len(empty))
            numerator[empty] = combined_differences[empty, fallback]
            denominator[empty] = seed_count * references[fallback]
        draws[start : start + width] = 100 * numerator / denominator

    seed_effects = 100 * differences.sum(axis=1) / references.sum()
    point_estimate = 100 * differences.sum() / (seed_count * references.sum())
    probability_nonpositive = (np.count_nonzero(draws <= 0) + 1) / (samples + 1)
    probability_nonnegative = (np.count_nonzero(draws >= 0) + 1) / (samples + 1)
    return {
        "method": (
            "paired seed-speaker-utterance Poisson bootstrap"
            if has_speakers
            else "paired seed-utterance Poisson bootstrap"
        ),
        "samples": samples,
        "seed_count": seed_count,
        "speaker_count": int(len(unique_speakers)) if has_speakers else None,
        "utterance_count": len(references),
        "difference_wer_points": float(point_estimate),
        "difference_by_seed_wer_points": {
            str(seed_id): float(value) for seed_id, value in zip(SEEDS, seed_effects, strict=True)
        },
        "ci95_wer_points": [float(value) for value in np.quantile(draws, (0.025, 0.975))],
        "probability_first_nonpositive": float(probability_nonpositive),
        "probability_first_better": float(np.mean(draws < 0)),
        "p_value_two_sided": float(
            min(1.0, 2 * min(probability_nonpositive, probability_nonnegative))
        ),
    }


def holm_adjust(p_values):
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (total - rank) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def run_inference(registry_path, root=None, samples=10_000, seed=42, chunk_size=32):
    registry_path = Path(registry_path).resolve()
    root = Path(root).resolve() if root else registry_path.parents[2]
    registry = yaml.safe_load(registry_path.read_text())
    results = {"samples": samples, "random_seed": seed, "corpora": {}}
    families = {"primary": {}, "secondary": {}}
    for corpus, patterns in registry["corpora"].items():
        corpus_results = {}
        for name, (first, second, family) in COMPARISONS.items():
            first_paths = [
                root / patterns[first].format(seed=seed_id) / "predictions.jsonl"
                for seed_id in SEEDS
            ]
            second_paths = [
                root / patterns[second].format(seed=seed_id) / "predictions.jsonl"
                for seed_id in SEEDS
            ]
            arrays = paired_error_arrays(first_paths, second_paths)
            result = paired_poisson_bootstrap(
                arrays[0], arrays[1], arrays[2], samples, seed, chunk_size
            )
            result["first_condition"] = first
            result["second_condition"] = second
            result["family"] = family
            corpus_results[name] = result
            families[family][corpus] = result["p_value_two_sided"]
        results["corpora"][corpus] = corpus_results
    for family, p_values in families.items():
        adjusted = holm_adjust(p_values)
        for corpus, value in adjusted.items():
            comparison = "full_minus_ab3" if family == "primary" else "ab3_minus_ab1"
            results["corpora"][corpus][comparison]["p_value_holm"] = value
    return results


def markdown_report(results):
    lines = [
        "# Paired hierarchical inference",
        "",
        "Negative differences favor the first condition. Holm correction is applied",
        "separately to the three primary Full−AB3 tests and three secondary AB3−AB1 tests.",
        "",
        "| Family | Corpus | Comparison | Difference | 95% CI | p | Holm p | Method |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for comparison in ("full_minus_ab3", "ab3_minus_ab1"):
        for corpus, corpus_results in results["corpora"].items():
            item = corpus_results[comparison]
            low, high = item["ci95_wer_points"]
            lines.append(
                f"| {item['family']} | {corpus} | {item['first_condition']}−"
                f"{item['second_condition']} | {item['difference_wer_points']:+.3f} | "
                f"[{low:+.3f}, {high:+.3f}] | {item['p_value_two_sided']:.4f} | "
                f"{item['p_value_holm']:.4f} | {item['method']} |"
            )
    lines.extend(
        [
            "",
            "Switchboard has no usable speaker labels in the stored predictions, so its",
            "bootstrap resamples seeds and utterances but not speakers. With only three training",
            "seeds, seed-level uncertainty is necessarily estimated imprecisely.",
            "",
        ]
    )
    return "\n".join(lines)
