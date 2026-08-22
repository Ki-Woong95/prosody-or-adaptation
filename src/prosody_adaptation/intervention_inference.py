"""Bootstrap inference for test-time feature interventions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .inference import holm_adjust, paired_poisson_bootstrap

CONTRASTS = {
    "true_minus_zero": ("true", "zero"),
    "true_minus_time_shuffle": ("true", "time_shuffle"),
    "true_minus_utterance_shuffle": ("true", "utterance_shuffle"),
}
SEEDS = (1, 2, 3)
MISSING_SPEAKER = "speaker-metadata-unavailable"


def _load(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            segment_id = row["segment_id"]
            if segment_id in rows:
                raise ValueError(f"Duplicate segment ID in {path}: {segment_id}")
            rows[segment_id] = row
    return rows


def _paired_arrays(first_paths, second_paths):
    """Seed-by-utterance error differences, aligned on segment ID."""
    first = [_load(path) for path in first_paths]
    second = [_load(path) for path in second_paths]
    segment_ids = sorted(first[0])
    expected = set(segment_ids)
    for rows in (*first, *second):
        if set(rows) != expected:
            raise ValueError("Intervention records are not paired across modes and seeds")

    differences = np.empty((len(SEEDS), len(segment_ids)), dtype=np.float64)
    references = np.empty(len(segment_ids), dtype=np.float64)
    speakers = []
    for column, segment_id in enumerate(segment_ids):
        metadata = {
            (rows[segment_id]["reference_words"], rows[segment_id]["speaker_id"])
            for rows in (*first, *second)
        }
        if len(metadata) != 1:
            raise ValueError(f"Reference metadata differs for {segment_id}")
        reference_words, speaker = metadata.pop()
        references[column] = reference_words
        speakers.append(speaker)
        for index in range(len(SEEDS)):
            differences[index, column] = (
                first[index][segment_id]["word_errors"]
                - second[index][segment_id]["word_errors"]
            )
    return differences, references, np.asarray(speakers), segment_ids


def run_intervention_inference(
    analysis_path, output_path, samples=100_000, seed=42, chunk_size=32, root=None
):
    analysis = json.loads(Path(analysis_path).read_text())
    directory = Path(analysis["utterance_record_directory"])
    if not directory.is_absolute():
        # The recorded path is repo-relative. Resolve against an explicit root,
        # the current directory, or the repository the analysis file sits in.
        candidates = [Path(root) / directory] if root is not None else []
        candidates.append(directory)
        candidates.append(Path(analysis_path).resolve().parents[1] / directory)
        directory = next((item for item in candidates if item.is_dir()), candidates[0])

    results = {
        "samples": samples,
        "random_seed": seed,
        "family": "exploratory",
        "family_size": 0,
        "note": (
            "Exploratory within-model contrasts. Holm correction spans all corpora "
            "and contrasts jointly. A p-value above 0.05 is not evidence of no effect; "
            "read the effect size and interval."
        ),
        "source_analysis": str(analysis_path),
        "corpora": {},
    }
    p_values = {}
    for corpus in analysis["corpora"]:
        corpus_results = {}
        for name, (first_mode, second_mode) in CONTRASTS.items():
            first_paths = [
                directory / f"{corpus}_seed{s}_{first_mode}.jsonl" for s in SEEDS
            ]
            second_paths = [
                directory / f"{corpus}_seed{s}_{second_mode}.jsonl" for s in SEEDS
            ]
            missing = [p for p in (*first_paths, *second_paths) if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing intervention records: {missing[0]}")
            differences, references, speakers, _ = _paired_arrays(first_paths, second_paths)
            result = paired_poisson_bootstrap(
                differences, references, speakers, samples, seed, chunk_size
            )
            result["first_condition"] = first_mode
            result["second_condition"] = second_mode
            corpus_results[name] = result
            p_values[f"{corpus}:{name}"] = result["p_value_two_sided"]
        results["corpora"][corpus] = corpus_results

    results["family_size"] = len(p_values)
    for key, value in holm_adjust(p_values).items():
        corpus, name = key.split(":", 1)
        results["corpora"][corpus][name]["p_value_holm_exploratory"] = value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def markdown_report(results):
    lines = [
        "# Feature-intervention inference (exploratory)",
        "",
        f"Paired hierarchical bootstrap, {results['samples']:,} draws. Negative differences",
        "favour the true learned representation. Holm correction spans the",
        f"{results['family_size']}-test exploratory family (3 corpora x 3 contrasts).",
        "",
        "A p-value above 0.05 is not evidence that an intervention has no effect;",
        "the effect size and its interval carry the information.",
        "",
        "| Corpus | Contrast | ΔWER (points) | 95% CI | p | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus, corpus_results in results["corpora"].items():
        for name in CONTRASTS:
            item = corpus_results[name]
            low, high = item["ci95_wer_points"]
            lines.append(
                f"| {corpus} | {item['first_condition']}−{item['second_condition']} | "
                f"{item['difference_wer_points']:+.3f} | [{low:+.3f}, {high:+.3f}] | "
                f"{item['p_value_two_sided']:.4f} | "
                f"{item['p_value_holm_exploratory']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)
