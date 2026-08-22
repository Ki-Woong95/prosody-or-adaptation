from __future__ import annotations

import argparse
import json
from pathlib import Path

from .buckeye import load_manifest, prepare_buckeye, validate_rows
from .cache import materialize_tar_cache, validate_cache


def _write_reports(data, markdown, json_path, markdown_path):
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown)


def build_parser():
    parser = argparse.ArgumentParser(prog="prosody-adaptation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    corpora = prepare.add_subparsers(dest="corpus", required=True)
    buckeye = corpora.add_parser("buckeye")
    buckeye.add_argument("--raw-archive", required=True)
    buckeye.add_argument("--output", required=True)
    buckeye.add_argument("--config", required=True)
    casper = corpora.add_parser("casper")
    casper.add_argument("--recordings", required=True)
    casper.add_argument("--output", required=True)
    casper.add_argument("--device", default="cuda")
    casper.add_argument("--version", default="v2", help="Suffix for the split directories")
    validate = commands.add_parser("validate-data")
    validate.add_argument("--manifest", required=True)
    cache = commands.add_parser("materialize-cache")
    cache.add_argument("--manifest", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--shard-size", type=int, default=500)
    validate_cache_parser = commands.add_parser("validate-cache")
    validate_cache_parser.add_argument("--manifest", required=True)
    validate_cache_parser.add_argument("--cache", required=True)
    validate_cache_parser.add_argument("--verify-audio", action="store_true")
    for name in ("train-prosody", "train-asr"):
        item = commands.add_parser(name)
        item.add_argument("--config", required=True)
    summary = commands.add_parser("summarize-results")
    summary.add_argument("--registry", default="configs/results/paper_runs.yaml")
    summary.add_argument("--json", default="results/paper_results.json")
    summary.add_argument("--markdown", default="results/paper_results.md")
    inference = commands.add_parser("infer")
    inference.add_argument("--registry", default="configs/results/paper_runs.yaml")
    inference.add_argument("--samples", type=int, default=10_000)
    inference.add_argument("--seed", type=int, default=42)
    inference.add_argument("--chunk-size", type=int, default=32)
    inference.add_argument("--json", default="results/paper_inference.json")
    inference.add_argument("--markdown", default="results/paper_inference.md")
    features = commands.add_parser("analyze-features")
    features.add_argument("--registry", default="configs/results/paper_runs.yaml")
    features.add_argument("--output", default="results/feature_interventions.json")
    features.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    features.add_argument("--max-batches", type=int)
    features.add_argument("--corpus", choices=("buckeye", "switchboard", "ami_ihm"))
    features.add_argument("--seed", type=int, choices=(1, 2, 3), action="append")
    features.add_argument("--batch-size", type=int)
    intervention = commands.add_parser("infer-interventions")
    intervention.add_argument("--analysis", default="results/feature_interventions.json")
    intervention.add_argument("--samples", type=int, default=100_000)
    intervention.add_argument("--seed", type=int, default=42)
    intervention.add_argument("--chunk-size", type=int, default=32)
    intervention.add_argument("--json", default="results/intervention_inference.json")
    intervention.add_argument("--markdown", default="results/intervention_inference.md")
    residuals = commands.add_parser("analyze-residuals")
    residuals.add_argument("--registry", default="configs/results/paper_runs.yaml")
    residuals.add_argument("--output", default="results/residual_analysis.json")
    residuals.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    residuals.add_argument("--max-batches", type=int)
    residuals.add_argument("--corpus", choices=("buckeye", "switchboard", "ami_ihm"))
    residuals.add_argument("--seed", type=int, choices=(1, 2, 3), action="append")
    residuals.add_argument("--batch-size", type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "prepare" and args.corpus == "buckeye":
        print(json.dumps(prepare_buckeye(args.raw_archive, args.output, args.config), indent=2))
        return 0
    if args.command == "prepare" and args.corpus == "casper":
        from .casper_prepare import prepare_casper

        print(
            json.dumps(
                prepare_casper(args.recordings, args.output, args.device, args.version),
                indent=2,
            )
        )
        return 0
    if args.command == "validate-data":
        print(json.dumps(validate_rows(load_manifest(args.manifest)), indent=2))
        return 0
    if args.command == "materialize-cache":
        print(json.dumps(materialize_tar_cache(args.manifest, args.output, args.shard_size), indent=2))
        return 0
    if args.command == "validate-cache":
        print(json.dumps(validate_cache(args.manifest, args.cache, args.verify_audio), indent=2))
        return 0
    if args.command == "summarize-results":
        from .results import markdown_report, summarize_results

        summary = summarize_results(args.registry)
        _write_reports(summary, markdown_report(summary), args.json, args.markdown)
        return 0
    if args.command == "infer":
        from .inference import markdown_report, run_inference

        results = run_inference(
            args.registry,
            samples=args.samples,
            seed=args.seed,
            chunk_size=args.chunk_size,
        )
        _write_reports(results, markdown_report(results), args.json, args.markdown)
        return 0
    if args.command == "analyze-features":
        from .feature_analysis import run_feature_analysis

        run_feature_analysis(
            args.registry,
            args.output,
            device=args.device,
            max_batches=args.max_batches,
            corpus_filter=args.corpus,
            seeds=tuple(args.seed or (1, 2, 3)),
            batch_size=args.batch_size,
        )
        return 0
    if args.command == "infer-interventions":
        from .intervention_inference import markdown_report, run_intervention_inference

        results = run_intervention_inference(
            args.analysis, args.json, samples=args.samples, seed=args.seed,
            chunk_size=args.chunk_size,
        )
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(markdown_report(results))
        return 0
    if args.command == "analyze-residuals":
        from .feature_analysis import run_residual_analysis

        run_residual_analysis(
            args.registry,
            args.output,
            device=args.device,
            max_batches=args.max_batches,
            corpus_filter=args.corpus,
            seeds=tuple(args.seed or (1, 2, 3)),
            batch_size=args.batch_size,
        )
        return 0
    config = Path(args.config)
    if not config.exists():
        raise FileNotFoundError(config)
    if args.command == "train-prosody":
        from .phase1_train import train_phase1
        print(train_phase1(config))
        return 0
    if args.command == "train-asr":
        from .asr_train import train_asr
        print(train_asr(config))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
