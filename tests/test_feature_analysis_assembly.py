"""Structural tests for the feature-analysis report assembly.

The expensive parts (HuBERT, decoding, the test corpora) are stubbed. What is
exercised is the report-building code around them: path recording, per-seed
accumulation, and that the returned object is a JSON-serialisable dict.

Both defects this guards against cost a multi-hour GPU run to discover:
  * recording a relative --output path against a resolved root
  * shadowing the report accumulator with a per-seed result list
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("transformers")
import torch
import yaml

import prosody_adaptation.feature_analysis as fa

CORPORA = ("buckeye", "ami_ihm")
SEEDS = (1, 2, 3)


class _StubModel:
    condition = "full"
    prosody_encoder = None
    fusion = None

    def to(self, device):
        return self

    def load_state_dict(self, payload):
        return None


def _make_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "configs" / "results").mkdir(parents=True)
    registry = {
        "phase1": "outputs/phase1/encoder",
        "corpora": {
            corpus: {
                "ab1": f"outputs/phase2/{corpus}_ab1_seed{{seed}}",
                "ab3": f"outputs/phase2/{corpus}_ab3_seed{{seed}}",
                "full": f"outputs/phase2/{corpus}_full_seed{{seed}}",
            }
            for corpus in CORPORA
        },
    }
    path = root / "configs" / "results" / "paper_runs.yaml"
    path.write_text(yaml.safe_dump(registry))

    resolved = {
        "experiment": {"batch_size": 2, "prosody_checkpoint": "p.pt"},
        "model": {"condition": "full"},
        "base": {"processor_model": "stub", "processor_revision": "stub"},
    }
    for corpus in CORPORA:
        for condition in ("ab3", "full"):
            for seed in SEEDS:
                run = root / "outputs" / "phase2" / f"{corpus}_{condition}_seed{seed}"
                run.mkdir(parents=True)
                (run / "config.resolved.yaml").write_text(yaml.safe_dump(resolved))
                (run / "checkpoint_best.pt").write_text("stub")
                (run / "checkpoint_best_sha256.txt").write_text(f"sha-{corpus}-{condition}-{seed}\n")
    return root, path


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    root, registry = _make_repo(tmp_path)

    class _Tokenizer:
        vocab_size = 5
        pad_token_id = 0

    class _Processor:
        tokenizer = _Tokenizer()

    # transformers is a _LazyModule, so patching the module attribute does not bind
    # for `from transformers import Wav2Vec2Processor`; patch the classmethod.
    from transformers import Wav2Vec2Processor

    monkeypatch.setattr(
        Wav2Vec2Processor, "from_pretrained",
        staticmethod(lambda *args, **kwargs: _Processor()),
    )
    monkeypatch.setattr(fa, "load_asr_splits", lambda experiment: (None, None, [0, 1, 2, 3]))
    monkeypatch.setattr(fa, "ASRCollator", lambda processor: (lambda items: items))
    monkeypatch.setattr(fa, "ProductionASR", lambda *a, **k: _StubModel())
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"model": {}})

    def fake_evaluate(model, loader, donor_loader, processor, device,
                      max_batches=None, modes=fa.MODES, utterance_sink=None):
        if utterance_sink is not None:
            for mode in modes:
                utterance_sink(mode, {
                    "segment_id": "utt0", "speaker_id": "spk0",
                    "word_errors": 1, "reference_words": 10, "reference": "a",
                })
        return [{"mode": mode, "wer": 0.3} for mode in modes]

    monkeypatch.setattr(fa, "evaluate_interventions", fake_evaluate)
    return root, registry


def test_report_is_a_serialisable_dict_keyed_by_corpus(stubbed, monkeypatch):
    """Catches shadowing the accumulator with the per-seed result list."""
    root, registry = stubbed
    monkeypatch.chdir(root)
    output = "results/feature_interventions.json"   # deliberately relative

    results = fa.run_feature_analysis(registry, output, device="cpu")

    assert isinstance(results, dict), "the report accumulator must stay a dict"
    assert set(results["corpora"]) == set(CORPORA)
    for corpus in CORPORA:
        assert set(results["corpora"][corpus]) == {str(s) for s in SEEDS}
        entry = results["corpora"][corpus]["1"]
        assert isinstance(entry["results"], list)
        assert entry["checkpoint_sha256"] == f"sha-{corpus}-full-1"
        assert entry["ab3_analysis"]["checkpoint_sha256"] == f"sha-{corpus}-ab3-1"
    json.dumps(results)                                    # must be serialisable
    assert json.loads((root / output).read_text())["corpora"].keys()


def test_relative_output_records_repo_relative_paths(stubbed, monkeypatch):
    """Catches Path.relative_to failing on a relative --output."""
    root, registry = stubbed
    monkeypatch.chdir(root)

    results = fa.run_feature_analysis(
        registry, "results/feature_interventions.json", device="cpu"
    )

    assert results["utterance_record_directory"] == "results/interventions"
    records = results["corpora"]["buckeye"]["2"]["utterance_records"]
    assert set(records) == set(fa.MODES)
    for mode, recorded in records.items():
        assert recorded == f"results/interventions/buckeye_seed2_{mode}.jsonl"
        assert (root / recorded).is_file()


def test_every_seed_and_mode_gets_its_own_record_file(stubbed, monkeypatch):
    root, registry = stubbed
    monkeypatch.chdir(root)
    fa.run_feature_analysis(registry, "results/out.json", device="cpu")

    written = sorted(p.name for p in (root / "results" / "interventions").glob("*.jsonl"))
    expected = sorted(
        f"{corpus}_seed{seed}_{mode}.jsonl"
        for corpus in CORPORA for seed in SEEDS for mode in fa.MODES
    )
    assert written == expected
