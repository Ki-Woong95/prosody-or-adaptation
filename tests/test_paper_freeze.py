import json
from pathlib import Path

import pytest

from prosody_adaptation.asr_data import normalize_transcript_v1
from prosody_adaptation.config import file_sha256, load_yaml

ROOT = Path(__file__).parents[1]
FROZEN_HASH = "6a90195dec1d3000eea714e206bbae416a7688d4cf571b73984e6df05b879698"
MANIFEST = ROOT / "data/buckeye_v2_1/manifest.jsonl"


def test_every_buckeye_config_pins_the_frozen_manifest_hash(monkeypatch):
    """Runs everywhere: the configs must agree with the frozen descriptor."""
    monkeypatch.setenv("PROSODY_ADAPTATION_BUCKEYE_CACHE", "/test/buckeye-cache")
    frozen = json.loads((ROOT / "data/buckeye_v2_1/FROZEN.json").read_text())
    assert frozen["manifest_sha256"] == FROZEN_HASH
    configs = list((ROOT / "configs/experiment").glob("buckeye_*.yaml"))
    assert configs, "expected Buckeye experiment configs"
    for path in configs:
        config = load_yaml(path)
        assert config["data_manifest"] == "data/buckeye_v2_1/manifest.jsonl"
        assert config["data_manifest_sha256"] == FROZEN_HASH


@pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason="Buckeye manifest is licensed and not redistributed; regenerate it with "
           "`prosody-adaptation prepare buckeye` to run this check (see LICENSED_DATA.md)",
)
def test_frozen_buckeye_manifest_reproduces_the_recorded_hash():
    assert file_sha256(MANIFEST) == FROZEN_HASH


def test_transcript_v1_is_character_only_and_preserves_lexical_apostrophes():
    assert normalize_transcript_v1("I’m DON'T we`ll they´ve") == "i'm don't we'll they've"
    assert normalize_transcript_v1("don't expand contractions") == "don't expand contractions"
    assert normalize_transcript_v1("um-hum peopuh=people it?") == "um hum peopuh people it"


def test_buckeye_configs_use_paper_training_schedule(monkeypatch):
    monkeypatch.setenv("PROSODY_ADAPTATION_BUCKEYE_CACHE", "/test/buckeye-cache")
    for condition in ("baseline", "null", "learned"):
        path = ROOT / f"configs/experiment/buckeye_{condition}_seed1.yaml"
        config = load_yaml(path)
        assert config["epochs"] == 100
        assert config["batch_size"] == 8
        assert config["gradient_accumulation_steps"] == 4
        assert config["learning_rate"] == 1e-4
        assert config["adam_betas"] == [0.9, 0.98]
        assert config["scheduler"] == "none"
        assert config["mixed_precision"] is True
        assert config["gradient_clip"] == 1.0
        assert config["early_stopping_patience"] == 10
        assert config["early_stopping_min_epochs"] == 25
        assert config["eval_every_steps"] == 0
        assert "max_train_batches" not in config
