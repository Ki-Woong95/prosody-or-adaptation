import pytest

from prosody_adaptation.config import load_yaml


def test_load_yaml_expands_environment_variables(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("train_data: ${PROSODY_ADAPTATION_TEST_DATA}/train\n")
    monkeypatch.setenv("PROSODY_ADAPTATION_TEST_DATA", "/datasets/casper")

    assert load_yaml(config)["train_data"] == "/datasets/casper/train"


def test_load_yaml_rejects_missing_environment_variables(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("train_data: ${PROSODY_ADAPTATION_TEST_DATA}/train\n")
    monkeypatch.delenv("PROSODY_ADAPTATION_TEST_DATA", raising=False)

    with pytest.raises(ValueError, match="PROSODY_ADAPTATION_TEST_DATA"):
        load_yaml(config)
