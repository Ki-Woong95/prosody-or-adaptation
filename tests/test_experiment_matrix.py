from pathlib import Path

from prosody_adaptation.config import load_yaml


ROOT = Path(__file__).parents[1]
CONDITIONS = {
    "baseline": ("ab1", "baseline"),
    "null": ("ab3", "null"),
    "learned": ("full", "learned"),
}
CORPORA = ("buckeye", "switchboard", "ami_ihm")


def test_complete_three_condition_three_corpus_three_seed_matrix(monkeypatch):
    monkeypatch.setenv("PROSODY_ADAPTATION_BUCKEYE_CACHE", "/test/buckeye-cache")
    for corpus in CORPORA:
        for label, (condition, model_name) in CONDITIONS.items():
            for seed in (1, 2, 3):
                path = ROOT / f"configs/experiment/{corpus}_{label}_seed{seed}.yaml"
                config = load_yaml(path)
                assert config["experiment"] == f"{corpus}_{label}_seed{seed}"
                assert config["seed"] == seed
                assert config["model_config"] == f"configs/model/{model_name}.yaml"
                assert load_yaml(ROOT / config["model_config"])["condition"] == condition
                assert config.get("manifest_status", "frozen") == "frozen"
                assert config["data_manifest_sha256"] != "PENDING_FREEZE"
                if corpus != "buckeye":
                    assert config["data_backend"] == "arrow_disk"


def test_inherited_configs_do_not_mutate_parent_values(monkeypatch):
    monkeypatch.setenv("PROSODY_ADAPTATION_BUCKEYE_CACHE", "/test/buckeye-cache")
    seed1 = load_yaml(ROOT / "configs/experiment/buckeye_learned_seed1.yaml")
    seed2 = load_yaml(ROOT / "configs/experiment/buckeye_learned_seed2.yaml")
    assert seed1["seed"] == 1 and seed2["seed"] == 2
    assert seed1["experiment"] != seed2["experiment"]
    assert seed1["log_every_steps"] == seed2["log_every_steps"] == 200
