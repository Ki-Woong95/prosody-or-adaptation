from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import file_sha256

REQUIRED_RUN_FILES = {
    "config.resolved.yaml", "metrics.json", "predictions.jsonl", "checkpoint_best.pt",
    "environment.txt", "git_commit.txt", "dataset_manifest_hash.txt", "training_log.jsonl",
}


def create_run_directory(root: str | Path, name: str) -> Path:
    root = Path(root)
    target = root / name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {target}")
    target.mkdir(parents=True)
    return target


def initialize_run(target: Path, config: dict, manifest: str | Path):
    (target / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target / "environment.txt").write_text(json.dumps(environment, indent=2) + "\n")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unavailable"
    (target / "git_commit.txt").write_text(commit + "\n")
    (target / "dataset_manifest_hash.txt").write_text(file_sha256(manifest) + "\n")


def append_jsonl(path: str | Path, record: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
