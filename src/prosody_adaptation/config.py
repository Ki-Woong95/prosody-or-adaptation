from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path, seen: frozenset[Path]) -> dict[str, Any]:
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Cyclic YAML inheritance involving {path}")
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    parent = value.pop("extends", None)
    if parent is None:
        return value
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _merge(_load_yaml(parent_path, seen | {path}), value)


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    missing = sorted(set(_ENVIRONMENT_VARIABLE.findall(value)) - os.environ.keys())
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Environment variable not set: {names}")
    return _ENVIRONMENT_VARIABLE.sub(lambda match: os.environ[match.group(1)], value)


def load_yaml(path: str | Path) -> dict[str, Any]:
    return _expand_environment(_load_yaml(Path(path), frozenset()))


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
