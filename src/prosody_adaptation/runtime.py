from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
from pathlib import Path

import torch

from .config import file_sha256


def runtime_metadata(manifest=None, audio_cache=None, prosody_checkpoint=None):
    packages = {}
    for name in ("numpy", "PyYAML", "torch", "torchaudio", "transformers", "datasets"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    cache_metadata = Path(audio_cache) / "cache_metadata.json" if audio_cache else None
    return {
        "hostname": socket.gethostname(),
        "worker": "colab" if "COLAB_RELEASE_TAG" in os.environ else "local",
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "packages": packages,
        "manifest_sha256": file_sha256(manifest) if manifest else None,
        "audio_cache_metadata_sha256": (
            file_sha256(cache_metadata)
            if cache_metadata is not None and cache_metadata.exists()
            else None
        ),
        "prosody_checkpoint_sha256": (
            file_sha256(prosody_checkpoint) if prosody_checkpoint else None
        ),
    }
