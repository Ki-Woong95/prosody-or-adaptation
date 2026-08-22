from __future__ import annotations

import hashlib
import io
import json
import tarfile
import wave
import zipfile
from collections import OrderedDict
from pathlib import Path

from .buckeye import load_manifest
from .config import file_sha256


def _read_nested_wav(source_file: str) -> bytes:
    outer_path, nested_name = source_file.split("::", 1)
    recording = Path(nested_name).stem
    with zipfile.ZipFile(outer_path) as outer:
        nested_bytes = outer.read(nested_name)
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
        return nested.read(f"{recording}.wav")


def _clip_wav(payload: bytes, start_time: float, end_time: float) -> bytes:
    with wave.open(io.BytesIO(payload), "rb") as source:
        params = source.getparams()
        start = round(start_time * params.framerate)
        stop = round(end_time * params.framerate)
        source.setpos(start)
        frames = source.readframes(stop - start)
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams(params)
        target.setnframes(stop - start)
        target.writeframes(frames)
    return output.getvalue()


def _tar_info(name: str, size: int):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    return info


def materialize_tar_cache(manifest_path, output_dir, shard_size: int = 500):
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    rows = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    index = []
    source_cache: OrderedDict[str, bytes] = OrderedDict()
    shard_hashes = {}
    for shard_index in range((len(rows) + shard_size - 1) // shard_size):
        shard_rows = rows[shard_index * shard_size : (shard_index + 1) * shard_size]
        shard_name = f"buckeye-{shard_index:05d}.tar"
        shard_path = output_dir / shard_name
        with tarfile.open(shard_path, "w", format=tarfile.PAX_FORMAT) as archive:
            for row in shard_rows:
                if row.source_file not in source_cache:
                    source_cache[row.source_file] = _read_nested_wav(row.source_file)
                    while len(source_cache) > 2:
                        source_cache.popitem(last=False)
                audio = _clip_wav(
                    source_cache[row.source_file], row.start_time, row.end_time
                )
                member = f"{row.segment_id}.wav"
                archive.addfile(_tar_info(member, len(audio)), io.BytesIO(audio))
                index.append({
                    "segment_id": row.segment_id,
                    "shard": shard_name,
                    "member": member,
                    "audio_sha256": hashlib.sha256(audio).hexdigest(),
                    "sample_count": row.sample_count,
                })
        shard_hashes[shard_name] = file_sha256(shard_path)
    index_path = output_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for item in index:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    metadata = {
        "format": "deterministic-wav-tar-shards-v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "index_sha256": file_sha256(index_path),
        "shard_size": shard_size,
        "segment_count": len(index),
        "shard_sha256": shard_hashes,
    }
    (output_dir / "cache_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def validate_cache(manifest_path, cache_dir, verify_audio: bool = False):
    manifest_path, cache_dir = Path(manifest_path), Path(cache_dir)
    metadata = json.loads((cache_dir / "cache_metadata.json").read_text())
    if metadata["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("Cache manifest hash does not match the supplied manifest")
    index_path = cache_dir / "index.jsonl"
    if metadata["index_sha256"] != file_sha256(index_path):
        raise ValueError("Cache index hash mismatch")
    for shard, expected in metadata["shard_sha256"].items():
        if file_sha256(cache_dir / shard) != expected:
            raise ValueError(f"Cache shard hash mismatch: {shard}")
    manifest_ids = {row.segment_id for row in load_manifest(manifest_path)}
    index = [json.loads(line) for line in index_path.read_text().splitlines() if line]
    if {item["segment_id"] for item in index} != manifest_ids:
        raise ValueError("Cache index segment IDs do not match the manifest")
    if verify_audio:
        by_shard = {}
        for item in index:
            by_shard.setdefault(item["shard"], []).append(item)
        for shard, items in by_shard.items():
            with tarfile.open(cache_dir / shard) as archive:
                for item in items:
                    payload = archive.extractfile(item["member"]).read()
                    if hashlib.sha256(payload).hexdigest() != item["audio_sha256"]:
                        raise ValueError(f"Cached audio hash mismatch: {item['segment_id']}")
    return {
        "manifest_sha256": metadata["manifest_sha256"],
        "index_sha256": metadata["index_sha256"],
        "segments": len(index),
        "shards": len(metadata["shard_sha256"]),
    }


class TarCacheReader:
    def __init__(self, manifest_path, cache_dir):
        validate_cache(manifest_path, cache_dir)
        self.cache_dir = Path(cache_dir)
        self.index = {
            item["segment_id"]: item
            for item in (
                json.loads(line)
                for line in (self.cache_dir / "index.jsonl").read_text().splitlines()
                if line
            )
        }
        self._open_shard = None
        self._archive = None

    def read(self, segment_id: str) -> bytes:
        item = self.index[segment_id]
        if item["shard"] != self._open_shard:
            if self._archive is not None:
                self._archive.close()
            self._open_shard = item["shard"]
            # Kept open across reads to avoid reopening a shard for every sample.
            self._archive = tarfile.open(self.cache_dir / self._open_shard)  # noqa: SIM115
        payload = self._archive.extractfile(item["member"]).read()
        if hashlib.sha256(payload).hexdigest() != item["audio_sha256"]:
            raise ValueError(f"Cached audio hash mismatch: {segment_id}")
        return payload
