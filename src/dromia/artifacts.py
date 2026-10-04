"""Small, shared helpers for run-local artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dromia import dto


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON without exposing a partially written artifact."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Resolve artifacts without mutating their portable manifest paths."""

    root: Path

    @classmethod
    def open(cls, root: Path) -> RunLayout:
        return cls(root.expanduser().resolve())

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def load_manifest(self) -> dto.RunManifest:
        return dto.RunManifest.model_validate_json(self.manifest_path.read_text())

    def artifact(self, name: str, manifest: dto.RunManifest | None = None) -> Path:
        current = manifest or self.load_manifest()
        try:
            relative = current.artifacts[name]
        except KeyError as exc:
            raise ValueError(f"Run manifest has no {name!r} artifact") from exc
        path = (self.root / relative).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"Artifact escapes run directory: {name}")
        return path

    def relative(self, path: Path) -> str:
        return path.expanduser().resolve().relative_to(self.root).as_posix()

