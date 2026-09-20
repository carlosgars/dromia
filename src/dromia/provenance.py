"""Reproducibility metadata for scientific DromIA runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dromia import config as dromia_config

COTRACKER_CHECKPOINT_SHA256 = "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"


def build_run_provenance(
    *,
    cfg: dromia_config.DromiaConfig,
    video_sha256: str,
    stage_durations_s: dict[str, float],
) -> dict[str, Any]:
    checkpoint = selected_checkpoint(cfg)
    config_json = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(cfg.repo_root),
        "git_dirty": git_dirty(cfg.repo_root),
        "cvat_git_commit": git_commit(cfg.repo_root.parent / "dromia-cvat"),
        "cvat_git_dirty": git_dirty(cfg.repo_root.parent / "dromia-cvat"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "video_sha256": video_sha256,
        "pose_variant": cfg.pose.variant,
        "sam_model": {
            "backend": "mlx",
            "model_id": cfg.sam.model_id,
            "revision": cfg.sam.revision,
            "sha256": cfg.sam.model_sha256,
        },
        "cotracker_model": {
            "name": cfg.cotracker.model_name,
            "sha256": COTRACKER_CHECKPOINT_SHA256,
        },
        "bboxmaskpose_git_commit": git_commit(cfg.pose.pmpose_root),
        "model_checkpoint": None
        if checkpoint is None
        else {
            "name": checkpoint.name,
            "size_bytes": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
        },
        "random_seed": cfg.random_seed,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "stage_durations_s": {key: round(value, 6) for key, value in stage_durations_s.items()},
    }


def save_run_provenance(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = run_dir / "provenance.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def selected_checkpoint(cfg: dromia_config.DromiaConfig) -> Path | None:
    path = cfg.pose.pmpose_checkpoint_path
    resolved = path.expanduser().resolve()
    return resolved if resolved.is_file() else None


def checkpoint_sha256(path: Path) -> str | None:
    resolved = path.expanduser().resolve()
    return file_sha256(resolved) if resolved.is_file() else None


def git_commit(repo_root: Path) -> str | None:
    if not repo_root.is_dir():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def git_dirty(repo_root: Path) -> bool | None:
    if not repo_root.is_dir():
        return None
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return None if result.returncode else bool(result.stdout.strip())


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
