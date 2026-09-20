"""Persist the curator workflow independently from generated artifact existence."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from dromia import config as dromia_config

WORKFLOW_SCHEMA_VERSION = 1
POSE_FINGERPRINT_KEYS = (
    "frame_indices",
    "object_ids",
    "reviewed_keypoints_xy",
    "visibility",
    "frame_review_status",
    "frame_ground_truth",
    "keypoint_ground_truth",
    "keypoint_source",
    "was_manually_moved",
    "proposal_applied",
)
METRIC_CODE_PATHS = (
    dromia_config.REPO_ROOT / "src" / "dromia" / "gait" / "analysis.py",
    dromia_config.REPO_ROOT / "src" / "dromia" / "timebase.py",
    dromia_config.REPO_ROOT / "src" / "dromia" / "calibration.py",
)
TIMEBASE_CONTEXT_KEYS = (
    "schema_version",
    "video_sha256",
    "source_fps",
    "source_frame_count",
    "metadata_frame_count",
    "decoded_frame_count",
    "frame_count_mismatch",
    "variable_frame_rate",
    "real_world_fps",
    "real_world_timing_resolved",
    "timing_source",
    "temporal_calibration_source",
    "fps_override",
    "quicktime_full_frame_rate_playback_intent",
    "media_to_real_time_scale",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class PoseStage(BaseModel):
    status: Literal["not_updated", "current", "failed"] = "not_updated"
    source: Literal["none", "automatic_first_pass", "reviewed"] = "none"
    fingerprint: str | None = None
    revision: int = 0
    updated_at: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class MetricsStage(BaseModel):
    status: Literal["not_generated", "current", "stale", "failed"] = "not_generated"
    input_pose_fingerprint: str | None = None
    input_context_fingerprint: str | None = None
    timebase_fingerprint: str | None = None
    calibration_fingerprint: str | None = None
    config_fingerprint: str | None = None
    generated_at: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    stale_reason: str | None = None
    error: str | None = None


class TaskWorkflow(BaseModel):
    schema_version: int = WORKFLOW_SCHEMA_VERSION
    task_id: int
    scope: str
    runner_id: int | None = None
    pose: PoseStage = Field(default_factory=PoseStage)
    metrics: MetricsStage = Field(default_factory=MetricsStage)


def workflow_path(run: Path, task_id: int) -> Path:
    return run / "annotations" / "workflow" / f"task_{task_id}.json"


def load_workflow(
    run: Path,
    *,
    task_id: int,
    scope: str,
    runner_id: int | None,
) -> TaskWorkflow:
    path = workflow_path(run, task_id)
    if path.exists():
        return TaskWorkflow.model_validate_json(path.read_text(encoding="utf-8"))
    return TaskWorkflow(task_id=task_id, scope=scope, runner_id=runner_id)


def save_workflow(run: Path, state: TaskWorkflow) -> Path:
    path = workflow_path(run, state.task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def reviewed_pose_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as payload:
        for key in POSE_FINGERPRINT_KEYS:
            if key not in payload:
                continue
            array = np.ascontiguousarray(payload[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def record_pose_update(
    run: Path,
    state: TaskWorkflow,
    *,
    reviewed_pose_path: Path,
    artifacts: dict[str, str],
    source: Literal["automatic_first_pass", "reviewed"] = "reviewed",
) -> TaskWorkflow:
    fingerprint = reviewed_pose_fingerprint(reviewed_pose_path)
    parent_fingerprint = state.pose.fingerprint
    changed = fingerprint != parent_fingerprint
    revision = 0 if parent_fingerprint is None else state.pose.revision + (1 if changed else 0)
    state.pose = PoseStage(
        status="current",
        source=source,
        fingerprint=fingerprint,
        revision=revision,
        updated_at=utc_now(),
        artifacts=dict(artifacts),
    )
    if state.metrics.status == "current" and state.metrics.input_pose_fingerprint != fingerprint:
        state.metrics.status = "stale"
        state.metrics.stale_reason = "The reviewed pose changed after metrics were generated."
    save_pose_revision(run, state, reviewed_pose_path, parent_fingerprint=parent_fingerprint)
    save_workflow(run, state)
    return state


def record_metrics_success(
    run: Path,
    state: TaskWorkflow,
    *,
    artifacts: dict[str, str],
    context_fingerprint: str | None = None,
) -> TaskWorkflow:
    if state.pose.status != "current" or state.pose.fingerprint is None:
        raise ValueError("No pose is available for metric generation")
    state.metrics = MetricsStage(
        status="current",
        input_pose_fingerprint=state.pose.fingerprint,
        input_context_fingerprint=context_fingerprint,
        timebase_fingerprint=file_fingerprint(run / "timebase.json"),
        calibration_fingerprint=file_fingerprint(run / "calibration" / "ground_calibration.json"),
        config_fingerprint=file_fingerprint(run / "config.json"),
        generated_at=utc_now(),
        artifacts=dict(artifacts),
    )
    save_workflow(run, state)
    return state


def record_metrics_failure(run: Path, state: TaskWorkflow, error: Exception) -> TaskWorkflow:
    state.metrics.status = "failed"
    state.metrics.error = str(error)
    state.metrics.stale_reason = None
    save_workflow(run, state)
    return state


def metrics_are_current(state: TaskWorkflow, run: Path | None = None) -> bool:
    current = bool(
        state.metrics.status == "current"
        and state.pose.fingerprint is not None
        and state.metrics.input_pose_fingerprint == state.pose.fingerprint
    )
    if not current or run is None:
        return current
    return bool(
        state.metrics.input_context_fingerprint
        and state.metrics.input_context_fingerprint
        == metrics_context_fingerprint(run, runner_id=state.runner_id)
    )


def invalidate_if_context_changed(run: Path, state: TaskWorkflow) -> TaskWorkflow:
    if state.metrics.status == "current" and not metrics_are_current(state, run):
        return invalidate_metrics(
            run,
            state,
            "Timebase, calibration, shoe assignments, biomechanical configuration, "
            "or metric code changed.",
        )
    return state


def metrics_context_fingerprint(run: Path, *, runner_id: int | None = None) -> str:
    """Fingerprint every non-pose input that can change derived metrics."""

    digest = hashlib.sha256()
    _update_timebase_context(digest, run / "timebase.json")
    for relative in (
        "calibration/ground_calibration.json",
        "shoes/shoe_assignments.json",
        "manifest.json",
    ):
        path = run / relative
        digest.update(relative.encode())
        if path.exists():
            digest.update(path.read_bytes())
        else:
            digest.update(b"missing")
    for path in METRIC_CODE_PATHS:
        digest.update(str(path.relative_to(dromia_config.REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _update_timebase_context(digest: hashlib._Hash, path: Path) -> None:
    """Hash stable video timing, not runner-specific derived frame arrays."""

    digest.update(b"timebase.json")
    if not path.exists():
        digest.update(b"missing")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        digest.update(path.read_bytes())
        return
    stable = {key: payload.get(key) for key in TIMEBASE_CONTEXT_KEYS}
    digest.update(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode())


def invalidate_metrics(run: Path, state: TaskWorkflow, reason: str) -> TaskWorkflow:
    if state.metrics.status == "current":
        state.metrics.status = "stale"
        state.metrics.stale_reason = reason
        save_workflow(run, state)
    return state


def save_pose_revision(
    run: Path,
    state: TaskWorkflow,
    source: Path,
    *,
    parent_fingerprint: str | None,
) -> Path:
    assert state.pose.fingerprint is not None
    revision_dir = run / "annotations" / "revisions" / state.scope
    revision_dir.mkdir(parents=True, exist_ok=True)
    destination = revision_dir / f"pose_{state.pose.fingerprint[:16]}.npz"
    if not destination.exists():
        shutil.copy2(source, destination)
    manifest = revision_dir / f"pose_{state.pose.fingerprint[:16]}.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_id": state.task_id,
                    "scope": state.scope,
                    "runner_id": state.runner_id,
                    "fingerprint": state.pose.fingerprint,
                    "parent_fingerprint": parent_fingerprint,
                    "revision": state.pose.revision,
                    "created_at": state.pose.updated_at,
                    "artifact": destination.relative_to(run).as_posix(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return destination


def file_fingerprint(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
