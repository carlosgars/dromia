"""Persistent single-worker analysis queue for the local CVAT bridge."""

from __future__ import annotations

import json
import queue
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import cv2
from pydantic import BaseModel, Field

from dromia import calibration as dromia_calibration
from dromia import config as dromia_config
from dromia import dto as dromia_dto
from dromia import temporal_calibration as dromia_temporal_calibration
from dromia import timebase as dromia_timebase
from dromia.pipeline import runner as dromia_pipeline
from dromia.review import cvat as cvat_annotation


class AnalysisCancelled(RuntimeError):
    pass


class AnalysisRecord(BaseModel):
    schema_version: int = 2
    analysis_id: str
    name: str
    view_name: str
    input_video: str
    status: Literal["ready", "queued", "running", "complete", "failed", "cancelled"] = "queued"
    stage: str = "queued"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    source_fps: float | None = Field(default=None, gt=0)
    source_frame_count: int | None = Field(default=None, gt=0)
    source_duration_s: float | None = Field(default=None, gt=0)
    capture_fps: float | None = Field(default=None, gt=0)
    capture_fps_reviewed: bool = False
    suggested_capture_fps: float | None = Field(default=None, gt=0)
    capture_fps_estimate_confidence: Literal["low", "medium", "high"] | None = None
    capture_fps_estimate_evidence: list[str] = Field(default_factory=list)
    capture_fps_auto_fill_recommended: bool = False
    real_world_duration_s: float | None = Field(default=None, gt=0)
    quicktime_full_frame_rate_playback_intent: bool | None = None
    preset: str = dromia_config.DEFAULT_POSE_VARIANT
    calibration_path: str | None = None
    run_dir: str | None = None
    cvat_project_id: int | None = None
    error: str | None = None
    cancel_requested: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ExistingRun(BaseModel):
    run_name: str
    video_name: str
    frame_count: int
    runner_ids: list[int]
    cvat_project_id: int | None = None
    calibrated: bool = False
    analysis_id: str | None = None


class AnalysisManager:
    def __init__(self, runs_dir: Path, *, start_worker: bool = True) -> None:
        self.runs_dir = runs_dir.resolve()
        self.state_dir = self.runs_dir / "_analyses"
        self.upload_dir = self.runs_dir / "_uploads"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.pending: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.state_lock = threading.RLock()
        if start_worker:
            self.recover_pending()
            threading.Thread(
                target=self._worker, name="dromia-analysis-worker", daemon=True
            ).start()

    def recover_pending(self) -> None:
        """Requeue work interrupted by a bridge restart without losing its run cache."""

        for record in reversed(self.list()):
            if record.status not in {"queued", "running"}:
                continue
            if record.status == "running":
                record.status = "queued"
                record.stage = "recovering_after_bridge_restart"
                record.progress = min(record.progress, 0.94)
                record.error = None
                self.save(record)
            self.pending.put(record.analysis_id)

    def create(
        self,
        video_path: Path,
        *,
        name: str,
        view_name: str,
        source_fps: float | None = None,
        source_frame_count: int | None = None,
        capture_fps: float | None = None,
        enqueue: bool = True,
    ) -> AnalysisRecord:
        analysis_id = uuid.uuid4().hex
        slow_motion_intent = dromia_timebase.quicktime_full_frame_rate_playback_intent(video_path)
        record = AnalysisRecord(
            analysis_id=analysis_id,
            name=name.strip() or video_path.stem,
            view_name=view_name.strip() or "sagittal",
            input_video=str(video_path.resolve()),
            status="queued" if enqueue else "ready",
            stage="queued" if enqueue else "awaiting_confirmation",
            source_fps=source_fps,
            source_frame_count=source_frame_count,
            source_duration_s=(
                source_frame_count / source_fps
                if source_fps is not None and source_frame_count is not None
                else None
            ),
            capture_fps=capture_fps,
            capture_fps_reviewed=capture_fps is not None,
            real_world_duration_s=(
                source_frame_count / capture_fps
                if capture_fps is not None and source_frame_count is not None
                else None
            ),
            quicktime_full_frame_rate_playback_intent=slow_motion_intent,
        )
        self.save(record)
        if capture_fps is None and source_fps is not None:
            try:
                self.estimate_capture_fps(record.analysis_id)
                record = self.get(record.analysis_id)
            except (OSError, ValueError):
                pass
        if enqueue:
            self.pending.put(analysis_id)
        return record

    def list_existing_runs(self) -> list[ExistingRun]:
        """List completed pipeline runs whose source video is still locally available."""

        analyses_by_run = {
            str(Path(record.run_dir).resolve()): record.analysis_id
            for record in self.list()
            if record.run_dir is not None and record.status == "complete"
        }
        result: list[ExistingRun] = []
        for manifest_path in self.runs_dir.glob("*/manifest.json"):
            try:
                run = manifest_path.parent.resolve()
                manifest = dromia_dto.RunManifest.model_validate_json(manifest_path.read_text())
                video = (run / manifest.source_video.path).resolve()
                if not video.is_file():
                    continue
                registry_path = run / "annotations" / "cvat" / "tasks.json"
                if not registry_path.is_file():
                    continue
                project_id = cvat_annotation.CvatTaskRegistry.model_validate_json(
                    registry_path.read_text()
                ).project_id
                result.append(
                    ExistingRun(
                        run_name=run.name,
                        video_name=video.name,
                        frame_count=manifest.frame_count,
                        runner_ids=manifest.accepted_runner_ids,
                        cvat_project_id=project_id,
                        calibrated=(run / "calibration" / "ground_calibration.json").is_file(),
                        analysis_id=analyses_by_run.get(str(run)),
                    )
                )
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda item: item.run_name, reverse=True)

    def attach_existing_run(self, run_name: str) -> AnalysisRecord:
        """Expose a synchronized run in the analysis UI without copying or reprocessing it."""

        if not run_name or Path(run_name).name != run_name:
            raise ValueError("Invalid DromIA run selection")
        run = (self.runs_dir / run_name).resolve()
        if run.parent != self.runs_dir:
            raise ValueError("Invalid DromIA run selection")
        manifest_path = run / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"DromIA run has no manifest: {run_name}")
        manifest = dromia_dto.RunManifest.model_validate_json(manifest_path.read_text())
        video = (run / manifest.source_video.path).resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Source video is not available: {video.name}")

        with self.state_lock:
            for existing in self.list():
                if (
                    existing.status == "complete"
                    and existing.run_dir is not None
                    and Path(existing.run_dir).resolve() == run
                ):
                    return existing

            capture = cv2.VideoCapture(str(video))
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                decoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()
            frame_count = manifest.frame_count or decoded_count
            fps_value = fps if fps > 0 else None
            registry_path = run / "annotations" / "cvat" / "tasks.json"
            if not registry_path.is_file():
                raise FileNotFoundError(f"DromIA run is not synchronized with CVAT: {run_name}")
            project_id = cvat_annotation.CvatTaskRegistry.model_validate_json(
                registry_path.read_text()
            ).project_id
            calibration_path = run / "calibration" / "ground_calibration.json"
            timebase_path = run / "timebase.json"
            persisted_timing = (
                dromia_timebase.VideoTimebase.model_validate_json(timebase_path.read_text())
                if timebase_path.is_file()
                else None
            )
            capture_fps = None if persisted_timing is None else persisted_timing.real_world_fps
            record = AnalysisRecord(
                analysis_id=uuid.uuid4().hex,
                name=video.stem,
                view_name="sagittal",
                input_video=str(video),
                status="complete",
                stage="complete",
                progress=1.0,
                source_fps=fps_value,
                source_frame_count=frame_count,
                source_duration_s=(frame_count / fps_value if fps_value else None),
                capture_fps=capture_fps,
                capture_fps_reviewed=capture_fps is not None,
                real_world_duration_s=(
                    frame_count / capture_fps if capture_fps is not None else None
                ),
                quicktime_full_frame_rate_playback_intent=(
                    None
                    if persisted_timing is None
                    else persisted_timing.quicktime_full_frame_rate_playback_intent
                ),
                calibration_path=str(calibration_path) if calibration_path.is_file() else None,
                run_dir=str(run),
                cvat_project_id=project_id,
            )
            self.save(record)
            try:
                self.estimate_capture_fps(record.analysis_id)
                return self.get(record.analysis_id)
            except (OSError, ValueError):
                return record

    def start(self, analysis_id: str, *, capture_fps: float | None = None) -> AnalysisRecord:
        with self.state_lock:
            record = self.get(analysis_id)
            if record.status != "ready":
                raise ValueError("Only an uploaded analysis awaiting confirmation can be started")
            if capture_fps is not None:
                if capture_fps <= 0:
                    raise ValueError("Capture FPS must be positive")
                record.capture_fps = capture_fps
                record.capture_fps_reviewed = True
                record.real_world_duration_s = (
                    record.source_frame_count / capture_fps
                    if record.source_frame_count is not None
                    else None
                )
            if (
                record.quicktime_full_frame_rate_playback_intent is False
                and record.capture_fps is None
            ):
                raise ValueError(
                    "This is an Apple slow-motion video. Enter its capture FPS before analysis."
                )
            record.status = "queued"
            record.stage = "queued"
            record.progress = 0.0
            self.save(record)
        self.pending.put(analysis_id)
        return record

    def get(self, analysis_id: str) -> AnalysisRecord:
        with self.state_lock:
            path = self.state_dir / f"{analysis_id}.json"
            if not path.exists():
                raise FileNotFoundError(f"Unknown analysis: {analysis_id}")
            return AnalysisRecord.model_validate_json(path.read_text())

    def list(self) -> list[AnalysisRecord]:
        with self.state_lock:
            return sorted(
                (
                    AnalysisRecord.model_validate_json(path.read_text())
                    for path in self.state_dir.glob("*.json")
                ),
                key=lambda record: record.created_at,
                reverse=True,
            )

    def save(self, record: AnalysisRecord) -> None:
        with self.state_lock:
            record.updated_at = datetime.now(UTC).isoformat()
            path = self.state_dir / f"{record.analysis_id}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(record.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            temporary.replace(path)

    def retry(self, analysis_id: str) -> AnalysisRecord:
        with self.state_lock:
            record = self.get(analysis_id)
            if record.status not in {"failed", "cancelled"}:
                raise ValueError("Only failed or cancelled analyses can be retried")
            record.status = "queued"
            record.stage = "queued"
            record.progress = 0.0
            record.error = None
            record.cancel_requested = False
            self.save(record)
        self.pending.put(analysis_id)
        return record

    def cancel(self, analysis_id: str) -> AnalysisRecord:
        with self.state_lock:
            record = self.get(analysis_id)
            if record.status in {"complete", "failed", "cancelled"}:
                return record
            record.cancel_requested = True
            if record.status in {"ready", "queued"}:
                record.status = "cancelled"
                record.stage = "cancelled"
            self.save(record)
            return record

    def attach_calibration(self, analysis_id: str, calibration_path: Path) -> AnalysisRecord:
        with self.state_lock:
            record = self.get(analysis_id)
            if record.status not in {"ready", "complete"}:
                raise ValueError("Calibration is available before start or after completion")
            record.calibration_path = str(calibration_path.resolve())
            self.save(record)
            return record

    def recalibrate_timing(self, analysis_id: str, capture_fps: float) -> AnalysisRecord:
        with self.state_lock:
            record = self.get(analysis_id)
            if record.status != "complete" or record.run_dir is None:
                raise ValueError("Timing can only be recalibrated for a completed analysis")
            dromia_temporal_calibration.recalibrate_run(Path(record.run_dir), capture_fps)
            record.capture_fps = capture_fps
            record.capture_fps_reviewed = True
            record.real_world_duration_s = (
                record.source_frame_count / capture_fps
                if record.source_frame_count is not None
                else None
            )
            self.save(record)
            return record

    def set_capture_fps(self, analysis_id: str, capture_fps: float) -> AnalysisRecord:
        with self.state_lock:
            record = self.get(analysis_id)
            if capture_fps <= 0:
                raise ValueError("Capture FPS must be positive")
            record.capture_fps = capture_fps
            record.capture_fps_reviewed = True
            record.real_world_duration_s = (
                record.source_frame_count / capture_fps
                if record.source_frame_count is not None
                else None
            )
            self.save(record)
            return record

    def apply_capture_fps(self, analysis_id: str, capture_fps: float) -> AnalysisRecord:
        record = self.get(analysis_id)
        return (
            self.recalibrate_timing(analysis_id, capture_fps)
            if record.status == "complete"
            else self.set_capture_fps(analysis_id, capture_fps)
            if record.status == "ready"
            else self._reject_capture_fps_status()
        )

    @staticmethod
    def _reject_capture_fps_status():
        raise ValueError("Capture FPS can be applied before start or after completion")

    def estimate_capture_fps(self, analysis_id: str) -> dict[str, object]:
        record = self.get(analysis_id)
        if record.source_fps is None:
            raise ValueError("Playback FPS is unavailable")
        estimate = dromia_temporal_calibration.estimate_capture_fps(
            Path(record.input_video),
            record.source_fps,
            run_dir=Path(record.run_dir) if record.run_dir is not None else None,
            current_capture_fps=(record.capture_fps if record.capture_fps_reviewed else None),
        )
        with self.state_lock:
            current = self.get(analysis_id)
            current.suggested_capture_fps = float(estimate["suggested_capture_fps"])
            current.capture_fps_estimate_confidence = str(estimate["confidence"])
            current.capture_fps_estimate_evidence = [str(value) for value in estimate["evidence"]]
            current.capture_fps_auto_fill_recommended = bool(estimate["auto_fill_recommended"])
            self.save(current)
        return estimate

    def _worker(self) -> None:
        while True:
            analysis_id = self.pending.get()
            try:
                try:
                    self.run_one(analysis_id)
                except Exception as exc:
                    # A malformed state file must not permanently kill the only worker.
                    try:
                        record = self.get(analysis_id)
                        record.status = "failed"
                        record.stage = "failed"
                        record.error = str(exc)
                        self.save(record)
                    except Exception:
                        pass
            finally:
                self.pending.task_done()

    def run_one(self, analysis_id: str) -> AnalysisRecord:
        with self.lock:
            record = self.get(analysis_id)
            if record.status == "cancelled":
                return record
            record.status = "running"
            record.stage = "starting"
            self.save(record)

            cfg = dromia_config.DromiaConfig(
                runs_dir=self.runs_dir,
                gait_analysis=dromia_config.GaitAnalysisConfig(
                    capture_fps_override=record.capture_fps
                ),
            )
            if record.run_dir is None:
                record.run_dir = str(
                    dromia_pipeline.make_run_dir(
                        Path(record.input_video),
                        self.runs_dir,
                        pose_model=cfg.pose.variant,
                    ).resolve()
                )
                self.save(record)
            if record.calibration_path is not None:
                calibration = dromia_calibration.GroundCalibration.model_validate_json(
                    Path(record.calibration_path).read_text()
                )
                dromia_calibration.save_ground_calibration(Path(record.run_dir), calibration)

            def progress(stage: str, fraction: float) -> None:
                with self.state_lock:
                    current = self.get(analysis_id)
                    if current.cancel_requested:
                        raise AnalysisCancelled("Analysis cancelled by user")
                    current.stage = stage
                    current.progress = fraction
                    self.save(current)

            try:
                run_dir = Path(record.run_dir)
                manifest_path = run_dir / "manifest.json"
                if not manifest_path.exists():
                    dromia_pipeline.run(
                        Path(record.input_video),
                        cfg=cfg,
                        progress_callback=progress,
                        resume_run_dir=run_dir,
                    )
                record = self.get(analysis_id)
                record.run_dir = str(run_dir)
                self.save(record)
                progress("creating_cvat_tasks", 0.95)
                registry_path = run_dir / "annotations" / "cvat" / "tasks.json"
                registry = (
                    cvat_annotation.CvatTaskRegistry.model_validate_json(registry_path.read_text())
                    if registry_path.exists()
                    else cvat_annotation.push_run(
                        run_dir,
                        cvat_annotation.CvatConnection.from_environment(),
                    )
                )
                progress("finalizing", 0.99)
                record = self.get(analysis_id)
                record.cvat_project_id = registry.project_id
                record.status = "complete"
                record.stage = "complete"
                record.progress = 1.0
            except AnalysisCancelled:
                record = self.get(analysis_id)
                record.status = "cancelled"
                record.stage = "cancelled"
            except Exception as exc:
                record = self.get(analysis_id)
                record.status = "failed"
                record.stage = "failed"
                record.error = str(exc)
            self.save(record)
            return record
