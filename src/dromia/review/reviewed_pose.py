"""Build and render the current CVAT-reviewed pose."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from dromia import dto as dromia_dto
from dromia.gait import analysis as gait_analysis_postprocess
from dromia.models import cache as sam_cache
from dromia.review import cvat as cvat_annotation
from dromia.review import propagation as annotation_propagation
from dromia.review import workflow as annotation_workflow
from dromia.viz import overlay as overlay_viz

SOURCE_POSTERIOR = "POSTERIOR"
SOURCE_ACCEPTED = "HUMAN_ACCEPTED"
SOURCE_CORRECTED = "HUMAN_CORRECTED"
SOURCE_FRAME_GROUND_TRUTH = "HUMAN_FRAME_GROUND_TRUTH"
SOURCE_PROPAGATED = "TRACKER_PROPAGATED"
SOURCE_UNREVIEWED_EDIT = "UNREVIEWED_EDIT"
SOURCE_AMBIGUOUS = "AMBIGUOUS"
SOURCE_NOT_VISIBLE = "NOT_VISIBLE"
SOURCE_OUT_OF_FRAME = "OUT_OF_FRAME"

SOURCE_COLORS = {
    SOURCE_POSTERIOR: (165, 165, 165),
    SOURCE_ACCEPTED: (70, 220, 80),
    SOURCE_CORRECTED: (230, 220, 30),
    SOURCE_FRAME_GROUND_TRUTH: (40, 255, 40),
    SOURCE_PROPAGATED: (30, 220, 255),
    SOURCE_UNREVIEWED_EDIT: (220, 80, 220),
    SOURCE_AMBIGUOUS: (20, 145, 255),
}


class ReviewedPoseData(dromia_dto.ArrayModel):
    frame_indices: np.ndarray
    object_ids: np.ndarray
    posterior_xy: np.ndarray
    reviewed_xy: np.ndarray
    visibility: np.ndarray
    review_status: np.ndarray
    frame_ground_truth: np.ndarray
    keypoint_ground_truth: np.ndarray
    keypoint_source: np.ndarray
    was_manually_moved: np.ndarray
    proposal_applied: np.ndarray


class ReviewedPoseSummary(BaseModel):
    task_id: int
    scope: str = "full_runner"
    runner_id: int | None = None
    frame_count: int
    runner_ids: list[int]
    source_counts: dict[str, int]
    human_reviewed_frame_count: int
    propagated_keypoint_count: int
    unreviewed_edit_count: int
    ground_truth_frame_count: int = 0
    ground_truth_keypoint_count: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)


class SyncSummary(BaseModel):
    propagation: annotation_propagation.PropagationDiagnostics
    reviewed_pose: ReviewedPoseSummary
    workflow: annotation_workflow.TaskWorkflow | None = None


class MetricsGenerationSummary(BaseModel):
    task_id: int
    runner_id: int | None = None
    artifacts: dict[str, str]
    workflow: annotation_workflow.TaskWorkflow


def render_run(
    run_dir: Path,
    connection: cvat_annotation.CvatConnection,
    task_id: int | None = None,
) -> ReviewedPoseSummary:
    run = run_dir.expanduser().resolve()
    record = cvat_annotation.find_task_record(run, task_id)
    bundle = cvat_annotation.bundle_for_record(run, record)
    reviewed = fetch_annotations(record, connection)
    reviewed = cvat_annotation.remap_reviewed_frames_to_source(reviewed, bundle)
    cvat_annotation.write_ground_truth(run, reviewed, task_id=record.task_id, bundle=bundle)
    state = cvat_annotation.annotation_state(bundle, reviewed)
    data = build_reviewed_pose(run, bundle, state)
    artifacts = save_reviewed_pose(run, data, bundle)
    artifacts.update(write_reviewed_videos(run, data, bundle))
    summary = summarize(record, data, artifacts)
    manifest_path = reviewed_manifest_path(run, bundle)
    summary.artifacts["reviewed_pose_manifest"] = str(manifest_path.resolve())
    manifest_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    state = annotation_workflow.load_workflow(
        run,
        task_id=record.task_id,
        scope=record.scope,
        runner_id=record.runner_id,
    )
    annotation_workflow.record_pose_update(
        run,
        state,
        reviewed_pose_path=Path(artifacts["reviewed_pose_npz"]),
        artifacts=summary.artifacts,
    )
    return summary


def sync_run(
    run_dir: Path,
    connection: cvat_annotation.CvatConnection,
    task_id: int | None = None,
) -> SyncSummary:
    propagation = annotation_propagation.propagate_run(run_dir, connection, None, task_id)
    reviewed = render_run(run_dir, connection, task_id)
    run = run_dir.expanduser().resolve()
    try:
        record = cvat_annotation.find_task_record(run, task_id)
        state = annotation_workflow.load_workflow(
            run,
            task_id=record.task_id,
            scope=record.scope,
            runner_id=record.runner_id,
        )
    except (OSError, ValueError):
        state = None
    return SyncSummary(propagation=propagation, reviewed_pose=reviewed, workflow=state)


def generate_metrics_run(
    run_dir: Path,
    task_id: int | None = None,
) -> MetricsGenerationSummary:
    """Generate gait metrics from reviewed pose or untouched canonical first pass."""

    run = run_dir.expanduser().resolve()
    record = cvat_annotation.find_task_record(run, task_id)
    bundle = cvat_annotation.bundle_for_record(run, record)
    state = annotation_workflow.load_workflow(
        run,
        task_id=record.task_id,
        scope=record.scope,
        runner_id=record.runner_id,
    )
    pose_path = run / "annotations" / f"reviewed_pose{scope_suffix(bundle)}.npz"
    if state.pose.status != "current" or state.pose.fingerprint is None:
        state = initialize_automatic_metrics_pose(run, bundle, state, pose_path)
    if not pose_path.exists():
        raise FileNotFoundError(f"Reviewed pose does not exist: {pose_path}")
    current_fingerprint = annotation_workflow.reviewed_pose_fingerprint(pose_path)
    if current_fingerprint != state.pose.fingerprint:
        raise ValueError("The reviewed pose changed outside Update Pose; update it again")
    try:
        artifacts = gait_analysis_postprocess.analyze_run_pose(
            run,
            pose_path,
            runner_id=bundle.runner_id,
        )
        context_fingerprint = annotation_workflow.metrics_context_fingerprint(
            run, runner_id=bundle.runner_id
        )
        state = annotation_workflow.record_metrics_success(
            run,
            state,
            artifacts=artifacts,
            context_fingerprint=context_fingerprint,
        )
    except Exception as exc:
        annotation_workflow.record_metrics_failure(run, state, exc)
        raise
    metrics_manifest = run / "gait" / f"metrics_manifest{scope_suffix(bundle)}.json"
    metrics_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "task_id": record.task_id,
                "runner_id": bundle.runner_id,
                "input_pose_source": state.pose.source,
                "input_pose_fingerprint": state.pose.fingerprint,
                "input_context_fingerprint": state.metrics.input_context_fingerprint,
                "timebase_fingerprint": state.metrics.timebase_fingerprint,
                "calibration_fingerprint": state.metrics.calibration_fingerprint,
                "config_fingerprint": state.metrics.config_fingerprint,
                "generated_at": state.metrics.generated_at,
                "artifacts": artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts = {**artifacts, "metrics_manifest": str(metrics_manifest.resolve())}
    state.metrics.artifacts = artifacts
    annotation_workflow.save_workflow(run, state)
    return MetricsGenerationSummary(
        task_id=record.task_id,
        runner_id=bundle.runner_id,
        artifacts=artifacts,
        workflow=state,
    )


def initialize_automatic_metrics_pose(
    run: Path,
    bundle: cvat_annotation.CvatBundle,
    state: annotation_workflow.TaskWorkflow,
    pose_path: Path,
) -> annotation_workflow.TaskWorkflow:
    """Materialize an immutable metrics input from the canonical automatic pose."""

    automatic_state = cvat_annotation.annotation_state(bundle)
    data = build_reviewed_pose(run, bundle, automatic_state)
    artifacts = save_reviewed_pose(run, data, bundle)
    if Path(artifacts["reviewed_pose_npz"]) != pose_path:
        raise ValueError("Automatic metrics pose was written to an unexpected scope")
    return annotation_workflow.record_pose_update(
        run,
        state,
        reviewed_pose_path=pose_path,
        artifacts=artifacts,
        source="automatic_first_pass",
    )


def fetch_annotations(
    record: cvat_annotation.CvatTaskRecord,
    connection: cvat_annotation.CvatConnection,
) -> list[cvat_annotation.ReviewedFrame]:
    try:
        from cvat_sdk import make_client
    except ImportError as exc:
        raise RuntimeError("Install the annotation extra with: uv sync --extra annotation") from exc
    with make_client(host=connection.host) as client:
        client.login((connection.username, connection.password))
        task = client.tasks.retrieve(record.task_id)
        return cvat_annotation.parse_annotations(task.get_annotations(), task.get_labels())


def build_reviewed_pose(
    run: Path,
    bundle: cvat_annotation.CvatBundle,
    state: cvat_annotation.AnnotationState,
) -> ReviewedPoseData:
    manifest = cvat_annotation.load_manifest(run)
    automatic_path = manifest.artifacts.get("first_pass_pose_npz") or (
        cvat_annotation.required_artifact(manifest, "posterior_npz")
    )
    posterior_data = np.load(Path(automatic_path))
    posterior = select_posterior_pose(posterior_data, bundle)
    reviewed = posterior.copy()
    visibility = (np.isfinite(posterior).all(axis=-1) * 2).astype(np.uint8)
    source = np.full(posterior.shape[:-1], SOURCE_POSTERIOR, dtype="<U24")
    manually_moved = np.zeros(posterior.shape[:-1], dtype=bool)
    propagated = load_propagation_flags(run, bundle, posterior.shape[:-1])

    for local_idx, joint_id in enumerate(cvat_annotation.JOINT_IDS):
        if joint_id == cvat_annotation.NECK_JOINT_ID:
            continue
        points = state.points_xy[:, :, local_idx]
        point_visibility = state.visibility[:, :, local_idx]
        delta = np.linalg.norm(points - posterior[:, :, joint_id], axis=-1)
        finite = np.isfinite(points).all(axis=-1)
        reviewed[:, :, joint_id] = points
        visibility[:, :, joint_id] = point_visibility
        manually_moved[:, :, joint_id] = (
            finite & (delta > 0.5) & (state.review_status == "CORRECTED")
        )
        source[:, :, joint_id] = source_for_points(
            state.review_status,
            manually_moved[:, :, joint_id],
            propagated[:, :, local_idx],
            finite & (delta > 0.5),
        )
        source[:, :, joint_id][state.keypoint_ground_truth[:, :, local_idx]] = (
            SOURCE_FRAME_GROUND_TRUTH
        )

    hidden = np.isin(source, [SOURCE_NOT_VISIBLE, SOURCE_OUT_OF_FRAME]) | (visibility == 0)
    reviewed[hidden] = np.nan
    visibility[hidden] = 0
    return ReviewedPoseData(
        frame_indices=np.asarray(bundle.frame_indices, dtype=np.int32),
        object_ids=np.asarray(bundle.runner_ids, dtype=np.int32),
        posterior_xy=posterior,
        reviewed_xy=reviewed,
        visibility=visibility,
        review_status=state.review_status.copy(),
        frame_ground_truth=state.frame_ground_truth.copy(),
        keypoint_ground_truth=state.keypoint_ground_truth.copy(),
        keypoint_source=source,
        was_manually_moved=manually_moved,
        proposal_applied=expand_lower_body(propagated, posterior.shape[:-1]),
    )


def select_posterior_pose(posterior_data: object, bundle: cvat_annotation.CvatBundle) -> np.ndarray:
    frame_lookup = {int(value): idx for idx, value in enumerate(posterior_data["frame_indices"])}
    id_key = "object_ids" if "object_ids" in posterior_data else "runner_ids"
    runner_lookup = {int(value): idx for idx, value in enumerate(posterior_data[id_key])}
    try:
        frames = [frame_lookup[value] for value in bundle.frame_indices]
        runners = [runner_lookup[value] for value in bundle.runner_ids]
    except KeyError as exc:
        raise ValueError("Posterior pose does not align with the CVAT task scope") from exc
    key = (
        "first_pass_keypoints_xy"
        if "first_pass_keypoints_xy" in posterior_data
        else "posterior_keypoints_xy"
    )
    posterior = np.asarray(posterior_data[key], dtype=np.float32)
    return posterior[np.ix_(frames, runners)]


def source_for_points(
    status: np.ndarray,
    moved_corrected: np.ndarray,
    propagated: np.ndarray,
    differs_from_posterior: np.ndarray,
) -> np.ndarray:
    source = np.full(status.shape, SOURCE_POSTERIOR, dtype="<U24")
    source[(status == "UNREVIEWED") & propagated] = SOURCE_PROPAGATED
    source[(status == "UNREVIEWED") & ~propagated & differs_from_posterior] = SOURCE_UNREVIEWED_EDIT
    source[status == "ACCEPTED"] = SOURCE_ACCEPTED
    source[status == "CORRECTED"] = SOURCE_ACCEPTED
    source[moved_corrected] = SOURCE_CORRECTED
    source[status == "AMBIGUOUS"] = SOURCE_AMBIGUOUS
    source[status == "NOT_VISIBLE"] = SOURCE_NOT_VISIBLE
    source[status == "OUT_OF_FRAME"] = SOURCE_OUT_OF_FRAME
    return source


def load_propagation_flags(
    run: Path,
    bundle: cvat_annotation.CvatBundle,
    full_shape: tuple[int, ...],
) -> np.ndarray:
    shape = (*full_shape[:2], len(cvat_annotation.JOINT_IDS))
    path = annotation_propagation.propagation_output_dir(run, bundle) / "propagation.npz"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    data = np.load(path)
    aligned = (
        np.array_equal(data["frame_indices"], np.asarray(bundle.frame_indices))
        and np.array_equal(data["runner_ids"], np.asarray(bundle.runner_ids))
        and data["proposal_applied"].shape == shape
    )
    return data["proposal_applied"].astype(bool) if aligned else np.zeros(shape, dtype=bool)


def expand_lower_body(lower: np.ndarray, full_shape: tuple[int, ...]) -> np.ndarray:
    output = np.zeros(full_shape, dtype=bool)
    for local_idx, joint_id in enumerate(cvat_annotation.JOINT_IDS):
        if joint_id == cvat_annotation.NECK_JOINT_ID:
            continue
        output[:, :, joint_id] = lower[:, :, local_idx]
    return output


def save_reviewed_pose(
    run: Path,
    data: ReviewedPoseData,
    bundle: cvat_annotation.CvatBundle,
) -> dict[str, str]:
    path = run / "annotations" / f"reviewed_pose{scope_suffix(bundle)}.npz"
    np.savez_compressed(
        path,
        frame_indices=data.frame_indices,
        object_ids=data.object_ids,
        posterior_keypoints_xy=data.posterior_xy,
        reviewed_keypoints_xy=data.reviewed_xy,
        visibility=data.visibility,
        frame_review_status=data.review_status,
        frame_ground_truth=data.frame_ground_truth,
        keypoint_ground_truth=data.keypoint_ground_truth,
        keypoint_source=data.keypoint_source,
        was_manually_moved=data.was_manually_moved,
        proposal_applied=data.proposal_applied,
    )
    resolved = str(path.resolve())
    return {"reviewed_pose_npz": resolved, "metrics_pose_npz": resolved}


def summarize(
    record: cvat_annotation.CvatTaskRecord,
    data: ReviewedPoseData,
    artifacts: dict[str, str],
) -> ReviewedPoseSummary:
    unique, counts = np.unique(data.keypoint_source, return_counts=True)
    source_counts = {str(name): int(count) for name, count in zip(unique, counts, strict=True)}
    return ReviewedPoseSummary(
        task_id=record.task_id,
        scope=record.scope,
        runner_id=record.runner_id,
        frame_count=len(data.frame_indices),
        runner_ids=data.object_ids.tolist(),
        source_counts=source_counts,
        human_reviewed_frame_count=int(
            np.count_nonzero((data.review_status != "UNREVIEWED") | data.frame_ground_truth)
        ),
        propagated_keypoint_count=source_counts.get(SOURCE_PROPAGATED, 0),
        unreviewed_edit_count=source_counts.get(SOURCE_UNREVIEWED_EDIT, 0),
        ground_truth_frame_count=int(np.count_nonzero(data.frame_ground_truth)),
        ground_truth_keypoint_count=int(np.count_nonzero(data.keypoint_ground_truth)),
        artifacts=artifacts,
    )


def write_reviewed_videos(
    run: Path,
    data: ReviewedPoseData,
    bundle: cvat_annotation.CvatBundle,
) -> dict[str, str]:
    manifest = cvat_annotation.load_manifest(run)
    video_path = Path(manifest.input_video)
    sam_frames = [
        sam_cache.load_frame(path) for path in sam_cache.frame_paths(Path(manifest.sam_cache_dir))
    ]
    frames_by_idx = {frame.frame_idx: frame for frame in sam_frames}
    suffix = scope_suffix(bundle)
    reviewed_path = run / "videos" / f"reviewed_pose{suffix}.mp4"
    comparison_path = run / "videos" / f"posterior_vs_reviewed{suffix}.mp4"
    write_videos(video_path, reviewed_path, comparison_path, data, frames_by_idx)
    reviewed_browser = browser_video(reviewed_path)
    comparison_browser = browser_video(comparison_path)
    return {
        "reviewed_pose_video": str(reviewed_path.resolve()),
        "posterior_vs_reviewed_video": str(comparison_path.resolve()),
        "reviewed_pose_browser_video": str(reviewed_browser.resolve()),
        "posterior_vs_reviewed_browser_video": str(comparison_browser.resolve()),
    }


def scope_suffix(bundle: cvat_annotation.CvatBundle) -> str:
    return "" if bundle.runner_id is None else f"_runner_{bundle.runner_id}"


def reviewed_manifest_path(run: Path, bundle: cvat_annotation.CvatBundle) -> Path:
    return run / "annotations" / f"reviewed_pose_manifest{scope_suffix(bundle)}.json"


def browser_video(source: Path) -> Path:
    avconvert = shutil.which("avconvert")
    if avconvert is None:
        return source
    output = source.with_name(f"{source.stem}_browser.m4v")
    subprocess.run(
        [
            avconvert,
            "--source",
            str(source),
            "--preset",
            "PresetAppleM4V1080pHD",
            "--output",
            str(output),
            "--replace",
        ],
        check=True,
        capture_output=True,
    )
    return output


def write_videos(
    video_path: Path,
    reviewed_path: Path,
    comparison_path: Path,
    data: ReviewedPoseData,
    frames_by_idx: dict[int, dromia_dto.SamFrame],
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    panel_w = min(width, 960)
    panel_h = max(2, int(round(height * panel_w / max(width, 1))))
    panel_h += panel_h % 2
    reviewed_path.parent.mkdir(parents=True, exist_ok=True)
    reviewed_writer = video_writer(reviewed_path, fps, (width, height))
    comparison_writer = video_writer(comparison_path, fps, (panel_w * 2, panel_h))
    try:
        for t, frame_idx in enumerate(data.frame_indices.tolist()):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                continue
            sam_frame = frames_by_idx.get(frame_idx)
            base = frame
            if sam_frame is not None:
                runners = [item for item in sam_frame.runners if item.obj_id in data.object_ids]
                base = overlay_viz.draw_runner_masks(frame, runners)
            posterior = draw_posterior(base, data.posterior_xy[t])
            reviewed = draw_reviewed_frame(base, data, t)
            reviewed_writer.write(reviewed)
            left = cv2.resize(posterior, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            right = cv2.resize(reviewed, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            cv2.putText(
                left, "POSTERIOR", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                right, "CVAT REVIEWED", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            comparison_writer.write(np.concatenate([left, right], axis=1))
    finally:
        capture.release()
        reviewed_writer.release()
        comparison_writer.release()


def video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not write video: {path}")
    return writer


def draw_posterior(frame: np.ndarray, poses: np.ndarray) -> np.ndarray:
    output = frame.copy()
    for pose in poses:
        output = overlay_viz.draw_pose(
            output,
            pose,
            line_color=(165, 165, 165),
            point_color=(165, 165, 165),
            line_thickness=2,
            point_radius=3,
        )
    return output


def draw_reviewed_frame(frame: np.ndarray, data: ReviewedPoseData, t: int) -> np.ndarray:
    output = frame.copy()
    for obj_idx, runner_id in enumerate(data.object_ids.tolist()):
        output = draw_source_pose(
            output,
            data.reviewed_xy[t, obj_idx],
            data.visibility[t, obj_idx],
            data.keypoint_source[t, obj_idx],
        )
        cv2.putText(
            output,
            f"R#{runner_id} {data.review_status[t, obj_idx]}"
            f"{' GT' if data.frame_ground_truth[t, obj_idx] else ''}",
            (12, 52 + obj_idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return draw_legend(output)


def draw_source_pose(
    frame: np.ndarray,
    points: np.ndarray,
    visibility: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    output = frame.copy()
    for a, b in overlay_viz.SKELETON:
        if not visible_point(points[a], visibility[a]) or not visible_point(
            points[b], visibility[b]
        ):
            continue
        color = SOURCE_COLORS.get(str(sources[b]), SOURCE_COLORS[SOURCE_POSTERIOR])
        cv2.line(output, tuple(points[a].astype(int)), tuple(points[b].astype(int)), color, 2)
    for point, visible, source in zip(points, visibility, sources, strict=True):
        if not visible_point(point, visible):
            continue
        color = SOURCE_COLORS.get(str(source), SOURCE_COLORS[SOURCE_POSTERIOR])
        cv2.circle(output, tuple(point.astype(int)), 4, color, -1)
    return output


def visible_point(point: np.ndarray, visibility: int) -> bool:
    return bool(visibility > 0 and np.isfinite(point).all())


def draw_legend(frame: np.ndarray) -> np.ndarray:
    output = frame.copy()
    labels = (
        (SOURCE_CORRECTED, "human corrected"),
        (SOURCE_ACCEPTED, "human accepted"),
        (SOURCE_PROPAGATED, "tracker proposal"),
        (SOURCE_POSTERIOR, "posterior"),
    )
    x = max(12, output.shape[1] - 185)
    for idx, (source, label) in enumerate(labels):
        y = 20 + idx * 18
        cv2.circle(output, (x, y - 4), 4, SOURCE_COLORS[source], -1)
        cv2.putText(output, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 245, 245), 1)
    return output
