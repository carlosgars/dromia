"""Propagate reviewed CVAT keypoint corrections with CoTracker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from dromia import config as dromia_config
from dromia import dto as dromia_dto
from dromia.models import cotracker as cotracker_tracking
from dromia.review import cvat as cvat_annotation
from dromia.review import human_conditioning as human_conditioning

TRACKED_JOINTS = (13, 14, 15, 16)
PROPAGATION_ALGORITHM_VERSION = "local_correction_windows_v4"
JOINT_TO_LOCAL = {joint_id: idx for idx, joint_id in enumerate(cvat_annotation.JOINT_IDS)}
CONNECTED = {
    13: ((11, "femur"), (15, "tibia")),
    14: ((12, "femur"), (16, "tibia")),
    15: ((13, "tibia"),),
    16: ((14, "tibia"),),
}


class PropagationConfig(BaseModel):
    correction_min_px: float = 0.5
    tracker_correction_min_px: float = 3.0
    tracker_window_radius: int = Field(default=24, ge=1)
    max_mask_distance_norm: float = 0.08
    max_step_norm: float = 0.20
    max_bone_relative_error: float = 0.55
    input_max_width: int = 640
    conditioning: human_conditioning.HumanConditioningConfig = Field(
        default_factory=human_conditioning.HumanConditioningConfig
    )


class CorrectionSeed(BaseModel):
    runner_id: int
    joint_id: int
    frame_idx: int
    time_idx: int
    xy: tuple[float, float]
    posterior_delta_px: float


class PropagationDiagnostics(BaseModel):
    task_id: int
    status: str
    seeds: list[CorrectionSeed] = Field(default_factory=list)
    seed_hashes: list[str] = Field(default_factory=list)
    applied_count: int = 0
    applied_without_tracker_visibility: int = 0
    applied_frames: list[int] = Field(default_factory=list)
    rejected_reasons: dict[str, int] = Field(default_factory=dict)
    quality_warnings: dict[str, int] = Field(default_factory=dict)
    trusted_observation_count: int = 0
    tracker_window_count: int = 0
    tracker_frame_count: int = 0
    identity_swap_frame_count: int = 0
    identity_swap_frames: dict[str, list[int]] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)


def propagate_run(
    run_dir: Path,
    connection: cvat_annotation.CvatConnection,
    cfg: PropagationConfig | None = None,
    task_id: int | None = None,
) -> PropagationDiagnostics:
    config = cfg or PropagationConfig()
    run = run_dir.expanduser().resolve()
    record = cvat_annotation.find_task_record(run, task_id)
    bundle = cvat_annotation.bundle_for_record(run, record)
    try:
        from cvat_sdk import make_client, models
    except ImportError as exc:
        raise RuntimeError("Install the annotation extra with: uv sync --extra annotation") from exc

    with make_client(host=connection.host) as client:
        client.login((connection.username, connection.password))
        task = client.tasks.retrieve(record.task_id)
        labels = task.get_labels()
        existing_annotations = task.get_annotations()
        reviewed = cvat_annotation.parse_annotations(existing_annotations, labels)
        reviewed = cvat_annotation.remap_reviewed_frames_to_source(reviewed, bundle)
        cvat_annotation.write_ground_truth(run, reviewed, task_id=record.task_id, bundle=bundle)
        state = cvat_annotation.annotation_state(bundle, reviewed)
        previous = previous_proposal_mask(run, bundle, state.points_xy)
        state = restore_previous_proposals(bundle, state, previous)
        trusted = human_conditioning.trusted_observations(
            bundle,
            state,
            previous_proposal=previous,
            cfg=config.conditioning,
        )
        tracker_observations = tracker_observations_for_corrections(trusted, config)
        seeds = [correction_seed(item) for item in tracker_observations]
        hashes = [trusted_hash(item) for item in trusted]
        diagnostics = PropagationDiagnostics(
            task_id=record.task_id,
            status="no_corrections",
            seeds=seeds,
            seed_hashes=hashes,
            trusted_observation_count=len(trusted),
        )
        if not trusted:
            write_diagnostics(run, diagnostics, bundle)
            return diagnostics
        if not has_new_seed_hashes(run, hashes, bundle):
            diagnostics.status = "no_new_corrections"
            write_diagnostics(run, diagnostics, bundle)
            return diagnostics

        tracker_cfg = dromia_config.CoTrackerConfig(
            enabled=True,
            input_max_width=config.input_max_width,
        )
        tracker_seeds = [
            cotracker_tracking.PointSeed(
                runner_id=seed.runner_id,
                keypoint_id=seed.joint_id,
                keypoint_name=cotracker_tracking.JOINT_NAMES[seed.joint_id],
                seed_frame_idx=seed.frame_idx,
                seed_time_idx=seed.time_idx,
                xy=seed.xy,
                confidence=1.0,
                mask_distance_norm=0.0,
            )
            for seed in seeds
        ]
        tracked_xy = np.empty((0, len(bundle.frame_indices), 2), dtype=np.float32)
        tracker_visibility = np.empty((0, len(bundle.frame_indices)), dtype=bool)
        if tracker_seeds:
            try:
                result, window_count, tracked_frame_count = track_correction_windows(
                    Path(bundle.input_video),
                    tracker_media_frame_indices(bundle),
                    tracker_seeds,
                    tracker_cfg,
                    radius=config.tracker_window_radius,
                )
                tracked_xy = result.xy
                tracker_visibility = result.visibility
                diagnostics.tracker_window_count = window_count
                diagnostics.tracker_frame_count = tracked_frame_count
            except Exception as exc:
                diagnostics.quality_warnings["TRACKER_FAILED"] = 1
                diagnostics.quality_warnings[f"TRACKER_ERROR:{type(exc).__name__}"] = 1
        manifest = cvat_annotation.load_manifest(run)
        with np.load(
            cvat_annotation.required_artifact(run, manifest, "pose_npz"),
            allow_pickle=False,
        ) as pose:
            bboxes = cvat_annotation.select_pose_bboxes(
                run,
                pose,
                np.asarray(bundle.frame_indices, dtype=np.int32),
                np.asarray(bundle.runner_ids, dtype=np.int32),
            )
        conditioned = human_conditioning.condition_pose(
            run_dir=run,
            bundle=bundle,
            state=state,
            observations=trusted,
            tracker_observations=tracker_observations,
            tracker_xy=tracked_xy,
            tracker_visibility=tracker_visibility,
            bboxes_xyxy=bboxes,
            cfg=config.conditioning,
            propagation_observations=tracker_observations,
        )
        updated = conditioned.state
        applied = conditioned.applied
        payload = cvat_annotation.build_annotations(
            models,
            labels,
            bundle,
            updated,
            existing_annotations=existing_annotations,
        )
        if any(track.id is None for track in payload.tracks):
            raise RuntimeError(
                "Cannot update CVAT annotations in place: existing DromIA track not found"
            )
        task.update_annotations(models.PatchedLabeledDataRequest(**payload.to_dict()))
        artifacts = save_propagation(
            run,
            bundle,
            updated,
            tracked_xy,
            tracker_visibility,
            applied,
            conditioned.source_anchor,
        )
        artifacts.update(human_conditioning.save_result(run, bundle, conditioned))
        diagnostics.status = "complete"
        diagnostics.applied_count = int(np.count_nonzero(applied))
        diagnostics.applied_without_tracker_visibility = int(
            np.count_nonzero(conditioned.source == "TEMPORAL_PRIOR")
        )
        diagnostics.applied_frames = sorted(
            {bundle.frame_indices[t] for t in np.where(np.any(applied, axis=(1, 2)))[0].tolist()}
        )
        diagnostics.identity_swap_frame_count = conditioned.diagnostics.identity_swap_frame_count
        diagnostics.identity_swap_frames = conditioned.diagnostics.identity_swap_frames
        diagnostics.quality_warnings.update(
            {
                "TRACKER_PRIOR_USED": conditioned.diagnostics.tracker_used_count,
                "TEMPORAL_PRIOR_USED": conditioned.diagnostics.transport_used_count,
            }
        )
        diagnostics.artifacts = artifacts
        save_processed_seed_hashes(run, hashes, bundle)
        write_diagnostics(run, diagnostics, bundle)
        return diagnostics


def tracker_media_frame_indices(bundle: cvat_annotation.CvatBundle) -> list[int]:
    """Return frame numbers in the media file read by CoTracker.

    Runner-specific CVAT tasks use a cropped video whose frames start at zero,
    while ``bundle.frame_indices`` deliberately retains the original source-video
    frame numbers. Seeking the source numbers in the cropped clip shifts every
    propagation window. Older/full-video bundles may not have a separate local
    timebase, so they safely fall back to the source indices.
    """

    media_frames = bundle.task_frame_indices or bundle.frame_indices
    if len(media_frames) != len(bundle.frame_indices):
        raise ValueError(
            "CVAT task/source frame mappings must have the same length "
            f"({len(media_frames)} != {len(bundle.frame_indices)})"
        )
    return list(media_frames)


def tracker_observations_for_corrections(
    observations: list[human_conditioning.TrustedObservation],
    cfg: PropagationConfig,
) -> list[human_conditioning.TrustedObservation]:
    material_corrections = [
        item
        for item in observations
        if item.joint_id in TRACKED_JOINTS
        and item.source != "HUMAN_ACCEPTED"
        and item.posterior_delta_px >= cfg.tracker_correction_min_px
    ]
    return human_conditioning.select_tracker_observations(
        material_corrections,
        cfg.conditioning,
    )


def track_correction_windows(
    video_path: Path,
    frame_indices: list[int],
    seeds: list[cotracker_tracking.PointSeed],
    cfg: dromia_config.CoTrackerConfig,
    *,
    radius: int,
) -> tuple[cotracker_tracking.TrackResult, int, int]:
    """Track corrections only where the downstream conditioner can use them.

    CoTracker cost grows sharply with video length.  Human conditioning is local,
    so tracking hundreds or thousands of frames outside every correction window is
    both expensive and unable to influence the result.  Overlapping windows are
    merged so nearby corrected joints share one model invocation.
    """

    frame_count = len(frame_indices)
    full_xy = np.full((len(seeds), frame_count, 2), np.nan, dtype=np.float32)
    full_visibility = np.zeros((len(seeds), frame_count), dtype=bool)
    if not seeds or frame_count == 0:
        return (
            cotracker_tracking.TrackResult(
                xy=full_xy,
                visibility=full_visibility,
                input_size_wh=(0, 0),
                inference_seconds=0.0,
            ),
            0,
            0,
        )

    intervals = sorted(
        (
            max(0, seed.seed_time_idx - radius),
            min(frame_count - 1, seed.seed_time_idx + radius),
            seed_idx,
        )
        for seed_idx, seed in enumerate(seeds)
    )
    windows: list[tuple[int, int, list[int]]] = []
    for start, end, seed_idx in intervals:
        if windows and start <= windows[-1][1] + 1:
            previous_start, previous_end, previous_seeds = windows[-1]
            windows[-1] = (previous_start, max(previous_end, end), [*previous_seeds, seed_idx])
        else:
            windows.append((start, end, [seed_idx]))

    input_size_wh = (0, 0)
    inference_seconds = 0.0
    tracked_frame_count = 0
    for start, end, seed_indices in windows:
        local_seeds = [
            seeds[seed_idx].model_copy(
                update={"seed_time_idx": seeds[seed_idx].seed_time_idx - start}
            )
            for seed_idx in seed_indices
        ]
        local_result = cotracker_tracking.track_points(
            video_path,
            frame_indices[start : end + 1],
            local_seeds,
            cfg,
        )
        full_xy[np.asarray(seed_indices), start : end + 1] = local_result.xy
        full_visibility[np.asarray(seed_indices), start : end + 1] = local_result.visibility
        input_size_wh = local_result.input_size_wh
        inference_seconds += local_result.inference_seconds
        tracked_frame_count += end - start + 1

    return (
        cotracker_tracking.TrackResult(
            xy=full_xy,
            visibility=full_visibility,
            input_size_wh=input_size_wh,
            inference_seconds=inference_seconds,
        ),
        len(windows),
        tracked_frame_count,
    )


def trusted_hash(observation: human_conditioning.TrustedObservation) -> str:
    value = (
        f"{PROPAGATION_ALGORITHM_VERSION}:{human_conditioning.ALGORITHM_VERSION}:"
        f"{observation.source}:"
        f"{observation.runner_id}:{observation.joint_id}:{observation.frame_idx}:"
        f"{observation.xy[0]:.3f}:{observation.xy[1]:.3f}"
    )
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def correction_seed(observation: human_conditioning.TrustedObservation) -> CorrectionSeed:
    return CorrectionSeed(
        runner_id=observation.runner_id,
        joint_id=observation.joint_id,
        frame_idx=observation.frame_idx,
        time_idx=observation.time_idx,
        xy=observation.xy,
        posterior_delta_px=observation.posterior_delta_px,
    )


def propagation_output_dir(run: Path, bundle: cvat_annotation.CvatBundle | None = None) -> Path:
    root = run / "annotations" / "propagation"
    return root if bundle is None or bundle.runner_id is None else root / bundle.scope


def load_processed_seed_hashes(
    run: Path,
    bundle: cvat_annotation.CvatBundle | None = None,
) -> set[str]:
    path = propagation_output_dir(run, bundle) / "processed_seeds.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("seed_hashes", []))


def has_new_seed_hashes(
    run: Path,
    hashes: list[str],
    bundle: cvat_annotation.CvatBundle | None = None,
) -> bool:
    return bool(hashes) and not set(hashes).issubset(load_processed_seed_hashes(run, bundle))


def save_processed_seed_hashes(
    run: Path,
    hashes: list[str],
    bundle: cvat_annotation.CvatBundle | None = None,
) -> None:
    path = propagation_output_dir(run, bundle) / "processed_seeds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_processed_seed_hashes(run, bundle)
    existing.update(hashes)
    path.write_text(json.dumps({"seed_hashes": sorted(existing)}, indent=2), encoding="utf-8")


def previous_proposal_mask(
    run: Path,
    bundle: cvat_annotation.CvatBundle,
    current_points: np.ndarray,
) -> np.ndarray:
    shape = current_points.shape[:-1]
    path = propagation_output_dir(run, bundle) / "propagation.npz"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    with np.load(path, allow_pickle=False) as data:
        aligned = (
            np.array_equal(data["frame_indices"], np.asarray(bundle.frame_indices))
            and np.array_equal(data["runner_ids"], np.asarray(bundle.runner_ids))
            and data["proposal_applied"].shape == shape
        )
        if not aligned:
            return np.zeros(shape, dtype=bool)
        unchanged = np.isclose(data["propagated_xy"], current_points, atol=0.25).all(axis=-1)
        return data["proposal_applied"].astype(bool) & unchanged


def restore_previous_proposals(
    bundle: cvat_annotation.CvatBundle,
    state: cvat_annotation.AnnotationState,
    previous_proposal: np.ndarray,
) -> cvat_annotation.AnnotationState:
    """Remove unchanged DromIA proposals before recomputing a propagation revision."""

    if not np.any(previous_proposal):
        return state
    with np.load(bundle.preannotations_npz, allow_pickle=False) as data:
        posterior = np.asarray(data["posterior_keypoints_xy"], dtype=np.float32)
    restored = cvat_annotation.AnnotationState(
        points_xy=state.points_xy.copy(),
        visibility=state.visibility.copy(),
        review_status=state.review_status.copy(),
        frame_ground_truth=state.frame_ground_truth.copy(),
        keypoint_ground_truth=state.keypoint_ground_truth.copy(),
    )
    restored.points_xy[previous_proposal] = posterior[previous_proposal]
    restored.visibility[previous_proposal] = 2
    return restored


def correction_seeds(
    bundle: cvat_annotation.CvatBundle,
    state: cvat_annotation.AnnotationState,
    cfg: PropagationConfig,
) -> list[CorrectionSeed]:
    with np.load(bundle.preannotations_npz, allow_pickle=False) as data:
        posterior = np.asarray(data["posterior_keypoints_xy"], dtype=np.float32)
    seeds: list[CorrectionSeed] = []
    for t, frame_idx in enumerate(bundle.frame_indices):
        for obj_idx, runner_id in enumerate(bundle.runner_ids):
            if state.review_status[t, obj_idx] != "CORRECTED":
                continue
            for joint_id in TRACKED_JOINTS:
                joint_idx = JOINT_TO_LOCAL[joint_id]
                point = state.points_xy[t, obj_idx, joint_idx]
                original = posterior[t, obj_idx, joint_idx]
                if state.visibility[t, obj_idx, joint_idx] == 0:
                    continue
                if not np.isfinite(point).all() or not np.isfinite(original).all():
                    continue
                delta = float(np.linalg.norm(point - original))
                if delta <= cfg.correction_min_px:
                    continue
                seeds.append(
                    CorrectionSeed(
                        runner_id=runner_id,
                        joint_id=joint_id,
                        frame_idx=frame_idx,
                        time_idx=t,
                        xy=(float(point[0]), float(point[1])),
                        posterior_delta_px=delta,
                    )
                )
    return seeds


def apply_tracks(
    *,
    bundle: cvat_annotation.CvatBundle,
    state: cvat_annotation.AnnotationState,
    seeds: list[CorrectionSeed],
    tracked_xy: np.ndarray,
    tracker_visibility: np.ndarray,
    sam_frames: list[dromia_dto.SamFrame],
    bboxes_xyxy: np.ndarray,
    cfg: PropagationConfig,
) -> tuple[cvat_annotation.AnnotationState, np.ndarray, dict[str, int], dict[str, int], np.ndarray]:
    output = cvat_annotation.AnnotationState(
        points_xy=state.points_xy.copy(),
        visibility=state.visibility.copy(),
        review_status=state.review_status.copy(),
        frame_ground_truth=state.frame_ground_truth.copy(),
        keypoint_ground_truth=state.keypoint_ground_truth.copy(),
    )
    applied = np.zeros(state.visibility.shape, dtype=bool)
    source_seed = np.full(state.visibility.shape, -1, dtype=np.int32)
    rejected: dict[str, int] = {}
    warnings: dict[str, int] = {}
    frame_lookup = {
        (frame.frame_idx, detection.obj_id): detection
        for frame in sam_frames
        for detection in frame.runners
    }
    medians = bone_medians(state.points_xy)
    for t, frame_idx in enumerate(bundle.frame_indices):
        for obj_idx, runner_id in enumerate(bundle.runner_ids):
            if state.review_status[t, obj_idx] != "UNREVIEWED":
                continue
            detection = frame_lookup.get((frame_idx, runner_id))
            candidate_bbox = np.asarray(bboxes_xyxy[t, obj_idx], dtype=np.float32)
            bbox = candidate_bbox if np.isfinite(candidate_bbox).all() else None
            for joint_id in TRACKED_JOINTS:
                joint_idx = JOINT_TO_LOCAL[joint_id]
                candidates = [
                    idx
                    for idx, seed in enumerate(seeds)
                    if seed.runner_id == runner_id and seed.joint_id == joint_id
                ]
                candidates.sort(key=lambda idx: abs(t - seeds[idx].time_idx))
                for seed_idx in candidates:
                    reason = candidate_rejection(
                        candidate=tracked_xy[seed_idx, t],
                        detection=detection,
                        bbox=bbox,
                    )
                    if reason is not None:
                        rejected[reason] = rejected.get(reason, 0) + 1
                        continue
                    for warning in candidate_warnings(
                        t=t,
                        joint_id=joint_id,
                        candidate=tracked_xy[seed_idx, t],
                        track=tracked_xy[seed_idx],
                        visible=tracker_visibility[seed_idx],
                        detection=detection,
                        bbox=bbox,
                        pose=output.points_xy[t, obj_idx],
                        bone_median=medians[obj_idx],
                        cfg=cfg,
                    ):
                        warnings[warning] = warnings.get(warning, 0) + 1
                    output.points_xy[t, obj_idx, joint_idx] = tracked_xy[seed_idx, t]
                    output.visibility[t, obj_idx, joint_idx] = 2
                    applied[t, obj_idx, joint_idx] = True
                    source_seed[t, obj_idx, joint_idx] = seed_idx
                    break
    return output, applied, rejected, warnings, source_seed


def candidate_rejection(
    *,
    candidate: np.ndarray,
    detection: object | None,
    bbox: np.ndarray | None,
) -> str | None:
    if not np.isfinite(candidate).all():
        return "TRACKER_POINT_MISSING"
    if detection is None or bbox is None:
        return "RUNNER_NOT_PRESENT"
    return None


def candidate_warnings(
    *,
    t: int,
    joint_id: int,
    candidate: np.ndarray,
    track: np.ndarray,
    visible: np.ndarray,
    detection: object,
    bbox: np.ndarray,
    pose: np.ndarray,
    bone_median: dict[tuple[int, int], float],
    cfg: PropagationConfig,
) -> list[str]:
    warnings: list[str] = []
    if not bool(visible[t]):
        warnings.append("TRACKER_NOT_VISIBLE")
    bbox_h = max(float(bbox[3] - bbox[1]), 1.0)
    if (
        point_mask_distance_norm(candidate, np.asarray(detection.mask) > 0, bbox_h)
        > cfg.max_mask_distance_norm
    ):
        warnings.append("OUTSIDE_RUNNER_MASK")
    if t > 0 and np.isfinite(track[t - 1]).all():
        if float(np.linalg.norm(candidate - track[t - 1]) / max(bbox_h, 1.0)) > cfg.max_step_norm:
            warnings.append("LARGE_TRACKER_STEP")
    for neighbor_id, bone_name in CONNECTED[joint_id]:
        neighbor = pose[JOINT_TO_LOCAL[neighbor_id]]
        median = bone_median.get((joint_id, neighbor_id))
        if median is None or not np.isfinite(neighbor).all():
            continue
        length = float(np.linalg.norm(candidate - neighbor))
        if abs(length - median) / max(median, 1.0) > cfg.max_bone_relative_error:
            warnings.append(f"{bone_name.upper()}_LENGTH_OUTLIER")
    return warnings


def bone_medians(points: np.ndarray) -> list[dict[tuple[int, int], float]]:
    pairs = ((13, 11), (13, 15), (14, 12), (14, 16), (15, 13), (16, 14))
    output: list[dict[tuple[int, int], float]] = []
    for obj_idx in range(points.shape[1]):
        values: dict[tuple[int, int], float] = {}
        for joint_id, neighbor_id in pairs:
            a = points[:, obj_idx, JOINT_TO_LOCAL[joint_id]]
            b = points[:, obj_idx, JOINT_TO_LOCAL[neighbor_id]]
            lengths = np.linalg.norm(a - b, axis=-1)
            finite = lengths[np.isfinite(lengths)]
            if finite.size:
                values[(joint_id, neighbor_id)] = float(np.median(finite))
        output.append(values)
    return output


def point_mask_distance_norm(point: np.ndarray, mask: np.ndarray, bbox_h: float) -> float:
    x, y = np.rint(point).astype(int)
    if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return float("inf")
    if mask[y, x]:
        return 0.0
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    return float(distance[y, x] / max(bbox_h, 1.0))


def save_propagation(
    run: Path,
    bundle: cvat_annotation.CvatBundle,
    after: cvat_annotation.AnnotationState,
    tracker_xy: np.ndarray,
    tracker_visibility: np.ndarray,
    applied: np.ndarray,
    source_seed: np.ndarray,
) -> dict[str, str]:
    output_dir = propagation_output_dir(run, bundle)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "propagation.npz"
    np.savez_compressed(
        path,
        frame_indices=np.asarray(bundle.frame_indices, dtype=np.int32),
        runner_ids=np.asarray(bundle.runner_ids, dtype=np.int32),
        joint_ids=np.asarray(cvat_annotation.JOINT_IDS, dtype=np.int32),
        propagated_xy=after.points_xy,
        tracker_xy=tracker_xy,
        tracker_visibility=tracker_visibility,
        proposal_applied=applied,
        source_seed_index=source_seed,
    )
    return {"propagation_npz": str(path.resolve())}


def write_diagnostics(
    run: Path,
    diagnostics: PropagationDiagnostics,
    bundle: cvat_annotation.CvatBundle | None = None,
) -> None:
    output_dir = propagation_output_dir(run, bundle)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "diagnostics.json"
    diagnostics.artifacts["propagation_diagnostics"] = str(path.resolve())
    path.write_text(json.dumps(diagnostics.model_dump(mode="json"), indent=2), encoding="utf-8")
