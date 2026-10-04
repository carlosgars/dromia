"""Detect and conservatively repair catastrophic per-joint pose drift.

The module is deliberately array-oriented and side-effect free. Model and tracker
candidates are supplied by the caller, which makes the acceptance policy testable
without loading either model. Only flagged joint-frames can ever be changed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dromia import config as dromia_config

METHOD_NONE = np.uint8(0)
METHOD_PLAIN = np.uint8(1)
METHOD_TRACKER = np.uint8(2)
METHOD_INTERPOLATION = np.uint8(3)
METHOD_NAMES = {0: "none", 1: "plain_pmpose", 2: "cotracker", 3: "boundary_interpolation"}
REASON_BBOX = np.uint8(1)
REASON_VELOCITY = np.uint8(2)
REASON_HEATMAP_ALIGNMENT = np.uint8(8)
REASON_BILATERAL_COLLAPSE = np.uint8(16)
REASON_MIXED_POSE = np.uint8(32)
REASON_IDENTITY_ANCHOR_MISMATCH = np.uint8(64)
REASON_NAMES = {
    int(REASON_BBOX): "bbox_exit",
    int(REASON_VELOCITY): "velocity",
    int(REASON_HEATMAP_ALIGNMENT): "heatmap_alignment",
    int(REASON_BILATERAL_COLLAPSE): "bilateral_collapse",
    int(REASON_MIXED_POSE): "mixed_pose",
    int(REASON_IDENTITY_ANCHOR_MISMATCH): "identity_anchor_mismatch",
}


@dataclass(slots=True, frozen=True)
class JointSegment:
    object_index: int
    joint_id: int
    start: int
    end: int
    context_start: int
    context_end: int


@dataclass(slots=True)
class RepairResult:
    keypoints_xy: np.ndarray
    flagged: np.ndarray
    reasons: np.ndarray
    method: np.ndarray
    displacement_px: np.ndarray
    tracker_visibility: np.ndarray
    segments: list[JointSegment]
    runtime_seconds: float


def unchanged_repair_result(pose_xy: np.ndarray) -> RepairResult:
    """Return zero-valued provenance for a deliberately skipped repair stage."""

    pose = np.asarray(pose_xy, dtype=np.float32)
    shape = pose.shape[:3]
    return RepairResult(
        keypoints_xy=pose.copy(),
        flagged=np.zeros(shape, dtype=bool),
        reasons=np.zeros(shape, dtype=np.uint8),
        method=np.zeros(shape, dtype=np.uint8),
        displacement_px=np.zeros(shape, dtype=np.float32),
        tracker_visibility=np.full(shape, np.nan, dtype=np.float32),
        segments=[],
        runtime_seconds=0.0,
    )


def detect_catastrophic_drift(
    pose_xy: np.ndarray,
    bboxes_xyxy: np.ndarray,
    alignment_error_px: np.ndarray | None,
    cfg: dromia_config.AutoPoseRepairConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flags and a reason bitmask, including transient bilateral collapse."""

    pose = np.asarray(pose_xy, dtype=np.float32)
    boxes = np.asarray(bboxes_xyxy, dtype=np.float32)
    reasons = np.zeros(pose.shape[:3], dtype=np.uint8)
    heights = np.maximum(boxes[..., 3] - boxes[..., 1], 1.0)
    finite = np.all(np.isfinite(pose), axis=-1)

    pad = cfg.bbox_padding_height_fraction * heights
    outside = finite & (
        (pose[..., 0] < boxes[..., 0, None] - pad[..., None])
        | (pose[..., 0] > boxes[..., 2, None] + pad[..., None])
        | (pose[..., 1] < boxes[..., 1, None] - pad[..., None])
        | (pose[..., 1] > boxes[..., 3, None] + pad[..., None])
    )
    reasons[outside] |= REASON_BBOX

    velocity = np.full(pose.shape[:3], np.nan, dtype=np.float32)
    velocity[1:] = np.linalg.norm(pose[1:] - pose[:-1], axis=-1) / heights[1:, :, None]
    severe_jump = velocity > cfg.severe_jump_threshold_norm
    weak_jump = velocity > cfg.weak_jump_threshold_norm
    reasons[weak_jump] |= REASON_VELOCITY

    heatmap_bad = np.zeros_like(finite)
    if alignment_error_px is not None:
        errors = np.asarray(alignment_error_px, dtype=np.float32)
        heatmap_bad = np.isfinite(errors) & (errors > cfg.heatmap_alignment_threshold_px)
        reasons[heatmap_bad] |= REASON_HEATMAP_ALIGNMENT

    bilateral_collapse = transient_bilateral_collapse(pose, boxes, cfg)
    reasons[bilateral_collapse] |= REASON_BILATERAL_COLLAPSE

    allowed = np.zeros(pose.shape[2], dtype=bool)
    allowed[list(cfg.joint_ids)] = True
    core_ids = tuple(
        joint_id
        for joint_id in cfg.mixed_pose_core_joint_ids
        if joint_id < pose.shape[2] and allowed[joint_id]
    )
    mixed_pose = np.zeros(pose.shape[:2], dtype=bool)
    if len(core_ids) >= cfg.mixed_pose_min_core_jumps:
        mixed_pose = (
            np.count_nonzero(severe_jump[..., list(core_ids)], axis=-1)
            >= cfg.mixed_pose_min_core_jumps
        )
    mixed_pose_scope = np.zeros(pose.shape[2], dtype=bool)
    mixed_pose_scope[list(core_ids)] = True
    mixed_pose_joint = mixed_pose[..., None] & mixed_pose_scope[None, None, :]
    reasons[mixed_pose_joint] |= REASON_MIXED_POSE

    severe = outside | severe_jump | heatmap_bad | bilateral_collapse | mixed_pose_joint
    weak_count = weak_jump.astype(np.uint8)
    per_joint_scope = np.zeros(pose.shape[2], dtype=bool)
    per_joint_scope[
        [joint_id for joint_id in cfg.per_joint_trigger_ids if joint_id < pose.shape[2]]
    ] = True
    flagged = finite & (
        ((severe | (weak_count >= 2)) & per_joint_scope[None, None, :]) | mixed_pose_joint
    )
    flagged &= allowed[None, None, :]
    return flagged, reasons


def transient_bilateral_collapse(
    pose_xy: np.ndarray,
    bboxes_xyxy: np.ndarray,
    cfg: dromia_config.AutoPoseRepairConfig,
) -> np.ndarray:
    """Flag a brief knee/ankle channel collapse while preserving real occlusion spans."""

    pose = np.asarray(pose_xy, dtype=np.float32)
    boxes = np.asarray(bboxes_xyxy, dtype=np.float32)
    heights = np.maximum(boxes[..., 3] - boxes[..., 1], 1.0)
    output = np.zeros(pose.shape[:3], dtype=bool)
    window = cfg.bilateral_neighbor_window
    for left, right in ((13, 14), (15, 16)):
        pair = pose[..., [left, right], :]
        finite = np.isfinite(pair).all(axis=(-1, -2))
        separation = np.linalg.norm(pair[..., 0, :] - pair[..., 1, :], axis=-1) / heights
        collapsed = finite & (separation < cfg.bilateral_collapse_max_separation_norm)
        for t, obj_idx in np.argwhere(collapsed):
            previous = separation[max(0, t - window) : t, obj_idx]
            following = separation[t + 1 : min(len(separation), t + window + 1), obj_idx]
            previous = previous[np.isfinite(previous)]
            following = following[np.isfinite(following)]
            if not previous.size or not following.size:
                continue
            if (
                float(np.median(previous)) >= cfg.bilateral_neighbor_min_separation_norm
                and float(np.median(following)) >= cfg.bilateral_neighbor_min_separation_norm
            ):
                output[t, obj_idx, left] = True
                output[t, obj_idx, right] = True
    return output


def detect_unstable_intervals(
    pose_xy: np.ndarray,
    bboxes_xyxy: np.ndarray,
    model_quality: np.ndarray | None,
    cfg: dromia_config.AutoPoseRepairConfig,
    *,
    alignment_error_px: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local anomalies into intervals bracketed by stable anchor blocks.

    A strong variation opens an interval.  It remains open through apparently
    smooth-but-uncertain frames and closes at the first frame of a stable future
    block.  The recovery frame is therefore retained as the right anchor rather
    than being repaired merely because the return itself creates a large step.
    """

    pose = np.asarray(pose_xy, dtype=np.float32)
    boxes = np.asarray(bboxes_xyxy, dtype=np.float32)
    _, reasons = detect_catastrophic_drift(pose, boxes, alignment_error_px, cfg)
    heights = np.maximum(boxes[..., 3] - boxes[..., 1], 1.0)
    finite = np.all(np.isfinite(pose), axis=-1)
    velocity = np.full(pose.shape[:3], np.nan, dtype=np.float32)
    velocity[1:] = np.linalg.norm(pose[1:] - pose[:-1], axis=-1) / heights[1:, :, None]

    quality_good = np.ones(pose.shape[:3], dtype=bool)
    low_quality = np.zeros(pose.shape[:3], dtype=bool)
    if model_quality is not None:
        quality = np.asarray(model_quality, dtype=np.float32)
        if quality.shape != pose.shape[:3]:
            raise ValueError("model_quality must align with pose [T,O,K]")
        for obj_idx in range(pose.shape[1]):
            for joint_id in cfg.joint_ids:
                values = quality[:, obj_idx, joint_id]
                usable = values[np.isfinite(values)]
                if not usable.size:
                    continue
                floor = float(np.median(usable)) * cfg.stable_quality_median_fraction
                good = np.isfinite(values) & (values >= floor)
                quality_good[:, obj_idx, joint_id] = good
                low_quality[:, obj_idx, joint_id] = ~good

    pair_separated = np.ones(pose.shape[:3], dtype=bool)
    collapsed = np.zeros(pose.shape[:3], dtype=bool)
    for left, right in ((13, 14), (15, 16)):
        separation = np.linalg.norm(pose[:, :, left] - pose[:, :, right], axis=-1) / heights
        separated = np.isfinite(separation) & (
            separation >= cfg.bilateral_neighbor_min_separation_norm
        )
        close = np.isfinite(separation) & (separation < cfg.bilateral_collapse_max_separation_norm)
        pair_separated[..., left] = separated
        pair_separated[..., right] = separated
        collapsed[..., left] = close
        collapsed[..., right] = close
    reasons[collapsed] |= REASON_BILATERAL_COLLAPSE

    # Open an interval only on an event boundary.  Treating every frame of a
    # close bilateral pair as a new seed turns genuine occlusions into long
    # repairs.  A collapse can therefore strengthen a large variation only on
    # the frame where the pair first converges; otherwise model quality must
    # independently indicate that the large variation is unreliable.
    collapse_entry = collapsed.copy()
    collapse_entry[1:] &= ~collapsed[:-1]
    pair_large_variation = np.zeros_like(collapsed)
    for left, right in ((13, 14), (15, 16)):
        pair_event = (velocity[..., left] > cfg.severe_jump_threshold_norm) | (
            velocity[..., right] > cfg.severe_jump_threshold_norm
        )
        pair_large_variation[..., left] = pair_event
        pair_large_variation[..., right] = pair_event
    hard_invalid = (reasons & (REASON_BBOX | REASON_HEATMAP_ALIGNMENT)) != 0
    mixed_pose_event = (reasons & REASON_MIXED_POSE) != 0
    per_joint_scope = np.zeros(pose.shape[2], dtype=bool)
    per_joint_scope[
        [joint_id for joint_id in cfg.per_joint_trigger_ids if joint_id < pose.shape[2]]
    ] = True
    seeds = (
        hard_invalid | ((velocity > cfg.severe_jump_threshold_norm) & low_quality)
    ) & per_joint_scope[None, None, :]
    # A multi-core jump must open an interval even when the pose model is
    # confidently attached to the wrong person. It is not excluded from a
    # future stable block, however, so the equally large jump back to the
    # correct runner can remain the right-hand anchor.
    seeds |= mixed_pose_event
    seeds |= collapse_entry & pair_large_variation
    allowed = np.zeros(pose.shape[2], dtype=bool)
    allowed[list(cfg.joint_ids)] = True
    seeds &= allowed[None, None, :]
    output = np.zeros_like(seeds)
    block = cfg.stable_anchor_frames
    for obj_idx in range(pose.shape[1]):
        for joint_id in cfg.joint_ids:
            stable_frame = (
                finite[:, obj_idx, joint_id]
                & pair_separated[:, obj_idx, joint_id]
                & quality_good[:, obj_idx, joint_id]
                & ~hard_invalid[:, obj_idx, joint_id]
            )
            stable_start = np.zeros(len(pose), dtype=bool)
            stable_end = np.zeros(len(pose), dtype=bool)
            for start in range(0, len(pose) - block + 1):
                end = start + block - 1
                internal_steps = velocity[start + 1 : end + 1, obj_idx, joint_id]
                stable = bool(np.all(stable_frame[start : end + 1])) and bool(
                    np.all(np.isfinite(internal_steps))
                    and np.all(internal_steps <= cfg.weak_jump_threshold_norm)
                )
                if stable:
                    stable_start[start] = True
                    stable_end[end] = True

            seed_indices = np.flatnonzero(seeds[:, obj_idx, joint_id])
            cursor = 0
            while cursor < len(seed_indices):
                seed = int(seed_indices[cursor])
                left_candidates = np.flatnonzero(stable_end[:seed])
                right_candidates = np.flatnonzero(stable_start[seed + 1 :])
                left = int(left_candidates[-1]) if left_candidates.size else None
                right = int(seed + 1 + right_candidates[0]) if right_candidates.size else None
                if (
                    left is not None
                    and right is not None
                    and 0 < right - left - 1 <= cfg.interpolation_max_gap
                ):
                    # A velocity spike on the first recovered frame describes
                    # the return from the excursion; keep it as the right anchor.
                    output[left + 1 : right + 1, obj_idx, joint_id] = False
                    output[left + 1 : right, obj_idx, joint_id] = True
                    reasons[left + 1 : right, obj_idx, joint_id] |= REASON_VELOCITY
                    while cursor < len(seed_indices) and seed_indices[cursor] <= right:
                        cursor += 1
                    continue
                output[seed, obj_idx, joint_id] = True
                cursor += 1
    return output, reasons


def coalesce_incompatible_bilateral_intervals(
    pose_xy: np.ndarray,
    bboxes_xyxy: np.ndarray,
    flagged: np.ndarray,
    reasons: np.ndarray,
    cfg: dromia_config.AutoPoseRepairConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Join bilateral repairs until their outer anchors preserve identity.

    Per-joint interval detection can leave a short, smooth plateau between two
    anomalies. That plateau is not a safe anchor when the model has exchanged
    the two anatomical channels. For each knee/ankle pair, this function finds
    stable outer anchor blocks and compares direct physical assignment
    (left-to-left plus right-to-right) with crossed assignment. Both joints are
    repaired together only when a direct-compatible pair of anchors exists.

    This intentionally does not inspect the HMM state: a state change can be the
    correct response to raw model channels changing identity.
    """

    pose = np.asarray(pose_xy, dtype=np.float32)
    boxes = np.asarray(bboxes_xyxy, dtype=np.float32)
    output = np.asarray(flagged, dtype=bool).copy()
    reason_bits = np.asarray(reasons, dtype=np.uint8).copy()
    if output.shape != pose.shape[:3] or reason_bits.shape != pose.shape[:3]:
        raise ValueError("flagged and reasons must align with pose [T,O,K]")

    total = len(pose)
    block = cfg.stable_anchor_frames
    allowed = set(cfg.joint_ids)
    heights = np.maximum(boxes[..., 3] - boxes[..., 1], 1.0)

    for left_joint, right_joint in ((13, 14), (15, 16)):
        if left_joint not in allowed or right_joint not in allowed:
            continue
        for obj_idx in range(pose.shape[1]):
            pair_flags = output[:, obj_idx, left_joint] | output[:, obj_idx, right_joint]
            if not np.any(pair_flags):
                continue

            pair_points = pose[:, obj_idx, [left_joint, right_joint]]
            pair_finite = np.all(np.isfinite(pair_points), axis=(-1, -2))
            separation = (
                np.linalg.norm(pair_points[:, 0] - pair_points[:, 1], axis=-1) / heights[:, obj_idx]
            )
            pair_steps = np.full((total, 2), np.nan, dtype=np.float32)
            pair_steps[1:] = (
                np.linalg.norm(pair_points[1:] - pair_points[:-1], axis=-1)
                / heights[1:, obj_idx, None]
            )

            stable_start = np.zeros(total, dtype=bool)
            stable_end = np.zeros(total, dtype=bool)
            for start in range(0, total - block + 1):
                end = start + block - 1
                stable = (
                    bool(np.all(pair_finite[start : end + 1]))
                    and bool(np.all(~pair_flags[start : end + 1]))
                    and bool(
                        np.all(
                            separation[start : end + 1]
                            >= cfg.bilateral_collapse_max_separation_norm
                        )
                    )
                    and bool(np.all(np.isfinite(pair_steps[start + 1 : end + 1])))
                    and bool(
                        np.all(pair_steps[start + 1 : end + 1] <= cfg.weak_jump_threshold_norm)
                    )
                )
                if stable:
                    stable_start[start] = True
                    stable_end[end] = True

            original_runs = contiguous_index_runs(np.flatnonzero(pair_flags))
            for run_start, run_end in original_runs:
                left_candidates = np.flatnonzero(stable_end[:run_start])
                right_candidates = np.flatnonzero(stable_start[run_end + 1 :]) + run_end + 1
                compatible: list[tuple[int, int]] = []
                for left_anchor in map(int, left_candidates):
                    for right_anchor in map(int, right_candidates):
                        gap = right_anchor - left_anchor - 1
                        if gap <= 0 or gap > cfg.interpolation_max_gap:
                            continue
                        height = max(
                            float(
                                0.5
                                * (heights[left_anchor, obj_idx] + heights[right_anchor, obj_idx])
                            ),
                            1.0,
                        )
                        left_pair = pair_points[left_anchor]
                        right_pair = pair_points[right_anchor]
                        direct = (
                            float(
                                np.linalg.norm(left_pair[0] - right_pair[0])
                                + np.linalg.norm(left_pair[1] - right_pair[1])
                            )
                            / height
                        )
                        crossed = (
                            float(
                                np.linalg.norm(left_pair[0] - right_pair[1])
                                + np.linalg.norm(left_pair[1] - right_pair[0])
                            )
                            / height
                        )
                        if direct + cfg.bilateral_anchor_assignment_margin_norm <= crossed:
                            compatible.append((left_anchor, right_anchor))

                if not compatible:
                    continue
                left_anchor, right_anchor = min(
                    compatible,
                    key=lambda item: (
                        item[1] - item[0],
                        run_start - item[0] + item[1] - run_end,
                    ),
                )
                repair_slice = slice(left_anchor + 1, right_anchor)
                old_pair_flags = output[repair_slice, obj_idx, [left_joint, right_joint]].copy()
                output[repair_slice, obj_idx, left_joint] = True
                output[repair_slice, obj_idx, right_joint] = True
                new_pair_flags = (
                    output[repair_slice, obj_idx, [left_joint, right_joint]] & ~old_pair_flags
                )
                expanded_frames = np.any(new_pair_flags, axis=-1)
                if np.any(expanded_frames):
                    expanded_indices = np.flatnonzero(expanded_frames) + left_anchor + 1
                    for joint_id in (left_joint, right_joint):
                        reason_bits[expanded_indices, obj_idx, joint_id] |= (
                            REASON_IDENTITY_ANCHOR_MISMATCH
                        )

    return output, reason_bits


def contiguous_segments(flagged: np.ndarray, boundary_frames: int = 2) -> list[JointSegment]:
    segments: list[JointSegment] = []
    total = flagged.shape[0]
    for obj_idx in range(flagged.shape[1]):
        for joint_id in range(flagged.shape[2]):
            indices = np.flatnonzero(flagged[:, obj_idx, joint_id])
            if not len(indices):
                continue
            start = previous = int(indices[0])
            for value in map(int, indices[1:]):
                if value != previous + 1:
                    segments.append(
                        _segment(obj_idx, joint_id, start, previous, total, boundary_frames)
                    )
                    start = value
                previous = value
            segments.append(_segment(obj_idx, joint_id, start, previous, total, boundary_frames))
    return segments


def _segment(obj: int, joint: int, start: int, end: int, total: int, boundary: int) -> JointSegment:
    return JointSegment(
        obj, joint, start, end, max(0, start - boundary), min(total - 1, end + boundary)
    )


def candidate_is_valid(
    candidate: np.ndarray,
    *,
    t: int,
    obj: int,
    joint: int,
    reference_xy: np.ndarray,
    bboxes_xyxy: np.ndarray,
    cfg: dromia_config.AutoPoseRepairConfig,
) -> bool:
    if not np.all(np.isfinite(candidate)):
        return False
    box = bboxes_xyxy[t, obj]
    height = max(float(box[3] - box[1]), 1.0)
    pad = cfg.bbox_padding_height_fraction * height
    if not (
        box[0] - pad <= candidate[0] <= box[2] + pad
        and box[1] - pad <= candidate[1] <= box[3] + pad
    ):
        return False
    return True


def repair_pose(
    pose_xy: np.ndarray,
    bboxes_xyxy: np.ndarray,
    alignment_error_px: np.ndarray | None,
    cfg: dromia_config.AutoPoseRepairConfig,
    *,
    plain_candidate_xy: np.ndarray | None = None,
    tracker_candidate_xy: np.ndarray | None = None,
    tracker_visibility: np.ndarray | None = None,
    allow_interpolation: bool = True,
    detected_flags: np.ndarray | None = None,
    detected_reasons: np.ndarray | None = None,
) -> RepairResult:
    started = time.perf_counter()
    original = np.asarray(pose_xy, dtype=np.float32)
    repaired = original.copy()
    if detected_flags is None or detected_reasons is None:
        flagged, reasons = detect_catastrophic_drift(original, bboxes_xyxy, alignment_error_px, cfg)
    else:
        flagged = np.asarray(detected_flags, dtype=bool).copy()
        reasons = np.asarray(detected_reasons, dtype=np.uint8).copy()
    method = np.zeros(flagged.shape, dtype=np.uint8)
    tracker_vis = np.full(flagged.shape, np.nan, dtype=np.float32)
    segments = contiguous_segments(flagged, cfg.boundary_frames)

    for segment in segments:
        o, j = segment.object_index, segment.joint_id
        for t in range(segment.start, segment.end + 1):
            if not flagged[t, o, j]:
                continue
            if plain_candidate_xy is not None:
                candidate = plain_candidate_xy[t, o, j]
                if candidate_is_valid(
                    candidate,
                    t=t,
                    obj=o,
                    joint=j,
                    reference_xy=original,
                    bboxes_xyxy=bboxes_xyxy,
                    cfg=cfg,
                ):
                    repaired[t, o, j] = candidate
                    method[t, o, j] = METHOD_PLAIN
                    continue
            if tracker_candidate_xy is not None and tracker_visibility is not None:
                visibility = float(tracker_visibility[t, o, j])
                tracker_vis[t, o, j] = visibility
                candidate = tracker_candidate_xy[t, o, j]
                boundary_ok = tracker_boundary_agrees(
                    candidate,
                    t=t,
                    segment=segment,
                    obj=o,
                    joint=j,
                    reference_xy=repaired,
                    bboxes_xyxy=bboxes_xyxy,
                    cfg=cfg,
                )
                if (
                    visibility >= cfg.tracker_visibility_threshold
                    and boundary_ok
                    and candidate_is_valid(
                        candidate,
                        t=t,
                        obj=o,
                        joint=j,
                        reference_xy=repaired,
                        bboxes_xyxy=bboxes_xyxy,
                        cfg=cfg,
                    )
                ):
                    repaired[t, o, j] = candidate
                    method[t, o, j] = METHOD_TRACKER

        if allow_interpolation:
            interpolate_unresolved_segment(repaired, flagged, method, segment)

    displacement = np.linalg.norm(repaired - original, axis=-1).astype(np.float32)
    # Hard invariant: no unflagged coordinate may move.
    repaired[~flagged] = original[~flagged]
    displacement[~flagged] = 0.0
    return RepairResult(
        repaired,
        flagged,
        reasons,
        method,
        displacement,
        tracker_vis,
        segments,
        time.perf_counter() - started,
    )


def interpolate_unresolved_segment(
    repaired: np.ndarray,
    flagged: np.ndarray,
    method: np.ndarray,
    segment: JointSegment,
) -> None:
    """Fill every unresolved anomaly, using a one-sided hold at track boundaries."""

    o, j = segment.object_index, segment.joint_id
    unresolved = np.asarray(
        [
            t
            for t in range(segment.start, segment.end + 1)
            if flagged[t, o, j] and method[t, o, j] == METHOD_NONE
        ],
        dtype=np.int32,
    )
    if not unresolved.size:
        return
    runs = contiguous_index_runs(unresolved)
    for start, end in runs:
        left = next_finite_index(repaired[:, o, j], start - 1, direction=-1)
        right = next_finite_index(repaired[:, o, j], end + 1, direction=1)
        if left is None and right is None:
            continue
        if left is None:
            repaired[start : end + 1, o, j] = repaired[right, o, j]
        elif right is None:
            repaired[start : end + 1, o, j] = repaired[left, o, j]
        else:
            for t in range(start, end + 1):
                alpha = (t - left) / max(right - left, 1)
                repaired[t, o, j] = (1.0 - alpha) * repaired[left, o, j] + alpha * repaired[
                    right, o, j
                ]
        method[start : end + 1, o, j] = METHOD_INTERPOLATION


def contiguous_index_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if not len(indices):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in map(int, indices[1:]):
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def next_finite_index(points: np.ndarray, start: int, *, direction: int) -> int | None:
    index = start
    while 0 <= index < len(points):
        if np.all(np.isfinite(points[index])):
            return index
        index += direction
    return None


def enforce_final_step_bound(
    result: RepairResult,
    bboxes_xyxy: np.ndarray,
    cfg: dromia_config.AutoPoseRepairConfig,
) -> RepairResult:
    """Guarantee a hard per-frame displacement bound on configured joints.

    Violating endpoints are expanded and interpolated first. A final forward projection
    provides a deterministic safety net for edge cases where two anchors cannot satisfy
    the bound simultaneously.
    """

    original = result.keypoints_xy.copy()
    repaired = result.keypoints_xy.copy()
    boxes = np.asarray(bboxes_xyxy, dtype=np.float32)
    heights = np.maximum(boxes[..., 3] - boxes[..., 1], 1.0)
    flagged = result.flagged.copy()
    reasons = result.reasons.copy()
    method = result.method.copy()
    tracker_visibility = result.tracker_visibility.copy()
    allowed = set(cfg.joint_ids) & set(cfg.final_step_joint_ids)

    for obj_idx in range(repaired.shape[1]):
        for joint_id in allowed:
            for _iteration in range(repaired.shape[0]):
                steps = normalized_steps(repaired[:, obj_idx, joint_id], heights[:, obj_idx])
                violations = np.flatnonzero(steps > cfg.final_max_step_norm)
                if not violations.size:
                    break
                points = sorted(
                    {value for t in map(int, violations) for value in (max(0, t - 1), t)}
                )
                changed = False
                for start, end in contiguous_index_runs(np.asarray(points, dtype=np.int32)):
                    left = start - 1
                    right = end + 1
                    if left >= 0 and right < len(repaired):
                        p0 = repaired[left, obj_idx, joint_id].copy()
                        p1 = repaired[right, obj_idx, joint_id].copy()
                        for t in range(start, end + 1):
                            alpha = (t - left) / (right - left)
                            repaired[t, obj_idx, joint_id] = (1.0 - alpha) * p0 + alpha * p1
                    elif left >= 0:
                        repaired[start : end + 1, obj_idx, joint_id] = repaired[
                            left, obj_idx, joint_id
                        ]
                    elif right < len(repaired):
                        repaired[start : end + 1, obj_idx, joint_id] = repaired[
                            right, obj_idx, joint_id
                        ]
                    else:
                        continue
                    flagged[start : end + 1, obj_idx, joint_id] = True
                    reasons[start : end + 1, obj_idx, joint_id] |= REASON_VELOCITY
                    method[start : end + 1, obj_idx, joint_id] = METHOD_INTERPOLATION
                    changed = True
                if not changed:
                    break

            # Absolute safety net: clipping sequentially guarantees the invariant.
            for t in range(1, repaired.shape[0]):
                previous = repaired[t - 1, obj_idx, joint_id]
                current = repaired[t, obj_idx, joint_id]
                if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
                    continue
                delta = current - previous
                distance = float(np.linalg.norm(delta))
                maximum = cfg.final_max_step_norm * float(heights[t, obj_idx])
                if distance <= maximum or distance <= 1e-9:
                    continue
                repaired[t, obj_idx, joint_id] = previous + delta * (maximum / distance)
                flagged[t, obj_idx, joint_id] = True
                reasons[t, obj_idx, joint_id] |= REASON_VELOCITY
                method[t, obj_idx, joint_id] = METHOD_INTERPOLATION

    displacement = np.linalg.norm(repaired - original, axis=-1).astype(np.float32)
    segments = contiguous_segments(flagged, cfg.boundary_frames)
    return RepairResult(
        repaired,
        flagged,
        reasons,
        method,
        displacement,
        tracker_visibility,
        segments,
        result.runtime_seconds,
    )


def normalized_steps(points_xy: np.ndarray, heights: np.ndarray) -> np.ndarray:
    steps = np.full(len(points_xy), np.nan, dtype=np.float32)
    steps[1:] = np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=-1) / np.maximum(
        heights[1:], 1.0
    )
    return steps


def merge_repair_results(
    before: RepairResult,
    after: RepairResult,
    *,
    original_xy: np.ndarray,
) -> RepairResult:
    """Combine pre- and post-HMM provenance while retaining the final coordinates."""

    flagged = before.flagged | after.flagged
    reasons = before.reasons | after.reasons
    method = before.method.copy()
    selected = after.method != METHOD_NONE
    method[selected] = after.method[selected]
    tracker_visibility = before.tracker_visibility.copy()
    finite_after = np.isfinite(after.tracker_visibility)
    tracker_visibility[finite_after] = after.tracker_visibility[finite_after]
    displacement = np.linalg.norm(after.keypoints_xy - original_xy, axis=-1).astype(np.float32)
    return RepairResult(
        after.keypoints_xy,
        flagged,
        reasons,
        method,
        displacement,
        tracker_visibility,
        [*before.segments, *after.segments],
        before.runtime_seconds + after.runtime_seconds,
    )


def tracker_boundary_agrees(
    candidate: np.ndarray,
    *,
    t: int,
    segment: JointSegment,
    obj: int,
    joint: int,
    reference_xy: np.ndarray,
    bboxes_xyxy: np.ndarray,
    cfg: dromia_config.AutoPoseRepairConfig,
) -> bool:
    """Reject a visible track that attaches to the wrong runner/body part."""

    left, right = segment.start - 1, segment.end + 1
    if left < 0 or right >= len(reference_xy):
        return True
    p0, p1 = reference_xy[left, obj, joint], reference_xy[right, obj, joint]
    if (
        not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(p0))
        or not np.all(np.isfinite(p1))
    ):
        return False
    alpha = (t - left) / max(right - left, 1)
    expected = (1.0 - alpha) * p0 + alpha * p1
    height = max(float(bboxes_xyxy[t, obj, 3] - bboxes_xyxy[t, obj, 1]), 1.0)
    return (
        float(np.linalg.norm(candidate - expected) / height) <= cfg.tracker_boundary_agreement_norm
    )


def run_bounded_cotracker(
    *,
    video_path: Path,
    frame_indices: list[int],
    object_ids: list[int],
    pose_xy: np.ndarray,
    bboxes_xyxy: np.ndarray,
    unresolved: np.ndarray,
    segments: list[JointSegment],
    cfg: dromia_config.AutoPoseRepairConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Track unresolved anomaly segments, batching overlapping runner windows.

    Each segment is seeded once from its stable left anchor.  The complete track
    is accepted only when it reaches the stable right anchor within tolerance;
    batching merely shares video decoding and a model pass across local windows.
    """

    from dromia.models.bounded_cotracker import (
        CoTracker3PointTracker,
        PointTrackSeed,
    )

    candidate = np.full_like(pose_xy, np.nan, dtype=np.float32)
    visibility = np.zeros(pose_xy.shape[:3], dtype=np.float32)
    warnings: list[str] = []
    tracker = CoTracker3PointTracker()
    if not tracker.is_available():
        return candidate, visibility, ["cotracker_backend_unavailable"]
    tracker_cfg = cfg
    eligible = [
        segment
        for segment in segments
        if np.any(
            unresolved[
                segment.start : segment.end + 1,
                segment.object_index,
                segment.joint_id,
            ]
        )
    ]
    for object_index, window_start, window_end, window_segments in merge_tracker_windows(eligible):
        local_frames = frame_indices[window_start : window_end + 1]
        seeds: list[PointTrackSeed] = []
        segment_seeds: list[tuple[JointSegment, int, int, int]] = []
        for segment in window_segments:
            joint_id = segment.joint_id
            left = segment.start - 1
            right = segment.end + 1
            anchors_valid = (
                left >= segment.context_start
                and right <= segment.context_end
                and not unresolved[left, object_index, joint_id]
                and not unresolved[right, object_index, joint_id]
                and np.all(np.isfinite(pose_xy[left, object_index, joint_id]))
                and np.all(np.isfinite(pose_xy[right, object_index, joint_id]))
            )
            if not anchors_valid:
                warnings.append(
                    f"no_reliable_anchors:runner={object_ids[object_index]}:joint={joint_id}"
                )
                continue
            seed_index = len(seeds)
            seeds.append(
                PointTrackSeed(
                    runner_id=object_ids[object_index],
                    keypoint_id=joint_id,
                    keypoint_name=str(joint_id),
                    seed_frame_idx=frame_indices[left],
                    xy=pose_xy[left, object_index, joint_id].copy(),
                    local_point_type="left_anchor",
                    source_confidence=1.0,
                    inside_sam_mask=True,
                )
            )
            segment_seeds.append((segment, seed_index, left, right))
        if not seeds:
            continue
        try:
            tracks = tracker.track_points(
                video_path=video_path, frame_indices=local_frames, seeds=seeds, cfg=tracker_cfg
            )
        except Exception as exc:  # keep interpolation available if the optional backend fails
            joints = ",".join(str(item[0].joint_id) for item in segment_seeds)
            warnings.append(
                f"tracker_failed:runner={object_ids[object_index]}:joints={joints}:{exc}"
            )
            continue
        warnings.extend(
            f"tracker_warning:runner={object_ids[object_index]}:{warning}"
            for warning in tracks.warnings
        )
        for segment, seed_index, left, right in segment_seeds:
            joint_id = segment.joint_id
            left_local = left - window_start
            right_local = right - window_start
            endpoint_visible = bool(tracks.visibility[seed_index, right_local]) and (
                float(tracks.confidence[seed_index, right_local])
                >= cfg.tracker_visibility_threshold
            )
            tracked_right = tracks.xy[seed_index, right_local]
            expected_right = pose_xy[right, object_index, joint_id]
            endpoint_height = max(
                float(bboxes_xyxy[right, object_index, 3] - bboxes_xyxy[right, object_index, 1]),
                1.0,
            )
            endpoint_error_norm = float(
                np.linalg.norm(tracked_right - expected_right) / endpoint_height
            )
            endpoint_finite = bool(np.all(np.isfinite(tracked_right)))
            if not endpoint_visible or not endpoint_finite:
                warnings.append(
                    "tracker_endpoint_not_visible:"
                    f"runner={object_ids[object_index]}:joint={joint_id}"
                )
                continue
            if endpoint_error_norm > cfg.tracker_endpoint_agreement_norm:
                warnings.append(
                    "tracker_endpoint_mismatch:"
                    f"runner={object_ids[object_index]}:joint={joint_id}:"
                    f"error_norm={endpoint_error_norm:.6f}"
                )
                continue
            tracked_left = tracks.xy[seed_index, left_local]
            expected_left = pose_xy[left, object_index, joint_id]
            for global_t in range(segment.start, segment.end + 1):
                if not unresolved[global_t, object_index, joint_id]:
                    continue
                local_t = global_t - window_start
                confidence = float(tracks.confidence[seed_index, local_t])
                if (
                    not tracks.visibility[seed_index, local_t]
                    or confidence < cfg.tracker_visibility_threshold
                ):
                    continue
                alpha = (global_t - left) / max(right - left, 1)
                correction = (1.0 - alpha) * (expected_left - tracked_left) + alpha * (
                    expected_right - tracked_right
                )
                candidate[global_t, object_index, joint_id] = (
                    tracks.xy[seed_index, local_t] + correction
                )
                visibility[global_t, object_index, joint_id] = confidence
    return candidate, visibility, warnings


def merge_tracker_windows(
    segments: list[JointSegment],
) -> list[tuple[int, int, int, tuple[JointSegment, ...]]]:
    """Merge overlapping/adjacent context windows without mixing runners."""

    by_object: dict[int, list[JointSegment]] = {}
    for segment in segments:
        by_object.setdefault(segment.object_index, []).append(segment)
    windows: list[tuple[int, int, int, tuple[JointSegment, ...]]] = []
    for object_index in sorted(by_object):
        ordered = sorted(
            by_object[object_index],
            key=lambda item: (item.context_start, item.context_end, item.joint_id),
        )
        current: list[JointSegment] = []
        current_start = current_end = -1
        for segment in ordered:
            if not current or segment.context_start > current_end + 1:
                if current:
                    windows.append((object_index, current_start, current_end, tuple(current)))
                current = [segment]
                current_start = segment.context_start
                current_end = segment.context_end
            else:
                current.append(segment)
                current_end = max(current_end, segment.context_end)
        if current:
            windows.append((object_index, current_start, current_end, tuple(current)))
    return windows


def write_artifacts(
    result: RepairResult,
    *,
    run_dir: Path,
    frame_indices: np.ndarray,
    object_ids: np.ndarray,
    original_xy: np.ndarray,
    video_path: Path | None = None,
    sam_frames: list[object] | None = None,
    fps: float = 30.0,
    cfg: dromia_config.AutoPoseRepairConfig | None = None,
    raw_confidence: np.ndarray | None = None,
    alignment_error_px: np.ndarray | None = None,
    bboxes_xyxy: np.ndarray | None = None,
    pre_hmm_result: RepairResult | None = None,
    post_hmm_result: RepairResult | None = None,
    tracker_warnings: list[str] | None = None,
) -> dict[str, str]:
    output = run_dir / "postprocess"
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / "first_pass_pose.npz"
    diagnostic_arrays: dict[str, np.ndarray] = {}
    if raw_confidence is not None:
        diagnostic_arrays["raw_confidence"] = np.asarray(raw_confidence, dtype=np.float32)
    if alignment_error_px is not None:
        diagnostic_arrays["heatmap_alignment_error_px"] = np.asarray(
            alignment_error_px, dtype=np.float32
        )
    if bboxes_xyxy is not None:
        boxes = np.asarray(bboxes_xyxy, dtype=np.float32)
        heights = np.maximum(boxes[..., 3] - boxes[..., 1], 1.0)
        velocity = np.full(result.method.shape, np.nan, dtype=np.float32)
        velocity[1:] = (
            np.linalg.norm(original_xy[1:] - original_xy[:-1], axis=-1) / heights[1:, :, None]
        )
        diagnostic_arrays["normalized_velocity"] = velocity
        diagnostic_arrays["bboxes_xyxy"] = boxes
        tibia = np.full(result.method.shape[:2] + (2,), np.nan, dtype=np.float32)
        tibia[..., 0] = (
            np.linalg.norm(original_xy[..., 13, :] - original_xy[..., 15, :], axis=-1) / heights
        )
        tibia[..., 1] = (
            np.linalg.norm(original_xy[..., 14, :] - original_xy[..., 16, :], axis=-1) / heights
        )
        diagnostic_arrays["normalized_tibia_length"] = tibia
        final_velocity = np.full(result.method.shape, np.nan, dtype=np.float32)
        final_velocity[1:] = (
            np.linalg.norm(result.keypoints_xy[1:] - result.keypoints_xy[:-1], axis=-1)
            / heights[1:, :, None]
        )
        diagnostic_arrays["final_normalized_velocity"] = final_velocity
    if pre_hmm_result is not None:
        diagnostic_arrays.update(
            {
                "pre_hmm_repair_flagged": pre_hmm_result.flagged,
                "pre_hmm_repair_reason_bits": pre_hmm_result.reasons,
                "pre_hmm_repair_method": pre_hmm_result.method,
            }
        )
    if post_hmm_result is not None:
        diagnostic_arrays.update(
            {
                "post_hmm_repair_flagged": post_hmm_result.flagged,
                "post_hmm_repair_reason_bits": post_hmm_result.reasons,
                "post_hmm_repair_method": post_hmm_result.method,
            }
        )
    np.savez_compressed(
        npz_path,
        frame_indices=frame_indices,
        object_ids=object_ids,
        original_keypoints_xy=original_xy,
        first_pass_keypoints_xy=result.keypoints_xy,
        repair_flagged=result.flagged,
        repair_reason_bits=result.reasons,
        repair_method=result.method,
        repair_displacement_px=result.displacement_px,
        tracker_visibility=result.tracker_visibility,
        **diagnostic_arrays,
    )
    events = []
    for t, o, j in np.argwhere(result.flagged):
        bits = int(result.reasons[t, o, j])
        events.append(
            {
                "frame_idx": int(frame_indices[t]),
                "runner_id": int(object_ids[o]),
                "joint_id": int(j),
                "reasons": [name for bit, name in REASON_NAMES.items() if bits & bit],
                "method": METHOD_NAMES[int(result.method[t, o, j])],
                "displacement_px": float(result.displacement_px[t, o, j]),
                "tracker_visibility": None
                if not np.isfinite(result.tracker_visibility[t, o, j])
                else float(result.tracker_visibility[t, o, j]),
                "source_boundary_frames": [
                    int(
                        frame_indices[
                            max(
                                0,
                                next(
                                    s.start
                                    for s in result.segments
                                    if s.object_index == o
                                    and s.joint_id == j
                                    and s.start <= t <= s.end
                                )
                                - 1,
                            )
                        ]
                    ),
                    int(
                        frame_indices[
                            min(
                                len(frame_indices) - 1,
                                next(
                                    s.end
                                    for s in result.segments
                                    if s.object_index == o
                                    and s.joint_id == j
                                    and s.start <= t <= s.end
                                )
                                + 1,
                            )
                        ]
                    ),
                ],
            }
        )
    diagnostics_path = output / "first_pass_pose_diagnostics.json"
    final_velocity = diagnostic_arrays.get("final_normalized_velocity")
    selected_joints = list((cfg or dromia_config.AutoPoseRepairConfig()).final_step_joint_ids)
    final_selected = (
        final_velocity[..., selected_joints]
        if final_velocity is not None
        else np.asarray([], dtype=np.float32)
    )
    finite_final_selected = final_selected[np.isfinite(final_selected)]
    diagnostics_path.write_text(
        json.dumps(
            {
                "runtime_seconds": result.runtime_seconds,
                "flagged_joint_frames": int(np.count_nonzero(result.flagged)),
                "repaired_joint_frames": int(np.count_nonzero(result.method)),
                "repairs_by_method": {
                    METHOD_NAMES[k]: int(np.count_nonzero(result.method == k)) for k in (1, 2, 3)
                },
                "pre_hmm": repair_stage_summary(pre_hmm_result),
                "post_hmm": repair_stage_summary(post_hmm_result),
                "tracker_warnings": list(tracker_warnings or []),
                "final_max_step_norm": (
                    None if not finite_final_selected.size else float(np.max(finite_final_selected))
                ),
                "final_step_violation_count": (
                    0
                    if not final_selected.size
                    else int(
                        np.count_nonzero(
                            final_selected
                            > (cfg or dromia_config.AutoPoseRepairConfig()).final_max_step_norm
                            + 1e-6
                        )
                    )
                ),
                "bilateral_collapse_joint_frames": int(
                    np.count_nonzero((result.reasons & REASON_BILATERAL_COLLAPSE) != 0)
                ),
                "segments": [
                    segment.__dict__
                    if hasattr(segment, "__dict__")
                    else {
                        "runner_index": segment.object_index,
                        "joint_id": segment.joint_id,
                        "start_time_index": segment.start,
                        "end_time_index": segment.end,
                        "context_start_time_index": segment.context_start,
                        "context_end_time_index": segment.context_end,
                    }
                    for segment in result.segments
                ],
                "events": events,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts = {
        "first_pass_pose_npz": str(npz_path.resolve()),
        "first_pass_pose_diagnostics_json": str(diagnostics_path.resolve()),
    }
    return artifacts


def repair_stage_summary(result: RepairResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "flagged_joint_frames": int(np.count_nonzero(result.flagged)),
        "repaired_joint_frames": int(np.count_nonzero(result.method)),
        "repairs_by_method": {
            METHOD_NAMES[k]: int(np.count_nonzero(result.method == k)) for k in (1, 2, 3)
        },
        "bilateral_collapse_joint_frames": int(
            np.count_nonzero((result.reasons & REASON_BILATERAL_COLLAPSE) != 0)
        ),
    }
