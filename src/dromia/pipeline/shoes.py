"""Assign SAM shoe masks to runner ankles."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from dromia import config as dromia_config
from dromia import dto as dromia_dto


@dataclass(frozen=True, slots=True)
class ShoeTrackAssignment:
    shoe_track_id: int
    first_frame: int
    last_frame: int
    observed_frame_count: int
    missing_intervals: str
    median_mask_area_px: float
    mean_detection_score: float
    co_visible_runner_ids: str
    runner_id: int | None
    runner_status: str
    runner_reason: str
    runner_overlap: float
    runner_second_overlap: float
    runner_margin: float
    runner_supporting_frames: int
    side: str | None
    side_status: str
    side_reason: str
    left_median_distance_norm: float
    right_median_distance_norm: float
    side_margin: float
    side_consistency_fraction: float
    side_supporting_frames: int
    source_pose_configuration: str
    source_cache: str
    tracker_provenance: str


@dataclass(slots=True)
class ShoeFrameAssignment:
    frame_idx: int
    shoe_track_id: int
    runner_id: int | None
    side: str | None
    mask_available: bool
    distance_norm: float
    selected_for_frame: bool
    conflict_status: str


@dataclass(slots=True)
class ShoeTrackAssignmentResult:
    tracks: list[ShoeTrackAssignment]
    frames: list[ShoeFrameAssignment]
    downstream_assignments: list[dromia_dto.ShoeAssignment]


def assign_shoe_tracks(
    *,
    frames: list[dromia_dto.SamFrame],
    runner_ids: list[int],
    pose_by_frame_runner: dict[tuple[int, int], np.ndarray],
    cfg: dromia_config.ShoeTrackAssignmentConfig,
    source_pose_configuration: str,
    source_cache: str,
    tracker_provenance: str = "",
) -> ShoeTrackAssignmentResult:
    """Assign complete tracker fragments without framewise ownership changes."""

    accepted_runners = set(int(item) for item in runner_ids)
    frame_lookup = {int(frame.frame_idx): frame for frame in frames}
    runner_lookup: dict[tuple[int, int], dromia_dto.SamDetection] = {}
    shoe_tracks: dict[int, dict[int, dromia_dto.SamDetection]] = {}
    for frame in sorted(frames, key=lambda item: item.frame_idx):
        for runner in sorted(frame.runners, key=lambda item: item.obj_id):
            if runner.obj_id in accepted_runners:
                runner_lookup[(frame.frame_idx, runner.obj_id)] = runner
        for shoe in sorted(frame.shoes, key=lambda item: (item.obj_id, -item.score)):
            # A tracker ID should occur once per frame. If a malformed cache has
            # duplicates, retain the highest-score detection deterministically.
            shoe_tracks.setdefault(shoe.obj_id, {}).setdefault(frame.frame_idx, shoe)

    track_rows: list[ShoeTrackAssignment] = []
    observations_by_track: dict[int, list[dromia_dto.SamDetection]] = {}
    for shoe_id, detections_by_frame in sorted(shoe_tracks.items()):
        observations = [detections_by_frame[key] for key in sorted(detections_by_frame)]
        observations_by_track[shoe_id] = observations
        track_rows.append(
            _assign_track(
                shoe_id=shoe_id,
                observations=observations,
                runner_lookup=runner_lookup,
                pose_by_frame_runner=pose_by_frame_runner,
                cfg=cfg,
                source_pose_configuration=source_pose_configuration,
                source_cache=source_cache,
                tracker_provenance=tracker_provenance,
            )
        )

    frame_rows = _expand_frame_assignments(
        track_rows,
        observations_by_track=observations_by_track,
        runner_lookup=runner_lookup,
        pose_by_frame_runner=pose_by_frame_runner,
    )
    downstream = _downstream_assignments(
        frame_lookup=frame_lookup,
        runner_ids=sorted(accepted_runners),
        frame_rows=frame_rows,
        observations_by_track=observations_by_track,
    )
    return ShoeTrackAssignmentResult(track_rows, frame_rows, downstream)


def _assign_track(
    *,
    shoe_id: int,
    observations: list[dromia_dto.SamDetection],
    runner_lookup: dict[tuple[int, int], dromia_dto.SamDetection],
    pose_by_frame_runner: dict[tuple[int, int], np.ndarray],
    cfg: dromia_config.ShoeTrackAssignmentConfig,
    source_pose_configuration: str,
    source_cache: str,
    tracker_provenance: str,
) -> ShoeTrackAssignment:
    frames = [int(item.frame_idx) for item in observations]
    co_visible = sorted(
        {runner_id for frame_idx, runner_id in runner_lookup if frame_idx in set(frames)}
    )
    runner_scores: list[tuple[float, int, int]] = []
    for runner_id in co_visible:
        intersections = 0
        shoe_pixels = 0
        supporting_frames = 0
        for shoe in observations:
            runner = runner_lookup.get((shoe.frame_idx, runner_id))
            if runner is None:
                continue
            intersection, area = mask_intersection_counts(shoe.mask, runner.mask)
            if area <= 0:
                continue
            intersections += intersection
            shoe_pixels += area
            supporting_frames += 1
        overlap = float(intersections / shoe_pixels) if shoe_pixels else 0.0
        runner_scores.append((overlap, int(runner_id), supporting_frames))
    runner_scores.sort(key=lambda item: (-item[0], item[1]))
    best_overlap, best_runner, best_support = runner_scores[0] if runner_scores else (0.0, -1, 0)
    second_overlap = runner_scores[1][0] if len(runner_scores) > 1 else 0.0
    runner_margin = float(best_overlap - second_overlap)
    runner_id: int | None = int(best_runner) if best_overlap > 0 else None
    runner_status = "assigned"
    runner_reason = "positive_maximum_overlap"
    if best_overlap <= 0:
        runner_id = None
        runner_status = "unassigned_runner"
        runner_reason = "no_positive_overlap"
    elif best_support < cfg.runner_min_supporting_frames:
        runner_id = None
        runner_status = "ambiguous_runner"
        runner_reason = "insufficient_support"
    elif np.isclose(runner_margin, 0.0) or runner_margin < cfg.runner_min_overlap_margin:
        runner_id = None
        runner_status = "ambiguous_runner"
        runner_reason = "insufficient_overlap_margin"

    side: str | None = None
    side_status = "unassigned_side"
    side_reason = "runner_unavailable"
    left_median = float("nan")
    right_median = float("nan")
    side_margin = float("nan")
    consistency = float("nan")
    side_support = 0
    if runner_id is not None:
        left_distances: list[float] = []
        right_distances: list[float] = []
        for shoe in observations:
            runner = runner_lookup.get((shoe.frame_idx, runner_id))
            pose = pose_by_frame_runner.get((shoe.frame_idx, runner_id))
            if runner is None or pose is None or np.asarray(pose).shape[0] < 17:
                continue
            height = bbox_height(runner)
            left = point_to_mask_distance(np.asarray(pose)[15], shoe.mask) / height
            right = point_to_mask_distance(np.asarray(pose)[16], shoe.mask) / height
            if np.isfinite(left) and np.isfinite(right):
                left_distances.append(float(left))
                right_distances.append(float(right))
        side_support = len(left_distances)
        if side_support:
            left_median = float(np.median(left_distances))
            right_median = float(np.median(right_distances))
            side = "left" if left_median < right_median else "right"
            side_margin = abs(left_median - right_median)
            selected = left_distances if side == "left" else right_distances
            alternative = right_distances if side == "left" else left_distances
            consistency = float(np.mean(np.asarray(selected) < np.asarray(alternative)))
        if side_support < cfg.side_min_supporting_frames:
            side = None
            side_status = "unassigned_side"
            side_reason = "insufficient_valid_ankle_observations"
        elif np.isclose(side_margin, 0.0) or side_margin < cfg.side_min_distance_margin_norm:
            side = None
            side_status = "ambiguous_side"
            side_reason = "insufficient_distance_margin"
        else:
            side_status = "assigned"
            side_reason = "minimum_median_point_to_mask_distance"

    mask_areas = [mask_area(item.mask) for item in observations]
    return ShoeTrackAssignment(
        shoe_track_id=int(shoe_id),
        first_frame=min(frames),
        last_frame=max(frames),
        observed_frame_count=len(frames),
        missing_intervals=format_missing_intervals(frames),
        median_mask_area_px=float(np.median(mask_areas)) if mask_areas else 0.0,
        mean_detection_score=float(np.mean([item.score for item in observations])),
        co_visible_runner_ids=";".join(str(item) for item in co_visible),
        runner_id=runner_id,
        runner_status=runner_status,
        runner_reason=runner_reason,
        runner_overlap=float(best_overlap),
        runner_second_overlap=float(second_overlap),
        runner_margin=runner_margin,
        runner_supporting_frames=int(best_support),
        side=side,
        side_status=side_status,
        side_reason=side_reason,
        left_median_distance_norm=left_median,
        right_median_distance_norm=right_median,
        side_margin=side_margin,
        side_consistency_fraction=consistency,
        side_supporting_frames=side_support,
        source_pose_configuration=source_pose_configuration,
        source_cache=source_cache,
        tracker_provenance=tracker_provenance,
    )


def _expand_frame_assignments(
    tracks: list[ShoeTrackAssignment],
    *,
    observations_by_track: dict[int, list[dromia_dto.SamDetection]],
    runner_lookup: dict[tuple[int, int], dromia_dto.SamDetection],
    pose_by_frame_runner: dict[tuple[int, int], np.ndarray],
) -> list[ShoeFrameAssignment]:
    rows: list[ShoeFrameAssignment] = []
    for track in tracks:
        for shoe in observations_by_track[track.shoe_track_id]:
            distance = float("nan")
            if track.runner_id is not None and track.side is not None:
                runner = runner_lookup.get((shoe.frame_idx, track.runner_id))
                pose = pose_by_frame_runner.get((shoe.frame_idx, track.runner_id))
                ankle_id = 15 if track.side == "left" else 16
                if runner is not None and pose is not None and np.asarray(pose).shape[0] > ankle_id:
                    distance = point_to_mask_distance(
                        np.asarray(pose)[ankle_id], shoe.mask
                    ) / bbox_height(runner)
            status = (
                track.runner_status
                if track.runner_status != "assigned"
                else track.side_status
                if track.side_status != "assigned"
                else "no_conflict"
            )
            rows.append(
                ShoeFrameAssignment(
                    frame_idx=int(shoe.frame_idx),
                    shoe_track_id=track.shoe_track_id,
                    runner_id=track.runner_id,
                    side=track.side,
                    mask_available=mask_area(shoe.mask) > 0,
                    distance_norm=float(distance),
                    selected_for_frame=False,
                    conflict_status=status,
                )
            )
    groups: dict[tuple[int, int, str], list[ShoeFrameAssignment]] = {}
    for row in rows:
        if row.runner_id is not None and row.side is not None and row.mask_available:
            groups.setdefault((row.frame_idx, row.runner_id, row.side), []).append(row)
    for candidates in groups.values():
        if len(candidates) == 1:
            candidates[0].selected_for_frame = True
            continue
        finite_candidates = [item for item in candidates if np.isfinite(item.distance_norm)]
        if not finite_candidates:
            for unresolved in candidates:
                unresolved.conflict_status = "unresolved_conflict_no_valid_distance"
            continue
        ordered = sorted(
            finite_candidates,
            key=lambda item: (
                item.distance_norm,
                item.shoe_track_id,
            ),
        )
        ordered[0].selected_for_frame = True
        ordered[0].conflict_status = "selected_conflict"
        for rejected in candidates:
            if rejected is not ordered[0]:
                rejected.conflict_status = "rejected_conflict"
    return sorted(rows, key=lambda item: (item.frame_idx, item.shoe_track_id))


def _downstream_assignments(
    *,
    frame_lookup: dict[int, dromia_dto.SamFrame],
    runner_ids: list[int],
    frame_rows: list[ShoeFrameAssignment],
    observations_by_track: dict[int, list[dromia_dto.SamDetection]],
) -> list[dromia_dto.ShoeAssignment]:
    selected = {
        (row.frame_idx, int(row.runner_id), str(row.side)): row
        for row in frame_rows
        if row.selected_for_frame and row.runner_id is not None and row.side is not None
    }
    shoes = {
        (item.frame_idx, shoe_id): item
        for shoe_id, observations in observations_by_track.items()
        for item in observations
    }
    output: list[dromia_dto.ShoeAssignment] = []
    for frame_idx in sorted(frame_lookup):
        visible_runners = {
            item.obj_id for item in frame_lookup[frame_idx].runners if item.obj_id in runner_ids
        }
        for runner_id in sorted(visible_runners):
            for side in ("left", "right"):
                row = selected.get((frame_idx, runner_id, side))
                if row is None:
                    output.append(empty_assignment(frame_idx, runner_id, side))
                    continue
                shoe = shoes[(frame_idx, row.shoe_track_id)]
                score = (
                    float(np.clip(1.0 - row.distance_norm, 0.0, 1.0))
                    if np.isfinite(row.distance_norm)
                    else 0.0
                )
                output.append(
                    dromia_dto.ShoeAssignment(
                        frame_idx=frame_idx,
                        runner_id=runner_id,
                        side=side,
                        shoe_obj_id=row.shoe_track_id,
                        score=score,
                        mask=shoe.mask,
                    )
                )
    return output


def write_track_assignment_artifacts(
    result: ShoeTrackAssignmentResult,
    run_dir: Path,
) -> dict[str, str]:
    shoes_dir = run_dir / "shoes"
    shoes_dir.mkdir(parents=True, exist_ok=True)
    track_path = shoes_dir / "shoe_track_assignments.csv"
    frame_path = shoes_dir / "shoe_frame_assignments.csv"
    _write_dataclass_csv(track_path, result.tracks, ShoeTrackAssignment)
    _write_dataclass_csv(frame_path, result.frames, ShoeFrameAssignment)
    return {
        "shoe_track_assignments_csv": str(track_path.resolve()),
        "shoe_frame_assignments_csv": str(frame_path.resolve()),
    }


def _write_dataclass_csv(path: Path, rows: list[Any], row_type: type[Any]) -> None:
    fieldnames = list(row_type.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def format_missing_intervals(frame_indices: list[int]) -> str:
    frames = sorted(set(int(item) for item in frame_indices))
    if len(frames) < 2:
        return ""
    missing = sorted(set(range(frames[0], frames[-1] + 1)) - set(frames))
    if not missing:
        return ""
    intervals: list[str] = []
    start = previous = missing[0]
    for frame_idx in missing[1:]:
        if frame_idx == previous + 1:
            previous = frame_idx
            continue
        intervals.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = frame_idx
    intervals.append(str(start) if start == previous else f"{start}-{previous}")
    return ";".join(intervals)


def bbox_height(runner: dromia_dto.SamDetection) -> float:
    bbox = np.asarray(runner.bbox_xyxy, dtype=np.float32)
    return max(float(bbox[3] - bbox[1]), 1.0)


def mask_area(mask: Any) -> int:
    return int(np.count_nonzero(np.asarray(mask) > 0))


def mask_intersection_counts(a: Any, b: Any) -> tuple[int, int]:
    mask_a = np.asarray(a) > 0
    mask_b = np.asarray(b) > 0
    area = int(np.count_nonzero(mask_a))
    if mask_a.shape != mask_b.shape:
        return 0, area
    return int(np.count_nonzero(mask_a & mask_b)), area


def point_to_mask_distance(point_xy: np.ndarray, mask: Any) -> float:
    point = np.asarray(point_xy, dtype=np.float32)
    if point.shape != (2,) or not np.isfinite(point).all():
        return float("nan")
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or not np.any(binary):
        return float("nan")
    ys, xs = np.nonzero(binary)
    squared = (xs.astype(np.float32) - point[0]) ** 2 + (ys.astype(np.float32) - point[1]) ** 2
    return float(np.sqrt(np.min(squared)))


def assign_shoes_for_frame(
    *,
    frame_idx: int,
    runners: list[dromia_dto.SamDetection],
    shoes: list[dromia_dto.SamDetection],
    pose_by_runner: dict[int, np.ndarray],
    max_norm_distance: float = 0.45,
) -> list[dromia_dto.ShoeAssignment]:
    assignments: list[dromia_dto.ShoeAssignment] = []
    used_shoes: set[int] = set()
    for runner in sorted(runners, key=lambda item: item.obj_id):
        pose = pose_by_runner.get(runner.obj_id)
        if pose is None or pose.shape[0] < 17:
            continue
        for side, ankle_id in (("left", 15), ("right", 16)):
            best = best_shoe_for_ankle(runner, shoes, pose[ankle_id], used_shoes, max_norm_distance)
            if best is None:
                assignments.append(empty_assignment(frame_idx, runner.obj_id, side))
                continue
            shoe, score = best
            used_shoes.add(shoe.obj_id)
            assignments.append(
                dromia_dto.ShoeAssignment(
                    frame_idx=frame_idx,
                    runner_id=runner.obj_id,
                    side=side,
                    shoe_obj_id=shoe.obj_id,
                    score=score,
                    mask=shoe.mask,
                )
            )
    return assignments


def best_shoe_for_ankle(
    runner: dromia_dto.SamDetection,
    shoes: list[dromia_dto.SamDetection],
    ankle_xy: np.ndarray,
    used_shoes: set[int],
    max_norm_distance: float,
) -> tuple[dromia_dto.SamDetection, float] | None:
    if not np.isfinite(ankle_xy).all():
        return None
    bbox = np.asarray(runner.bbox_xyxy, dtype=np.float32)
    height = max(float(bbox[3] - bbox[1]), 1.0)
    candidates: list[tuple[float, dromia_dto.SamDetection]] = []
    for shoe in shoes:
        if shoe.obj_id in used_shoes:
            continue
        center = mask_center(shoe.mask)
        if center is None:
            continue
        overlap = mask_overlap(shoe.mask, runner.mask)
        # TODO: Consider replacing strict overlap with runner-mask/keypoint distance tolerance.
        lower_body = center[1] >= bbox[1] + 0.42 * height
        distance_norm = float(np.linalg.norm(center - ankle_xy)) / height
        if overlap <= 0.0 or not lower_body or distance_norm > max_norm_distance:
            continue
        cost = distance_norm - 0.25 * overlap - 0.05 * float(shoe.score)
        candidates.append((cost, shoe))
    if not candidates:
        return None
    cost, shoe = min(candidates, key=lambda item: item[0])
    score = float(np.clip(1.0 - cost, 0.0, 1.0))
    return shoe, score


def assignments_by_frame_runner_side(
    assignments: list[dromia_dto.ShoeAssignment],
) -> dict[tuple[int, int, str], dromia_dto.ShoeAssignment]:
    return {(item.frame_idx, item.runner_id, item.side): item for item in assignments}


def stabilize_assignments(
    assignments: list[dromia_dto.ShoeAssignment],
) -> list[dromia_dto.ShoeAssignment]:
    by_runner: dict[int, dict[int, dict[str, dromia_dto.ShoeAssignment]]] = {}
    for item in assignments:
        by_runner.setdefault(item.runner_id, {}).setdefault(item.frame_idx, {})[item.side] = item

    stable: list[dromia_dto.ShoeAssignment] = []
    for _runner_id, frame_map in sorted(by_runner.items()):
        frames = sorted(frame_map)
        states = viterbi_swap_states([frame_map[frame_idx] for frame_idx in frames])
        for frame_idx, state in zip(frames, states, strict=True):
            pair = frame_map[frame_idx]
            stable.extend(apply_swap_state(pair, swapped=bool(state)))
    return sorted(stable, key=lambda item: (item.frame_idx, item.runner_id, item.side))


def viterbi_swap_states(frames: list[dict[str, dromia_dto.ShoeAssignment]]) -> list[int]:
    if not frames:
        return []
    costs = np.full((len(frames), 2), np.inf, dtype=np.float32)
    previous = np.zeros((len(frames), 2), dtype=np.int8)
    costs[0] = [unary_cost(frames[0], 0), unary_cost(frames[0], 1)]
    for idx in range(1, len(frames)):
        for state in (0, 1):
            options = [
                costs[idx - 1, prior] + transition_cost(frames[idx - 1], prior, frames[idx], state)
                for prior in (0, 1)
            ]
            best = int(np.argmin(options))
            costs[idx, state] = float(options[best]) + unary_cost(frames[idx], state)
            previous[idx, state] = best
    states = np.zeros(len(frames), dtype=np.int8)
    states[-1] = int(np.argmin(costs[-1]))
    for idx in range(len(frames) - 1, 0, -1):
        states[idx - 1] = previous[idx, states[idx]]
    return [int(item) for item in states]


def unary_cost(pair: dict[str, dromia_dto.ShoeAssignment], state: int) -> float:
    left, right = state_pair(pair, state)
    if left is None or right is None or left.mask is None or right.mask is None:
        return 0.0
    confidence_cost = (1.0 - float(left.score)) + (1.0 - float(right.score))
    swap_bias = 0.08 if state == 1 else 0.0
    return confidence_cost + swap_bias


def transition_cost(
    prev_pair: dict[str, dromia_dto.ShoeAssignment],
    prev_state: int,
    curr_pair: dict[str, dromia_dto.ShoeAssignment],
    curr_state: int,
) -> float:
    prev_left, prev_right = state_pair(prev_pair, prev_state)
    curr_left, curr_right = state_pair(curr_pair, curr_state)
    costs = [
        center_distance(prev_left, curr_left),
        center_distance(prev_right, curr_right),
    ]
    finite = [item for item in costs if np.isfinite(item)]
    switch_penalty = 0.18 if prev_state != curr_state else 0.0
    return (float(np.mean(finite)) / 80.0 if finite else 0.0) + switch_penalty


def state_pair(
    pair: dict[str, dromia_dto.ShoeAssignment],
    state: int,
) -> tuple[dromia_dto.ShoeAssignment | None, dromia_dto.ShoeAssignment | None]:
    left = pair.get("left")
    right = pair.get("right")
    if state == 1 and left is not None and right is not None:
        return right, left
    return left, right


def apply_swap_state(
    pair: dict[str, dromia_dto.ShoeAssignment],
    *,
    swapped: bool,
) -> list[dromia_dto.ShoeAssignment]:
    left = pair.get("left")
    right = pair.get("right")
    if not swapped or left is None or right is None:
        return [item for item in (left, right) if item is not None]
    return [
        copy_with_side(right, "left"),
        copy_with_side(left, "right"),
    ]


def copy_with_side(item: dromia_dto.ShoeAssignment, side: str) -> dromia_dto.ShoeAssignment:
    return item.model_copy(update={"side": side})


def center_distance(
    a: dromia_dto.ShoeAssignment | None,
    b: dromia_dto.ShoeAssignment | None,
) -> float:
    if a is None or b is None or a.mask is None or b.mask is None:
        return float("inf")
    center_a = mask_center(a.mask)
    center_b = mask_center(b.mask)
    if center_a is None or center_b is None:
        return float("inf")
    return float(np.linalg.norm(center_a - center_b))


def empty_assignment(frame_idx: int, runner_id: int, side: str) -> dromia_dto.ShoeAssignment:
    return dromia_dto.ShoeAssignment(
        frame_idx=frame_idx,
        runner_id=runner_id,
        side=side,
        shoe_obj_id=None,
        score=0.0,
        mask=None,
    )


def mask_center(mask: np.ndarray) -> np.ndarray | None:
    lazy_path = getattr(mask, "path", None)
    lazy_index = getattr(mask, "index", None)
    if lazy_path is not None and lazy_index is not None:
        return _lazy_mask_center(str(lazy_path), int(lazy_index))
    ys, xs = np.where(np.asarray(mask) > 0)
    if not len(xs):
        return None
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float32)


@lru_cache(maxsize=8192)
def _lazy_mask_center(path: str, index: int) -> np.ndarray | None:
    """Cache the tiny center statistic without repeatedly scanning a lazy mask."""

    with np.load(path, allow_pickle=False) as data:
        mask = np.asarray(data["masks"][index]) > 0
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def mask_overlap(a: np.ndarray, b: np.ndarray) -> float:
    mask_a = np.asarray(a) > 0
    mask_b = np.asarray(b) > 0
    if mask_a.shape != mask_b.shape:
        return 0.0
    denom = max(int(np.count_nonzero(mask_a)), 1)
    return float(np.count_nonzero(mask_a & mask_b) / denom)
