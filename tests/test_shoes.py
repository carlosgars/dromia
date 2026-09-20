from __future__ import annotations

import numpy as np

from dromia import config as dromia_config
from dromia import dto as dromia_dto
from dromia.pipeline import shoes as shoe_assignment


def test_assign_shoe_to_nearest_runner_ankle() -> None:
    runner_mask = np.zeros((50, 50), dtype=np.uint8)
    runner_mask[5:45, 10:35] = 1
    shoe_mask = np.zeros((50, 50), dtype=np.uint8)
    shoe_mask[38:44, 14:24] = 1
    runner = dromia_dto.SamDetection(
        obj_id=1,
        frame_idx=0,
        label="Runner running",
        score=0.9,
        bbox_xyxy=np.asarray([10, 5, 35, 45], dtype=np.float32),
        mask=runner_mask,
    )
    shoe = dromia_dto.SamDetection(
        obj_id=7,
        frame_idx=0,
        label="Running shoe",
        score=0.8,
        bbox_xyxy=np.asarray([14, 38, 24, 44], dtype=np.float32),
        mask=shoe_mask,
    )
    pose = np.full((17, 2), np.nan, dtype=np.float32)
    pose[15] = [19, 41]
    assignments = shoe_assignment.assign_shoes_for_frame(
        frame_idx=0,
        runners=[runner],
        shoes=[shoe],
        pose_by_runner={1: pose},
    )
    left = [item for item in assignments if item.side == "left"][0]
    assert left.shoe_obj_id == 7
    assert left.score > 0.5


def test_stabilize_assignments_corrects_one_frame_swap() -> None:
    raw = [
        _shoe_assignment(0, "left", 10, 12),
        _shoe_assignment(0, "right", 20, 32),
        _shoe_assignment(1, "left", 20, 33),
        _shoe_assignment(1, "right", 10, 13),
        _shoe_assignment(2, "left", 10, 14),
        _shoe_assignment(2, "right", 20, 34),
    ]

    stable = shoe_assignment.stabilize_assignments(raw)
    lookup = shoe_assignment.assignments_by_frame_runner_side(stable)

    assert lookup[(1, 1, "left")].shoe_obj_id == 10
    assert lookup[(1, 1, "right")].shoe_obj_id == 20


def test_track_assignment_uses_aggregate_mask_overlap_and_point_to_mask_side() -> None:
    frames: list[dromia_dto.SamFrame] = []
    poses: dict[tuple[int, int], np.ndarray] = {}
    for frame_idx in range(3):
        runner = _detection(1, frame_idx, "Runner running", (4, 3, 25, 46))
        shoe = _detection(10, frame_idx, "Running shoe", (7, 36, 14, 43))
        frames.append(dromia_dto.SamFrame(frame_idx=frame_idx, detections=[shoe, runner]))
        poses[(frame_idx, 1)] = _pose(left=(10, 39), right=(21, 39))

    result = shoe_assignment.assign_shoe_tracks(
        frames=frames,
        runner_ids=[1],
        pose_by_frame_runner=poses,
        cfg=_track_cfg(),
        source_pose_configuration="identity_on",
        source_cache="test-cache",
    )

    track = result.tracks[0]
    assert track.runner_id == 1
    assert track.runner_status == "assigned"
    assert track.side == "left"
    assert track.side_status == "assigned"
    assert track.runner_supporting_frames == 3
    assert all(row.selected_for_frame for row in result.frames)


def test_track_assignment_uses_masks_not_overlapping_runner_boxes() -> None:
    frames: list[dromia_dto.SamFrame] = []
    poses: dict[tuple[int, int], np.ndarray] = {}
    for frame_idx in range(3):
        left_runner = _detection(1, frame_idx, "Runner running", (2, 3, 17, 46))
        right_runner = _detection(2, frame_idx, "Runner running", (20, 3, 35, 46))
        # Deliberately make their boxes overlap while their masks stay distinct.
        left_runner.bbox_xyxy = np.asarray([2, 3, 28, 46], dtype=np.float32)
        right_runner.bbox_xyxy = np.asarray([10, 3, 35, 46], dtype=np.float32)
        shoe = _detection(10, frame_idx, "Running shoe", (6, 37, 13, 43))
        frames.append(
            dromia_dto.SamFrame(
                frame_idx=frame_idx,
                detections=[right_runner, shoe, left_runner],
            )
        )
        poses[(frame_idx, 1)] = _pose(left=(9, 39), right=(15, 39))
        poses[(frame_idx, 2)] = _pose(left=(22, 39), right=(30, 39))

    result = shoe_assignment.assign_shoe_tracks(
        frames=frames,
        runner_ids=[2, 1],
        pose_by_frame_runner=poses,
        cfg=_track_cfg(),
        source_pose_configuration="identity_on",
        source_cache="test-cache",
    )

    assert result.tracks[0].runner_id == 1
    assert result.tracks[0].runner_overlap == 1.0
    assert result.tracks[0].runner_second_overlap == 0.0


def test_track_assignment_keeps_no_overlap_and_invalid_side_unavailable() -> None:
    frames: list[dromia_dto.SamFrame] = []
    poses: dict[tuple[int, int], np.ndarray] = {}
    for frame_idx in range(3):
        runner = _detection(1, frame_idx, "Runner running", (2, 3, 15, 46))
        far_shoe = _detection(10, frame_idx, "Running shoe", (30, 37, 38, 43))
        frames.append(dromia_dto.SamFrame(frame_idx=frame_idx, detections=[runner, far_shoe]))
        poses[(frame_idx, 1)] = _pose(left=(np.nan, np.nan), right=(np.nan, np.nan))

    result = shoe_assignment.assign_shoe_tracks(
        frames=frames,
        runner_ids=[1],
        pose_by_frame_runner=poses,
        cfg=_track_cfg(),
        source_pose_configuration="identity_on",
        source_cache="test-cache",
    )

    assert result.tracks[0].runner_status == "unassigned_runner"
    assert result.tracks[0].runner_id is None
    assert result.tracks[0].side_status == "unassigned_side"
    assert not any(row.selected_for_frame for row in result.frames)


def test_track_assignment_reports_ambiguous_runner_and_side() -> None:
    runner_frames: list[dromia_dto.SamFrame] = []
    poses: dict[tuple[int, int], np.ndarray] = {}
    for frame_idx in range(3):
        runner_a = _detection(1, frame_idx, "Runner running", (2, 3, 20, 46))
        runner_b = _detection(2, frame_idx, "Runner running", (12, 3, 30, 46))
        shoe = _detection(10, frame_idx, "Running shoe", (12, 37, 20, 43))
        runner_frames.append(
            dromia_dto.SamFrame(frame_idx=frame_idx, detections=[runner_a, runner_b, shoe])
        )
        poses[(frame_idx, 1)] = _pose(left=(14, 39), right=(18, 39))
        poses[(frame_idx, 2)] = _pose(left=(14, 39), right=(18, 39))
    runner_result = shoe_assignment.assign_shoe_tracks(
        frames=runner_frames,
        runner_ids=[1, 2],
        pose_by_frame_runner=poses,
        cfg=_track_cfg(),
        source_pose_configuration="identity_on",
        source_cache="test-cache",
    )
    assert runner_result.tracks[0].runner_status == "ambiguous_runner"

    side_frames: list[dromia_dto.SamFrame] = []
    side_poses: dict[tuple[int, int], np.ndarray] = {}
    for frame_idx in range(3):
        runner = _detection(1, frame_idx, "Runner running", (2, 3, 30, 46))
        shoe = _detection(10, frame_idx, "Running shoe", (12, 37, 20, 43))
        side_frames.append(dromia_dto.SamFrame(frame_idx=frame_idx, detections=[runner, shoe]))
        side_poses[(frame_idx, 1)] = _pose(left=(10, 39), right=(22, 39))
    side_result = shoe_assignment.assign_shoe_tracks(
        frames=side_frames,
        runner_ids=[1],
        pose_by_frame_runner=side_poses,
        cfg=_track_cfg(side_min_distance_margin_norm=0.2),
        source_pose_configuration="identity_on",
        source_cache="test-cache",
    )
    assert side_result.tracks[0].side_status == "ambiguous_side"
    assert side_result.tracks[0].side is None


def test_fragmented_tracks_and_frame_conflicts_are_explicit() -> None:
    runner = _detection(1, 0, "Runner running", (2, 3, 30, 46))
    shoe_a = _detection(10, 0, "Running shoe", (7, 36, 14, 43))
    shoe_b = _detection(11, 0, "Running shoe", (9, 36, 16, 43))
    frames = [dromia_dto.SamFrame(frame_idx=0, detections=[runner, shoe_b, shoe_a])]
    result = shoe_assignment.assign_shoe_tracks(
        frames=frames,
        runner_ids=[1],
        pose_by_frame_runner={(0, 1): _pose(left=(10, 39), right=(25, 39))},
        cfg=_track_cfg(
            runner_min_supporting_frames=1,
            side_min_supporting_frames=1,
        ),
        source_pose_configuration="identity_on",
        source_cache="test-cache",
    )

    assert [item.shoe_track_id for item in result.tracks] == [10, 11]
    selected = [item for item in result.frames if item.selected_for_frame]
    rejected = [item for item in result.frames if item.conflict_status == "rejected_conflict"]
    assert len(selected) == 1
    assert selected[0].shoe_track_id == 10
    assert len(rejected) == 1
    assert result.downstream_assignments[0].shoe_obj_id == 10


def test_track_assignment_is_iteration_order_independent() -> None:
    frames: list[dromia_dto.SamFrame] = []
    poses: dict[tuple[int, int], np.ndarray] = {}
    for frame_idx in range(3):
        runner = _detection(2, frame_idx, "Runner running", (2, 3, 30, 46))
        shoe = _detection(12, frame_idx, "Running shoe", (7, 36, 14, 43))
        frames.append(dromia_dto.SamFrame(frame_idx=frame_idx, detections=[runner, shoe]))
        poses[(frame_idx, 2)] = _pose(left=(10, 39), right=(25, 39))
    kwargs = {
        "runner_ids": [2],
        "pose_by_frame_runner": poses,
        "cfg": _track_cfg(),
        "source_pose_configuration": "identity_on",
        "source_cache": "test-cache",
    }

    forward = shoe_assignment.assign_shoe_tracks(frames=frames, **kwargs)
    reverse = shoe_assignment.assign_shoe_tracks(
        frames=[
            frame.model_copy(update={"detections": list(reversed(frame.detections))})
            for frame in reversed(frames)
        ],
        **kwargs,
    )

    assert forward.tracks == reverse.tracks
    assert forward.frames == reverse.frames


def _shoe_assignment(
    frame_idx: int, side: str, shoe_id: int, center_x: int
) -> dromia_dto.ShoeAssignment:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[35:42, center_x - 2 : center_x + 3] = 1
    return dromia_dto.ShoeAssignment(
        frame_idx=frame_idx,
        runner_id=1,
        side=side,
        shoe_obj_id=shoe_id,
        score=0.9,
        mask=mask,
    )


def _track_cfg(**updates: object) -> dromia_config.ShoeTrackAssignmentConfig:
    return dromia_config.ShoeTrackAssignmentConfig(**updates)


def _pose(*, left: tuple[float, float], right: tuple[float, float]) -> np.ndarray:
    pose = np.full((17, 2), np.nan, dtype=np.float32)
    pose[15] = left
    pose[16] = right
    return pose


def _detection(
    obj_id: int,
    frame_idx: int,
    label: str,
    box: tuple[int, int, int, int],
) -> dromia_dto.SamDetection:
    mask = np.zeros((50, 50), dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1
    return dromia_dto.SamDetection(
        obj_id=obj_id,
        frame_idx=frame_idx,
        label=label,
        score=0.9,
        bbox_xyxy=np.asarray(box, dtype=np.float32),
        mask=mask,
    )
