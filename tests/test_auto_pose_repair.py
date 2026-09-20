from __future__ import annotations

import numpy as np

from dromia.config import AutoPoseRepairConfig
from dromia.pipeline.auto_pose_repair import (
    METHOD_INTERPOLATION,
    METHOD_PLAIN,
    REASON_BILATERAL_COLLAPSE,
    REASON_IDENTITY_ANCHOR_MISMATCH,
    REASON_MIXED_POSE,
    JointSegment,
    coalesce_incompatible_bilateral_intervals,
    contiguous_segments,
    detect_catastrophic_drift,
    detect_unstable_intervals,
    enforce_final_step_bound,
    merge_tracker_windows,
    repair_pose,
    run_bounded_cotracker,
)


def fixture_pose(frames: int = 40) -> tuple[np.ndarray, np.ndarray]:
    pose = np.zeros((frames, 1, 17, 2), dtype=np.float32)
    boxes = np.tile(np.asarray([100, 100, 300, 500], dtype=np.float32), (frames, 1, 1))
    for t in range(frames):
        x = 180 + t * 0.5
        pose[t, 0, 5] = (x - 30, 170)
        pose[t, 0, 6] = (x + 10, 170)
        pose[t, 0, 12] = (x, 250)
        pose[t, 0, 14] = (x + 5, 340)
        pose[t, 0, 16] = (x + 8, 430)
        pose[t, 0, 11] = (x - 25, 250)
        pose[t, 0, 13] = (x - 20, 340)
        pose[t, 0, 15] = (x - 17, 430)
    return pose, boxes


def test_mixed_person_torso_interval_repairs_shoulders_and_hips_only() -> None:
    pose, boxes = fixture_pose(frames=12)
    for t in range(len(pose)):
        x = 180 + t * 0.5
        pose[t, 0, 5] = (x - 24, 170)
        pose[t, 0, 6] = (x + 4, 170)
        pose[t, 0, 11] = (x - 20, 250)
        pose[t, 0, 12] = (x, 250)
    expected = pose.copy()
    pose[6, 0, 5] += (75, 65)
    pose[6, 0, 6] += (55, 65)
    pose[6, 0, 11] += (55, 35)
    pose[6, 0, 12] += (20, 20)
    # Arms are intentionally not repaired even when the mixed pose affects them.
    pose[6, 0, 7] = (290, 290)
    arm_before = pose[6, 0, 7].copy()
    quality = np.ones(pose.shape[:3], dtype=np.float32)

    flagged, reasons = detect_unstable_intervals(pose, boxes, quality, AutoPoseRepairConfig())

    assert np.all(flagged[6, 0, [5, 6, 11, 12]])
    assert np.all((reasons[6, 0, [5, 6, 11, 12]] & REASON_MIXED_POSE) != 0)
    assert not flagged[6, 0, 7]
    repaired = repair_pose(
        pose,
        boxes,
        None,
        AutoPoseRepairConfig(),
        detected_flags=flagged,
        detected_reasons=reasons,
    )
    np.testing.assert_allclose(
        repaired.keypoints_xy[6, 0, [5, 6, 11, 12]],
        expected[6, 0, [5, 6, 11, 12]],
        atol=1e-5,
    )
    np.testing.assert_array_equal(repaired.keypoints_xy[6, 0, 7], arm_before)
    np.testing.assert_array_equal(repaired.keypoints_xy[:6], pose[:6])
    np.testing.assert_array_equal(repaired.keypoints_xy[7:], pose[7:])


def test_single_fast_shoulder_does_not_open_mixed_pose_interval() -> None:
    pose, boxes = fixture_pose(frames=12)
    for t in range(len(pose)):
        x = 180 + t * 0.5
        pose[t, 0, 5] = (x - 24, 170)
        pose[t, 0, 6] = (x + 4, 170)
    pose[6, 0, 5] += (80, 0)
    quality = np.ones(pose.shape[:3], dtype=np.float32)

    flagged, reasons = detect_unstable_intervals(pose, boxes, quality, AutoPoseRepairConfig())

    assert not np.any(flagged[6, 0, [5, 6, 11, 12]])
    assert not np.any((reasons[6, 0, [5, 6, 11, 12]] & REASON_MIXED_POSE) != 0)


def test_single_offscreen_shoulder_is_not_repaired_without_core_mix() -> None:
    pose, boxes = fixture_pose(frames=12)
    pose[6, 0, 5] = (520, 40)
    quality = np.ones(pose.shape[:3], dtype=np.float32)

    flagged, _reasons = detect_unstable_intervals(pose, boxes, quality, AutoPoseRepairConfig())

    assert not flagged[6, 0, 5]


def test_high_confidence_style_offscreen_failure_is_flagged_per_joint() -> None:
    pose, boxes = fixture_pose()
    pose[12:18, 0, 16] = (520, 40)
    alignment = np.zeros(pose.shape[:3], dtype=np.float32)
    alignment[12:18, 0, 16] = 80

    flagged, reasons = detect_catastrophic_drift(pose, boxes, alignment, AutoPoseRepairConfig())

    assert np.all(flagged[12:18, 0, 16])
    assert np.all((reasons[12:18, 0, 16] & 1) != 0)
    assert not np.any(flagged[:10])


def test_production_defaults_disable_limb_length_prior_and_repair_gate() -> None:
    from dromia.config import TemporalBiomechanicsConfig

    assert not AutoPoseRepairConfig().limb_length_enabled
    assert TemporalBiomechanicsConfig().limb_length_weight == 0.0


def test_repairs_only_flagged_joint_with_plain_candidate() -> None:
    pose, boxes = fixture_pose()
    native = pose.copy()
    pose[12:18, 0, 16] = (520, 40)
    alignment = np.zeros(pose.shape[:3], dtype=np.float32)
    alignment[12:18, 0, 16] = 80

    result = repair_pose(pose, boxes, alignment, AutoPoseRepairConfig(), plain_candidate_xy=native)

    np.testing.assert_allclose(result.keypoints_xy[12:18, 0, 16], native[12:18, 0, 16])
    assert np.all(result.method[12:18, 0, 16] == METHOD_PLAIN)
    unchanged = np.ones(pose.shape[:3], dtype=bool)
    unchanged[12:18, 0, 16] = False
    np.testing.assert_array_equal(result.keypoints_xy[unchanged], pose[unchanged])


def test_boundary_interpolation_uses_two_anchors_or_one_sided_hold() -> None:
    pose, boxes = fixture_pose()
    pose[12:18, 0, 16] = (520, 40)
    alignment = np.zeros(pose.shape[:3], dtype=np.float32)
    alignment[12:18, 0, 16] = 80
    result = repair_pose(pose, boxes, alignment, AutoPoseRepairConfig())
    assert np.all(result.method[12:18, 0, 16] == METHOD_INTERPOLATION)
    assert np.all(result.keypoints_xy[12:18, 0, 16, 0] < 300)

    pose2, boxes2 = fixture_pose()
    pose2[:6, 0, 16] = (520, 40)
    alignment2 = np.zeros(pose2.shape[:3], dtype=np.float32)
    alignment2[:6, 0, 16] = 80
    result2 = repair_pose(pose2, boxes2, alignment2, AutoPoseRepairConfig())
    assert np.all(result2.method[:6, 0, 16] == METHOD_INTERPOLATION)
    np.testing.assert_allclose(
        result2.keypoints_xy[:6, 0, 16],
        np.repeat(result2.keypoints_xy[6:7, 0, 16], 6, axis=0),
    )


def test_transient_bilateral_knee_collapse_flags_both_channels() -> None:
    pose, boxes = fixture_pose()
    pose[:, 0, 13, 0] -= 30.0
    pose[20, 0, 13] = pose[20, 0, 14] + (1.0, 1.0)

    flagged, reasons = detect_catastrophic_drift(pose, boxes, None, AutoPoseRepairConfig())

    assert flagged[20, 0, 13]
    assert flagged[20, 0, 14]
    assert reasons[20, 0, 13] & REASON_BILATERAL_COLLAPSE
    assert reasons[20, 0, 14] & REASON_BILATERAL_COLLAPSE


def test_runner_36_frame_278_regression_flags_both_collapsed_knees() -> None:
    pose = np.zeros((8, 1, 17, 2), dtype=np.float32)
    boxes = np.tile(np.asarray([900, 300, 1200, 663.9158], dtype=np.float32), (8, 1, 1))
    pose[:, 0, 13] = np.asarray(
        [
            [1056.11, 505.93],
            [1051.96, 505.34],
            [1045.36, 508.15],
            [1040.84, 506.67],
            [1078.72, 512.95],
            [1033.71, 507.05],
            [1029.94, 507.14],
            [1025.15, 510.00],
        ],
        dtype=np.float32,
    )
    pose[:, 0, 14] = np.asarray(
        [
            [1115.72, 515.76],
            [1105.32, 516.10],
            [1100.62, 516.37],
            [1091.59, 513.39],
            [1081.65, 515.41],
            [1074.68, 517.23],
            [1068.41, 517.49],
            [1060.02, 517.51],
        ],
        dtype=np.float32,
    )

    flagged, reasons = detect_catastrophic_drift(pose, boxes, None, AutoPoseRepairConfig())

    assert flagged[4, 0, 13]
    assert flagged[4, 0, 14]
    assert reasons[4, 0, 13] & REASON_BILATERAL_COLLAPSE
    assert reasons[4, 0, 14] & REASON_BILATERAL_COLLAPSE

    # High model quality must not make the HMM output accept one physical knee
    # as both anatomical channels.
    quality = np.ones(pose.shape[:3], dtype=np.float32)
    interval_flags, interval_reasons = detect_unstable_intervals(
        pose, boxes, quality, AutoPoseRepairConfig()
    )
    assert interval_flags[4, 0, 13]
    assert interval_flags[4, 0, 14]
    repaired = repair_pose(
        pose,
        boxes,
        None,
        AutoPoseRepairConfig(),
        detected_flags=interval_flags,
        detected_reasons=interval_reasons,
    )
    repaired_separation = np.linalg.norm(
        repaired.keypoints_xy[4, 0, 13] - repaired.keypoints_xy[4, 0, 14]
    ) / (boxes[4, 0, 3] - boxes[4, 0, 1])
    assert repaired_separation > 0.08


def test_unstable_interval_keeps_stable_extremes_and_includes_smooth_errors() -> None:
    frames = 18
    pose, boxes = fixture_pose(frames)
    expected = pose.copy()
    pose[:, 0, 13, 0] -= 35.0
    expected[:, 0, 13, 0] -= 35.0
    quality = np.full(pose.shape[:3], 0.8, dtype=np.float32)
    for t in (5, 7, 8, 9, 10, 11):
        pose[t, 0, 13] = pose[t, 0, 14] + (1.0, 1.0)
    quality[5:12, 0, 13:15] = 0.4

    flagged, reasons = detect_unstable_intervals(pose, boxes, quality, AutoPoseRepairConfig())

    assert not flagged[4, 0, 13]
    assert np.all(flagged[5:12, 0, 13])
    assert np.all(flagged[5:12, 0, 14])
    assert not flagged[12, 0, 13]
    repaired = repair_pose(
        pose,
        boxes,
        None,
        AutoPoseRepairConfig(),
        detected_flags=flagged,
        detected_reasons=reasons,
    )
    np.testing.assert_allclose(repaired.keypoints_xy[5:12, 0, 13], expected[5:12, 0, 13], atol=1e-5)


def test_sustained_bilateral_overlap_is_not_forced_apart() -> None:
    pose, boxes = fixture_pose()
    pose[:, 0, 13] = pose[:, 0, 14] + (1.0, 1.0)

    flagged, reasons = detect_catastrophic_drift(pose, boxes, None, AutoPoseRepairConfig())

    assert not np.any((reasons[..., 13:15] & REASON_BILATERAL_COLLAPSE) != 0)
    assert not np.any(flagged[..., 13:15])


def test_crossed_middle_anchor_coalesces_two_ankle_intervals() -> None:
    pose, boxes = fixture_pose(frames=14)
    # Keep three stable frames on each side. Frame 7 is a smooth but
    # identity-crossed plateau between the two detected anomaly fragments.
    pose[:7, 0, 15] = (145, 430)
    pose[:7, 0, 16] = (225, 430)
    pose[7, 0, 15] = (225, 430)
    pose[7, 0, 16] = (145, 430)
    pose[8:10, 0, 15] = (220, 430)
    pose[8:10, 0, 16] = (150, 430)
    pose[10:, 0, 15] = (150, 430)
    pose[10:, 0, 16] = (220, 430)
    flagged = np.zeros(pose.shape[:3], dtype=bool)
    flagged[4:7, 0, 15:17] = True
    flagged[8:10, 0, 15:17] = True
    reasons = np.zeros(pose.shape[:3], dtype=np.uint8)

    merged, merged_reasons = coalesce_incompatible_bilateral_intervals(
        pose, boxes, flagged, reasons, AutoPoseRepairConfig()
    )

    assert not np.any(merged[3, 0, 15:17])
    assert np.all(merged[4:10, 0, 15:17])
    assert not np.any(merged[10, 0, 15:17])
    assert np.all((merged_reasons[7, 0, 15:17] & REASON_IDENTITY_ANCHOR_MISMATCH) != 0)
    result = repair_pose(
        pose,
        boxes,
        None,
        AutoPoseRepairConfig(),
        detected_flags=merged,
        detected_reasons=merged_reasons,
    )
    assert np.all(result.keypoints_xy[4:10, 0, 15, 0] < 200)
    assert np.all(result.keypoints_xy[4:10, 0, 16, 0] > 200)
    np.testing.assert_array_equal(result.keypoints_xy[..., :15, :], pose[..., :15, :])


def test_direct_compatible_bilateral_anchors_are_not_overexpanded() -> None:
    pose, boxes = fixture_pose(frames=14)
    flagged = np.zeros(pose.shape[:3], dtype=bool)
    flagged[5:7, 0, 15:17] = True
    reasons = np.zeros(pose.shape[:3], dtype=np.uint8)

    merged, _ = coalesce_incompatible_bilateral_intervals(
        pose, boxes, flagged, reasons, AutoPoseRepairConfig()
    )

    np.testing.assert_array_equal(merged, flagged)


def test_final_step_bound_removes_giant_knee_jump() -> None:
    pose, boxes = fixture_pose()
    pose[20, 0, 13, 0] += 180.0
    result = repair_pose(pose, boxes, None, AutoPoseRepairConfig())

    bounded = enforce_final_step_bound(result, boxes, AutoPoseRepairConfig())
    height = boxes[:, 0, 3] - boxes[:, 0, 1]
    steps = (
        np.linalg.norm(bounded.keypoints_xy[1:, 0, 13] - bounded.keypoints_xy[:-1, 0, 13], axis=-1)
        / height[1:]
    )

    assert float(np.max(steps)) <= 0.08 + 1e-6


def test_segments_are_joint_scoped_and_include_two_context_frames() -> None:
    flags = np.zeros((20, 1, 17), dtype=bool)
    flags[5:8, 0, 16] = True
    flags[10, 0, 16] = True
    segments = contiguous_segments(flags, 2)
    assert [(x.start, x.end, x.context_start, x.context_end) for x in segments] == [
        (5, 7, 3, 9),
        (10, 10, 8, 12),
    ]


def test_tracker_windows_merge_overlap_only_within_runner() -> None:
    segments = [
        JointSegment(0, 13, 5, 6, 1, 10),
        JointSegment(0, 14, 11, 12, 7, 16),
        JointSegment(0, 15, 30, 31, 26, 35),
        JointSegment(1, 13, 8, 9, 4, 13),
    ]

    windows = merge_tracker_windows(segments)

    assert [(item[0], item[1], item[2], len(item[3])) for item in windows] == [
        (0, 1, 16, 2),
        (0, 26, 35, 1),
        (1, 4, 13, 1),
    ]


def test_bounded_cotracker_batches_overlapping_segments(monkeypatch, tmp_path) -> None:
    from dromia.models import bounded_cotracker as reconstruction

    pose, _boxes = fixture_pose()
    unresolved = np.zeros(pose.shape[:3], dtype=bool)
    unresolved[5:7, 0, 13] = True
    unresolved[7:9, 0, 14] = True
    segments = [
        JointSegment(0, 13, 5, 6, 1, 10),
        JointSegment(0, 14, 7, 8, 3, 12),
    ]

    class FakeTracker:
        calls: list[tuple[list[int], int]] = []

        def is_available(self) -> bool:
            return True

        def track_points(self, *, video_path, frame_indices, seeds, cfg):
            _ = video_path, cfg
            self.calls.append((list(frame_indices), len(seeds)))
            xy = np.stack(
                [np.repeat(seed.xy[None], len(frame_indices), axis=0) for seed in seeds]
            ).astype(np.float32)
            shape = (len(seeds), len(frame_indices))
            return reconstruction.PointTrackResult(
                xy=xy,
                visibility=np.ones(shape, dtype=bool),
                confidence=np.ones(shape, dtype=np.float32),
                tracker_name="fake",
                tracker_config={},
            )

    monkeypatch.setattr(reconstruction, "CoTracker3PointTracker", FakeTracker)
    candidate, visibility, warnings = run_bounded_cotracker(
        video_path=tmp_path / "unused.mp4",
        frame_indices=list(range(len(pose))),
        object_ids=[36],
        pose_xy=pose,
        bboxes_xyxy=np.tile(np.asarray([100, 100, 300, 500], dtype=np.float32), (len(pose), 1, 1)),
        unresolved=unresolved,
        segments=segments,
        cfg=AutoPoseRepairConfig(),
    )

    assert FakeTracker.calls == [(list(range(1, 13)), 2)]
    assert np.all(np.isfinite(candidate[5:7, 0, 13]))
    assert np.all(np.isfinite(candidate[7:9, 0, 14]))
    assert np.all(visibility[5:7, 0, 13] == 1)
    assert not np.any(np.isfinite(candidate[:5, 0, 13]))
    assert warnings == []


def test_reliable_late_frames_remain_bitwise_unchanged() -> None:
    pose, boxes = fixture_pose()
    original = pose.copy()
    pose[12:18, 0, 16] = (520, 40)
    alignment = np.zeros(pose.shape[:3], dtype=np.float32)
    alignment[12:18, 0, 16] = 80
    result = repair_pose(pose, boxes, alignment, AutoPoseRepairConfig())
    np.testing.assert_array_equal(result.keypoints_xy[20:], original[20:])


def test_visible_tracker_on_wrong_body_part_is_rejected_by_boundaries() -> None:
    pose, boxes = fixture_pose()
    native = pose.copy()
    pose[12:18, 0, 16] = (520, 40)
    alignment = np.zeros(pose.shape[:3], dtype=np.float32)
    alignment[12:18, 0, 16] = 80
    tracker = np.full_like(pose, np.nan)
    tracker[12:18, 0, 16] = (290, 180)  # inside bbox, plausible-ish limb, wrong feature
    visible = np.ones(pose.shape[:3], dtype=np.float32)

    result = repair_pose(
        pose,
        boxes,
        alignment,
        AutoPoseRepairConfig(),
        tracker_candidate_xy=tracker,
        tracker_visibility=visible,
    )

    assert np.all(result.method[12:18, 0, 16] == METHOD_INTERPOLATION)
    assert (
        np.max(np.linalg.norm(result.keypoints_xy[12:18, 0, 16] - native[12:18, 0, 16], axis=1)) < 2
    )
