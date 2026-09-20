from __future__ import annotations

import numpy as np
import pytest

from dromia import calibration as dromia_calibration
from dromia import config as dromia_config
from dromia import dto as dromia_dto
from dromia.gait import analysis as gait_analysis


def test_default_contact_clearance_is_subpixel_for_shortened_video_scale() -> None:
    cfg = dromia_config.GaitAnalysisConfig()

    assert cfg.fps_override is None
    assert (
        max(
            cfg.min_contact_tolerance_px,
            cfg.contact_tolerance_bbox_fraction * 400,
        )
        < 1.0
    )


def excluded_local_step_ground_model_was_not_migrated() -> None:
    fps = 100.0
    frames = np.arange(90, dtype=np.int32)
    ids = np.asarray([5], dtype=np.int32)
    bboxes = np.tile(np.asarray([0, 0, 100, 100], np.float32), (len(frames), 1, 1))
    trajectories = {
        "left": interpolated_trajectory(
            [(0, 80), (8, 80), (20, 40), (35, 85), (43, 85), (55, 40), (70, 90), (89, 90)],
            frames,
        ),
        "right": interpolated_trajectory(
            [(0, 75), (12, 75), (27, 35), (42, 82), (50, 82), (65, 35), (80, 87), (89, 87)],
            frames,
        ),
    }
    assignments = []
    for side, bottoms in trajectories.items():
        x0 = 10 if side == "left" else 60
        for frame, bottom in zip(frames, bottoms, strict=True):
            mask = np.zeros((120, 100), np.uint8)
            mask[max(int(bottom) - 5, 0) : int(bottom) + 1, x0 : x0 + 20] = 1
            assignments.append(
                dromia_dto.ShoeAssignment(
                    frame_idx=int(frame),
                    runner_id=5,
                    side=side,
                    shoe_obj_id=int(frame),
                    score=1.0,
                    mask=mask,
                )
            )

    shoes = gait_analysis.analyze_shoes(
        assignments,
        bboxes,
        frames,
        ids,
        dromia_config.GaitAnalysisConfig(ground_model="local_steps"),
        fps=fps,
        directions={5: "right"},
    )

    by_side = {
        side: sorted((item for item in shoes if item.side == side), key=lambda item: item.frame_idx)
        for side in ("left", "right")
    }
    assert by_side["left"][0].ground_y == 80.0
    assert by_side["left"][39].ground_y == 85.0
    assert by_side["left"][-1].ground_y == 90.0
    assert by_side["right"][0].ground_y == 75.0
    assert by_side["right"][-1].ground_y == 87.0
    assert all(
        first.ground_y <= second.ground_y
        for first, second in zip(by_side["left"], by_side["left"][1:], strict=False)
    )
    assert all(
        first.ground_y <= second.ground_y
        for first, second in zip(by_side["right"], by_side["right"][1:], strict=False)
    )
    assert by_side["left"][20].ground_y != by_side["right"][20].ground_y
    assert by_side["left"][20].ground_step_index == 1
    assert by_side["right"][20].ground_step_index == 0
    for side in ("left", "right"):
        assert set(item.ground_step_index for item in by_side[side]) == {0, 1, 2}
        assert all(
            any(item.contact for item in by_side[side] if item.ground_step_index == step)
            for step in (0, 1, 2)
        )


def test_global_ground_line_rejects_temporal_outliers_and_falls_back_horizontal() -> None:
    cfg = dromia_config.GaitAnalysisConfig(
        ground_line_bin_count=6,
        ground_line_min_candidates_per_bin=4,
    )
    frames = []
    for x in np.linspace(0, 100, 6):
        for y in (98.0, 99.0, 100.0, 100.0, 100.0, 70.0):
            frames.append(
                gait_analysis.ShoeFrame(
                    frame_idx=len(frames),
                    runner_id=1,
                    side="left",
                    score=1.0,
                    curve=[[float(x), y]],
                    max_y=y,
                    ground_x=float(x),
                )
            )

    line = gait_analysis.fit_global_ground_line(frames, cfg)

    assert line.slope == 0.0
    assert line.intercept == pytest.approx(100.0)
    assert line.quality == "horizontal_fallback"


def test_global_ground_line_accepts_a_consistent_spatial_slope() -> None:
    cfg = dromia_config.GaitAnalysisConfig(
        ground_line_bin_count=6,
        ground_line_min_candidates_per_bin=4,
    )
    frames = []
    for x in np.linspace(0, 100, 6):
        ground = 90.0 + 0.1 * x
        for y in (ground, ground, ground, ground - 20.0, ground - 30.0):
            frames.append(
                gait_analysis.ShoeFrame(
                    frame_idx=len(frames),
                    runner_id=1,
                    side="left",
                    score=1.0,
                    curve=[[float(x), y]],
                    max_y=y,
                    ground_x=float(x),
                )
            )

    line = gait_analysis.fit_global_ground_line(frames, cfg)

    assert line.slope == pytest.approx(0.1)
    assert line.intercept == pytest.approx(90.0)
    assert line.quality == "robust_spatial_line"


def excluded_global_line_ablation_was_not_migrated() -> None:
    frames = np.asarray([0], dtype=np.int32)
    ids = np.asarray([1], dtype=np.int32)
    boxes = np.asarray([[[0, 0, 100, 100]]], dtype=np.float32)
    mask = np.zeros((120, 100), np.uint8)
    mask[80:91, 20:40] = 1
    assignment = dromia_dto.ShoeAssignment(
        frame_idx=0,
        runner_id=1,
        side="left",
        shoe_obj_id=1,
        score=1.0,
        mask=mask,
    )
    video_ground = gait_analysis.GroundLine(0.0, 100.0, 1.0, 10, 5, "video_global")

    shoes = gait_analysis.analyze_shoes(
        [assignment],
        boxes,
        frames,
        ids,
        dromia_config.GaitAnalysisConfig(ground_model="global_line"),
        global_ground=video_ground,
    )

    assert shoes[0].ground_y == 100.0
    assert shoes[0].ground_line_quality == "video_global"


def test_per_shoe_ground_lines_keep_offsets_and_accept_one_degree_tilt() -> None:
    cfg = dromia_config.GaitAnalysisConfig(
        ground_line_candidate_quantile=94,
        ground_line_bin_count=6,
        ground_line_min_candidates_per_bin=3,
    )
    positive_slope = float(np.tan(np.radians(1.0)))
    negative_slope = -positive_slope
    frames = []
    for runner_id, side, intercept, slope in (
        (1, "left", 90.0, positive_slope),
        (1, "right", 105.0, positive_slope),
        (2, "left", 120.0, negative_slope),
        (2, "right", 135.0, negative_slope),
    ):
        for x in np.linspace(0, 100, 6):
            ground = intercept + slope * x
            for y in (ground, ground, ground, ground - 15.0):
                frames.append(
                    gait_analysis.ShoeFrame(
                        frame_idx=len(frames),
                        runner_id=runner_id,
                        side=side,
                        score=1.0,
                        curve=[[float(x), float(y)]],
                        max_y=float(y),
                        ground_x=float(x),
                    )
                )

    lines = gait_analysis.fit_per_shoe_ground_lines(frames, cfg)

    assert set(lines) == {
        (1, "left"),
        (1, "right"),
        (2, "left"),
        (2, "right"),
    }
    assert lines[(1, "left")].slope == pytest.approx(positive_slope)
    assert lines[(1, "right")].slope == pytest.approx(positive_slope)
    assert lines[(2, "left")].slope == pytest.approx(negative_slope)
    assert lines[(2, "right")].slope == pytest.approx(negative_slope)
    assert lines[(1, "left")].intercept == pytest.approx(90.0)
    assert lines[(1, "right")].intercept == pytest.approx(105.0)
    assert lines[(2, "left")].intercept == pytest.approx(120.0)
    assert lines[(2, "right")].intercept == pytest.approx(135.0)
    assert all(line.quality == "robust_per_shoe_line" for line in lines.values())


def test_contact_rejects_large_mask_penetration_below_ground() -> None:
    frames = np.asarray([0, 1], dtype=np.int32)
    ids = np.asarray([1], dtype=np.int32)
    boxes = np.tile(np.asarray([0, 0, 100, 100], np.float32), (2, 1, 1))
    assignments = []
    for frame, bottom in enumerate((100, 120)):
        mask = np.zeros((130, 100), np.uint8)
        mask[90 : bottom + 1, 20:40] = 1
        assignments.append(
            dromia_dto.ShoeAssignment(
                frame_idx=frame,
                runner_id=1,
                side="left",
                shoe_obj_id=frame,
                score=1.0,
                mask=mask,
            )
        )

    shoes = gait_analysis.analyze_shoes(
        assignments,
        boxes,
        frames,
        ids,
        dromia_config.GaitAnalysisConfig(ground_line_candidate_quantile=50),
    )
    by_frame = {item.frame_idx: item for item in shoes if item.max_y is not None}

    assert by_frame[0].contact is False
    assert by_frame[1].contact is False


def test_high_fps_contact_filter_removes_short_blips_after_gap_closing() -> None:
    items = [gait_analysis.ShoeFrame(frame, 2, "left", 1.0, [], None) for frame in range(30)]
    for frame in (2, 3, 7, 8):
        items[frame].contact = True
    for frame in range(15, 25):
        items[frame].contact = True
    for frame in range(10, 13):
        opposite = gait_analysis.ShoeFrame(frame, 2, "right", 1.0, [], None)
        opposite.contact = True
        items.append(opposite)

    gait_analysis.stabilize_contacts(
        items,
        dromia_config.GaitAnalysisConfig(
            max_contact_gap_seconds=0.02,
            min_contact_seconds=0.04,
        ),
        fps=240.0,
    )

    assert not any(item.contact for item in items[:15])
    assert all(item.contact for item in items[15:25])


def test_shoe_kinematics_refines_contact_without_runner_specific_frames() -> None:
    items = []
    for frame in range(40, 106):
        if frame <= 47:
            bottom_y = 600.0 + frame - 40
        elif frame <= 90:
            bottom_y = min(623.0, 605.0 + (frame - 48) * 1.5)
        else:
            bottom_y = 623.0 - (frame - 90) * 4.0
        lower_mean_y = (
            620.0 + (frame - 61) * (2.0 / 9.0) if frame <= 70 else 622.0 - (frame - 70) * 0.195
        )
        item = gait_analysis.ShoeFrame(
            frame_idx=frame,
            runner_id=7,
            side="left",
            score=1.0,
            curve=[],
            max_y=bottom_y,
            mask_observed=True,
            lower_curve_mean_y=lower_mean_y,
            bbox_height=400.0,
            contact=61 <= frame <= 103,
            contact_source="direct" if 61 <= frame <= 103 else "observed_non_contact",
        )
        items.append(item)

    gait_analysis.refine_contacts_from_shoe_kinematics(
        items,
        dromia_config.GaitAnalysisConfig(),
        fps=240.0,
    )
    by_frame = {item.frame_idx: item for item in items}

    assert by_frame[47].contact is False
    assert by_frame[48].contact is True
    assert by_frame[48].contact_source == "kinematic_refinement"
    assert by_frame[91].contact is True
    assert by_frame[92].contact is False
    assert by_frame[92].contact_source == "kinematic_refinement"
    assert by_frame[103].contact is False


def test_shoe_kinematics_uses_bottom_clearance_without_a_turning_point() -> None:
    items = []
    for frame in range(110, 170):
        shoe_bottom_y = min(625.0, 594.0 + (frame - 110))
        item = gait_analysis.ShoeFrame(
            frame_idx=frame,
            runner_id=3,
            side="right",
            score=1.0,
            curve=[],
            max_y=shoe_bottom_y,
            mask_observed=True,
            lower_curve_mean_y=610.0,
            ground_y=626.0,
            clearance_px=626.0 - shoe_bottom_y,
            bbox_height=400.0,
            contact=138 <= frame <= 168,
            contact_source="direct" if 138 <= frame <= 168 else "observed_non_contact",
        )
        items.append(item)

    gait_analysis.refine_contacts_from_shoe_kinematics(
        items,
        dromia_config.GaitAnalysisConfig(),
        fps=240.0,
    )
    by_frame = {item.frame_idx: item for item in items}

    assert by_frame[135].contact is False
    assert by_frame[136].contact is True
    assert by_frame[136].contact_source == "kinematic_refinement"


def test_same_foot_fragments_bridge_only_without_opposite_contact() -> None:
    items = [
        gait_analysis.ShoeFrame(frame, 2, side, 1.0, [], None)
        for side in ("left", "right")
        for frame in range(90)
    ]
    by_key = {(item.frame_idx, item.side): item for item in items}
    for start, end in ((10, 20), (40, 50), (70, 80)):
        for frame in range(start, end + 1):
            by_key[frame, "left"].contact = True
    for frame in range(60, 66):
        by_key[frame, "right"].contact = True

    gait_analysis.stabilize_contacts(
        items,
        dromia_config.GaitAnalysisConfig(
            max_contact_gap_seconds=0.01,
            min_contact_seconds=0.04,
        ),
        fps=100.0,
    )

    assert all(by_key[frame, "left"].contact for frame in range(10, 51))
    assert not any(by_key[frame, "left"].contact for frame in range(51, 70))
    assert all(by_key[frame, "left"].contact for frame in range(70, 81))


def test_local_ground_does_not_invent_observations_and_splits_long_gaps() -> None:
    items = [
        gait_analysis.ShoeFrame(frame, 2, "left", 1.0, [[10.0, y]], y)
        for frame, y in [(0, 80.0), (2, 80.0), (20, 95.0), (21, 95.0)]
    ]
    bboxes = np.tile(np.asarray([0, 0, 100, 100], np.float32), (22, 1, 1))

    gait_analysis.assign_local_step_grounds(
        items,
        bboxes=bboxes,
        frame_pos={frame: frame for frame in range(22)},
        obj_idx=0,
        fps=100.0,
        cfg=dromia_config.GaitAnalysisConfig(),
    )

    assert [item.frame_idx for item in items] == [0, 2, 20, 21]
    assert [item.ground_y for item in items] == [80.0, 80.0, 95.0, 95.0]
    assert [item.ground_step_index for item in items] == [0, 0, 1, 1]


def test_heel_patch_can_land_before_midfoot_reaches_floor() -> None:
    frames = np.asarray([0, 1], np.int32)
    ids = np.asarray([1, 2], np.int32)
    bboxes = np.tile(np.asarray([0, 0, 100, 100], np.float32), (2, 2, 1))
    assignments = []
    for runner_id, contact_columns in ((1, range(10, 15)), (2, range(18, 23))):
        for frame in frames:
            mask = np.zeros((110, 100), np.uint8)
            for x in range(10, 30):
                bottom = 100 if frame == 1 else 97 if x in contact_columns else 90
                mask[80 : bottom + 1, x] = 1
            assignments.append(
                dromia_dto.ShoeAssignment(
                    frame_idx=int(frame),
                    runner_id=runner_id,
                    side="left",
                    shoe_obj_id=runner_id * 10 + int(frame),
                    score=1.0,
                    mask=mask,
                )
            )

    shoes = gait_analysis.analyze_shoes(
        assignments,
        bboxes,
        frames,
        ids,
        dromia_config.GaitAnalysisConfig(ground_percentile=100),
        directions={1: "right", 2: "right"},
    )
    by_runner_frame = {(x.runner_id, x.frame_idx): x for x in shoes}

    assert by_runner_frame[(1, 0)].contact is True
    assert by_runner_frame[(1, 0)].contact_position is not None
    assert by_runner_frame[(1, 0)].contact_position <= 0.34
    assert gait_analysis.classify_strike(by_runner_frame[(1, 0)], "right")[0] == "heel"
    assert by_runner_frame[(2, 0)].contact is False
    assert by_runner_frame[(2, 1)].contact is True
    assert gait_analysis.classify_strike(by_runner_frame[(2, 1)], "right")[0] == "midfoot"


def test_gait_events_strike_timing_angles_and_flight() -> None:
    frames = np.arange(8, dtype=np.int32)
    ids = np.asarray([7], dtype=np.int32)
    pose = synthetic_pose()
    bboxes = np.tile(np.asarray([0, 0, 160, 120], np.float32), (8, 1, 1))
    assignments = synthetic_shoes()

    result = gait_analysis.analyze_gait(
        frame_indices=frames,
        object_ids=ids,
        pose_xy=pose,
        bboxes_xyxy=bboxes,
        shoe_assignments=assignments,
        fps=10.0,
        cfg=dromia_config.GaitAnalysisConfig(
            ground_percentile=90,
            contact_tolerance_bbox_fraction=0.01,
            min_contact_tolerance_px=1.5,
        ),
    )

    runner = result["runners"]["7"]
    assert result["schema_version"] == 13
    assert set(runner["ground_y_by_side"]) == {"left", "right"}
    assert set(runner["ground_lines"]) == {"left", "right"}
    assert "ground_y" not in runner
    left = [event for event in runner["events"] if event["side"] == "left"]
    assert left[0]["landing_frame"] == 1
    assert left[0]["takeoff_frame"] == 2
    assert left[0]["contact_time_s"] == 0.2
    assert left[0]["same_foot_flight_time_s"] == 0.3
    assert left[0]["strike_type"] == "forefoot"
    assert left[0]["landing_knee_angle_deg"] == 180.0
    assert left[0]["landing_tibia_horizontal_angle_deg"] == 90.0
    assert left[1]["landing_frame"] == 6
    assert left[1]["takeoff_frame"] is None
    assert left[1]["contact_time_s"] is None
    right = [event for event in runner["events"] if event["side"] == "right"]
    assert right[0]["landing_frame"] is None
    assert right[0]["takeoff_frame"] == 0
    assert right[0]["strike_type"] == "not_observed"
    assert right[0]["contact_time_s"] is None
    assert "midflight_frame" not in runner["flight_intervals"][0]
    assert runner["frames"][1]["left_knee_angle_deg"] == 180.0
    assert runner["frames"][1]["left_tibia_horizontal_angle_deg"] == 90.0
    assert runner["frames"][1]["torso_posture"] == "straight"
    assert runner["global_contact"]["observed_frame_count"] > 0
    assert runner["asymmetry"]["diagnostic"] is False
    assert left[0]["strike_threshold_version"] == "outsole_thirds_v1_unvalidated"


def test_calibration_x_range_records_window_inclusion_without_discarding_observable_metrics() -> (
    None
):
    pose = synthetic_pose()
    pose[0, 0, 5, 0] = 49
    pose[7, 0, 16, 0] = 91
    calibration = dromia_calibration.fit_ground_calibration(
        video_sha256="video",
        image_points_xy=[(50, 110), (90, 110), (90, 0), (50, 0)],
        validation_error_m=0.0,
    )

    result = gait_analysis.analyze_gait(
        frame_indices=np.arange(8),
        object_ids=np.asarray([7]),
        pose_xy=pose,
        bboxes_xyxy=np.tile(np.asarray([0, 0, 160, 120], np.float32), (8, 1, 1)),
        shoe_assignments=synthetic_shoes(),
        fps=10,
        cfg=dromia_config.GaitAnalysisConfig(),
        calibration=calibration,
    )

    runner = result["runners"]["7"]
    assert runner["measurement_window"] == {
        "applied": True,
        "x_min": 50.0,
        "x_max": 90.0,
        "included_frame_count": 6,
        "excluded_frame_count": 2,
    }
    assert runner["frames"][0]["measurement_window_included"] is False
    assert runner["frames"][0]["left_knee_angle_deg"] == 180.0
    assert runner["frames"][1]["measurement_window_included"] is True
    assert runner["frames"][7]["measurement_window_included"] is False
    left_events = [event for event in runner["events"] if event["side"] == "left"]
    assert left_events[0]["landing_frame"] == 1
    assert left_events[0]["landing_contact_point_in_calibration"] is True


def test_contact_point_outside_calibration_bounds_excluded_from_distances() -> None:
    calibration = dromia_calibration.fit_ground_calibration(
        video_sha256="video",
        image_points_xy=[(50, 110), (90, 110), (90, 0), (50, 0)],
        longitudinal_m=4.0,
        transverse_m=6.0,
        validation_error_m=0.0,
    )
    # Point clearly outside the calibration rectangle in world coordinates
    outside_pt = (20.0, 50.0)  # x=20 is far to the left of x=50
    inside_pt = (70.0, 50.0)  # x=70 is between 50 and 90

    assert gait_analysis.project_point(calibration, outside_pt, require_inside=True) is None
    assert gait_analysis.project_point(calibration, outside_pt, require_inside=False) is not None
    assert gait_analysis.project_point(calibration, inside_pt, require_inside=True) is not None


def test_missing_shoes_and_pose_are_reported_without_inventing_metrics() -> None:
    pose = np.full((2, 1, 17, 2), np.nan, np.float32)
    result = gait_analysis.analyze_gait(
        frame_indices=np.arange(2),
        object_ids=np.asarray([3]),
        pose_xy=pose,
        bboxes_xyxy=np.tile(np.asarray([0, 0, 100, 100], np.float32), (2, 1, 1)),
        shoe_assignments=[],
        fps=30,
        cfg=dromia_config.GaitAnalysisConfig(),
    )
    runner = result["runners"]["3"]
    assert runner["events"] == []
    assert runner["frames"][0]["left_knee_angle_deg"] is None
    assert runner["frames"][0]["torso_posture"] == "unknown"
    assert runner["frames"][0]["left_contact"] is None
    assert runner["frames"][0]["left_contact_source"] == "unknown"
    assert runner["global_contact"]["observed_frame_count"] == 0
    assert runner["frames"][0]["left_shoe_clearance_px"] is None
    assert "left_shoe_axis_xy" not in runner["frames"][0]
    assert runner["frames"][0]["left_foot_tibia_angle_deg"] is None


def test_cadence_step_stride_and_stance_phases() -> None:
    timing = gait_analysis.synthetic_timebase(np.arange(100), 100.0)
    events = [
        {"landing_frame": 0, "side": "left", "landing_contact_point_m": (0.0, 0.0)},
        {"landing_frame": 30, "side": "right", "landing_contact_point_m": (1.0, 0.5)},
        {"landing_frame": 60, "side": "left", "landing_contact_point_m": (2.0, 0.0)},
    ]
    calibration = dromia_calibration.fit_ground_calibration(
        video_sha256="video",
        image_points_xy=[(0, 0), (4, 0), (4, 6), (0, 6)],
        validation_error_m=0.0,
    )

    cadence = gait_analysis.cadence_metrics(events, timing)
    spatial = gait_analysis.spatial_distance_metrics(events, calibration)

    assert cadence["cadence_spm"] == 200.0
    assert cadence["step_interval_count"] == 2
    assert spatial["mean_step_length_m"] == 1.0
    assert spatial["mean_stride_length_m"] == 2.0
    assert spatial["steps"][0]["distance_m"] > spatial["steps"][0]["longitudinal_distance_m"]


def test_two_contacts_are_marked_as_extrapolated_cadence() -> None:
    timing = gait_analysis.synthetic_timebase(np.arange(61), 100.0)
    cadence = gait_analysis.cadence_metrics(
        [
            {"landing_frame": 0, "side": "left"},
            {"landing_frame": 30, "side": "right"},
        ],
        timing,
    )

    assert cadence["cadence_spm"] == 200.0
    assert cadence["is_extrapolated_from_two_contacts"] is True


def test_slow_motion_calibration_restores_real_world_cadence() -> None:
    timing = gait_analysis.synthetic_timebase(np.arange(61), 30.0).model_copy(
        update={
            "real_world_fps": 240.0,
            "media_to_real_time_scale": 0.125,
            "timestamps_s": [value / 240.0 for value in range(61)],
            "frame_durations_s": [1.0 / 240.0 for _ in range(61)],
            "frame_duration_s": 1.0 / 240.0,
        }
    )
    cadence = gait_analysis.cadence_metrics(
        [
            {"landing_frame": 0, "side": "left"},
            {"landing_frame": 60, "side": "right"},
        ],
        timing,
    )

    assert cadence["cadence_spm"] == 240.0
    assert cadence["plausible"] is True


def test_implausible_cadence_is_retained_but_not_reported_as_valid() -> None:
    timing = gait_analysis.synthetic_timebase(np.arange(11), 30.0)
    cadence = gait_analysis.cadence_metrics(
        [{"landing_frame": 0, "side": "left"}, {"landing_frame": 1, "side": "right"}],
        timing,
    )

    assert cadence["cadence_spm"] is None
    assert cadence["unfiltered_cadence_spm"] == pytest.approx(1800.0)
    assert cadence["quality"] == "outside_plausible_range"


def test_global_contact_union_tracks_complete_censored_and_unknown_intervals() -> None:
    timing = gait_analysis.synthetic_timebase(np.arange(8), 10.0)
    rows = [
        {"frame_idx": 0, "left_contact": None, "right_contact": False},
        {"frame_idx": 1, "left_contact": False, "right_contact": False},
        {"frame_idx": 2, "left_contact": True, "right_contact": False},
        {"frame_idx": 3, "left_contact": True, "right_contact": True},
        {"frame_idx": 4, "left_contact": False, "right_contact": False},
        {"frame_idx": 5, "left_contact": None, "right_contact": False},
        {"frame_idx": 6, "left_contact": True, "right_contact": False},
        {"frame_idx": 7, "left_contact": False, "right_contact": False},
    ]

    result = gait_analysis.global_contact_metrics(rows, timing)

    assert result["complete_interval_count"] == 1
    assert result["censored_interval_count"] == 1
    assert result["mean_global_contact_time_s"] == 0.2
    assert result["unknown_frame_count"] == 2
    assert result["contact_duty_factor"] == 3 / 6


def test_missing_mask_is_unknown_and_bridged_contact_records_provenance() -> None:
    items = [gait_analysis.ShoeFrame(frame, 2, "left", 1.0, [], None) for frame in range(5)]
    for frame in (1, 3):
        items[frame].mask_observed = True
        items[frame].contact = True
        items[frame].contact_source = "direct"

    gait_analysis.stabilize_contacts(
        items,
        dromia_config.GaitAnalysisConfig(
            max_contact_gap_seconds=0.2,
            min_contact_seconds=0.0,
        ),
        fps=10.0,
    )

    assert items[0].contact is None
    assert items[0].contact_source == "unknown"
    assert items[2].contact is True
    assert items[2].contact_source == "short_gap_bridge"


def test_activity_window_excludes_clip_boundaries_from_flight_and_duty_factor() -> None:
    timing = gait_analysis.synthetic_timebase(np.arange(8), 10.0)
    rows = [
        {
            "frame_idx": frame,
            "left_contact": frame in {2, 3},
            "right_contact": frame in {5, 6},
            "left_mask_observed": True,
            "right_mask_observed": True,
        }
        for frame in range(8)
    ]
    events = [
        {"landing_frame": 2, "takeoff_frame": 3},
        {"landing_frame": 5, "takeoff_frame": 6},
    ]

    window = gait_analysis.apply_activity_window(rows, events)
    flights = gait_analysis.global_flight_intervals_from_rows(rows, timing)
    contact = gait_analysis.global_contact_metrics(rows, timing)

    assert window["start_frame"] == 2 and window["end_frame"] == 6
    assert contact["inactive_excluded_frame_count"] == 3
    assert contact["contact_duty_factor"] == 4 / 5
    assert [(x["start_frame"], x["end_frame"]) for x in flights] == [(4, 4)]


def test_endpoint_phase_and_low_quality_event_are_withheld_from_asymmetry() -> None:
    timing = gait_analysis.synthetic_timebase(np.arange(4), 10.0)
    rows = [
        {
            "frame_idx": frame,
            "knee_horizontal_separation_px": float(frame),
            "bbox_height_px": 100.0,
        }
        for frame in range(4)
    ]
    event = {
        "side": "left",
        "landing_frame": 1,
        "takeoff_frame": 3,
        "contact_time_s": 0.3,
        "endpoint_quality": "observed",
        "inferred_contact_fraction": 0.95,
    }

    gait_analysis.add_stance_phases(event, rows, timing)
    gait_analysis.add_event_quality(event, dromia_config.GaitAnalysisConfig())
    asymmetry = gait_analysis.asymmetry_metrics(
        [event, {**event, "side": "right", "contact_time_s": 0.2}],
        {
            "mean_step_length_m_by_side": {"left": None, "right": None},
            "mean_stride_length_m_by_side": {"left": None, "right": None},
        },
        dromia_config.GaitAnalysisConfig(),
    )

    assert event["phase_quality"] == "endpoint_alignment_withheld"
    assert event["braking_time_s"] is None and event["propulsion_time_s"] is None
    assert event["event_quality"] == "low_quality"
    assert asymmetry["metrics"]["contact_time_s"]["left_n"] == 0


def test_foot_axis_temporal_quality_rejects_isolated_outlier() -> None:
    frames = [
        gait_analysis.ShoeFrame(
            frame_idx=index,
            runner_id=7,
            side="left",
            score=1.0,
            curve=[],
            max_y=1.0,
            foot_axis_xy=(0.0, 1.0) if index == 2 else (1.0, 0.0),
            foot_axis_quality="valid",
        )
        for index in range(5)
    ]

    gait_analysis.stabilize_foot_axes(frames, dromia_config.GaitAnalysisConfig())

    assert frames[2].foot_axis_xy is None
    assert frames[2].foot_axis_quality == "temporally_unstable"
    assert frames[1].foot_axis_quality == "valid"


def test_asymmetry_reports_evidence_and_non_diagnostic_review_flag() -> None:
    events = [
        {"side": "left", "contact_time_s": 0.2},
        {"side": "left", "contact_time_s": 0.22},
        {"side": "right", "contact_time_s": 0.3},
        {"side": "right", "contact_time_s": 0.32},
    ]
    distances = {
        "mean_step_length_m_by_side": {"left": None, "right": None},
        "mean_stride_length_m_by_side": {"left": None, "right": None},
    }

    result = gait_analysis.asymmetry_metrics(
        events, distances, dromia_config.GaitAnalysisConfig(asymmetry_review_threshold_percent=10)
    )

    contact = result["metrics"]["contact_time_s"]
    assert contact["left_n"] == 2 and contact["right_n"] == 2
    assert contact["quality"] == "valid"
    assert contact["review_recommended"] is True
    assert result["diagnostic"] is False


def test_robust_outsole_axis_rejects_noisy_upper_points() -> None:
    x = np.linspace(0, 30, 31)
    y = 50 + 0.1 * x
    curve = np.stack((x, y), axis=1).astype(np.float32)
    curve[10, 1] += 20

    axis, rmse = gait_analysis.robust_outsole_axis(curve)

    assert axis is not None and rmse is not None
    assert abs(axis[1] / axis[0] - 0.1) < 0.01
    assert rmse < 0.1


def synthetic_pose() -> np.ndarray:
    points = np.full((8, 1, 17, 2), np.nan, np.float32)
    for t in range(8):
        points[t, 0, 5] = [60 + t, 20]
        points[t, 0, 6] = [80 + t, 20]
        points[t, 0, 11] = [60 + t, 50]
        points[t, 0, 12] = [80 + t, 50]
        points[t, 0, 13] = [60 + t, 75]
        points[t, 0, 14] = [80 + t, 75]
        points[t, 0, 15] = [60 + t, 100]
        points[t, 0, 16] = [80 + t, 100]
    points[3, 0, 13, 0] = 70
    points[3, 0, 14, 0] = 70
    return points


def synthetic_shoes() -> list[dromia_dto.ShoeAssignment]:
    contacts = {"left": {1, 2, 6, 7}, "right": {0, 4, 5}}
    output = []
    for frame in range(8):
        for side in ("left", "right"):
            mask = np.zeros((120, 160), np.uint8)
            x0 = 45 if side == "left" else 75
            y1 = 110 if frame in contacts[side] else 100
            # The rightmost portion sits lowest, yielding a forefoot strike for rightward motion.
            mask[y1 - 6 : y1 - 1, x0 : x0 + 20] = 1
            mask[y1 - 1 : y1 + 1, x0 + 14 : x0 + 20] = 1
            output.append(
                dromia_dto.ShoeAssignment(
                    frame_idx=frame,
                    runner_id=7,
                    side=side,
                    shoe_obj_id=frame * 2,
                    score=0.9,
                    mask=mask,
                )
            )
    return output


def interpolated_trajectory(anchors: list[tuple[int, float]], frames: np.ndarray) -> np.ndarray:
    return np.interp(
        frames,
        np.asarray([frame for frame, _ in anchors]),
        np.asarray([value for _, value in anchors]),
    )


def test_three_steps_varying_floor_depths_have_exact_landing_and_takeoff() -> None:
    fps = 100.0
    frames = np.arange(160, dtype=np.int32)
    ids = np.asarray([1], dtype=np.int32)
    bboxes = np.tile(np.asarray([0, 0, 100, 100], np.float32), (len(frames), 1, 1))

    # 3 distinct steps with non-linear floor heights: Step 1 (y=100), Step 2 (y=103), Step 3 (y=97)
    # The foot descends into stance and lifts off into flight
    anchors = [
        (0, 40.0),
        (15, 100.0),
        (25, 100.0),
        (45, 40.0),
        (65, 103.0),
        (75, 103.0),
        (95, 40.0),
        (115, 97.0),
        (125, 97.0),
        (159, 40.0),
    ]
    trajectory_y = interpolated_trajectory(anchors, frames)
    # x moves from 10 to 90 across the clip
    trajectory_x = np.linspace(10.0, 90.0, len(frames))

    assignments = []
    for frame, x, y in zip(frames, trajectory_x, trajectory_y, strict=True):
        mask = np.zeros((120, 120), np.uint8)
        bottom = int(round(y))
        x_center = int(round(x))
        mask[max(bottom - 5, 0) : bottom + 1, max(x_center - 10, 0) : min(x_center + 10, 120)] = 1
        assignments.append(
            dromia_dto.ShoeAssignment(
                frame_idx=int(frame),
                runner_id=1,
                side="left",
                shoe_obj_id=int(frame),
                score=1.0,
                mask=mask,
            )
        )

    shoes = gait_analysis.analyze_shoes(
        assignments,
        bboxes,
        frames,
        ids,
        dromia_config.GaitAnalysisConfig(ground_model="per_shoe_line"),
        fps=fps,
        directions={1: "right"},
    )

    left_shoes = sorted([s for s in shoes if s.side == "left"], key=lambda s: s.frame_idx)
    # Verify ground line adapts accurately to the 3 distinct step levels
    assert left_shoes[20].ground_y == pytest.approx(100.0, abs=0.5)
    assert left_shoes[70].ground_y == pytest.approx(103.0, abs=0.5)
    assert left_shoes[120].ground_y == pytest.approx(97.0, abs=0.5)

    # Verify contact states for all 3 steps
    assert all(left_shoes[f].contact is True for f in range(15, 26))
    assert not any(left_shoes[f].contact for f in range(35, 55))
    assert all(left_shoes[f].contact is True for f in range(65, 76))
    assert not any(left_shoes[f].contact for f in range(85, 105))
    assert all(left_shoes[f].contact is True for f in range(115, 126))
    assert not any(left_shoes[f].contact for f in range(135, 159))


def test_per_foot_ground_isolation() -> None:
    frames = np.asarray([0, 1, 2], dtype=np.int32)
    ids = np.asarray([1], dtype=np.int32)
    bboxes = np.tile(np.asarray([0, 0, 100, 100], np.float32), (3, 1, 1))
    assignments = []
    for side, floor in (("left", 100.0), ("right", 120.0)):
        for f in frames:
            mask = np.zeros((140, 100), np.uint8)
            mask[int(floor) - 5 : int(floor) + 1, 20:40] = 1
            assignments.append(
                dromia_dto.ShoeAssignment(
                    frame_idx=int(f),
                    runner_id=1,
                    side=side,
                    shoe_obj_id=int(f),
                    score=1.0,
                    mask=mask,
                )
            )

    shoes = gait_analysis.analyze_shoes(
        assignments,
        bboxes,
        frames,
        ids,
        dromia_config.GaitAnalysisConfig(),
        directions={1: "right"},
    )
    left_ground = next(s.ground_y for s in shoes if s.side == "left")
    right_ground = next(s.ground_y for s in shoes if s.side == "right")

    assert left_ground == pytest.approx(100.0, abs=0.5)
    assert right_ground == pytest.approx(120.0, abs=0.5)
    assert left_ground != right_ground


def test_strike_classification_on_kinematically_refined_landing() -> None:
    frames = np.arange(10, dtype=np.int32)
    # Heel is lowest at landing frame 1 (x=10..20, heel at x=10..12 for rightward motion)
    curve = np.asarray([[10.0, 100.0], [15.0, 95.0], [20.0, 90.0]], dtype=np.float32)
    shoe_1 = gait_analysis.ShoeFrame(
        frame_idx=1,
        runner_id=1,
        side="left",
        score=1.0,
        curve=curve.tolist(),
        max_y=100.0,
        ground_x=10.0,
        ground_y=100.0,
        clearance_px=0.0,
        bbox_height=100.0,
        contact=True,
        contact_source="kinematic_refinement",
        mask_observed=True,
    )
    shoe_2 = gait_analysis.ShoeFrame(
        frame_idx=2,
        runner_id=1,
        side="left",
        score=1.0,
        curve=curve.tolist(),
        max_y=100.0,
        ground_x=10.0,
        ground_y=100.0,
        clearance_px=0.0,
        bbox_height=100.0,
        contact=True,
        contact_source="direct",
        mask_observed=True,
    )
    shoe_0 = gait_analysis.ShoeFrame(
        frame_idx=0,
        runner_id=1,
        side="left",
        score=1.0,
        curve=[],
        max_y=None,
        contact=False,
        contact_source="observed_non_contact",
        mask_observed=True,
    )
    shoe_3 = gait_analysis.ShoeFrame(
        frame_idx=3,
        runner_id=1,
        side="left",
        score=1.0,
        curve=[],
        max_y=None,
        contact=False,
        contact_source="observed_non_contact",
        mask_observed=True,
    )
    lookup = {
        (0, 1, "left"): shoe_0,
        (1, 1, "left"): shoe_1,
        (2, 1, "left"): shoe_2,
        (3, 1, "left"): shoe_3,
    }
    events = gait_analysis.build_contact_events(
        1,
        frames[:4],
        lookup,
        timing=gait_analysis.synthetic_timebase(frames[:4], 30.0),
        direction="right",
        cfg=dromia_config.GaitAnalysisConfig(),
    )
    assert len(events) == 1
    assert events[0].landing_frame == 1
    assert events[0].strike_type == "heel"


def test_event_without_preceding_mask_censors_landing() -> None:
    frames = np.arange(10, dtype=np.int32)
    curve = np.asarray([[10.0, 100.0], [15.0, 95.0], [20.0, 90.0]], dtype=np.float32)
    # shoe_0 has mask_observed=False (e.g. unobserved entering the video)
    shoe_0 = gait_analysis.ShoeFrame(
        frame_idx=0,
        runner_id=1,
        side="left",
        score=1.0,
        curve=[],
        max_y=None,
        contact=False,
        contact_source="observed_non_contact",
        mask_observed=False,
    )
    shoe_1 = gait_analysis.ShoeFrame(
        frame_idx=1,
        runner_id=1,
        side="left",
        score=1.0,
        curve=curve.tolist(),
        max_y=100.0,
        ground_x=10.0,
        ground_y=100.0,
        clearance_px=0.0,
        bbox_height=100.0,
        contact=True,
        contact_source="direct",
        mask_observed=True,
    )
    shoe_2 = gait_analysis.ShoeFrame(
        frame_idx=2,
        runner_id=1,
        side="left",
        score=1.0,
        curve=[],
        max_y=None,
        contact=False,
        contact_source="observed_non_contact",
        mask_observed=True,
    )
    lookup = {
        (0, 1, "left"): shoe_0,
        (1, 1, "left"): shoe_1,
        (2, 1, "left"): shoe_2,
    }
    events = gait_analysis.build_contact_events(
        1,
        frames[:3],
        lookup,
        timing=gait_analysis.synthetic_timebase(frames[:3], 30.0),
        direction="right",
        cfg=dromia_config.GaitAnalysisConfig(),
    )
    assert len(events) == 1
    # Landing is censored / unobserved because preceding frame had no shoe mask
    assert events[0].landing_frame is None
    assert events[0].takeoff_frame == 1
    assert events[0].strike_type == "not_observed"
    assert events[0].endpoint_quality == "censored_unknown_or_boundary"
