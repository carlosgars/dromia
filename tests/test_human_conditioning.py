from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dromia.review import cvat as cvat_annotation
from dromia.review import human_conditioning
from dromia.review import propagation as annotation_propagation


def test_connected_native_anomaly_stops_before_late_reliable_frames() -> None:
    posterior = np.zeros((40, 1, 7, 2), dtype=np.float32)
    boxes = np.tile(np.asarray([100, 100, 300, 500], dtype=np.float32), (40, 1, 1))
    posterior[:, 0, 6] = (180, 430)
    posterior[10:20, 0, 6] = (520, 40)

    allowed = human_conditioning.connected_native_anomaly_mask(
        posterior, boxes, obj_idx=0, local_idx=6, anchor_time_idx=15
    )

    assert allowed[10:23].all()
    assert not allowed[23]
    assert not allowed[30]


def test_moved_point_becomes_implicit_human_anchor_unless_it_is_an_dromia_proposal(
    tmp_path: Path,
) -> None:
    bundle = make_bundle(tmp_path, frame_count=2)
    state = cvat_annotation.annotation_state(bundle)
    ankle = cvat_annotation.JOINT_IDS.index(15)
    state.points_xy[1, 0, ankle] += [6.0, 0.0]

    observations = human_conditioning.trusted_observations(bundle, state)

    assert [(item.frame_idx, item.joint_id, item.source) for item in observations] == [
        (1, 15, "HUMAN_CORRECTED_IMPLICIT")
    ]

    previous = np.zeros(state.visibility.shape, dtype=bool)
    previous[1, 0, ankle] = True
    assert human_conditioning.trusted_observations(bundle, state, previous_proposal=previous) == []


def test_accepted_frame_marks_each_visible_lower_body_point_trusted(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, frame_count=2)
    state = cvat_annotation.annotation_state(bundle)
    state.review_status[0, 0] = "ACCEPTED"

    observations = human_conditioning.trusted_observations(bundle, state)

    assert {item.joint_id for item in observations} == set(cvat_annotation.JOINT_IDS)
    assert all(item.confidence == 1.0 and item.source == "HUMAN_ACCEPTED" for item in observations)


def test_frame_ground_truth_promotes_all_visible_keypoints_to_human_anchors(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, frame_count=2)
    reviewed = [
        cvat_annotation.ReviewedFrame(
            runner_id=3,
            frame_idx=1,
            review_status="UNREVIEWED",
            frame_ground_truth=True,
            points=[
                cvat_annotation.ReviewedPoint(
                    joint_id=joint_id, xy=(10.0 + idx, 16.0), visibility=2
                )
                for idx, joint_id in enumerate(cvat_annotation.JOINT_IDS)
            ],
        )
    ]
    state = cvat_annotation.annotation_state(bundle, reviewed)

    observations = human_conditioning.trusted_observations(bundle, state)

    assert state.keypoint_ground_truth[1, 0].all()
    assert {item.joint_id for item in observations} == set(cvat_annotation.JOINT_IDS)
    assert all(
        item.source == "HUMAN_FRAME_GROUND_TRUTH" and item.confidence == 1.0
        for item in observations
    )


def test_editing_an_dromia_proposal_creates_a_new_human_observation(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, frame_count=2)
    state = cvat_annotation.annotation_state(bundle)
    ankle = cvat_annotation.JOINT_IDS.index(15)
    state.points_xy[1, 0, ankle] += [5.0, 0.0]
    applied = np.zeros(state.visibility.shape, dtype=bool)
    applied[1, 0, ankle] = True
    path = annotation_propagation.propagation_output_dir(tmp_path, bundle) / "propagation.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        frame_indices=np.asarray(bundle.frame_indices),
        runner_ids=np.asarray(bundle.runner_ids),
        propagated_xy=state.points_xy,
        proposal_applied=applied,
    )

    previous = annotation_propagation.previous_proposal_mask(tmp_path, bundle, state.points_xy)
    assert human_conditioning.trusted_observations(bundle, state, previous_proposal=previous) == []

    state.points_xy[1, 0, ankle] += [3.0, 0.0]
    previous = annotation_propagation.previous_proposal_mask(tmp_path, bundle, state.points_xy)
    observations = human_conditioning.trusted_observations(
        bundle, state, previous_proposal=previous
    )
    assert [(item.frame_idx, item.joint_id) for item in observations] == [(1, 15)]


def test_tracker_prior_updates_only_local_window_and_preserves_anchor(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, frame_count=5)
    state = cvat_annotation.annotation_state(bundle)
    knee = cvat_annotation.JOINT_IDS.index(13)
    state.points_xy[2, 0, knee] = [20.0, 16.0]
    state.review_status[2, 0] = "CORRECTED"
    observations = human_conditioning.trusted_observations(bundle, state)
    tracker_observations = [item for item in observations if item.joint_id == 13]
    tracker_xy = np.full((1, 5, 2), [20.0, 16.0], dtype=np.float32)
    tracker_visibility = np.ones((1, 5), dtype=bool)
    write_posterior_maps(tmp_path, frame_count=5, runner_id=3, peak_xy=(10, 16))
    outside_window = state.points_xy[[0, 4], 0, knee].copy()

    result = human_conditioning.condition_pose(
        run_dir=tmp_path,
        bundle=bundle,
        state=state,
        observations=observations,
        tracker_observations=tracker_observations,
        tracker_xy=tracker_xy,
        tracker_visibility=tracker_visibility,
        bboxes_xyxy=np.tile(np.asarray([0, 0, 32, 32], dtype=np.float32), (5, 1, 1)),
        cfg=human_conditioning.HumanConditioningConfig(
            window_radius=1,
            tracker_sigma_norm=0.1,
            tracker_weight=3.0,
            geometry_weight=0.0,
            enforce_constant_limb_length=False,
        ),
    )

    assert result.state.points_xy[2, 0, knee].tolist() == [20.0, 16.0]
    assert result.trusted[2, 0, knee]
    assert result.state.points_xy[1, 0, knee, 0] >= 18.0
    assert result.source[1, 0, knee] == "BAYESIAN_SMOOTHER_PRIOR"
    assert result.state.points_xy[3, 0, knee, 0] >= 18.0
    assert result.source[3, 0, knee] == "BAYESIAN_SMOOTHER_PRIOR"
    assert np.array_equal(result.state.points_xy[[0, 4], 0, knee], outside_window)


def test_contralateral_exclusion_penalizes_duplicate_ankle_location() -> None:
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    points = np.full((len(cvat_annotation.JOINT_IDS), 2), np.nan, dtype=np.float32)
    points[cvat_annotation.JOINT_IDS.index(16)] = [10.0, 16.0]

    support = human_conditioning.contralateral_exclusion_grid(
        xx,
        yy,
        np.asarray([0, 0, 32, 32], dtype=np.float32),
        15,
        points,
        100.0,
        human_conditioning.HumanConditioningConfig(),
    )

    assert support is not None
    assert support[16, 10] == 0.0
    assert support[16, 14] > 0.8


def test_ground_truth_identity_detects_an_adjacent_left_right_swap(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, frame_count=3)
    posterior = bilateral_pose(frame_count=3)
    swap_bilateral_pairs(posterior[2, 0])
    replace_bundle_posterior(bundle, posterior)
    state = cvat_annotation.annotation_state(bundle)
    state.frame_ground_truth[0, 0] = True
    state.keypoint_ground_truth[0, 0] = True
    observations = human_conditioning.trusted_observations(bundle, state)
    tracker_observations = human_conditioning.select_tracker_observations(
        [item for item in observations if item.joint_id in (13, 14, 15, 16)],
        human_conditioning.HumanConditioningConfig(),
    )
    tracker_xy = np.asarray(
        [[item.xy for _ in bundle.frame_indices] for item in tracker_observations],
        dtype=np.float32,
    )
    tracker_visibility = np.ones(tracker_xy.shape[:2], dtype=bool)

    result = human_conditioning.infer_identity_assignments(
        run_dir=tmp_path,
        bundle=bundle,
        state=state,
        posterior=posterior,
        observations=observations,
        tracker_observations=tracker_observations,
        tracker_xy=tracker_xy,
        tracker_visibility=tracker_visibility,
        bboxes_xyxy=np.tile(np.asarray([0, 0, 32, 32], dtype=np.float32), (3, 1, 1)),
        cfg=human_conditioning.HumanConditioningConfig(),
    )

    assert result.swapped[:, 0].tolist() == [False, False, True]
    assert result.swap_probability[2, 0] > 0.99
    assert result.evidence_pair_count[2, 0] == 2


def test_conditioning_decodes_swapped_channels_in_anatomical_identity(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, frame_count=2)
    posterior = bilateral_pose(frame_count=2)
    swap_bilateral_pairs(posterior[1, 0])
    replace_bundle_posterior(bundle, posterior)
    write_pose_maps(tmp_path, posterior, runner_id=3)
    state = cvat_annotation.annotation_state(bundle)
    state.frame_ground_truth[0, 0] = True
    state.keypoint_ground_truth[0, 0] = True
    observations = human_conditioning.trusted_observations(bundle, state)
    tracker_observations = human_conditioning.select_tracker_observations(
        [item for item in observations if item.joint_id in (13, 14, 15, 16)],
        human_conditioning.HumanConditioningConfig(),
    )
    tracker_xy = np.asarray(
        [[item.xy for _ in bundle.frame_indices] for item in tracker_observations],
        dtype=np.float32,
    )

    result = human_conditioning.condition_pose(
        run_dir=tmp_path,
        bundle=bundle,
        state=state,
        observations=observations,
        tracker_observations=tracker_observations,
        tracker_xy=tracker_xy,
        tracker_visibility=np.ones(tracker_xy.shape[:2], dtype=bool),
        bboxes_xyxy=np.tile(np.asarray([0, 0, 32, 32], dtype=np.float32), (2, 1, 1)),
        cfg=human_conditioning.HumanConditioningConfig(
            window_radius=1,
            geometry_weight=0.0,
            motion_weight=0.0,
            contralateral_exclusion_weight=0.0,
            enforce_constant_limb_length=False,
        ),
    )

    left_knee = cvat_annotation.JOINT_IDS.index(13)
    right_knee = cvat_annotation.JOINT_IDS.index(14)
    assert result.identity_swapped[:, 0].tolist() == [False, True]
    assert result.state.points_xy[0, 0].tolist() == posterior[0, 0].tolist()
    assert result.state.points_xy[1, 0, left_knee, 0] < 12.0
    assert result.state.points_xy[1, 0, right_knee, 0] > 20.0
    assert result.source[1, 0, left_knee] == "LEFT_RIGHT_SWAP_PRIOR"
    assert result.source[1, 0, right_knee] == "LEFT_RIGHT_SWAP_PRIOR"


def test_switching_smoother_assigns_each_bilateral_pair_independently(
    tmp_path: Path,
) -> None:
    bundle = make_bundle(tmp_path, frame_count=2)
    posterior = bilateral_pose(frame_count=2)
    swap_joint_pair(posterior[1, 0], 13, 14)
    replace_bundle_posterior(bundle, posterior)
    state = cvat_annotation.annotation_state(bundle)
    state.frame_ground_truth[0, 0] = True
    state.keypoint_ground_truth[0, 0] = True
    observations = human_conditioning.trusted_observations(bundle, state)
    tracker_observations = human_conditioning.select_tracker_observations(
        [item for item in observations if item.joint_id in (13, 14, 15, 16)],
        human_conditioning.HumanConditioningConfig(),
    )
    tracker_xy = np.asarray(
        [[item.xy for _ in bundle.frame_indices] for item in tracker_observations],
        dtype=np.float32,
    )

    result = human_conditioning.infer_identity_assignments(
        run_dir=tmp_path,
        bundle=bundle,
        state=state,
        posterior=posterior,
        observations=observations,
        tracker_observations=tracker_observations,
        tracker_xy=tracker_xy,
        tracker_visibility=np.ones(tracker_xy.shape[:2], dtype=bool),
        bboxes_xyxy=np.tile(np.asarray([0, 0, 32, 32], dtype=np.float32), (2, 1, 1)),
        cfg=human_conditioning.HumanConditioningConfig(),
    )

    assert result.pair_swapped[1, 0].tolist() == [False, True, False]
    assert result.pair_swap_probability[1, 0, 1] > 0.99
    assert result.pair_swap_probability[1, 0, 2] < 0.01


def test_ground_truth_is_channel_evidence_but_its_coordinates_remain_anatomical(
    tmp_path: Path,
) -> None:
    bundle = make_bundle(tmp_path, frame_count=2)
    anatomical = bilateral_pose(frame_count=2)
    posterior = anatomical.copy()
    swap_joint_pair(posterior[1, 0], 15, 16)
    replace_bundle_posterior(bundle, posterior)
    state = cvat_annotation.annotation_state(bundle)
    state.points_xy[1, 0] = anatomical[1, 0]
    state.frame_ground_truth[1, 0] = True
    state.keypoint_ground_truth[1, 0] = True
    observations = human_conditioning.trusted_observations(bundle, state)
    tracker_observations = human_conditioning.select_tracker_observations(
        [item for item in observations if item.joint_id in (13, 14, 15, 16)],
        human_conditioning.HumanConditioningConfig(),
    )
    tracker_xy = np.asarray(
        [[item.xy for _ in bundle.frame_indices] for item in tracker_observations],
        dtype=np.float32,
    )

    identity = human_conditioning.infer_identity_assignments(
        run_dir=tmp_path,
        bundle=bundle,
        state=state,
        posterior=posterior,
        observations=observations,
        tracker_observations=tracker_observations,
        tracker_xy=tracker_xy,
        tracker_visibility=np.ones(tracker_xy.shape[:2], dtype=bool),
        bboxes_xyxy=np.tile(np.asarray([0, 0, 32, 32], dtype=np.float32), (2, 1, 1)),
        cfg=human_conditioning.HumanConditioningConfig(),
    )

    assert identity.pair_swapped[1, 0].tolist() == [False, False, True]
    assert state.points_xy[1, 0].tolist() == anatomical[1, 0].tolist()


def test_point_tracker_reanchors_at_each_promoted_frame_and_spans_video() -> None:
    observations = [
        human_conditioning.TrustedObservation(
            runner_id=3,
            joint_id=15,
            frame_idx=0,
            time_idx=0,
            xy=(0.0, 10.0),
            source="HUMAN_FRAME_GROUND_TRUTH",
            posterior_delta_px=0.0,
        ),
        human_conditioning.TrustedObservation(
            runner_id=3,
            joint_id=15,
            frame_idx=4,
            time_idx=4,
            xy=(40.0, 10.0),
            source="HUMAN_FRAME_GROUND_TRUTH",
            posterior_delta_px=0.0,
        ),
    ]
    tracker_xy = np.asarray(
        [
            [[0, 10], [1, 10], [2, 10], [3, 10], [4, 10]],
            [[36, 10], [37, 10], [38, 10], [39, 10], [40, 10]],
        ],
        dtype=np.float32,
    )

    centers = human_conditioning.smooth_identity_centers(
        runner_id=3,
        frame_count=5,
        observations=observations,
        tracker_observations=observations,
        tracker_xy=tracker_xy,
        tracker_visibility=np.ones((2, 5), dtype=bool),
        bboxes_xyxy=np.tile(np.asarray([0, 0, 32, 100], dtype=np.float32), (5, 1)),
        cfg=human_conditioning.HumanConditioningConfig(window_radius=1),
    )

    ankle = cvat_annotation.JOINT_IDS.index(15)
    assert np.isfinite(centers[:, ankle]).all()
    assert centers[0, ankle].tolist() == [0.0, 10.0]
    assert centers[4, ankle].tolist() == [40.0, 10.0]
    assert centers[1, ankle, 0] < centers[3, ankle, 0]


def test_constant_limb_projection_is_exact_and_preserves_human_ground_truth() -> None:
    points = np.zeros((2, len(cvat_annotation.JOINT_IDS), 2), dtype=np.float32)
    lower = np.asarray([[0, 0], [0, 10], [3, 0], [3, 10], [20, 0], [20, 10]], dtype=np.float32)
    lower_indices = [cvat_annotation.JOINT_IDS.index(joint) for joint in (11, 12, 13, 14, 15, 16)]
    points[:, lower_indices] = lower
    original_ground_truth = points[0].copy()
    trusted = np.zeros((2, len(cvat_annotation.JOINT_IDS)), dtype=bool)
    trusted[0] = True
    confidence = np.ones((2, len(cvat_annotation.JOINT_IDS)), dtype=np.float32)
    prior = human_conditioning.RunnerPrior(
        runner_id=3,
        bone_length_px={name: 8.0 for name in ("11_13", "13_15", "12_14", "14_16")},
    )

    changed = human_conditioning.enforce_constant_limb_lengths(
        points,
        trusted,
        confidence,
        prior,
        human_conditioning.HumanConditioningConfig(),
    )

    assert points[0].tolist() == original_ground_truth.tolist()
    assert not changed[0].any()
    for a, b in human_conditioning.BONES:
        a_idx = cvat_annotation.JOINT_IDS.index(a)
        b_idx = cvat_annotation.JOINT_IDS.index(b)
        assert np.linalg.norm(points[1, a_idx] - points[1, b_idx]) == pytest.approx(8.0, abs=0.06)


def test_limb_projection_moves_only_selected_corrected_joint() -> None:
    points = np.zeros((1, len(cvat_annotation.JOINT_IDS), 2), dtype=np.float32)
    knee = cvat_annotation.JOINT_IDS.index(13)
    ankle = cvat_annotation.JOINT_IDS.index(15)
    points[0, knee] = [0.0, 0.0]
    points[0, ankle] = [20.0, 0.0]
    before = points.copy()
    movable = np.zeros(points.shape[:-1], dtype=bool)
    movable[0, ankle] = True
    prior = human_conditioning.RunnerPrior(
        runner_id=3,
        bone_length_px={"13_15": 8.0},
    )

    changed = human_conditioning.enforce_constant_limb_lengths(
        points,
        np.zeros(points.shape[:-1], dtype=bool),
        np.ones(points.shape[:-1], dtype=np.float32),
        prior,
        human_conditioning.HumanConditioningConfig(),
        active_frames=np.asarray([True]),
        movable_points=movable,
    )

    assert np.array_equal(points[0, knee], before[0, knee])
    assert np.linalg.norm(points[0, knee] - points[0, ankle]) == pytest.approx(8.0, abs=0.06)
    assert changed[0, ankle]
    assert not changed[0, knee]


def make_bundle(root: Path, *, frame_count: int) -> cvat_annotation.CvatBundle:
    run = root
    (run / "posterior").mkdir(parents=True, exist_ok=True)
    joint_count = len(cvat_annotation.JOINT_IDS)
    points = np.zeros((frame_count, 1, joint_count, 2), dtype=np.float32)
    for local_idx in range(joint_count):
        points[:, 0, local_idx] = [10.0 + local_idx, 16.0]
    for lower_idx, joint_id in enumerate(human_conditioning.REFINEMENT_JOINT_IDS):
        points[:, 0, cvat_annotation.JOINT_IDS.index(joint_id)] = [10.0 + lower_idx, 16.0]
    preannotations = run / "preannotations.npz"
    np.savez_compressed(
        preannotations,
        frame_indices=np.arange(frame_count, dtype=np.int32),
        runner_ids=np.asarray([3], dtype=np.int32),
        joint_ids=np.asarray(cvat_annotation.JOINT_IDS, dtype=np.int32),
        posterior_keypoints_xy=points,
        posterior_peak_probability=np.ones((frame_count, 1, joint_count), dtype=np.float32),
        posterior_entropy=np.zeros((frame_count, 1, joint_count), dtype=np.float32),
    )
    return cvat_annotation.CvatBundle(
        run_dir=str(run),
        input_video=str(run / "source.mp4"),
        task_name="test",
        frame_indices=list(range(frame_count)),
        runner_ids=[3],
        joint_ids=list(cvat_annotation.JOINT_IDS),
        joint_names=list(cvat_annotation.JOINT_NAMES),
        preannotations_npz=str(preannotations),
        scope="runner_3",
        runner_id=3,
    )


def write_posterior_maps(
    root: Path, *, frame_count: int, runner_id: int, peak_xy: tuple[int, int]
) -> None:
    directory = root / "posterior" / "posteriors"
    directory.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    probability = np.exp(-0.5 * ((xx - peak_xy[0]) ** 2 + (yy - peak_xy[1]) ** 2) / 4.0)
    maps = np.repeat(probability[None], 17, axis=0).astype(np.float32)
    maps /= maps.sum(axis=(1, 2), keepdims=True)
    for frame_idx in range(frame_count):
        np.savez_compressed(
            directory / f"frame_{frame_idx:06d}_runner_{runner_id:04d}.npz",
            posterior=maps[11:17],
            joint_ids=np.arange(11, 17, dtype=np.int16),
            crop_xyxy=np.asarray([0, 0, 32, 32], dtype=np.float32),
        )


def bilateral_pose(*, frame_count: int) -> np.ndarray:
    lower = np.asarray(
        [[8.0, 8.0], [24.0, 8.0], [8.0, 16.0], [24.0, 16.0], [8.0, 24.0], [24.0, 24.0]],
        dtype=np.float32,
    )
    pose = np.zeros((len(cvat_annotation.JOINT_IDS), 2), dtype=np.float32)
    indices = [cvat_annotation.JOINT_IDS.index(joint) for joint in (11, 12, 13, 14, 15, 16)]
    pose[indices] = lower
    return np.tile(pose, (frame_count, 1, 1, 1))


def swap_joint_pair(pose: np.ndarray, left: int, right: int) -> None:
    left_idx = cvat_annotation.JOINT_IDS.index(left)
    right_idx = cvat_annotation.JOINT_IDS.index(right)
    pose[[left_idx, right_idx]] = pose[[right_idx, left_idx]]


def swap_bilateral_pairs(pose: np.ndarray) -> None:
    for left, right in human_conditioning.BILATERAL_PAIRS:
        swap_joint_pair(pose, left, right)


def replace_bundle_posterior(bundle: cvat_annotation.CvatBundle, posterior: np.ndarray) -> None:
    with np.load(bundle.preannotations_npz) as current:
        payload = {key: current[key] for key in current.files}
    payload["posterior_keypoints_xy"] = posterior
    np.savez_compressed(bundle.preannotations_npz, **payload)


def write_pose_maps(root: Path, posterior: np.ndarray, *, runner_id: int) -> None:
    directory = root / "posterior" / "posteriors"
    directory.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    for frame_idx in range(len(posterior)):
        maps = np.full((17, 32, 32), 1.0 / (32 * 32), dtype=np.float32)
        for local_idx, joint_id in enumerate(cvat_annotation.JOINT_IDS):
            if joint_id == cvat_annotation.NECK_JOINT_ID:
                continue
            point = posterior[frame_idx, 0, local_idx]
            probability = np.exp(-0.5 * ((xx - point[0]) ** 2 + (yy - point[1]) ** 2) / 4.0)
            maps[joint_id] = probability / probability.sum()
        np.savez_compressed(
            directory / f"frame_{frame_idx:06d}_runner_{runner_id:04d}.npz",
            posterior=maps[11:17],
            joint_ids=np.arange(11, 17, dtype=np.int16),
            crop_xyxy=np.asarray([0, 0, 32, 32], dtype=np.float32),
        )
