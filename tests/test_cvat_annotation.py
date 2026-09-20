from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dromia import dto as dromia_dto
from dromia.review import cvat as cvat_annotation
from dromia.review import propagation as annotation_propagation


def test_export_and_ground_truth_round_trip(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    bundle = cvat_annotation.export_run(run_dir)

    exported = np.load(bundle.preannotations_npz)
    assert exported["posterior_keypoints_xy"].shape == (
        2,
        1,
        len(cvat_annotation.JOINT_IDS),
        2,
    )
    assert exported["joint_ids"].tolist() == list(cvat_annotation.JOINT_IDS)
    neck = cvat_annotation.JOINT_IDS.index(cvat_annotation.NECK_JOINT_ID)
    np.testing.assert_allclose(exported["posterior_keypoints_xy"][0, 0, neck], [11.0, 16.5])

    ankle = cvat_annotation.JOINT_IDS.index(15)
    corrected = exported["posterior_keypoints_xy"][1, 0, ankle] + [4.0, 0.0]
    reviewed = [
        cvat_annotation.ReviewedFrame(
            runner_id=3,
            frame_idx=0,
            review_status="ACCEPTED",
            frame_ground_truth=True,
            points=[
                cvat_annotation.ReviewedPoint(joint_id=joint_id, xy=tuple(point), visibility=2)
                for joint_id, point in zip(
                    cvat_annotation.JOINT_IDS, exported["posterior_keypoints_xy"][0, 0], strict=True
                )
            ],
        ),
        cvat_annotation.ReviewedFrame(
            runner_id=3,
            frame_idx=1,
            review_status="CORRECTED",
            points=[
                cvat_annotation.ReviewedPoint(
                    joint_id=joint_id,
                    xy=tuple(corrected) if joint_id == 15 else tuple(point),
                    visibility=2,
                )
                for joint_id, point in zip(
                    cvat_annotation.JOINT_IDS, exported["posterior_keypoints_xy"][1, 0], strict=True
                )
            ],
        ),
    ]
    summary = cvat_annotation.write_ground_truth(run_dir, reviewed, task_id=12)

    ground_truth = np.load(summary.artifacts["ground_truth_npz"])
    assert summary.reviewed_frame_count == 2
    assert summary.corrected_keypoint_count == 1
    assert summary.ground_truth_frame_count == 1
    assert summary.ground_truth_keypoint_count == len(cvat_annotation.JOINT_IDS)
    assert ground_truth["frame_ground_truth"].tolist() == [[True], [False]]
    assert ground_truth["keypoint_ground_truth"][0, 0].all()
    assert np.isfinite(ground_truth["verified_ground_truth_keypoints_xy"][0, 0]).all()
    assert np.isnan(ground_truth["verified_ground_truth_keypoints_xy"][1, 0]).all()
    assert ground_truth["was_corrected"][1, 0, ankle]
    assert ground_truth["posterior_error_px"][1, 0, ankle] == 4.0
    coco = json.loads(Path(summary.artifacts["ground_truth_coco"]).read_text())
    assert coco["categories"][0]["keypoints"] == list(cvat_annotation.JOINT_NAMES)


def test_runner_export_is_temporally_cropped_and_remaps_review_frames(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    frame_indices = np.arange(6, dtype=np.int32)
    posterior_xy = np.full((6, 1, 17, 2), np.nan, dtype=np.float32)
    posterior_xy[2:5, 0] = 20.0
    np.savez_compressed(
        run / "posterior" / "posterior_pose.npz",
        frame_indices=frame_indices,
        object_ids=np.asarray([3], dtype=np.int32),
        posterior_keypoints_xy=posterior_xy,
        posterior_peak_probability=np.ones((6, 1, 17), dtype=np.float32),
        posterior_entropy=np.zeros((6, 1, 17), dtype=np.float32),
    )
    bboxes = np.full((6, 1, 4), np.nan, dtype=np.float32)
    bboxes[2:5, 0] = [10, 10, 90, 90]
    np.savez_compressed(
        run / "pose" / "pose_observations.npz",
        frame_indices=frame_indices,
        object_ids=np.asarray([3], dtype=np.int32),
        bboxes_xyxy=bboxes,
    )

    bundle = cvat_annotation.export_run(run, runner_id=3)
    exported = np.load(bundle.preannotations_npz)

    assert bundle.frame_indices == [2, 3, 4]
    assert bundle.task_frame_indices == [0, 1, 2]
    assert bundle.media_start_frame is None
    assert bundle.media_stop_frame is None
    assert Path(bundle.input_video).name == "runner_3.mp4"
    capture = cv2.VideoCapture(bundle.input_video)
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()
    assert exported["frame_indices"].tolist() == [2, 3, 4]
    reviewed = [
        cvat_annotation.ReviewedFrame(
            runner_id=3,
            frame_idx=1,
            review_status="CORRECTED",
            points=[],
        )
    ]
    remapped = cvat_annotation.remap_reviewed_frames_to_source(reviewed, bundle)
    assert remapped[0].frame_idx == 3


def test_skeleton_svg_names_every_joint() -> None:
    svg = cvat_annotation.skeleton_svg()
    assert cvat_annotation.JOINT_NAMES == (
        "neck",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    for name in cvat_annotation.JOINT_NAMES:
        assert f'data-label-name="{name}"' in svg
    assert svg.count("<circle") == len(cvat_annotation.JOINT_IDS)
    assert svg.count("<line") == len(cvat_annotation.EDGES)
    assert "nose" not in svg
    assert "left_elbow" not in svg


def test_task_label_exposes_frame_ground_truth_checkbox() -> None:
    models = pytest.importorskip("cvat_sdk.models")
    label = cvat_annotation.task_label(models)
    attributes = {item.name: item for item in label.attributes}

    checkbox = attributes[cvat_annotation.FRAME_GROUND_TRUTH_ATTRIBUTE]
    assert str(checkbox.input_type) == "checkbox"
    assert checkbox.mutable
    assert checkbox.default_value == "false"


def test_builds_cvat_skeleton_track_payload(tmp_path: Path) -> None:
    models = pytest.importorskip("cvat_sdk.models")
    bundle = cvat_annotation.export_run(make_run(tmp_path))
    label = models.PatchedLabelRequest(
        id=10,
        name="runner_lower_body",
        type="skeleton",
        attributes=[
            models.AttributeRequest(
                id=20,
                name="runner_id",
                mutable=False,
                input_type="text",
                values=[],
                default_value="",
            ),
            models.AttributeRequest(
                id=21,
                name="review_status",
                mutable=True,
                input_type="select",
                values=list(cvat_annotation.REVIEW_STATUSES),
                default_value="UNREVIEWED",
            ),
            models.AttributeRequest(
                id=22,
                name=cvat_annotation.FRAME_GROUND_TRUTH_ATTRIBUTE,
                mutable=True,
                input_type="checkbox",
                values=["true"],
                default_value="false",
            ),
        ],
        sublabels=[
            models.SublabelRequest(id=30 + idx, name=name, type="points")
            for idx, name in enumerate(cvat_annotation.JOINT_NAMES)
        ],
    )

    state = cvat_annotation.annotation_state(bundle)
    state.frame_ground_truth[1, 0] = True
    state.keypoint_ground_truth[1, 0] = True
    payload = cvat_annotation.build_annotations(models, [label], bundle, state)

    assert len(payload.tracks) == 1
    assert len(payload.tracks[0].shapes) == 2
    assert len(payload.tracks[0].elements) == len(cvat_annotation.JOINT_IDS)
    assert all(len(element.shapes) == 2 for element in payload.tracks[0].elements)
    parsed = cvat_annotation.parse_annotations(payload, [label])
    assert [(item.frame_idx, item.frame_ground_truth) for item in parsed] == [(0, False), (1, True)]


def test_build_annotations_preserves_existing_track_and_shape_ids(tmp_path: Path) -> None:
    models = pytest.importorskip("cvat_sdk.models")
    bundle = cvat_annotation.export_run(make_run(tmp_path))
    label = models.PatchedLabelRequest(
        id=10,
        name="runner_lower_body",
        type="skeleton",
        attributes=[
            models.AttributeRequest(
                id=20,
                name="runner_id",
                mutable=False,
                input_type="text",
                values=[],
                default_value="",
            ),
            models.AttributeRequest(
                id=21,
                name="review_status",
                mutable=True,
                input_type="select",
                values=list(cvat_annotation.REVIEW_STATUSES),
                default_value="UNREVIEWED",
            ),
            models.AttributeRequest(
                id=22,
                name=cvat_annotation.FRAME_GROUND_TRUTH_ATTRIBUTE,
                mutable=True,
                input_type="checkbox",
                values=["true"],
                default_value="false",
            ),
        ],
        sublabels=[
            models.SublabelRequest(id=30 + idx, name=name, type="points")
            for idx, name in enumerate(cvat_annotation.JOINT_NAMES)
        ],
    )
    existing = cvat_annotation.build_annotations(models, [label], bundle)
    existing.tracks[0].id = 100
    for shape_idx, shape in enumerate(existing.tracks[0].shapes):
        shape.id = 200 + shape_idx
    for element_idx, element in enumerate(existing.tracks[0].elements):
        element.id = 300 + element_idx
        for shape_idx, shape in enumerate(element.shapes):
            shape.id = 400 + 10 * element_idx + shape_idx

    updated = cvat_annotation.build_annotations(
        models,
        [label],
        bundle,
        existing_annotations=existing,
    )

    assert updated.tracks[0].id == 100
    assert [shape.id for shape in updated.tracks[0].shapes] == [200, 201]
    assert [element.id for element in updated.tracks[0].elements] == list(
        range(300, 300 + len(cvat_annotation.JOINT_IDS))
    )
    assert [shape.id for shape in updated.tracks[0].elements[0].shapes] == [400, 401]


def test_annotation_state_preserves_reviewed_points_and_status(tmp_path: Path) -> None:
    bundle = cvat_annotation.export_run(make_run(tmp_path))
    reviewed = [
        cvat_annotation.ReviewedFrame(
            runner_id=3,
            frame_idx=1,
            review_status="CORRECTED",
            frame_ground_truth=True,
            points=[cvat_annotation.ReviewedPoint(joint_id=15, xy=(41.0, 52.0), visibility=1)],
        )
    ]

    state = cvat_annotation.annotation_state(bundle, reviewed)

    assert state.review_status[1, 0] == "CORRECTED"
    ankle = cvat_annotation.JOINT_IDS.index(15)
    assert state.points_xy[1, 0, ankle].tolist() == [41.0, 52.0]
    assert state.visibility[1, 0, ankle] == 1
    assert state.frame_ground_truth[1, 0]
    assert state.keypoint_ground_truth[1, 0, ankle]


def test_annotation_state_merges_duplicate_tracks_without_losing_distinct_edits(
    tmp_path: Path,
) -> None:
    bundle = cvat_annotation.export_run(make_run(tmp_path))
    original = np.load(bundle.preannotations_npz)["posterior_keypoints_xy"][1, 0]
    first = [
        cvat_annotation.ReviewedPoint(
            joint_id=joint_id,
            xy=tuple(original[idx] + ([5.0, 0.0] if joint_id == 13 else [0.0, 0.0])),
            visibility=2,
        )
        for idx, joint_id in enumerate(cvat_annotation.JOINT_IDS)
    ]
    second = [
        cvat_annotation.ReviewedPoint(
            joint_id=joint_id,
            xy=tuple(original[idx] + ([0.0, 7.0] if joint_id == 15 else [0.0, 0.0])),
            visibility=2,
        )
        for idx, joint_id in enumerate(cvat_annotation.JOINT_IDS)
    ]
    reviewed = [
        cvat_annotation.ReviewedFrame(
            runner_id=3, frame_idx=1, review_status="UNREVIEWED", points=first
        ),
        cvat_annotation.ReviewedFrame(
            runner_id=3, frame_idx=1, review_status="UNREVIEWED", points=second
        ),
    ]

    state = cvat_annotation.annotation_state(bundle, reviewed)

    knee = cvat_annotation.JOINT_IDS.index(13)
    ankle = cvat_annotation.JOINT_IDS.index(15)
    assert state.points_xy[1, 0, knee].tolist() == (original[knee] + [5.0, 0.0]).tolist()
    assert state.points_xy[1, 0, ankle].tolist() == (original[ankle] + [0.0, 7.0]).tolist()


def test_correction_seeds_include_only_moved_corrected_lower_limbs(tmp_path: Path) -> None:
    bundle = cvat_annotation.export_run(make_run(tmp_path))
    state = cvat_annotation.annotation_state(bundle)
    state.review_status[1, 0] = "CORRECTED"
    state.points_xy[1, 0, cvat_annotation.JOINT_IDS.index(13)] += [3.0, 0.0]
    state.points_xy[1, 0, cvat_annotation.JOINT_IDS.index(11)] += [5.0, 0.0]

    seeds = annotation_propagation.correction_seeds(
        bundle,
        state,
        annotation_propagation.PropagationConfig(),
    )

    assert [(seed.frame_idx, seed.runner_id, seed.joint_id) for seed in seeds] == [(1, 3, 13)]


def test_propagation_updates_only_unreviewed_valid_points(tmp_path: Path) -> None:
    bundle = cvat_annotation.export_run(make_run(tmp_path))
    state = cvat_annotation.annotation_state(bundle)
    state.review_status[1, 0] = "CORRECTED"
    ankle = cvat_annotation.JOINT_IDS.index(15)
    state.points_xy[1, 0, ankle] = [30.0, 40.0]
    seed = annotation_propagation.CorrectionSeed(
        runner_id=3,
        joint_id=15,
        frame_idx=1,
        time_idx=1,
        xy=(30.0, 40.0),
        posterior_delta_px=10.0,
    )
    tracked_xy = np.asarray([[[30.0, 38.0], [30.0, 40.0]]], dtype=np.float32)
    visibility = np.zeros((1, 2), dtype=bool)
    mask = np.ones((100, 100), dtype=np.uint8)
    sam_frames = [
        dromia_dto.SamFrame(
            frame_idx=frame_idx,
            detections=[
                dromia_dto.SamDetection(
                    obj_id=3,
                    frame_idx=frame_idx,
                    label="Runner running",
                    score=1.0,
                    bbox_xyxy=np.asarray([0, 0, 100, 100], dtype=np.float32),
                    mask=mask,
                )
            ],
        )
        for frame_idx in (0, 1)
    ]
    bboxes = np.tile(np.asarray([0, 0, 100, 100], dtype=np.float32), (2, 1, 1))

    updated, applied, rejected, warnings, _ = annotation_propagation.apply_tracks(
        bundle=bundle,
        state=state,
        seeds=[seed],
        tracked_xy=tracked_xy,
        tracker_visibility=visibility,
        sam_frames=sam_frames,
        bboxes_xyxy=bboxes,
        cfg=annotation_propagation.PropagationConfig(max_bone_relative_error=10.0),
    )

    assert applied[0, 0, ankle]
    assert updated.points_xy[0, 0, ankle].tolist() == [30.0, 38.0]
    assert rejected == {}
    assert warnings == {"TRACKER_NOT_VISIBLE": 1}
    assert not applied[1, 0, ankle]
    assert updated.points_xy[1, 0, ankle].tolist() == [30.0, 40.0]


def make_run(root: Path) -> Path:
    run = root / "runs" / "20260101_000000_tiny"
    (run / "posterior").mkdir(parents=True)
    (run / "pose").mkdir(parents=True)
    posterior_xy = np.zeros((2, 1, 17, 2), dtype=np.float32)
    for joint_id in cvat_annotation.MODEL_JOINT_IDS:
        posterior_xy[:, 0, joint_id] = [joint_id * 2, joint_id * 3]
    posterior_path = run / "posterior" / "posterior_pose.npz"
    np.savez_compressed(
        posterior_path,
        frame_indices=np.asarray([0, 1], dtype=np.int32),
        object_ids=np.asarray([3], dtype=np.int32),
        posterior_keypoints_xy=posterior_xy,
        posterior_peak_probability=np.ones((2, 1, 17), dtype=np.float32),
        posterior_entropy=np.zeros((2, 1, 17), dtype=np.float32),
    )
    pose_path = run / "pose" / "pose_observations.npz"
    np.savez_compressed(
        pose_path,
        frame_indices=np.asarray([0, 1], dtype=np.int32),
        object_ids=np.asarray([3], dtype=np.int32),
        bboxes_xyxy=np.asarray([[[0, 0, 100, 100]], [[0, 0, 100, 100]]], dtype=np.float32),
    )
    video = run / "input" / "tiny.mp4"
    video.parent.mkdir()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24))
    for value in range(10):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()
    manifest = dromia_dto.RunManifestV1(
        run_id=run.name,
        source_video=dromia_dto.SourceVideo(
            name=video.name,
            sha256="0" * 64,
            path="input/tiny.mp4",
        ),
        frame_count=2,
        accepted_runner_ids=[3],
        runner_decisions=[],
        config_fingerprint="1" * 64,
        model_fingerprints={},
        timing={},
        artifacts={
            "posterior_npz": "posterior/posterior_pose.npz",
            "pose_npz": "pose/pose_observations.npz",
        },
    )
    (run / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json")))
    return run
