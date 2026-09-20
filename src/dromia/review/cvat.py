"""CVAT round trip for DromIA posterior pose annotations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from dromia import dto as dromia_dto

# The pose model has no native neck output. ID 17 is an DromIA-only landmark
# derived as the midpoint of the two COCO shoulders (5 and 6).
NECK_JOINT_ID = 17
JOINT_IDS = (NECK_JOINT_ID, 5, 6, 11, 12, 13, 14, 15, 16)
MODEL_JOINT_IDS = tuple(joint_id for joint_id in JOINT_IDS if joint_id != NECK_JOINT_ID)
JOINT_NAMES = (
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
JOINT_COLORS = (
    "#ffd166",
    "#06d6a0",
    "#118ab2",
    "#ff9f1c",
    "#2ec4b6",
    "#ffbf69",
    "#3a86ff",
    "#ef476f",
    "#8338ec",
)
EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 4),
    (3, 5),
    (5, 7),
    (4, 6),
    (6, 8),
)
REVIEW_STATUSES = (
    "UNREVIEWED",
    "ACCEPTED",
    "CORRECTED",
    "AMBIGUOUS",
    "NOT_VISIBLE",
    "OUT_OF_FRAME",
)
FRAME_GROUND_TRUTH_ATTRIBUTE = "frame_ground_truth"
SCHEMA_VERSION = 5


class CvatConnection(BaseModel):
    host: str = "http://localhost:8080"
    username: str
    password: str = Field(repr=False)

    @classmethod
    def from_environment(cls) -> CvatConnection:
        username = os.getenv("CVAT_USERNAME")
        password = os.getenv("CVAT_PASSWORD")
        if not username or not password:
            raise RuntimeError("Set CVAT_USERNAME and CVAT_PASSWORD before connecting to CVAT")
        return cls(
            host=os.getenv("CVAT_HOST", "http://localhost:8080"),
            username=username,
            password=password,
        )


class CvatBundle(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_dir: str
    input_video: str
    task_name: str
    frame_indices: list[int]
    runner_ids: list[int]
    joint_ids: list[int]
    joint_names: list[str]
    preannotations_npz: str
    scope: str = "full_runner"
    runner_id: int | None = None
    # DromIA keeps source-frame identity in ``frame_indices`` while CVAT uses
    # task-local frame numbers for temporally cropped runner tasks.
    task_frame_indices: list[int] = Field(default_factory=list)
    media_start_frame: int | None = None
    media_stop_frame: int | None = None


class CvatTaskRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    task_id: int
    task_url: str
    host: str
    run_dir: str
    input_video: str
    project_id: int | None = None
    scope: str = "full_runner"
    runner_id: int | None = None
    bundle_path: str | None = None


class CvatTaskRegistry(BaseModel):
    schema_version: int = SCHEMA_VERSION
    project_id: int
    project_url: str
    run_dir: str
    full_runner: CvatTaskRecord
    runners: dict[str, CvatTaskRecord]


def registry_records(registry: CvatTaskRegistry) -> list[CvatTaskRecord]:
    return [registry.full_runner, *registry.runners.values()]


class ReviewedPoint(BaseModel):
    joint_id: int
    xy: tuple[float, float] | None
    visibility: int


class ReviewedFrame(BaseModel):
    runner_id: int
    frame_idx: int
    review_status: str
    frame_ground_truth: bool = False
    points: list[ReviewedPoint]


class ReviewSummary(BaseModel):
    schema_version: int = SCHEMA_VERSION
    task_id: int | None
    scope: str = "full_runner"
    runner_id: int | None = None
    runner_ids: list[int]
    frame_count: int
    reviewed_frame_count: int
    unreviewed_frame_count: int
    corrected_keypoint_count: int
    missing_keypoint_count: int
    ground_truth_frame_count: int = 0
    ground_truth_keypoint_count: int = 0
    status_counts: dict[str, int]
    artifacts: dict[str, str]


class SchemaUpgradeSummary(BaseModel):
    task_id: int
    label_id: int
    attribute: str = FRAME_GROUND_TRUTH_ATTRIBUTE
    added: bool


@dataclass(slots=True)
class AnnotationState:
    points_xy: np.ndarray
    visibility: np.ndarray
    review_status: np.ndarray
    frame_ground_truth: np.ndarray
    keypoint_ground_truth: np.ndarray


def annotation_pose_arrays(
    posterior: Any,
    selected_runners: np.ndarray,
    *,
    preferred_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return model landmarks plus DromIA's derived shoulder-midpoint neck."""

    source_xy = posterior["posterior_keypoints_xy"] if preferred_xy is None else preferred_xy
    full_xy = np.asarray(source_xy[:, selected_runners])
    full_peak = np.asarray(posterior["posterior_peak_probability"][:, selected_runners])
    full_entropy = np.asarray(posterior["posterior_entropy"][:, selected_runners])
    left_shoulder = full_xy[:, :, 5]
    right_shoulder = full_xy[:, :, 6]
    shoulders_finite = np.isfinite(left_shoulder).all(axis=-1) & np.isfinite(right_shoulder).all(
        axis=-1
    )
    neck_xy = np.where(
        shoulders_finite[..., None],
        0.5 * (left_shoulder + right_shoulder),
        np.nan,
    )
    neck_peak = np.minimum(full_peak[:, :, 5], full_peak[:, :, 6])
    neck_entropy = np.maximum(full_entropy[:, :, 5], full_entropy[:, :, 6])
    neck_peak = np.where(shoulders_finite, neck_peak, 0.0)
    neck_entropy = np.where(shoulders_finite, neck_entropy, np.nan)

    keypoints = np.stack(
        [
            neck_xy if joint_id == NECK_JOINT_ID else full_xy[:, :, joint_id]
            for joint_id in JOINT_IDS
        ],
        axis=2,
    )
    peak = np.stack(
        [
            neck_peak if joint_id == NECK_JOINT_ID else full_peak[:, :, joint_id]
            for joint_id in JOINT_IDS
        ],
        axis=2,
    )
    entropy = np.stack(
        [
            neck_entropy if joint_id == NECK_JOINT_ID else full_entropy[:, :, joint_id]
            for joint_id in JOINT_IDS
        ],
        axis=2,
    )
    return keypoints, peak, entropy


def export_run(
    run_dir: Path,
    runner_id: int | None = None,
    *,
    annotation_root: Path | None = None,
    preserve_frame_ground_truth: bool = True,
    task_name_suffix: str | None = None,
    crop_runner_media: bool = True,
) -> CvatBundle:
    run = run_dir.expanduser().resolve()
    manifest = load_manifest(run)
    posterior_path = Path(required_artifact(manifest, "posterior_npz"))
    posterior = np.load(posterior_path)
    frame_indices = posterior["frame_indices"].astype(np.int32)
    all_runner_ids = posterior["object_ids"].astype(np.int32)
    if runner_id is None:
        selected = np.arange(len(all_runner_ids))
    else:
        matches = np.where(all_runner_ids == runner_id)[0]
        if not matches.size:
            raise ValueError(f"Runner {runner_id} is not present in this DromIA run")
        selected = matches
    runner_ids = all_runner_ids[selected]
    preferred_xy = None
    first_pass_path = manifest.artifacts.get("first_pass_pose_npz")
    if first_pass_path and Path(first_pass_path).is_file():
        first_pass = np.load(Path(first_pass_path))
        preferred_xy = np.asarray(first_pass["first_pass_keypoints_xy"], dtype=np.float32)
    keypoints, peak, entropy = annotation_pose_arrays(
        posterior, selected, preferred_xy=preferred_xy
    )
    media_start_frame = None
    media_stop_frame = None
    task_frame_indices = frame_indices.copy()
    scope = "full_runner" if runner_id is None else f"runner_{runner_id}"
    root = annotation_root or run / "annotations" / "cvat"
    annotation_dir = root / scope
    annotation_dir.mkdir(parents=True, exist_ok=True)
    input_video = manifest.input_video
    if runner_id is not None and crop_runner_media:
        pose_path = Path(required_artifact(manifest, "pose_npz"))
        with np.load(pose_path) as pose:
            bboxes = select_pose_bboxes(run, pose, frame_indices, runner_ids)[:, 0]
        visible = (
            np.isfinite(bboxes).all(axis=1)
            & ((bboxes[:, 2] - bboxes[:, 0]) > 2)
            & ((bboxes[:, 3] - bboxes[:, 1]) > 2)
        )
        appearances = np.flatnonzero(visible)
        if not appearances.size:
            raise ValueError(f"Runner {runner_id} has no valid appearance frames")
        start = int(appearances[0])
        stop = int(appearances[-1]) + 1
        frame_indices = frame_indices[start:stop]
        keypoints = keypoints[start:stop]
        peak = peak[start:stop]
        entropy = entropy[start:stop]
        task_frame_indices = np.arange(len(frame_indices), dtype=np.int32)
        input_video = str(
            write_runner_media_clip(
                Path(manifest.input_video),
                annotation_dir / f"runner_{runner_id}.mp4",
                frame_indices,
            )
        )
    initial_frame_ground_truth = (
        load_existing_frame_ground_truth(run, frame_indices, runner_ids)
        if preserve_frame_ground_truth
        else np.zeros((len(frame_indices), len(runner_ids)), dtype=bool)
    )

    preannotations_path = annotation_dir / "preannotations.npz"
    np.savez_compressed(
        preannotations_path,
        frame_indices=frame_indices,
        runner_ids=runner_ids,
        joint_ids=np.asarray(JOINT_IDS, dtype=np.int32),
        joint_names=np.asarray(JOINT_NAMES, dtype="<U16"),
        posterior_keypoints_xy=keypoints.astype(np.float32),
        posterior_peak_probability=peak.astype(np.float32),
        posterior_entropy=entropy.astype(np.float32),
        initial_frame_ground_truth=initial_frame_ground_truth,
    )
    bundle = CvatBundle(
        run_dir=str(run),
        input_video=input_video,
        task_name=" | ".join(
            part
            for part in (
                Path(manifest.input_video).stem,
                scope,
                run.name.split("_")[0],
                task_name_suffix,
            )
            if part
        ),
        frame_indices=frame_indices.tolist(),
        runner_ids=runner_ids.tolist(),
        joint_ids=list(JOINT_IDS),
        joint_names=list(JOINT_NAMES),
        preannotations_npz=str(preannotations_path.resolve()),
        scope=scope,
        runner_id=runner_id,
        task_frame_indices=task_frame_indices.tolist(),
        media_start_frame=media_start_frame,
        media_stop_frame=media_stop_frame,
    )
    bundle_path = annotation_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8")
    write_annotation_readme(annotation_dir)
    return bundle


def write_runner_media_clip(
    source_video: Path,
    output_video: Path,
    frame_indices: np.ndarray,
) -> Path:
    """Write only the runner's visible source frames for its CVAT task."""

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"Could not open runner source video: {source_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise ValueError(f"Invalid runner source video metadata: {source_video}")
    temporary = output_video.with_suffix(".mp4.tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not create runner media clip: {output_video}")
    try:
        for frame_idx in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not decode source frame {int(frame_idx)}")
            writer.write(frame)
    finally:
        writer.release()
        capture.release()
    temporary.replace(output_video)
    return output_video.resolve()


def push_run(run_dir: Path, connection: CvatConnection) -> CvatTaskRegistry:
    run = run_dir.expanduser().resolve()
    full_bundle = export_run(run)
    try:
        from cvat_sdk import make_client, models
    except ImportError as exc:
        raise RuntimeError("Install the annotation extra with: uv sync --extra annotation") from exc

    with make_client(host=connection.host) as client:
        client.login((connection.username, connection.password))
        project = client.projects.create(
            models.ProjectWriteRequest(
                name=f"DromIA {Path(full_bundle.input_video).stem} {run.name.split('_')[0]}",
                labels=[task_label(models)],
            )
        )
        full_record = create_scoped_task(client, models, full_bundle, connection, int(project.id))
        runner_records = {
            str(runner_id): create_scoped_task(
                client,
                models,
                export_run(run, runner_id),
                connection,
                int(project.id),
            )
            for runner_id in full_bundle.runner_ids
        }
        registry = CvatTaskRegistry(
            project_id=int(project.id),
            project_url=f"{connection.host.rstrip('/')}/projects/{project.id}",
            run_dir=str(run),
            full_runner=full_record,
            runners=runner_records,
        )
    cvat_dir = run / "annotations" / "cvat"
    (cvat_dir / "tasks.json").write_text(
        json.dumps(registry.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    (cvat_dir / "task.json").write_text(
        json.dumps(full_record.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return registry


def create_scoped_task(
    client: Any,
    models: Any,
    bundle: CvatBundle,
    connection: CvatConnection,
    project_id: int,
) -> CvatTaskRecord:
    data_params: dict[str, Any] = {"image_quality": 100}
    if bundle.media_start_frame is not None:
        data_params["start_frame"] = bundle.media_start_frame
    if bundle.media_stop_frame is not None:
        data_params["stop_frame"] = bundle.media_stop_frame
    task = client.tasks.create_from_data(
        spec=models.TaskWriteRequest(name=bundle.task_name, project_id=project_id),
        resources=[bundle.input_video],
        data_params=data_params,
    )
    task.set_annotations(build_annotations(models, task.get_labels(), bundle))
    record = CvatTaskRecord(
        task_id=int(task.id),
        task_url=f"{connection.host.rstrip('/')}/tasks/{task.id}",
        host=connection.host,
        run_dir=bundle.run_dir,
        input_video=bundle.input_video,
        project_id=project_id,
        scope=bundle.scope,
        runner_id=bundle.runner_id,
        bundle_path=str((Path(bundle.preannotations_npz).parent / "bundle.json").resolve()),
    )
    path = Path(bundle.preannotations_npz).parent / "task.json"
    path.write_text(json.dumps(record.model_dump(mode="json"), indent=2), encoding="utf-8")
    return record


def pull_run(
    run_dir: Path,
    connection: CvatConnection,
    task_id: int | None = None,
) -> ReviewSummary:
    run = run_dir.expanduser().resolve()
    record = find_task_record(run, task_id)
    bundle = bundle_for_record(run, record)
    try:
        from cvat_sdk import make_client
    except ImportError as exc:
        raise RuntimeError("Install the annotation extra with: uv sync --extra annotation") from exc

    with make_client(host=connection.host) as client:
        client.login((connection.username, connection.password))
        task = client.tasks.retrieve(record.task_id)
        reviewed = remap_reviewed_frames_to_source(
            parse_annotations(task.get_annotations(), task.get_labels()), bundle
        )
    return write_ground_truth(run, reviewed, task_id=record.task_id, bundle=bundle)


def remap_reviewed_frames_to_source(
    reviewed: list[ReviewedFrame], bundle: CvatBundle
) -> list[ReviewedFrame]:
    """Translate task-local runner-clip frames back to source-video frames."""

    task_frames = bundle.task_frame_indices
    if not task_frames:
        return reviewed
    if len(task_frames) != len(bundle.frame_indices):
        raise ValueError("CVAT task-frame mapping does not align with source frames")
    mapping = dict(zip(task_frames, bundle.frame_indices, strict=True))
    output: list[ReviewedFrame] = []
    for frame in reviewed:
        source_frame = mapping.get(frame.frame_idx)
        if source_frame is None:
            continue
        output.append(frame.model_copy(update={"frame_idx": int(source_frame)}))
    return output


def find_task_record(run: Path, task_id: int | None = None) -> CvatTaskRecord:
    cvat_dir = run / "annotations" / "cvat"
    registry_path = cvat_dir / "tasks.json"
    if registry_path.exists():
        registry = CvatTaskRegistry.model_validate_json(registry_path.read_text())
        records = registry_records(registry)
        if task_id is None:
            return registry.full_runner
        for record in records:
            if record.task_id == task_id:
                return record
    raise ValueError(f"CVAT task {task_id} is not registered for {run}")


def bundle_for_record(run: Path, record: CvatTaskRecord) -> CvatBundle:
    if record.bundle_path and Path(record.bundle_path).exists():
        return CvatBundle.model_validate_json(Path(record.bundle_path).read_text())
    return export_run(run, record.runner_id)


def task_label(models: Any) -> Any:
    attributes = [
        models.AttributeRequest(
            name="runner_id",
            mutable=False,
            input_type="text",
            values=[],
            default_value="",
        ),
        models.AttributeRequest(
            name="review_status",
            mutable=True,
            input_type="select",
            values=list(REVIEW_STATUSES),
            default_value="UNREVIEWED",
        ),
        frame_ground_truth_attribute(models),
    ]
    sublabels = [
        models.SublabelRequest(name=name, color=color, type="points")
        for name, color in zip(JOINT_NAMES, JOINT_COLORS, strict=True)
    ]
    return models.PatchedLabelRequest(
        name="runner_lower_body",
        color="#22a06b",
        type="skeleton",
        attributes=attributes,
        sublabels=sublabels,
        svg=skeleton_svg(),
    )


def frame_ground_truth_attribute(models: Any) -> Any:
    return models.AttributeRequest(
        name=FRAME_GROUND_TRUTH_ATTRIBUTE,
        mutable=True,
        input_type="checkbox",
        values=["true"],
        default_value="false",
    )


def ensure_frame_ground_truth_attribute(
    run_dir: Path,
    connection: CvatConnection,
    task_id: int | None = None,
) -> SchemaUpgradeSummary:
    """Add the checkbox to an existing DromIA CVAT project without recreating tasks."""

    run = run_dir.expanduser().resolve()
    record = find_task_record(run, task_id)
    try:
        from cvat_sdk import make_client, models
    except ImportError as exc:
        raise RuntimeError("Install the annotation extra with: uv sync --extra annotation") from exc
    with make_client(host=connection.host) as client:
        client.login((connection.username, connection.password))
        task = client.tasks.retrieve(record.task_id)
        label = next(item for item in task.get_labels() if item.name == "runner_lower_body")
        if any(item.name == FRAME_GROUND_TRUTH_ATTRIBUTE for item in label.attributes):
            return SchemaUpgradeSummary(task_id=record.task_id, label_id=int(label.id), added=False)
        client.api_client.labels_api.partial_update(
            int(label.id),
            patched_label_request=models.PatchedLabelRequest(
                attributes=[frame_ground_truth_attribute(models)]
            ),
        )
        return SchemaUpgradeSummary(task_id=record.task_id, label_id=int(label.id), added=True)


def skeleton_svg() -> str:
    positions = (
        (50, 12),
        (30, 28),
        (70, 28),
        (32, 52),
        (68, 52),
        (32, 74),
        (68, 74),
        (32, 96),
        (68, 96),
    )
    lines = [
        f'<line x1="{positions[a][0]}" y1="{positions[a][1]}" x2="{positions[b][0]}" '
        f'y2="{positions[b][1]}" stroke="black" data-type="edge" data-node-from="{a + 1}" '
        f'stroke-width="0.5" data-node-to="{b + 1}"></line>'
        for a, b in EDGES
    ]
    circles = [
        f'<circle r="1.5" stroke="black" fill="{color}" cx="{x}" cy="{y}" stroke-width="0.1" '
        f'data-type="element node" data-element-id="{idx}" data-node-id="{idx}" '
        f'data-label-name="{name}"></circle>'
        for idx, (name, color, (x, y)) in enumerate(
            zip(JOINT_NAMES, JOINT_COLORS, positions, strict=True), start=1
        )
    ]
    return "".join([*lines, *circles])


def build_annotations(
    models: Any,
    labels: list[Any],
    bundle: CvatBundle,
    state: AnnotationState | None = None,
    existing_annotations: Any | None = None,
) -> Any:
    label = next(item for item in labels if item.name == "runner_lower_body")
    sublabels = {item.name: item.id for item in label.sublabels}
    attributes = {item.name: item.id for item in label.attributes}
    data = np.load(bundle.preannotations_npz)
    current = state or annotation_state(bundle)
    points = current.points_xy
    existing_ids = existing_annotation_id_index(
        existing_annotations,
        label_id=int(label.id),
        runner_attribute_id=int(attributes["runner_id"]),
        ground_truth_attribute_id=(
            int(attributes[FRAME_GROUND_TRUTH_ATTRIBUTE])
            if FRAME_GROUND_TRUTH_ATTRIBUTE in attributes
            else None
        ),
    )
    tracks = []
    task_frames = bundle.task_frame_indices or data["frame_indices"].astype(int).tolist()
    if len(task_frames) != len(data["frame_indices"]):
        raise ValueError("CVAT task-frame mapping does not align with preannotations")
    for obj_idx, runner_id in enumerate(data["runner_ids"].tolist()):
        ids = existing_ids.get(int(runner_id), {})
        parent_shape_ids = ids.get("parent_shapes", {})
        element_ids = ids.get("elements", {})
        element_shape_ids = ids.get("element_shapes", {})
        parent_shapes = []
        elements = []
        for t, frame_idx in enumerate(task_frames):
            visible = bool(np.any(current.visibility[t, obj_idx] > 0))
            shape_attributes = [
                models.AttributeValRequest(
                    spec_id=attributes["review_status"],
                    value=str(current.review_status[t, obj_idx]),
                )
            ]
            if FRAME_GROUND_TRUTH_ATTRIBUTE in attributes:
                shape_attributes.append(
                    models.AttributeValRequest(
                        spec_id=attributes[FRAME_GROUND_TRUTH_ATTRIBUTE],
                        value=checkbox_value(current.frame_ground_truth[t, obj_idx]),
                    )
                )
            parent_shapes.append(
                models.TrackedShapeRequest(
                    type="skeleton",
                    frame=int(frame_idx),
                    points=[],
                    outside=not visible,
                    occluded=False,
                    attributes=shape_attributes,
                    id=parent_shape_ids.get(int(frame_idx)),
                )
            )
        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            shapes = []
            for t, frame_idx in enumerate(task_frames):
                point = points[t, obj_idx, joint_idx]
                point_visibility = int(current.visibility[t, obj_idx, joint_idx])
                finite = bool(np.isfinite(point).all()) and point_visibility > 0
                shapes.append(
                    models.TrackedShapeRequest(
                        type="points",
                        frame=int(frame_idx),
                        points=point.astype(float).tolist() if finite else [0.0, 0.0],
                        outside=not finite,
                        occluded=point_visibility == 1,
                        attributes=[],
                        id=element_shape_ids.get((int(sublabels[joint_name]), int(frame_idx))),
                    )
                )
            elements.append(
                models.SubLabeledTrackRequest(
                    label_id=sublabels[joint_name],
                    frame=int(task_frames[0]),
                    source="auto",
                    shapes=shapes,
                    attributes=[],
                    id=element_ids.get(int(sublabels[joint_name])),
                )
            )
        tracks.append(
            models.LabeledTrackRequest(
                label_id=label.id,
                frame=int(task_frames[0]),
                source="auto",
                shapes=parent_shapes,
                attributes=[
                    models.AttributeValRequest(
                        spec_id=attributes["runner_id"], value=str(runner_id)
                    )
                ],
                elements=elements,
                id=ids.get("track"),
            )
        )
    return models.LabeledDataRequest(version=0, tags=[], shapes=[], tracks=tracks)


def existing_annotation_id_index(
    annotations: Any | None,
    *,
    label_id: int,
    runner_attribute_id: int,
    ground_truth_attribute_id: int | None,
) -> dict[int, dict[str, Any]]:
    """Index IDs from existing DromIA tracks so synchronization updates in place."""

    if annotations is None:
        return {}
    candidates: dict[int, list[Any]] = {}
    for track in annotations.tracks:
        if int(track.label_id) != label_id:
            continue
        runner_value = next(
            (
                attribute.value
                for attribute in track.attributes
                if int(attribute.spec_id) == runner_attribute_id
            ),
            None,
        )
        try:
            runner_id = int(runner_value)
        except (TypeError, ValueError):
            continue
        candidates.setdefault(runner_id, []).append(track)

    output: dict[int, dict[str, Any]] = {}
    for runner_id, tracks in candidates.items():
        # If retries produced duplicates, retain the track with the most explicit
        # ground-truth frames and then the most complete shape history.
        track = max(
            tracks,
            key=lambda item: (
                ground_truth_shape_count(item, ground_truth_attribute_id),
                len(item.shapes),
                -int(item.id or 0),
            ),
        )
        output[runner_id] = {
            "track": track.id,
            "parent_shapes": {int(shape.frame): shape.id for shape in track.shapes},
            "elements": {int(element.label_id): element.id for element in track.elements},
            "element_shapes": {
                (int(element.label_id), int(shape.frame)): shape.id
                for element in track.elements
                for shape in element.shapes
            },
        }
    return output


def ground_truth_shape_count(track: Any, attribute_id: int | None) -> int:
    if attribute_id is None:
        return 0
    return sum(
        parse_checkbox(attribute.value)
        for shape in track.shapes
        for attribute in shape.attributes
        if int(attribute.spec_id) == attribute_id
    )


def annotation_state(
    bundle: CvatBundle, reviewed: list[ReviewedFrame] | None = None
) -> AnnotationState:
    data = np.load(bundle.preannotations_npz)
    points = data["posterior_keypoints_xy"].astype(np.float32).copy()
    visibility = (np.isfinite(points).all(axis=-1) * 2).astype(np.uint8)
    status = np.full(points.shape[:2], "UNREVIEWED", dtype="<U16")
    frame_ground_truth = initial_frame_ground_truth_array(data, points.shape[:2])
    keypoint_ground_truth = (
        frame_ground_truth[:, :, None] & (visibility > 0) & np.isfinite(points).all(axis=-1)
    )
    if reviewed is None:
        return AnnotationState(
            points_xy=points,
            visibility=visibility,
            review_status=status,
            frame_ground_truth=frame_ground_truth,
            keypoint_ground_truth=keypoint_ground_truth,
        )
    frame_to_t = {int(value): idx for idx, value in enumerate(data["frame_indices"])}
    runner_to_idx = {int(value): idx for idx, value in enumerate(data["runner_ids"])}
    joint_to_idx = {joint_id: idx for idx, joint_id in enumerate(JOINT_IDS)}
    frame_seen = np.zeros(points.shape[:2], dtype=bool)
    ground_truth_seen = np.zeros(points.shape[:2], dtype=bool)
    point_seen = np.zeros(points.shape[:3], dtype=bool)
    for frame in reviewed:
        if frame.frame_idx not in frame_to_t or frame.runner_id not in runner_to_idx:
            continue
        t = frame_to_t[frame.frame_idx]
        obj_idx = runner_to_idx[frame.runner_id]
        previous_status = str(status[t, obj_idx])
        if not frame_seen[t, obj_idx] or review_priority(frame.review_status) >= review_priority(
            previous_status
        ):
            status[t, obj_idx] = frame.review_status
        frame_seen[t, obj_idx] = True
        if not ground_truth_seen[t, obj_idx]:
            frame_ground_truth[t, obj_idx] = frame.frame_ground_truth
        else:
            frame_ground_truth[t, obj_idx] |= frame.frame_ground_truth
        ground_truth_seen[t, obj_idx] = True
        for point in frame.points:
            if point.joint_id not in joint_to_idx:
                continue
            joint_idx = joint_to_idx[point.joint_id]
            candidate = (
                np.asarray(point.xy, dtype=np.float32)
                if point.xy is not None
                else np.full(2, np.nan)
            )
            should_replace = not point_seen[t, obj_idx, joint_idx]
            if point_seen[t, obj_idx, joint_idx]:
                original = data["posterior_keypoints_xy"][t, obj_idx, joint_idx]
                current_delta = point_delta(points[t, obj_idx, joint_idx], original)
                candidate_delta = point_delta(candidate, original)
                incoming_priority = review_priority(frame.review_status)
                previous_priority = review_priority(previous_status)
                should_replace = incoming_priority > previous_priority or (
                    incoming_priority == previous_priority
                    and candidate_delta > current_delta + 0.25
                )
            if should_replace:
                visibility[t, obj_idx, joint_idx] = point.visibility
                points[t, obj_idx, joint_idx] = candidate
            point_seen[t, obj_idx, joint_idx] = True
    keypoint_ground_truth = (
        frame_ground_truth[:, :, None] & (visibility > 0) & np.isfinite(points).all(axis=-1)
    )
    return AnnotationState(
        points_xy=points,
        visibility=visibility,
        review_status=status,
        frame_ground_truth=frame_ground_truth,
        keypoint_ground_truth=keypoint_ground_truth,
    )


def review_priority(status: str) -> int:
    return {
        "UNREVIEWED": 0,
        "ACCEPTED": 1,
        "CORRECTED": 2,
        "AMBIGUOUS": 3,
        "NOT_VISIBLE": 4,
        "OUT_OF_FRAME": 4,
    }.get(status, 0)


def point_delta(point: np.ndarray, original: np.ndarray) -> float:
    if not np.isfinite(point).all() or not np.isfinite(original).all():
        return float("inf") if not np.isfinite(point).all() else 0.0
    return float(np.linalg.norm(point - original))


def checkbox_value(value: object) -> str:
    return "true" if bool(value) else "false"


def parse_checkbox(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def initial_frame_ground_truth_array(data: object, shape: tuple[int, int]) -> np.ndarray:
    if "initial_frame_ground_truth" not in data:
        return np.zeros(shape, dtype=bool)
    values = np.asarray(data["initial_frame_ground_truth"], dtype=bool)
    return values.copy() if values.shape == shape else np.zeros(shape, dtype=bool)


def load_existing_frame_ground_truth(
    run: Path,
    frame_indices: np.ndarray,
    runner_ids: np.ndarray,
) -> np.ndarray:
    """Restore explicitly verified frames when rebuilding a CVAT bundle."""

    output = np.zeros((len(frame_indices), len(runner_ids)), dtype=bool)
    candidates = [run / "annotations" / "ground_truth_pose.npz"]
    candidates.extend(
        run / "annotations" / f"ground_truth_pose_runner_{int(runner_id)}.npz"
        for runner_id in runner_ids
    )
    frame_lookup = {int(value): idx for idx, value in enumerate(frame_indices)}
    runner_lookup = {int(value): idx for idx, value in enumerate(runner_ids)}
    for path in candidates:
        if not path.is_file():
            continue
        with np.load(path) as saved:
            if (
                "frame_ground_truth" not in saved
                or "frame_indices" not in saved
                or "object_ids" not in saved
            ):
                continue
            saved_flags = np.asarray(saved["frame_ground_truth"], dtype=bool)
            for saved_t, frame_idx in enumerate(np.asarray(saved["frame_indices"]).tolist()):
                for saved_obj, runner_id in enumerate(np.asarray(saved["object_ids"]).tolist()):
                    target_t = frame_lookup.get(int(frame_idx))
                    target_obj = runner_lookup.get(int(runner_id))
                    in_bounds = saved_t < saved_flags.shape[0] and saved_obj < saved_flags.shape[1]
                    if target_t is not None and target_obj is not None and in_bounds:
                        output[target_t, target_obj] = bool(saved_flags[saved_t, saved_obj])
    return output


def parse_annotations(annotations: Any, labels: list[Any]) -> list[ReviewedFrame]:
    label = next(item for item in labels if item.name == "runner_lower_body")
    sublabel_names = {item.id: item.name for item in label.sublabels}
    joint_ids = {name: joint_id for name, joint_id in zip(JOINT_NAMES, JOINT_IDS, strict=True)}
    attribute_names = {item.id: item.name for item in label.attributes}
    reviewed: list[ReviewedFrame] = []
    for track in annotations.tracks:
        if track.label_id != label.id:
            continue
        track_attrs = attributes_by_name(track.attributes, attribute_names)
        runner_id = int(track_attrs["runner_id"])
        attributes_by_frame = {
            int(shape.frame): attributes_by_name(shape.attributes, attribute_names)
            for shape in track.shapes
        }
        points_by_frame: dict[int, list[ReviewedPoint]] = {}
        for element in track.elements:
            joint_name = sublabel_names.get(element.label_id)
            if joint_name not in joint_ids:
                continue
            for shape in element.shapes:
                outside = bool(shape.outside)
                xy = (
                    None
                    if outside or len(shape.points) < 2
                    else (float(shape.points[0]), float(shape.points[1]))
                )
                visibility = 0 if outside else 1 if bool(shape.occluded) else 2
                points_by_frame.setdefault(int(shape.frame), []).append(
                    ReviewedPoint(joint_id=joint_ids[joint_name], xy=xy, visibility=visibility)
                )
        for frame_idx, points in sorted(points_by_frame.items()):
            reviewed.append(
                ReviewedFrame(
                    runner_id=runner_id,
                    frame_idx=frame_idx,
                    review_status=attributes_by_frame.get(frame_idx, {}).get(
                        "review_status", "UNREVIEWED"
                    ),
                    frame_ground_truth=parse_checkbox(
                        attributes_by_frame.get(frame_idx, {}).get(
                            FRAME_GROUND_TRUTH_ATTRIBUTE, "false"
                        )
                    ),
                    points=points,
                )
            )
    return reviewed


def write_ground_truth(
    run_dir: Path,
    reviewed: list[ReviewedFrame],
    *,
    task_id: int | None,
    bundle: CvatBundle | None = None,
) -> ReviewSummary:
    run = run_dir.expanduser().resolve()
    current_bundle = bundle or export_run(run)
    data = np.load(current_bundle.preannotations_npz)
    frame_indices = data["frame_indices"].astype(np.int32)
    runner_ids = data["runner_ids"].astype(np.int32)
    posterior = data["posterior_keypoints_xy"].astype(np.float32)
    shape = posterior.shape[:3]
    ground_truth = np.full((*shape, 2), np.nan, dtype=np.float32)
    visibility = np.zeros(shape, dtype=np.uint8)
    review_status = np.full(shape[:2], "UNREVIEWED", dtype="<U16")
    frame_ground_truth = np.zeros(shape[:2], dtype=bool)
    frame_to_t = {int(value): idx for idx, value in enumerate(frame_indices)}
    runner_to_idx = {int(value): idx for idx, value in enumerate(runner_ids)}
    joint_to_idx = {joint_id: idx for idx, joint_id in enumerate(JOINT_IDS)}
    for frame in reviewed:
        if frame.frame_idx not in frame_to_t or frame.runner_id not in runner_to_idx:
            continue
        t = frame_to_t[frame.frame_idx]
        obj_idx = runner_to_idx[frame.runner_id]
        review_status[t, obj_idx] = frame.review_status
        frame_ground_truth[t, obj_idx] = frame.frame_ground_truth
        for point in frame.points:
            if point.joint_id not in joint_to_idx:
                continue
            joint_idx = joint_to_idx[point.joint_id]
            visibility[t, obj_idx, joint_idx] = point.visibility
            if point.xy is not None:
                ground_truth[t, obj_idx, joint_idx] = point.xy

    error_px = np.linalg.norm(ground_truth - posterior, axis=-1)
    finite = np.isfinite(ground_truth).all(axis=-1) & np.isfinite(posterior).all(axis=-1)
    error_px[~finite] = np.nan
    corrected = finite & (error_px > 0.5) & (review_status[:, :, None] == "CORRECTED")
    keypoint_ground_truth = frame_ground_truth[:, :, None] & finite & (visibility > 0)
    verified_ground_truth = np.where(keypoint_ground_truth[..., None], ground_truth, np.nan).astype(
        np.float32
    )
    ground_truth_source = np.full(shape, "NONE", dtype="<U24")
    ground_truth_source[keypoint_ground_truth] = "HUMAN_FRAME"
    bbox_heights = annotation_bbox_heights(run, frame_indices, runner_ids)
    error_norm = error_px / np.maximum(bbox_heights[:, :, None], 1.0)
    output_dir = run / "annotations"
    suffix = "" if current_bundle.runner_id is None else f"_runner_{current_bundle.runner_id}"
    ground_truth_path = output_dir / f"ground_truth_pose{suffix}.npz"
    np.savez_compressed(
        ground_truth_path,
        frame_indices=frame_indices,
        object_ids=runner_ids,
        joint_ids=np.asarray(JOINT_IDS, dtype=np.int32),
        joint_names=np.asarray(JOINT_NAMES, dtype="<U16"),
        posterior_keypoints_xy=posterior,
        ground_truth_keypoints_xy=ground_truth,
        verified_ground_truth_keypoints_xy=verified_ground_truth,
        visibility=visibility,
        review_status=review_status,
        frame_ground_truth=frame_ground_truth,
        keypoint_ground_truth=keypoint_ground_truth,
        ground_truth_source=ground_truth_source,
        was_corrected=corrected,
        posterior_error_px=error_px.astype(np.float32),
        posterior_error_norm=error_norm.astype(np.float32),
    )
    coco_path = output_dir / f"ground_truth_coco{suffix}.json"
    write_coco_ground_truth(
        run,
        coco_path,
        frame_indices,
        runner_ids,
        ground_truth,
        visibility,
        review_status,
        frame_ground_truth,
        keypoint_ground_truth,
    )

    status_counts = {
        status: int(np.count_nonzero(review_status == status)) for status in REVIEW_STATUSES
    }
    reviewed_count = int(review_status.size - status_counts["UNREVIEWED"])
    summary_path = output_dir / f"review_manifest{suffix}.json"
    summary = ReviewSummary(
        task_id=task_id,
        scope=current_bundle.scope,
        runner_id=current_bundle.runner_id,
        runner_ids=runner_ids.tolist(),
        frame_count=int(len(frame_indices)),
        reviewed_frame_count=reviewed_count,
        unreviewed_frame_count=status_counts["UNREVIEWED"],
        corrected_keypoint_count=int(np.count_nonzero(corrected)),
        missing_keypoint_count=int(np.count_nonzero(visibility == 0)),
        ground_truth_frame_count=int(np.count_nonzero(frame_ground_truth)),
        ground_truth_keypoint_count=int(np.count_nonzero(keypoint_ground_truth)),
        status_counts=status_counts,
        artifacts={
            "ground_truth_npz": str(ground_truth_path.resolve()),
            "ground_truth_coco": str(coco_path.resolve()),
            "review_manifest": str(summary_path.resolve()),
        },
    )
    summary_path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")
    return summary


def write_coco_ground_truth(
    run: Path,
    path: Path,
    frame_indices: np.ndarray,
    runner_ids: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    review_status: np.ndarray,
    frame_ground_truth: np.ndarray,
    keypoint_ground_truth: np.ndarray,
) -> None:
    manifest = load_manifest(run)
    capture = cv2.VideoCapture(manifest.input_video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    pose = np.load(Path(required_artifact(manifest, "pose_npz")))
    bboxes = select_pose_bboxes(run, pose, frame_indices, runner_ids)
    images = [
        {
            "id": int(frame_idx),
            "file_name": f"{Path(manifest.input_video).stem}#frame_{frame_idx:06d}",
            "width": width,
            "height": height,
        }
        for frame_idx in frame_indices.tolist()
    ]
    annotations = []
    annotation_id = 1
    for t, frame_idx in enumerate(frame_indices.tolist()):
        for obj_idx, runner_id in enumerate(runner_ids.tolist()):
            xy = keypoints[t, obj_idx]
            vis = visibility[t, obj_idx]
            bbox = bboxes[t, obj_idx]
            if not np.isfinite(bbox).all():
                bbox = np.asarray([0, 0, 0, 0], dtype=np.float32)
            flattened = [
                value
                for point, visible in zip(xy, vis, strict=True)
                for value in (
                    float(point[0]) if np.isfinite(point).all() else 0.0,
                    float(point[1]) if np.isfinite(point).all() else 0.0,
                    int(visible),
                )
            ]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": int(frame_idx),
                    "category_id": 1,
                    "keypoints": flattened,
                    "num_keypoints": int(np.count_nonzero(vis > 0)),
                    "bbox": [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(max(bbox[2] - bbox[0], 0)),
                        float(max(bbox[3] - bbox[1], 0)),
                    ],
                    "area": float(max(bbox[2] - bbox[0], 0) * max(bbox[3] - bbox[1], 0)),
                    "iscrowd": 0,
                    "attributes": {
                        "track_id": int(runner_id),
                        "review_status": str(review_status[t, obj_idx]),
                        "frame_ground_truth": bool(frame_ground_truth[t, obj_idx]),
                        "keypoint_ground_truth": keypoint_ground_truth[t, obj_idx]
                        .astype(bool)
                        .tolist(),
                    },
                }
            )
            annotation_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "runner_lower_body",
                "keypoints": list(JOINT_NAMES),
                "skeleton": [[a + 1, b + 1] for a, b in EDGES],
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def annotation_bbox_heights(
    run: Path,
    frame_indices: np.ndarray,
    runner_ids: np.ndarray,
) -> np.ndarray:
    manifest = load_manifest(run)
    pose = np.load(Path(required_artifact(manifest, "pose_npz")))
    bboxes = select_pose_bboxes(run, pose, frame_indices, runner_ids)
    return np.maximum(bboxes[..., 3] - bboxes[..., 1], 1.0)


def select_pose_bboxes(
    run: Path,
    pose: Any,
    frame_indices: np.ndarray,
    runner_ids: np.ndarray,
) -> np.ndarray:
    manifest = load_manifest(run)
    pose_frames = np.asarray(pose["frame_indices"] if "frame_indices" in pose else frame_indices)
    pose_runners = np.asarray(
        pose["object_ids"] if "object_ids" in pose else manifest.accepted_runner_ids,
        dtype=np.int32,
    )
    frame_lookup = {int(value): idx for idx, value in enumerate(pose_frames)}
    runner_lookup = {int(value): idx for idx, value in enumerate(pose_runners)}
    try:
        frame_selection = [frame_lookup[int(value)] for value in frame_indices]
        runner_selection = [runner_lookup[int(value)] for value in runner_ids]
    except KeyError as exc:
        raise ValueError("Pose bboxes do not align with CVAT preannotations") from exc
    return np.asarray(pose["bboxes_xyxy"], dtype=np.float32)[
        np.ix_(frame_selection, runner_selection)
    ]


def attributes_by_name(values: list[Any], names: dict[int, str]) -> dict[str, str]:
    return {names[item.spec_id]: str(item.value) for item in values if item.spec_id in names}


def load_manifest(run_dir: Path) -> dromia_dto.RunManifest:
    manifest = dromia_dto.RunManifest.model_validate_json(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    manifest.source_video.path = str((run_dir / manifest.source_video.path).resolve())
    manifest.artifacts = {
        name: str((run_dir / path).resolve()) for name, path in manifest.artifacts.items()
    }
    return manifest


def required_artifact(manifest: dromia_dto.RunManifest, name: str) -> str:
    path = manifest.artifacts.get(name)
    if not path:
        raise ValueError(f"Run manifest is missing artifact: {name}")
    return path


def write_annotation_readme(annotation_dir: Path) -> None:
    text = """# CVAT Annotation Bundle

The editable skeleton is initialized from DromIA `posterior_pose` and contains hips,
knees, and ankles. Review every runner frame in CVAT and set `review_status`.

Check `frame_ground_truth` to promote every finite, visible keypoint on that runner-frame
to human ground truth and a maximum-confidence conditioning anchor.

- `ACCEPTED`: coordinates were already correct.
- `CORRECTED`: one or more points were moved.
- `AMBIGUOUS`: no defensible exact coordinate.
- `NOT_VISIBLE`: keypoint is occluded; use the point occlusion control.
- `OUT_OF_FRAME`: the anatomical point is outside the image.

DromIA retains the original posterior and computes corrections after `dromia-cvat pull`.
"""
    (annotation_dir / "README.md").write_text(text, encoding="utf-8")
