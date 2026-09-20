"""Simple pose and mask overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dromia import dto as dromia_dto

SKELETON = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def draw_runner_masks(frame: np.ndarray, runners: list[dromia_dto.SamDetection]) -> np.ndarray:
    out = frame.copy()
    overlay = frame.copy()
    for runner in runners:
        color = color_for(runner.obj_id)
        mask = np.asarray(runner.mask) > 0
        overlay[mask] = color
        x0, y0, x1, y1 = np.asarray(runner.bbox_xyxy, dtype=np.int32)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
        cv2.putText(
            out,
            f"R#{runner.obj_id}",
            (x0, max(14, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)
    return out


def draw_pose(
    frame: np.ndarray,
    keypoints_xy: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    line_color: tuple[int, int, int] = (60, 255, 255),
    point_color: tuple[int, int, int] = (30, 60, 255),
    line_thickness: int = 2,
    point_radius: int = 3,
) -> np.ndarray:
    out = frame.copy()
    points = np.asarray(keypoints_xy, dtype=np.float32)
    conf = (
        np.ones(points.shape[0], dtype=np.float32)
        if confidence is None
        else np.asarray(confidence, dtype=np.float32)
    )
    for a, b in SKELETON:
        if a >= len(points) or b >= len(points):
            continue
        if conf[a] <= 0.03 or conf[b] <= 0.03:
            continue
        if np.isfinite(points[[a, b]]).all():
            cv2.line(
                out,
                tuple(points[a].astype(int)),
                tuple(points[b].astype(int)),
                line_color,
                line_thickness,
            )
    for idx, point in enumerate(points):
        if conf[idx] > 0.03 and np.isfinite(point).all():
            cv2.circle(out, tuple(point.astype(int)), point_radius, point_color, -1)
    return out


def write_pose_video(
    *,
    video_path: Path,
    output_path: Path,
    frame_indices: list[int],
    object_ids: list[int],
    poses: np.ndarray,
    confidence: np.ndarray,
    frames_by_idx: dict[int, dromia_dto.SamFrame],
    fps: float,
    line_color: tuple[int, int, int] = (60, 255, 255),
    point_color: tuple[int, int, int] = (30, 60, 255),
    line_thickness: int = 2,
    point_radius: int = 3,
    codec: str = "mp4v",
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write video: {output_path}")
    try:
        for t, frame_idx in enumerate(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                continue
            sam_frame = frames_by_idx[frame_idx]
            out = draw_runner_masks(frame, [r for r in sam_frame.runners if r.obj_id in object_ids])
            for obj_idx, _obj_id in enumerate(object_ids):
                out = draw_pose(
                    out,
                    poses[t, obj_idx],
                    confidence[t, obj_idx],
                    line_color=line_color,
                    point_color=point_color,
                    line_thickness=line_thickness,
                    point_radius=point_radius,
                )
            writer.write(out)
    finally:
        capture.release()
        writer.release()


def write_mask_video(
    *,
    video_path: Path,
    output_path: Path,
    frame_indices: list[int],
    object_ids: list[int],
    frames_by_idx: dict[int, dromia_dto.SamFrame],
    fps: float,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write video: {output_path}")
    try:
        for frame_idx in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                continue
            sam_frame = frames_by_idx[frame_idx]
            out = draw_runner_masks(frame, [r for r in sam_frame.runners if r.obj_id in object_ids])
            writer.write(out)
    finally:
        capture.release()
        writer.release()


def color_for(obj_id: int) -> tuple[int, int, int]:
    return (80 + (obj_id * 53) % 140, 210, 90 + (obj_id * 31) % 120)
