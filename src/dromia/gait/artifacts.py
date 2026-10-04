"""Serialization and optional rendering for gait-analysis outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dromia import dto as dromia_dto
from dromia import gait_contract
from dromia import video as dromia_video


def write_gait_artifacts(
    payload: dict[str, Any],
    *,
    run_dir: Path,
    video_path: Path | None = None,
    frame_indices: np.ndarray | None = None,
    object_ids: np.ndarray | None = None,
    pose_xy: np.ndarray | None = None,
    shoe_assignments: list[dromia_dto.ShoeAssignment] | None = None,
    fps: float | None = None,
    draw_video: bool = True,
    suffix: str = "",
) -> dict[str, str]:
    output = run_dir / "gait"
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"gait_analysis{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    frame_rows = [row for runner in payload["runners"].values() for row in runner["frames"]]
    event_rows = [row for runner in payload["runners"].values() for row in runner["events"]]
    frames_csv = output / f"gait_frames{suffix}.csv"
    events_csv = output / f"gait_events{suffix}.csv"
    write_csv(frames_csv, frame_rows)
    write_csv(
        events_csv,
        [
            gait_contract.event_csv_row(
                {
                    **event,
                    "source_fps": payload.get("source_fps"),
                    "real_world_fps": payload.get("real_world_fps"),
                }
            )
            for event in event_rows
        ],
    )
    npz_path = output / f"gait_analysis{suffix}.npz"
    write_npz(npz_path, frame_rows, event_rows)
    artifacts = {
        "gait_analysis_json": str(json_path.resolve()),
        "gait_frames_csv": str(frames_csv.resolve()),
        "gait_events_csv": str(events_csv.resolve()),
        "gait_analysis_npz": str(npz_path.resolve()),
    }
    if (
        draw_video
        and video_path is not None
        and frame_indices is not None
        and object_ids is not None
        and pose_xy is not None
    ):
        video_out = output / f"gait_debug{suffix}.mp4"
        write_debug_video(
            video_path,
            video_out,
            frame_indices,
            object_ids,
            pose_xy,
            payload,
            shoe_assignments or [],
            fps or payload["fps"],
        )
        artifacts["gait_debug_video"] = str(video_out.resolve())
    return artifacts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_npz(path: Path, frames: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    np.savez_compressed(
        path,
        frame_idx=np.asarray([x["frame_idx"] for x in frames], np.int32),
        runner_id=np.asarray([x["runner_id"] for x in frames], np.int32),
        # Float contact arrays preserve unknown observations as NaN (0=flight, 1=contact).
        global_contact=numeric_array(frames, "global_contact"),
        left_contact=numeric_array(frames, "left_contact"),
        right_contact=numeric_array(frames, "right_contact"),
        left_ground_y=numeric_array(frames, "left_ground_y"),
        right_ground_y=numeric_array(frames, "right_ground_y"),
        left_ground_step_index=numeric_array(frames, "left_ground_step_index"),
        right_ground_step_index=numeric_array(frames, "right_ground_step_index"),
        left_knee_angle_deg=numeric_array(frames, "left_knee_angle_deg"),
        right_knee_angle_deg=numeric_array(frames, "right_knee_angle_deg"),
        left_tibia_horizontal_angle_deg=numeric_array(frames, "left_tibia_horizontal_angle_deg"),
        right_tibia_horizontal_angle_deg=numeric_array(frames, "right_tibia_horizontal_angle_deg"),
        left_foot_tibia_angle_deg=numeric_array(frames, "left_foot_tibia_angle_deg"),
        right_foot_tibia_angle_deg=numeric_array(frames, "right_foot_tibia_angle_deg"),
        torso_lean_deg=numeric_array(frames, "torso_lean_deg"),
        events_json=np.asarray(json.dumps(events)),
    )


def write_debug_video(
    video_path: Path,
    output_path: Path,
    frame_indices: np.ndarray,
    object_ids: np.ndarray,
    pose: np.ndarray,
    payload: dict[str, Any],
    assignments: list[dromia_dto.ShoeAssignment],
    fps: float,
) -> None:
    info = dromia_video.inspect(video_path)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (info.width, info.height)
    )
    masks = {(x.frame_idx, x.runner_id, x.side): x.mask for x in assignments if x.mask is not None}
    rows = {
        (int(rid), int(row["frame_idx"])): row
        for rid, runner in ((int(k), v) for k, v in payload["runners"].items())
        for row in runner["frames"]
    }
    events = {
        (int(key), event["landing_frame"]): (
            f"LANDING {event['side'][0].upper()} {event['strike_type']}"
        )
        for key, runner in payload["runners"].items()
        for event in runner["events"]
        if event["landing_frame"] is not None
    }
    ground_model = payload.get("ground_model", {})
    ground_slope = (
        ground_model.get("slope") if ground_model.get("model") == "global_spatial_line" else None
    )
    ground_intercept = (
        ground_model.get("intercept")
        if ground_model.get("model") == "global_spatial_line"
        else None
    )
    for key, runner in payload["runners"].items():
        for event in runner["events"]:
            if event["takeoff_frame"] is not None:
                events.setdefault((int(key), event["takeoff_frame"]), "TAKEOFF")
            if event["knee_alignment_frame"] is not None:
                event_key = (int(key), event["knee_alignment_frame"])
                label = f"KNEE ALIGNMENT {event['side'][0].upper()}"
                current = events.get(event_key)
                events[event_key] = label if current is None else f"{current} / {label}"
    frame_lookup = {int(frame): idx for idx, frame in enumerate(frame_indices)}
    try:
        for frame_idx, image in dromia_video.ordered_frames(video_path, frame_indices):
            t = frame_lookup[frame_idx]
            if ground_slope is not None and ground_intercept is not None:
                start_y = round(float(ground_intercept))
                end_y = round(float(ground_slope) * (info.width - 1) + float(ground_intercept))
                cv2.line(image, (0, start_y), (info.width - 1, end_y), (255, 255, 0), 3)
            for obj_idx, runner_id_raw in enumerate(object_ids):
                runner_id = int(runner_id_raw)
                row = rows[(runner_id, frame_idx)]
                if ground_slope is None or ground_intercept is None:
                    runner_ground = payload["runners"][str(runner_id)]
                    lines_by_side = runner_ground.get("ground_lines", {})
                    ground_by_side = runner_ground["ground_y_by_side"]
                    for side, color in (("left", (0, 220, 255)), ("right", (255, 160, 20))):
                        line = lines_by_side.get(side, {})
                        if line.get("slope") is not None and line.get("intercept") is not None:
                            start_y = round(float(line["intercept"]))
                            end_y = round(
                                float(line["slope"]) * (info.width - 1)
                                + float(line["intercept"])
                            )
                            cv2.line(image, (0, start_y), (info.width - 1, end_y), color, 2)
                        else:
                            ground = row.get(f"{side}_ground_y")
                            if ground is None:
                                ground = ground_by_side.get(side)
                            if ground is not None:
                                cv2.line(
                                    image,
                                    (0, round(ground)),
                                    (info.width - 1, round(ground)),
                                    color,
                                    2,
                                )
                for side, color in (("left", (0, 220, 255)), ("right", (255, 160, 20))):
                    mask = masks.get((frame_idx, runner_id, side))
                    if mask is not None and mask.shape == image.shape[:2]:
                        image[np.asarray(mask) > 0] = color
                draw_pose_geometry(image, pose[t, obj_idx])
                pose_x = (
                    int(np.nanmin(pose[t, obj_idx, :, 0]))
                    if np.isfinite(pose[t, obj_idx, :, 0]).any()
                    else 10
                )
                x = min(max(pose_x, 5), max(info.width - 720, 5))
                y = (
                    int(np.nanmin(pose[t, obj_idx, :, 1]))
                    if np.isfinite(pose[t, obj_idx, :, 1]).any()
                    else 30
                )
                label = (
                    events.get((runner_id, frame_idx), "")
                    if row.get("measurement_window_included", True)
                    else "OUTSIDE CALIBRATION WINDOW"
                )
                lines = [
                    f"runner {runner_id} {label}",
                    format_side_overlay("L", "left", row),
                    format_side_overlay("R", "right", row),
                    f"torso={row['torso_posture']} {format_angle(row['torso_lean_deg'])}",
                ]
                text_y = max(min(y - 12, info.height - 75), 18)
                panel = image.copy()
                cv2.rectangle(
                    panel,
                    (x - 5, text_y - 15),
                    (info.width - 5, text_y + 65),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(panel, 0.55, image, 0.45, 0, image)
                for line_idx, text in enumerate(lines):
                    cv2.putText(
                        image,
                        text,
                        (x, text_y + 20 * line_idx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            writer.write(image)
    finally:
        writer.release()


def draw_pose_geometry(image: np.ndarray, points: np.ndarray) -> None:
    for a, b in ((5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 12)):
        if finite(points[a], points[b]):
            cv2.line(
                image,
                tuple(np.rint(points[a]).astype(int)),
                tuple(np.rint(points[b]).astype(int)),
                (60, 255, 60),
                2,
            )


def numeric_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([np.nan if row.get(key) is None else row[key] for row in rows], np.float32)


def format_angle(value: Any) -> str:
    return "--" if value is None else f"{float(value):.1f}deg"


def format_side_overlay(label: str, side: str, row: dict[str, Any]) -> str:
    return (
        f"{label} c={row[f'{side}_contact']} "
        f"k={format_angle(row[f'{side}_knee_angle_deg'])} "
        f"tf={format_angle(row[f'{side}_tibia_horizontal_angle_deg'])} "
    )


def finite(*points: np.ndarray) -> bool:
    return all(np.isfinite(point).all() for point in points)
