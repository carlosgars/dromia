"""First-pass, inspectable 2D gait analysis from pose and shoe masks.

All geometry is measured in the image plane. The module deliberately keeps contact
detection, event extraction, geometry, serialization, and rendering separate so each
piece can be calibrated or replaced without changing the artifact contract.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy.stats import theilslopes

from dromia import calibration as dromia_calibration
from dromia import config as dromia_config
from dromia import dto as dromia_dto
from dromia import gait_contract
from dromia import timebase as dromia_timebase
from dromia.models import cache as sam31_cache

SIDES = ("left", "right")
SIDE_JOINTS = {"left": (11, 13, 15), "right": (12, 14, 16)}
MEASUREMENT_WINDOW_JOINTS = (5, 6, 11, 12, 13, 14, 15, 16)
EPS = 1e-8


@dataclass(slots=True)
class ShoeFrame:
    frame_idx: int
    runner_id: int
    side: str
    score: float
    curve: list[list[float]]
    max_y: float | None
    mask_observed: bool = False
    lower_curve_mean_y: float | None = None
    ground_x: float | None = None
    bbox_height: float | None = None
    floor_candidate_eligible: bool = True
    ground_y: float | None = None
    ground_step_index: int | None = None
    ground_line_slope: float | None = None
    ground_line_intercept: float | None = None
    ground_line_r2: float | None = None
    ground_line_quality: str | None = None
    ground_line_candidate_count: int | None = None
    ground_line_bin_count: int | None = None
    clearance_px: float | None = None
    contact_fraction: float = 0.0
    contact_position: float | None = None
    contact: bool | None = None
    contact_source: str = "unknown"
    contact_point_xy: tuple[float, float] | None = None
    foot_axis_xy: tuple[float, float] | None = None
    foot_axis_rmse_px: float | None = None
    foot_axis_quality: str = "not_available"


@dataclass(slots=True, frozen=True)
class GroundLine:
    slope: float
    intercept: float
    r2: float
    candidate_count: int
    bin_count: int
    quality: str
    anchors: tuple[tuple[float, float], ...] = ()

    def y_at(self, x: float) -> float:
        if len(self.anchors) >= 2:
            xs = [a[0] for a in self.anchors]
            ys = [a[1] for a in self.anchors]
            order = np.argsort(xs)
            xs_sorted = np.asarray(xs, dtype=np.float64)[order]
            ys_sorted = np.asarray(ys, dtype=np.float64)[order]
            if xs_sorted[-1] - xs_sorted[0] > EPS:
                if x < xs_sorted[0]:
                    return float(ys_sorted[0] + self.slope * (x - xs_sorted[0]))
                if x > xs_sorted[-1]:
                    return float(ys_sorted[-1] + self.slope * (x - xs_sorted[-1]))
                return float(np.interp(x, xs_sorted, ys_sorted))
        elif len(self.anchors) == 1:
            x0, y0 = self.anchors[0]
            return float(y0 + self.slope * (x - x0))
        return self.slope * x + self.intercept


@dataclass(slots=True)
class ContactEvent:
    runner_id: int
    side: str
    landing_frame: int | None
    takeoff_frame: int | None
    first_flight_frame: int | None
    contact_frames: int
    contact_time_s: float | None
    contact_time_ms: float | None
    same_foot_flight_frames: int | None
    same_foot_flight_time_s: float | None
    same_foot_flight_time_ms: float | None
    strike_type: str
    strike_position: float | None
    confidence: float
    strike_contact_fraction: float
    strike_shoe_score: float
    strike_direction_source: str
    inferred_contact_frames: int = 0
    inferred_contact_fraction: float = 0.0
    contact_sources: tuple[str, ...] = ()
    endpoint_quality: str = "unknown"
    strike_threshold_version: str = "outsole_thirds_v1_unvalidated"
    strike_source: str = "automatic"
    strike_quality: str = "experimental_unvalidated_thresholds"


def analyze_gait(
    *,
    frame_indices: list[int] | np.ndarray,
    object_ids: list[int] | np.ndarray,
    pose_xy: np.ndarray,
    bboxes_xyxy: np.ndarray,
    shoe_assignments: list[dromia_dto.ShoeAssignment],
    fps: float,
    cfg: dromia_config.GaitAnalysisConfig,
    source_pose: str = "temporal_biomechanics",
    fps_is_assumed: bool = False,
    timebase: dromia_timebase.VideoTimebase | None = None,
    calibration: dromia_calibration.GroundCalibration | None = None,
    ground_line: GroundLine | None = None,
    ground_lines: dict[tuple[int, str], GroundLine] | None = None,
) -> dict[str, Any]:
    """Compute per-frame geometry and per-contact gait events for every runner."""

    frames = np.asarray(frame_indices, dtype=np.int32)
    runners = np.asarray(object_ids, dtype=np.int32)
    pose = np.asarray(pose_xy, dtype=np.float32)
    bboxes = np.asarray(bboxes_xyxy, dtype=np.float32)
    if pose.shape != (len(frames), len(runners), 17, 2):
        raise ValueError("pose_xy must be [T,O,17,2]")
    if bboxes.shape != (len(frames), len(runners), 4):
        raise ValueError("bboxes_xyxy must be [T,O,4]")
    timing = timebase or synthetic_timebase(frames, fps)
    if timing.frame_indices != frames.tolist():
        raise ValueError("timebase frame indices must exactly match gait frame indices")

    confirmed_direction = calibration_direction(calibration)
    measurement_window = calibration_measurement_window(calibration)
    frame_is_measurable = {
        int(runner_id): measurement_window_frame_mask(pose[:, obj_idx], measurement_window)
        for obj_idx, runner_id in enumerate(runners)
    }
    directions: dict[int, str] = {}
    direction_sources: dict[int, str] = {}
    for obj_idx, runner_id in enumerate(runners):
        runner_pose = pose[:, obj_idx]
        observed = runner_direction(runner_pose)
        left = runner_pose[:, 11, 0]
        right = runner_pose[:, 12, 0]
        hip = np.where(np.isfinite(left) & np.isfinite(right), (left + right) * 0.5, np.nan)
        finite_values = hip[np.isfinite(hip)]
        has_clear_displacement = (
            len(finite_values) >= 2 and abs(float(finite_values[-1] - finite_values[0])) > 30.0
        )
        if has_clear_displacement:
            directions[int(runner_id)] = observed
            direction_sources[int(runner_id)] = (
                "ground_calibration" if confirmed_direction == observed else "pose_displacement"
            )
        else:
            directions[int(runner_id)] = confirmed_direction or observed
            direction_sources[int(runner_id)] = (
                "ground_calibration" if confirmed_direction else "pose_displacement"
            )
    shoe_frames = analyze_shoes(
        shoe_assignments,
        bboxes,
        frames,
        runners,
        cfg,
        fps=timing.real_world_fps or timing.source_fps,
        directions=directions,
        global_ground=ground_line,
        ground_lines=ground_lines,
    )
    ground_model = ground_model_summary(shoe_frames, cfg)
    shoe_lookup = {(x.frame_idx, x.runner_id, x.side): x for x in shoe_frames}
    runner_payload: dict[str, Any] = {}
    for obj_idx, runner_id_raw in enumerate(runners):
        runner_id = int(runner_id_raw)
        direction = directions[runner_id]
        direction_source = direction_sources[runner_id]
        measurable = frame_is_measurable[runner_id]
        events = build_contact_events(
            runner_id,
            frames,
            shoe_lookup,
            timing=timing,
            direction=direction,
            cfg=cfg,
            direction_source=direction_source,
        )
        frame_rows = []
        for t, frame_idx in enumerate(frames):
            points = pose[t, obj_idx]
            row = frame_metrics(
                int(frame_idx),
                runner_id,
                points,
                bboxes[t, obj_idx],
                shoe_lookup,
                direction,
                cfg,
            )
            row["measurement_window_included"] = bool(measurable[t])
            frame_rows.append(row)
        event_rows = enrich_events(events, frame_rows, timing, calibration)
        assign_event_ids(event_rows)
        refresh_event_geometry(event_rows, frame_rows, calibration)
        for event in event_rows:
            add_event_quality(event, cfg)
        recompute_event_flights(event_rows, timing, frame_rows)
        activity_window = apply_activity_window(frame_rows, event_rows)
        flights = global_flight_intervals_from_rows(frame_rows, timing)
        associate_global_flights(event_rows, flights)
        global_contact = global_contact_metrics(frame_rows, timing)
        cadence = cadence_metrics(event_rows, timing, cfg)
        distances = spatial_distance_metrics(event_rows, calibration)
        for event in event_rows:
            add_public_event_fields(event)
        asymmetry = asymmetry_metrics(event_rows, distances, cfg)
        runner_payload[str(runner_id)] = {
            "runner_id": runner_id,
            "source_fps": float(timing.source_fps),
            "real_world_fps": float(timing.real_world_fps or timing.source_fps),
            "direction": direction,
            "direction_source": direction_source,
            "measurement_window": {
                "applied": measurement_window is not None,
                "x_min": None if measurement_window is None else measurement_window[0],
                "x_max": None if measurement_window is None else measurement_window[1],
                "included_frame_count": int(np.count_nonzero(measurable)),
                "excluded_frame_count": int(len(measurable) - np.count_nonzero(measurable)),
            },
            "activity_window": activity_window,
            "ground_y_by_side": runner_ground_y_by_side(runner_id, shoe_frames),
            "ground_lines": runner_ground_lines(runner_id, shoe_frames),
            "events": event_rows,
            "flight_intervals": flights,
            "global_contact": global_contact,
            "frames": frame_rows,
            "cadence": cadence,
            "spatial": distances,
            "asymmetry": asymmetry,
            "summary": summarize(event_rows, flights, cadence, distances, global_contact),
        }
    return {
        "schema_version": gait_contract.SCHEMA_VERSION,
        "source_pose": source_pose,
        "fps": float(timing.source_fps),
        "source_fps": float(timing.source_fps),
        "real_world_fps": float(timing.real_world_fps or timing.source_fps),
        "fps_is_assumed": bool(fps_is_assumed),
        "timebase": timing.model_dump(mode="json"),
        "calibration": None if calibration is None else calibration.model_dump(mode="json"),
        "coordinate_system": "image_xy_x_right_y_down",
        "angle_convention": "2d_image_plane_degrees",
        "ground_model": ground_model,
        "frame_count": len(frames),
        "runners": runner_payload,
        "limitations": [
            "Angles are 2D image-plane projections and are not perspective corrected.",
            (
                "Ground contact uses one robust spatial line per runner and shoe, with a "
                "jointly estimated and physically bounded camera-tilt slope."
                if cfg.ground_model == "per_shoe_line"
                else "Ground contact uses one robust spatial line fitted to the global lower "
                "shoe envelope."
                if cfg.ground_model == "global_line"
                else "Each foot uses an independently interpolated ground path through local "
                "step anchors."
            ),
            "Contact timing precision is limited to one video frame.",
            "Contact intervals touching a clip boundary are reported as censored.",
            "Metric distances are reported only for valid ground-plane calibration.",
            "When calibration is valid, metrics exclude frames whose core body keypoints "
            "fall outside the calibration x-range.",
        ],
    }


def ground_model_summary(
    frames: list[ShoeFrame], cfg: dromia_config.GaitAnalysisConfig
) -> dict[str, Any]:
    if cfg.ground_model == "per_shoe_line":
        lines = {
            str(runner_id): runner_ground_lines(runner_id, frames)
            for runner_id in sorted({item.runner_id for item in frames})
        }
        runner_slopes = {
            runner_id: float(
                np.median(
                    [
                        float(line["slope"])
                        for line in runner.values()
                        if line.get("slope") is not None
                    ]
                )
            )
            for runner_id, runner in lines.items()
            if any(line.get("slope") is not None for line in runner.values())
        }
        return {
            "model": "per_runner_per_shoe_spatial_lines",
            "equation": "y = slope * x + intercept",
            "runner_slopes": runner_slopes,
            "runner_angles_deg": {
                runner_id: float(np.degrees(np.arctan(slope)))
                for runner_id, slope in runner_slopes.items()
            },
            "lines": lines,
        }
    fitted = next(
        (
            item
            for item in frames
            if item.ground_line_quality is not None
            and item.ground_line_slope is not None
            and item.ground_line_intercept is not None
        ),
        None,
    )
    if fitted is None:
        return {"model": cfg.ground_model, "quality": "per_step_or_not_available"}
    return {
        "model": "global_spatial_line",
        "equation": "y = slope * x + intercept",
        "slope": fitted.ground_line_slope,
        "intercept": fitted.ground_line_intercept,
        "r2": fitted.ground_line_r2,
        "quality": fitted.ground_line_quality,
        "candidate_count": fitted.ground_line_candidate_count,
        "bin_count": fitted.ground_line_bin_count,
    }


def calibration_measurement_window(
    calibration: dromia_calibration.GroundCalibration | None,
) -> tuple[float, float] | None:
    """Return the simple horizontal measurement limits requested by the field workflow."""

    if calibration is None or not calibration.valid:
        return None
    x_values = np.asarray(calibration.image_points_xy, dtype=np.float32)[:, 0]
    return float(np.min(x_values)), float(np.max(x_values))


def measurement_window_frame_mask(
    runner_pose: np.ndarray,
    window: tuple[float, float] | None,
) -> np.ndarray:
    if window is None:
        return np.ones(len(runner_pose), dtype=bool)
    core = np.asarray(runner_pose, dtype=np.float32)[:, MEASUREMENT_WINDOW_JOINTS]
    finite = np.isfinite(core).all(axis=2)
    x_min, x_max = window
    inside = (core[:, :, 0] >= x_min) & (core[:, :, 0] <= x_max)
    return np.all(finite & inside, axis=1)


def enrich_events(
    events: list[ContactEvent],
    rows: list[dict[str, Any]],
    timing: dromia_timebase.VideoTimebase,
    calibration: dromia_calibration.GroundCalibration | None,
) -> list[dict[str, Any]]:
    """Attach event-frame geometry so events are useful without joining the frame table."""

    by_frame = {int(row["frame_idx"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for event in events:
        item = asdict(event)
        add_stance_phases(item, rows, timing)
        for phase, frame_idx in (
            ("landing", event.landing_frame),
            ("takeoff", event.takeoff_frame),
        ):
            row = {} if frame_idx is None else by_frame.get(frame_idx, {})
            prefix = event.side
            for metric in (
                "foot_tibia_angle_deg",
                "tibia_horizontal_angle_deg",
                "knee_angle_deg",
            ):
                item[f"{phase}_{metric}"] = row.get(f"{prefix}_{metric}")
            item[f"{phase}_torso_lean_deg"] = row.get("torso_lean_deg")
            item[f"{phase}_torso_posture"] = row.get("torso_posture")
        landing_row = {} if event.landing_frame is None else by_frame.get(event.landing_frame, {})
        contact_point = landing_row.get(f"{event.side}_contact_point_xy")
        if contact_point is None and event.landing_frame is not None:
            max_search = (
                int(event.takeoff_frame)
                if event.takeoff_frame is not None
                else int(event.landing_frame) + 10
            )
            for search_frame in range(int(event.landing_frame), max_search + 1):
                candidate = by_frame.get(search_frame, {}).get(f"{event.side}_contact_point_xy")
                if candidate is not None:
                    contact_point = candidate
                    break
        item["landing_contact_point_xy"] = contact_point
        item["landing_contact_point_m"] = project_point(
            calibration, contact_point, require_inside=True
        )
        item["landing_contact_point_raw_m"] = project_point(
            calibration, contact_point, require_inside=False
        )
        item["landing_contact_point_in_calibration"] = bool(
            item["landing_contact_point_m"] is not None
        )
        output.append(item)
    return output


def refresh_event_geometry(
    events: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    calibration: dromia_calibration.GroundCalibration | None,
) -> None:
    by_frame = {int(row["frame_idx"]): row for row in rows}
    known_frames = sorted(by_frame)
    for event in events:
        side = str(event["side"])
        landing = event.get("landing_frame")
        takeoff = event.get("takeoff_frame")
        event["contact_frames"] = (
            (int(takeoff) - int(landing) + 1)
            if (landing is not None and takeoff is not None)
            else sum(
                (landing is None or frame >= int(landing))
                and (takeoff is None or frame <= int(takeoff))
                for frame in known_frames
            )
        )
        event["first_flight_frame"] = next(
            (frame for frame in known_frames if takeoff is not None and frame > int(takeoff)),
            None,
        )
        for phase in ("landing", "takeoff"):
            frame_idx = event.get(f"{phase}_frame")
            row = {} if frame_idx is None else by_frame.get(int(frame_idx), {})
            for metric in (
                "foot_tibia_angle_deg",
                "tibia_horizontal_angle_deg",
                "knee_angle_deg",
            ):
                event[f"{phase}_{metric}"] = row.get(f"{side}_{metric}")
            event[f"{phase}_torso_lean_deg"] = row.get("torso_lean_deg")
            event[f"{phase}_torso_posture"] = row.get("torso_posture")
        landing_row = {} if landing is None else by_frame.get(int(landing), {})
        contact_point = landing_row.get(f"{side}_contact_point_xy")
        if contact_point is None and landing is not None:
            max_search = int(takeoff) if takeoff is not None else int(landing) + 10
            for search_frame in range(int(landing), max_search + 1):
                candidate = by_frame.get(search_frame, {}).get(f"{side}_contact_point_xy")
                if candidate is not None:
                    contact_point = candidate
                    break
        event["landing_contact_point_xy"] = contact_point
        event["landing_contact_point_m"] = project_point(
            calibration, contact_point, require_inside=True
        )
        event["landing_contact_point_raw_m"] = project_point(
            calibration, contact_point, require_inside=False
        )
        event["landing_contact_point_in_calibration"] = bool(
            event["landing_contact_point_m"] is not None
        )
        alignment = event.get("knee_alignment_frame")
        alignment_row = {} if alignment is None else by_frame.get(int(alignment), {})
        separation = alignment_row.get("knee_horizontal_separation_px")
        height = alignment_row.get("bbox_height_px")
        event["knee_alignment_separation_norm"] = (
            float(separation) / float(height) if separation is not None and height else None
        )


def recompute_event_flights(
    events: list[dict[str, Any]],
    timing: dromia_timebase.VideoTimebase,
    rows: list[dict[str, Any]] | None = None,
) -> None:
    for event in events:
        event["same_foot_flight_frames"] = None
        event["same_foot_flight_time_s"] = None
        event["same_foot_flight_time_ms"] = None
        event["same_foot_flight_quality"] = "censored_or_unknown"
    for side in SIDES:
        side_events = sorted(
            [event for event in events if event.get("side") == side],
            key=lambda event: (
                event.get("landing_frame") if event.get("landing_frame") is not None else -1
            ),
        )
        for current, following in zip(side_events, side_events[1:], strict=False):
            takeoff = current.get("takeoff_frame")
            landing = following.get("landing_frame")
            if takeoff is None or landing is None or int(landing) <= int(takeoff) + 1:
                current["same_foot_flight_frames"] = None
                current["same_foot_flight_time_s"] = None
                current["same_foot_flight_time_ms"] = None
                continue
            between = [
                frame for frame in timing.frame_indices if int(takeoff) < frame < int(landing)
            ]
            if rows is not None:
                observed = {int(row["frame_idx"]): row.get(f"{side}_contact") for row in rows}
                if any(observed.get(frame) is not False for frame in between):
                    continue
            if len(between) != int(landing) - int(takeoff) - 1:
                continue
            current["same_foot_flight_quality"] = "complete"
            current["same_foot_flight_frames"] = len(between)
            current["same_foot_flight_time_s"] = (
                dromia_timebase.interval_duration_s(timing, between[0], between[-1])
                if between
                else None
            )
            current["same_foot_flight_time_ms"] = to_ms(current["same_foot_flight_time_s"])


def analyze_shoes(
    assignments: list[dromia_dto.ShoeAssignment],
    bboxes: np.ndarray,
    frame_indices: np.ndarray,
    object_ids: np.ndarray,
    cfg: dromia_config.GaitAnalysisConfig,
    *,
    fps: float = 30.0,
    directions: dict[int, str] | None = None,
    global_ground: GroundLine | None = None,
    ground_lines: dict[tuple[int, str], GroundLine] | None = None,
) -> list[ShoeFrame]:
    frame_pos = {int(value): idx for idx, value in enumerate(frame_indices)}
    obj_pos = {int(value): idx for idx, value in enumerate(object_ids)}
    output: list[ShoeFrame] = []
    for item in assignments:
        if item.runner_id not in obj_pos or item.frame_idx not in frame_pos:
            continue
        curve = lower_curve(item.mask) if item.mask is not None else np.empty((0, 2), np.float32)
        max_y = float(np.max(curve[:, 1])) if len(curve) else None
        ground_x = None
        if max_y is not None:
            bottom = curve[curve[:, 1] >= max_y - 1.0]
            ground_x = float(np.median(bottom[:, 0]))
        t, o = frame_pos[int(item.frame_idx)], obj_pos[int(item.runner_id)]
        bbox = np.asarray(bboxes[t, o], dtype=np.float64)
        bbox_height = (
            float(bbox[3] - bbox[1]) if np.isfinite(bbox).all() and bbox[3] > bbox[1] else None
        )
        frame_width = float(item.mask.shape[1]) if item.mask is not None else 0.0
        edge_margin = cfg.ground_line_frame_edge_margin_fraction * frame_width
        floor_candidate_eligible = bool(
            frame_width <= 0.0
            or not np.isfinite(bbox).all()
            or (bbox[0] >= edge_margin and bbox[2] <= frame_width - edge_margin)
        )
        output.append(
            ShoeFrame(
                frame_idx=int(item.frame_idx),
                runner_id=int(item.runner_id),
                side=item.side,
                score=float(item.score),
                curve=curve.tolist(),
                max_y=max_y,
                mask_observed=bool(len(curve)),
                lower_curve_mean_y=(float(np.mean(curve[:, 1])) if len(curve) else None),
                ground_x=ground_x,
                bbox_height=bbox_height,
                floor_candidate_eligible=floor_candidate_eligible,
            )
        )
    fitted_lines: dict[tuple[int, str], GroundLine] = {}
    if cfg.ground_model == "per_shoe_line":
        fitted_lines = ground_lines or fit_per_shoe_ground_lines(output, cfg, fps=fps)
    elif cfg.ground_model == "global_line":
        global_ground = global_ground or fit_global_ground_line(output, cfg)
        fitted_lines = {
            (int(runner_id), side): global_ground for runner_id in object_ids for side in SIDES
        }
    if fitted_lines:
        for item in output:
            if item.max_y is None or item.ground_x is None:
                continue
            line = fitted_lines.get((item.runner_id, item.side))
            if line is None:
                continue
            item.ground_y = line.y_at(item.ground_x)
            if line.anchors:
                item.ground_step_index = int(
                    np.argmin([abs(item.ground_x - a[0]) for a in line.anchors])
                )
            else:
                item.ground_step_index = 0
            item.ground_line_slope = line.slope
            item.ground_line_intercept = line.intercept
            item.ground_line_r2 = line.r2
            item.ground_line_quality = line.quality
            item.ground_line_candidate_count = line.candidate_count
            item.ground_line_bin_count = line.bin_count
    for runner_id in object_ids:
        for side in SIDES:
            foot = [
                item
                for item in output
                if item.runner_id == int(runner_id) and item.side == side and item.max_y is not None
            ]
            if not foot:
                continue
            if not fitted_lines:
                assign_local_step_grounds(
                    foot,
                    bboxes=bboxes,
                    frame_pos=frame_pos,
                    obj_idx=obj_pos[int(runner_id)],
                    fps=fps,
                    cfg=cfg,
                )
            for item in foot:
                t, o = frame_pos[item.frame_idx], obj_pos[item.runner_id]
                height = max(float(bboxes[t, o, 3] - bboxes[t, o, 1]), 1.0)
                strict_tolerance = max(
                    cfg.min_contact_tolerance_px, cfg.contact_tolerance_bbox_fraction * height
                )
                curve = np.asarray(item.curve, np.float32)
                direction = (directions or {}).get(item.runner_id, "right")
                strict_position, strict_fraction = contact_region(
                    curve, float(item.ground_y), strict_tolerance, direction
                )
                extremity_tolerance = max(
                    cfg.min_extremity_contact_tolerance_px,
                    cfg.extremity_contact_tolerance_bbox_fraction * height,
                )
                extremity_position, extremity_fraction = contact_region(
                    curve, float(item.ground_y), extremity_tolerance, direction
                )
                penetration_tolerance = max(
                    cfg.min_ground_penetration_tolerance_px,
                    cfg.ground_penetration_tolerance_bbox_fraction * height,
                )
                item.clearance_px = float(item.ground_y) - float(item.max_y)
                strict_contact = (
                    -penetration_tolerance <= item.clearance_px <= strict_tolerance
                    and strict_fraction >= cfg.min_contact_curve_fraction
                )
                extremity_contact = (
                    -penetration_tolerance <= item.clearance_px <= extremity_tolerance
                    and extremity_position is not None
                    and (extremity_position <= 0.34 or extremity_position >= 0.66)
                    and extremity_fraction >= cfg.min_extremity_contact_curve_fraction
                )
                item.contact = strict_contact or extremity_contact
                item.contact_source = "direct" if item.contact else "observed_non_contact"
                if extremity_contact and not strict_contact:
                    item.contact_fraction = extremity_fraction
                    item.contact_position = extremity_position
                else:
                    item.contact_fraction = strict_fraction
                    item.contact_position = strict_position
                near_ground = curve[
                    (curve[:, 1] >= float(item.ground_y) - extremity_tolerance)
                    & (curve[:, 1] <= float(item.ground_y) + penetration_tolerance)
                ]
                if len(near_ground):
                    point = np.mean(near_ground, axis=0)
                    item.contact_point_xy = (float(point[0]), float(point[1]))
                axis, rmse = robust_outsole_axis(curve)
                item.foot_axis_rmse_px = rmse
                if (
                    axis is not None
                    and np.ptp(curve[:, 0]) >= cfg.foot_axis_min_length_px
                    and rmse is not None
                    and rmse <= cfg.foot_axis_max_rmse_px
                ):
                    if direction == "left":
                        axis = -axis
                    item.foot_axis_xy = (float(axis[0]), float(axis[1]))
                    item.foot_axis_quality = "valid"
                elif axis is not None:
                    item.foot_axis_quality = "low_quality"
    stabilize_contacts(output, cfg, fps=fps)
    refine_contacts_from_shoe_kinematics(output, cfg, fps=fps)
    stabilize_foot_axes(output, cfg)
    return output


def fit_per_shoe_ground_lines(
    frames: list[ShoeFrame], cfg: dromia_config.GaitAnalysisConfig, *, fps: float = 30.0
) -> dict[tuple[int, str], GroundLine]:
    """Fit a distinct floor offset per runner/shoe with one robust camera-roll slope.

    Each shoe first contributes stable temporal stance peaks, so swing masks cannot
    define the floor. Within each runner, Theil--Sen slopes through those stance
    anchors are clipped to the configured camera-roll bound and combined by their
    median. The two shoes keep separate intercepts while sharing that runner's robust
    tilt. Small real slopes are retained even when their R² is low relative to
    gait-cycle oscillation.
    """

    tracks: dict[tuple[int, str], list[ShoeFrame]] = {}
    for item in frames:
        if item.ground_x is not None and item.max_y is not None:
            tracks.setdefault((item.runner_id, item.side), []).append(item)
    envelopes: dict[tuple[int, str], tuple[np.ndarray, int]] = {}
    raw_slopes: dict[int, list[float]] = {}
    max_slope = float(np.tan(np.radians(cfg.ground_line_max_abs_angle_deg)))
    for key, track in tracks.items():
        envelope, candidate_count = temporal_stance_anchors(track, cfg, fps=fps)
        if not len(envelope):
            envelope, candidate_count = spatial_ground_envelope(track, cfg)
        envelopes[key] = (envelope, candidate_count)
        if len(envelope) >= 2 and np.ptp(envelope[:, 0]) > EPS:
            slope = float(theilslopes(envelope[:, 1], envelope[:, 0]).slope)
            raw_slopes.setdefault(key[0], []).append(float(np.clip(slope, -max_slope, max_slope)))
    runner_slopes = {
        runner_id: float(np.median(slopes)) for runner_id, slopes in raw_slopes.items()
    }

    output: dict[tuple[int, str], GroundLine] = {}
    for key, (envelope, candidate_count) in envelopes.items():
        if not len(envelope):
            output[key] = GroundLine(0.0, 0.0, 0.0, 0, 0, "not_available")
            continue
        slope = runner_slopes.get(key[0], 0.0)
        intercept = float(np.median(envelope[:, 1] - slope * envelope[:, 0]))
        predicted = slope * envelope[:, 0] + intercept
        residual_sum = float(np.sum((envelope[:, 1] - predicted) ** 2))
        total_sum = float(np.sum((envelope[:, 1] - np.mean(envelope[:, 1])) ** 2))
        r2 = 1.0 if total_sum <= EPS else max(0.0, 1.0 - residual_sum / total_sum)
        anchor_tuples = tuple((float(pt[0]), float(pt[1])) for pt in envelope)
        output[key] = GroundLine(
            slope,
            intercept,
            r2,
            candidate_count,
            len(envelope),
            "robust_per_shoe_line" if len(envelope) >= 2 else "horizontal_fallback",
            anchors=anchor_tuples,
        )
    return output


def temporal_stance_anchors(
    frames: list[ShoeFrame],
    cfg: dromia_config.GaitAnalysisConfig,
    *,
    fps: float,
) -> tuple[np.ndarray, int]:
    """Extract spatial floor anchors only from stable temporal outsole lows."""

    ordered = sorted(
        (item for item in frames if item.ground_x is not None and item.max_y is not None),
        key=lambda item: item.frame_idx,
    )
    if not ordered:
        return np.empty((0, 2), dtype=np.float64), 0
    max_gap = max(0, round(cfg.ground_step_max_interpolation_gap_seconds * fps))
    blocks: list[list[ShoeFrame]] = []
    block: list[ShoeFrame] = []
    for item in ordered:
        if block and item.frame_idx - block[-1].frame_idx - 1 > max_gap:
            blocks.append(block)
            block = []
        block.append(item)
    if block:
        blocks.append(block)

    anchors: list[tuple[float, float]] = []
    minimum_cycle = max(
        1,
        round(cfg.ground_step_min_separation_seconds * fps),
        round(120.0 / cfg.cadence_max_spm * fps),
    )
    peak_window = max(1, round(cfg.ground_line_peak_window_seconds * fps))
    for visible in blocks:
        if len(visible) < 3:
            continue
        observed_frames = np.asarray([item.frame_idx for item in visible], dtype=np.int32)
        observed_x = np.asarray([float(item.ground_x) for item in visible], dtype=np.float64)
        observed_y = np.asarray([float(item.max_y) for item in visible], dtype=np.float64)
        observed_eligible = np.asarray(
            [item.floor_candidate_eligible for item in visible], dtype=bool
        )
        dense_frames = np.arange(observed_frames[0], observed_frames[-1] + 1, dtype=np.int32)
        dense_y = np.interp(dense_frames, observed_frames, observed_y)
        smoothing = max(1, round(cfg.ground_step_smoothing_seconds * fps))
        if smoothing % 2 == 0:
            smoothing += 1
        smoothed_y = median_filter(dense_y, size=smoothing, mode="nearest")
        heights = [
            float(item.bbox_height)
            for item in visible
            if item.bbox_height is not None and item.bbox_height > 0
        ]
        median_height = float(np.median(heights)) if heights else 500.0
        peaks, _ = find_peaks(
            smoothed_y,
            distance=minimum_cycle,
            prominence=cfg.ground_line_min_peak_prominence_bbox_fraction * median_height,
        )
        for peak in peaks:
            center = int(dense_frames[peak])
            selected = (np.abs(observed_frames - center) <= peak_window) & observed_eligible
            if not np.any(selected):
                selected = np.abs(observed_frames - center) <= peak_window
            if not np.any(selected):
                continue
            ground_y = float(
                np.percentile(observed_y[selected], cfg.ground_line_candidate_quantile)
            )
            near_ground = selected & (observed_y >= ground_y - 1.0)
            ground_x = float(np.median(observed_x[near_ground]))
            anchors.append((ground_x, ground_y))
    return np.asarray(anchors, dtype=np.float64).reshape(-1, 2), len(ordered)


def spatial_ground_envelope(
    frames: list[ShoeFrame], cfg: dromia_config.GaitAnalysisConfig
) -> tuple[np.ndarray, int]:
    """Return equal-width lower-envelope anchors for one runner/shoe track."""

    candidates = np.asarray(
        [
            (float(item.ground_x), float(item.max_y))
            for item in frames
            if item.ground_x is not None
            and item.max_y is not None
            and np.isfinite(item.ground_x)
            and np.isfinite(item.max_y)
        ],
        dtype=np.float64,
    )
    if not len(candidates):
        return np.empty((0, 2), dtype=np.float64), 0
    x_min, x_max = float(np.min(candidates[:, 0])), float(np.max(candidates[:, 0]))
    if x_max - x_min <= EPS:
        return np.asarray(
            [
                (
                    float(np.median(candidates[:, 0])),
                    float(np.percentile(candidates[:, 1], cfg.ground_line_candidate_quantile)),
                )
            ],
            dtype=np.float64,
        ), len(candidates)
    edges = np.linspace(x_min, x_max, cfg.ground_line_bin_count + 1)
    envelope: list[tuple[float, float]] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (candidates[:, 0] >= left) & (
            candidates[:, 0] < right if index < len(edges) - 2 else candidates[:, 0] <= right
        )
        values = candidates[selected]
        if len(values) < cfg.ground_line_min_candidates_per_bin:
            continue
        envelope.append(
            (
                float(np.median(values[:, 0])),
                float(np.percentile(values[:, 1], cfg.ground_line_candidate_quantile)),
            )
        )
    if not envelope:
        envelope.append(
            (
                float(np.median(candidates[:, 0])),
                float(np.percentile(candidates[:, 1], cfg.ground_line_candidate_quantile)),
            )
        )
    return np.asarray(envelope, dtype=np.float64), len(candidates)


def fit_global_ground_line(
    frames: list[ShoeFrame], cfg: dromia_config.GaitAnalysisConfig
) -> GroundLine:
    """Fit one robust spatial line to the global lower shoe envelope.

    Image ``y`` grows downwards, so contact candidates live near the upper
    quantile of outsole-bottom coordinates. Equal-width spatial bins prevent a
    long or densely sampled stance from dominating the fit. A non-horizontal
    slope is accepted only when the binned envelope supports it strongly.
    """

    candidates = np.asarray(
        [
            (float(item.ground_x), float(item.max_y))
            for item in frames
            if item.ground_x is not None
            and item.max_y is not None
            and np.isfinite(item.ground_x)
            and np.isfinite(item.max_y)
        ],
        dtype=np.float64,
    )
    if not len(candidates):
        return GroundLine(0.0, 0.0, 0.0, 0, 0, "not_available")
    x_min, x_max = float(np.min(candidates[:, 0])), float(np.max(candidates[:, 0]))
    if x_max - x_min <= EPS:
        intercept = float(np.percentile(candidates[:, 1], cfg.ground_line_candidate_quantile))
        return GroundLine(0.0, intercept, 1.0, len(candidates), 1, "horizontal_fallback")

    edges = np.linspace(x_min, x_max, cfg.ground_line_bin_count + 1)
    envelope: list[tuple[float, float]] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (candidates[:, 0] >= left) & (
            candidates[:, 0] < right if index < len(edges) - 2 else candidates[:, 0] <= right
        )
        values = candidates[selected]
        if len(values) < cfg.ground_line_min_candidates_per_bin:
            continue
        envelope.append(
            (
                float(np.median(values[:, 0])),
                float(np.percentile(values[:, 1], cfg.ground_line_candidate_quantile)),
            )
        )
    if len(envelope) < 2:
        intercept = float(np.percentile(candidates[:, 1], cfg.ground_line_candidate_quantile))
        return GroundLine(
            0.0,
            intercept,
            0.0,
            len(candidates),
            len(envelope),
            "horizontal_fallback",
        )

    envelope_xy = np.asarray(envelope, dtype=np.float64)
    slope, intercept = np.polyfit(envelope_xy[:, 0], envelope_xy[:, 1], 1)
    predicted = slope * envelope_xy[:, 0] + intercept
    residual_sum = float(np.sum((envelope_xy[:, 1] - predicted) ** 2))
    total_sum = float(np.sum((envelope_xy[:, 1] - np.mean(envelope_xy[:, 1])) ** 2))
    r2 = 1.0 if total_sum <= EPS else max(0.0, 1.0 - residual_sum / total_sum)
    total_rise = abs(float(slope)) * (x_max - x_min)
    if r2 < cfg.ground_line_min_r2 or total_rise < cfg.ground_line_min_total_rise_px:
        return GroundLine(
            0.0,
            float(np.median(envelope_xy[:, 1])),
            r2,
            len(candidates),
            len(envelope),
            "horizontal_fallback",
        )
    return GroundLine(
        float(slope),
        float(intercept),
        r2,
        len(candidates),
        len(envelope),
        "robust_spatial_line",
    )


def fit_global_ground_line_from_assignments(
    assignments: list[dromia_dto.ShoeAssignment], cfg: dromia_config.GaitAnalysisConfig
) -> GroundLine:
    """Fit the video-level floor without depending on a runner task's crop."""

    candidates: list[ShoeFrame] = []
    for item in assignments:
        curve = lower_curve(item.mask) if item.mask is not None else np.empty((0, 2), np.float32)
        if not len(curve):
            continue
        max_y = float(np.max(curve[:, 1]))
        bottom = curve[curve[:, 1] >= max_y - 1.0]
        candidates.append(
            ShoeFrame(
                frame_idx=int(item.frame_idx),
                runner_id=int(item.runner_id),
                side=item.side,
                score=float(item.score),
                curve=[],
                max_y=max_y,
                mask_observed=True,
                ground_x=float(np.median(bottom[:, 0])),
            )
        )
    return fit_global_ground_line(candidates, cfg)


def assign_local_step_grounds(
    foot: list[ShoeFrame],
    *,
    bboxes: np.ndarray,
    frame_pos: dict[int, int],
    obj_idx: int,
    fps: float,
    cfg: dromia_config.GaitAnalysisConfig,
) -> list[tuple[int, float]]:
    """Interpolate a per-foot ground path through robust step-level anchor points."""

    ordered = sorted(
        (item for item in foot if item.max_y is not None), key=lambda item: item.frame_idx
    )
    if not ordered:
        return []
    max_gap = max(0, round(cfg.ground_step_max_interpolation_gap_seconds * fps))
    blocks: list[list[ShoeFrame]] = []
    block: list[ShoeFrame] = []
    for item in ordered:
        if block and item.frame_idx - block[-1].frame_idx - 1 > max_gap:
            blocks.append(block)
            block = []
        block.append(item)
    if block:
        blocks.append(block)

    next_step_index = 0
    all_anchors: list[tuple[int, float]] = []
    for visible in blocks:
        observed_frames = np.asarray([item.frame_idx for item in visible], dtype=np.int32)
        observed_y = np.asarray([item.max_y for item in visible], dtype=np.float64)
        dense_frames = np.arange(observed_frames[0], observed_frames[-1] + 1, dtype=np.int32)
        dense_y = np.interp(dense_frames, observed_frames, observed_y)
        smoothing = max(1, round(cfg.ground_step_smoothing_seconds * fps))
        if smoothing % 2 == 0:
            smoothing += 1
        smoothed_y = median_filter(dense_y, size=smoothing, mode="nearest")
        heights = []
        for item in visible:
            bbox = bboxes[frame_pos[item.frame_idx], obj_idx]
            height = float(bbox[3] - bbox[1]) if np.isfinite(bbox).all() else np.nan
            if np.isfinite(height) and height > 0:
                heights.append(height)
        median_height = float(np.median(heights)) if heights else 1.0
        minima, _ = find_peaks(
            -smoothed_y,
            distance=max(1, round(cfg.ground_step_min_separation_seconds * fps)),
            prominence=cfg.ground_step_min_prominence_bbox_fraction * median_height,
        )
        boundaries = [0, *minima.tolist(), len(dense_frames)]
        block_anchors: list[tuple[int, float]] = []
        for start, stop in zip(boundaries, boundaries[1:], strict=False):
            if stop <= start:
                continue
            first_frame = int(dense_frames[start])
            last_frame = int(dense_frames[stop - 1])
            members = [item for item in visible if first_frame <= item.frame_idx <= last_frame]
            if not members:
                continue
            ground = float(
                np.percentile([float(item.max_y) for item in members], cfg.ground_percentile)
            )
            closest_distance = min(abs(float(item.max_y) - ground) for item in members)
            closest_frames = [
                item.frame_idx
                for item in members
                if abs(float(item.max_y) - ground) <= closest_distance + EPS
            ]
            anchor_frame = int(round(float(np.median(closest_frames))))
            block_anchors.append((anchor_frame, ground))
            for item in members:
                item.ground_step_index = next_step_index
            next_step_index += 1
        if block_anchors:
            anchor_frames = np.asarray([frame for frame, _ in block_anchors], dtype=np.float64)
            anchor_y = np.asarray([ground for _, ground in block_anchors], dtype=np.float64)
            for item in visible:
                item.ground_y = float(np.interp(item.frame_idx, anchor_frames, anchor_y))
            all_anchors.extend(block_anchors)
    return all_anchors


def stabilize_contacts(
    frames: list[ShoeFrame],
    cfg: dromia_config.GaitAnalysisConfig,
    *,
    fps: float = 30.0,
) -> None:
    """Close tiny missing-mask gaps and remove sub-physiological contact blips."""
    max_gap = max(
        cfg.max_contact_gap_frames,
        round(cfg.max_contact_gap_seconds * fps),
    )
    min_contact = max(
        cfg.min_contact_frames,
        round(cfg.min_contact_seconds * fps),
    )
    tracks: dict[tuple[int, str], list[ShoeFrame]] = {}
    for item in frames:
        tracks.setdefault((item.runner_id, item.side), []).append(item)
    for track in tracks.values():
        ordered = sorted(track, key=lambda x: x.frame_idx)
        contact_indices = [index for index, item in enumerate(ordered) if item.contact is True]
        for first, second in zip(contact_indices[:-1], contact_indices[1:], strict=False):
            frame_gap = ordered[second].frame_idx - ordered[first].frame_idx - 1
            if frame_gap <= max_gap:
                for item in ordered[first + 1 : second]:
                    item.contact = True
                    if item.contact_source != "direct":
                        item.contact_source = "short_gap_bridge"

    # A foot cannot complete another full gait cycle before two alternating
    # steps have elapsed. Join same-foot fragments inside that interval only
    # when the opposite foot has no contact between them. This repairs longer
    # sole-mask dropouts without bridging a real flight or the next stance.
    minimum_same_foot_cycle = round(120.0 / cfg.cadence_max_spm * fps)
    contact_snapshot = {
        key: {item.frame_idx for item in track if item.contact is True}
        for key, track in tracks.items()
    }
    bridges: list[list[ShoeFrame]] = []
    for (runner_id, side), track in tracks.items():
        ordered = sorted(track, key=lambda x: x.frame_idx)
        intervals = boolean_intervals([item.contact for item in ordered])
        opposite = contact_snapshot.get((runner_id, "right" if side == "left" else "left"), set())
        for (first_start, first_end), (second_start, _second_end) in zip(
            intervals[:-1], intervals[1:], strict=False
        ):
            cycle_frames = ordered[second_start].frame_idx - ordered[first_start].frame_idx
            gap_start = ordered[first_end].frame_idx + 1
            gap_end = ordered[second_start].frame_idx
            has_opposite_contact = any(gap_start <= frame < gap_end for frame in opposite)
            if cycle_frames < minimum_same_foot_cycle and not has_opposite_contact:
                bridges.append(ordered[first_end + 1 : second_start])
    for bridge in bridges:
        for item in bridge:
            item.contact = True
            if item.contact_source != "direct":
                item.contact_source = "same_foot_bridge"

    for track in tracks.values():
        ordered = sorted(track, key=lambda x: x.frame_idx)
        groups = contiguous_contact_groups(ordered, max_gap)
        for group in groups:
            if len(group) < min_contact:
                for item in group:
                    item.contact = False if item.mask_observed else None
                    item.contact_source = (
                        "observed_non_contact" if item.mask_observed else "unknown"
                    )


def refine_contacts_from_shoe_kinematics(
    frames: list[ShoeFrame],
    cfg: dromia_config.GaitAnalysisConfig,
    *,
    fps: float = 30.0,
) -> None:
    """Refine every stance boundary from the same temporal shoe-motion rules.

    The floor test supplies a conservative contact core. A local maximum of the
    shoe's bottom-most point immediately before that core marks initial impact;
    contact starts on the following frame as the shoe begins to load. At the other
    boundary, a persistent rise of the whole lower shoe curve marks unloading even
    when one stale mask pixel remains on the floor. All thresholds are normalized by
    runner height and are applied identically to every runner and every shoe.
    """

    tracks: dict[tuple[int, str], list[ShoeFrame]] = {}
    for item in frames:
        tracks.setdefault((item.runner_id, item.side), []).append(item)

    search_frames = max(1, round(cfg.contact_refine_window_seconds * fps))
    minimum_contact_frames = max(1, round(cfg.min_event_contact_seconds * fps))
    persistence_frames = cfg.contact_refine_persistence_frames

    for track in tracks.values():
        ordered = sorted(track, key=lambda item: item.frame_idx)
        intervals = boolean_intervals([item.contact is True for item in ordered])
        heights = [
            float(item.bbox_height)
            for item in ordered
            if item.bbox_height is not None and item.bbox_height > 0.0
        ]
        if not intervals or not heights:
            continue
        median_height = float(np.median(heights))
        turning_prominence = cfg.contact_refine_turning_prominence_bbox_fraction * median_height
        takeoff_lift = cfg.contact_refine_takeoff_lift_bbox_fraction * median_height

        for core_start, core_end in intervals:
            core_start_frame = ordered[core_start].frame_idx
            landing_observations = [
                (index, item)
                for index, item in enumerate(ordered)
                if core_start_frame - search_frames <= item.frame_idx <= core_start_frame
                and item.max_y is not None
            ]
            onset_frame = core_start_frame
            if len(landing_observations) >= persistence_frames + 1:
                onset_frame = next(
                    (
                        item.frame_idx
                        for _, item in landing_observations
                        if item.clearance_px is not None
                        and item.bbox_height is not None
                        and item.clearance_px
                        <= cfg.contact_refine_landing_clearance_bbox_fraction * item.bbox_height
                    ),
                    core_start_frame,
                )
            if len(landing_observations) >= 3:
                bottom_y = np.asarray(
                    [float(item.max_y) for _, item in landing_observations],
                    dtype=np.float64,
                )
                peaks, _ = find_peaks(
                    bottom_y,
                    prominence=turning_prominence,
                    distance=2,
                )
                qualifying_peaks = [
                    int(peak)
                    for peak in peaks
                    if landing_observations[int(peak)][1].frame_idx < core_start_frame
                    and landing_observations[int(peak)][1].frame_idx
                    - landing_observations[int(peak) - 1][1].frame_idx
                    == 1
                    and landing_observations[int(peak) + 1][1].frame_idx
                    - landing_observations[int(peak)][1].frame_idx
                    == 1
                ]
                if qualifying_peaks:
                    peak_index = qualifying_peaks[-1]
                    onset_frame = landing_observations[peak_index][1].frame_idx + 1
            for item in ordered:
                if onset_frame <= item.frame_idx < core_start_frame:
                    item.contact = True
                    item.contact_source = "kinematic_refinement"

            stance = ordered[core_start : core_end + 1]
            bottom_y = np.asarray(
                [np.nan if item.max_y is None else float(item.max_y) for item in stance],
                dtype=np.float64,
            )
            if np.count_nonzero(np.isfinite(bottom_y)) < persistence_frames:
                continue
            peak_offset = int(np.nanargmax(bottom_y))
            stance_floor_y = float(bottom_y[peak_offset])
            earliest_offset = max(peak_offset, minimum_contact_frames - 1)
            takeoff_offset: int | None = None
            for offset in range(earliest_offset, len(stance)):
                stop = offset + persistence_frames
                if stop > len(stance):
                    break
                window = bottom_y[offset:stop]
                if not np.isfinite(window).all():
                    continue
                lift = stance_floor_y - window
                if lift[0] >= takeoff_lift and np.all(lift >= 0.8 * takeoff_lift):
                    takeoff_offset = offset
                    break
            if takeoff_offset is None:
                continue
            for item in stance[takeoff_offset + 1 :]:
                item.contact = False if item.mask_observed else None
                item.contact_source = "kinematic_refinement" if item.mask_observed else "unknown"


def stabilize_foot_axes(frames: list[ShoeFrame], cfg: dromia_config.GaitAnalysisConfig) -> None:
    """Withhold outsole axes that lack local temporal support or are isolated outliers."""

    tracks: dict[tuple[int, str], list[ShoeFrame]] = {}
    for item in frames:
        tracks.setdefault((item.runner_id, item.side), []).append(item)
    for track in tracks.values():
        ordered = sorted(track, key=lambda item: item.frame_idx)
        angles = [
            None
            if item.foot_axis_xy is None or item.foot_axis_quality != "valid"
            else float(np.arctan2(item.foot_axis_xy[1], item.foot_axis_xy[0]))
            for item in ordered
        ]
        decisions: list[tuple[ShoeFrame, str]] = []
        for index, (item, angle) in enumerate(zip(ordered, angles, strict=True)):
            if angle is None:
                continue
            start = max(0, index - cfg.foot_axis_temporal_window_frames)
            end = min(len(ordered), index + cfg.foot_axis_temporal_window_frames + 1)
            local = [value for value in angles[start:end] if value is not None]
            if len(local) < cfg.foot_axis_min_temporal_samples:
                decisions.append((item, "insufficient_temporal_stability"))
                continue
            center = float(np.median(np.unwrap(np.asarray(local, np.float64))))
            difference = abs(
                float(np.degrees(np.arctan2(np.sin(angle - center), np.cos(angle - center))))
            )
            if difference > cfg.foot_axis_max_temporal_delta_deg:
                decisions.append((item, "temporally_unstable"))
        for item, quality in decisions:
            item.foot_axis_xy = None
            item.foot_axis_quality = quality


def build_contact_events(
    runner_id: int,
    frames: np.ndarray,
    lookup: dict[tuple[int, int, str], ShoeFrame],
    *,
    timing: dromia_timebase.VideoTimebase,
    direction: str,
    direction_source: str = "pose_displacement",
    cfg: dromia_config.GaitAnalysisConfig,
) -> list[ContactEvent]:
    events: list[ContactEvent] = []
    for side in SIDES:
        observations = [lookup.get((int(frame), runner_id, side)) for frame in frames]
        states = [None if item is None else item.contact for item in observations]
        contacts = [state is True for state in states]
        intervals = boolean_intervals(contacts)
        for event_idx, (start, end) in enumerate(intervals):
            landing = observations[start]
            assert landing is not None
            landing_observed = (
                start > 0
                and observations[start - 1] is not None
                and observations[start - 1].mask_observed is True
                and states[start - 1] is False
            )
            takeoff_observed = (
                end < len(frames) - 1
                and observations[end + 1] is not None
                and observations[end + 1].mask_observed is True
                and states[end + 1] is False
            )
            strike, position = (
                classify_strike(landing, direction)
                if landing_observed and (landing.mask_observed or len(landing.curve) >= 2)
                else ("not_observed", None)
            )
            next_start = intervals[event_idx + 1][0] if event_idx + 1 < len(intervals) else None
            flight_is_observed = next_start is not None and all(
                state is False for state in states[end + 1 : next_start]
            )
            same_foot_flight_frames = (
                max(next_start - end - 1, 0)
                if flight_is_observed and next_start is not None
                else None
            )
            count = end - start + 1
            event_items = [item for item in observations[start : end + 1] if item is not None]
            inferred_count = sum(item.contact_source != "direct" for item in event_items)
            contact_sources = tuple(sorted({item.contact_source for item in event_items}))
            inferred_fraction = inferred_count / count if count else 0.0
            contact_time = (
                dromia_timebase.interval_duration_s(timing, int(frames[start]), int(frames[end]))
                if landing_observed and takeoff_observed
                else None
            )
            flight_time = (
                None
                if not flight_is_observed
                or next_start is None
                or same_foot_flight_frames is None
                or same_foot_flight_frames == 0
                else dromia_timebase.interval_duration_s(
                    timing, int(frames[end + 1]), int(frames[next_start - 1])
                )
            )
            events.append(
                ContactEvent(
                    runner_id=runner_id,
                    side=side,
                    landing_frame=int(frames[start]) if landing_observed else None,
                    takeoff_frame=int(frames[end]) if takeoff_observed else None,
                    first_flight_frame=int(frames[end + 1]) if end + 1 < len(frames) else None,
                    contact_frames=(int(frames[end]) - int(frames[start]) + 1)
                    if landing_observed and takeoff_observed
                    else count,
                    contact_time_s=contact_time,
                    contact_time_ms=to_ms(contact_time),
                    same_foot_flight_frames=same_foot_flight_frames,
                    same_foot_flight_time_s=flight_time,
                    same_foot_flight_time_ms=to_ms(flight_time),
                    strike_type=strike,
                    strike_position=position,
                    confidence=float(
                        np.clip(0.6 * landing.score + 0.4 * landing.contact_fraction, 0, 1)
                    ),
                    strike_contact_fraction=float(landing.contact_fraction),
                    strike_shoe_score=float(landing.score),
                    strike_direction_source=direction_source,
                    inferred_contact_frames=inferred_count,
                    inferred_contact_fraction=float(inferred_fraction),
                    contact_sources=contact_sources,
                    endpoint_quality=(
                        "observed"
                        if landing_observed and takeoff_observed
                        else "censored_unknown_or_boundary"
                    ),
                )
            )
    return sorted(
        events,
        key=lambda x: (
            x.landing_frame if x.landing_frame is not None else -1,
            x.side,
        ),
    )


def frame_metrics(
    frame_idx: int,
    runner_id: int,
    points: np.ndarray,
    bbox: np.ndarray,
    shoes: dict[tuple[int, int, str], ShoeFrame],
    direction: str,
    cfg: dromia_config.GaitAnalysisConfig,
) -> dict[str, Any]:
    row: dict[str, Any] = {"frame_idx": frame_idx, "runner_id": runner_id}
    bbox_height = max(float(bbox[3] - bbox[1]), 1.0) if np.isfinite(bbox).all() else None
    shoulders = midpoint(points[5], points[6])
    hips = midpoint(points[11], points[12])
    torso = shoulders - hips if shoulders is not None and hips is not None else None
    signed_lean = torso_lean_deg(torso, direction)
    row["torso_lean_deg"] = signed_lean
    row["torso_posture"] = posture(signed_lean, cfg.torso_straight_tolerance_deg)
    for side in SIDES:
        hip_id, knee_id, ankle_id = SIDE_JOINTS[side]
        hip, knee, ankle = points[[hip_id, knee_id, ankle_id]]
        shoe = shoes.get((frame_idx, runner_id, side))
        tibia = knee - ankle if finite(ankle, knee) else None
        # Keep missing segmentation distinct from a confidently airborne foot.
        row[f"{side}_contact"] = None if shoe is None else shoe.contact
        row[f"{side}_mask_observed"] = bool(shoe is not None and shoe.mask_observed)
        row[f"{side}_contact_source"] = "unknown" if shoe is None else shoe.contact_source
        row[f"{side}_shoe_score"] = None if shoe is None else shoe.score
        row[f"{side}_shoe_clearance_px"] = None if shoe is None else shoe.clearance_px
        row[f"{side}_ground_y"] = None if shoe is None else shoe.ground_y
        row[f"{side}_ground_step_index"] = None if shoe is None else shoe.ground_step_index
        row[f"{side}_contact_point_xy"] = None if shoe is None else shoe.contact_point_xy
        foot_axis = (
            None if shoe is None or shoe.foot_axis_xy is None else np.asarray(shoe.foot_axis_xy)
        )
        row[f"{side}_foot_tibia_angle_deg"] = vector_angle_deg(foot_axis, tibia)
        row[f"{side}_foot_axis_rmse_px"] = None if shoe is None else shoe.foot_axis_rmse_px
        row[f"{side}_foot_axis_quality"] = (
            "not_available" if shoe is None else shoe.foot_axis_quality
        )
        row[f"{side}_tibia_horizontal_angle_deg"] = axis_angle_deg(tibia, np.asarray([1.0, 0.0]))
        row[f"{side}_knee_angle_deg"] = joint_angle_deg(hip, knee, ankle)
    left, right = row["left_contact"], row["right_contact"]
    row["global_contact"] = (
        True
        if left is True or right is True
        else (False if left is False and right is False else None)
    )
    row["knee_horizontal_separation_px"] = (
        abs(float(points[13, 0] - points[14, 0])) if finite(points[13], points[14]) else None
    )
    row["bbox_height_px"] = bbox_height
    return row


def apply_activity_window(
    rows: list[dict[str, Any]], events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Restrict aggregate timing to the reliably observed gait sequence."""

    landings = [int(x["landing_frame"]) for x in events if x.get("landing_frame") is not None]
    takeoffs = [int(x["takeoff_frame"]) for x in events if x.get("takeoff_frame") is not None]
    source = "complete_contact_endpoints"
    if landings and takeoffs and min(landings) <= max(takeoffs):
        start_frame, end_frame = min(landings), max(takeoffs)
    else:
        observed = [
            int(row["frame_idx"])
            for row in rows
            if row.get("left_mask_observed") or row.get("right_mask_observed")
        ]
        source = "observed_shoe_extent" if observed else "not_available"
        start_frame = min(observed) if observed else None
        end_frame = max(observed) if observed else None
    for row in rows:
        frame = int(row["frame_idx"])
        row["activity_window_included"] = bool(
            start_frame is not None and end_frame is not None and start_frame <= frame <= end_frame
        )
    included = sum(bool(row["activity_window_included"]) for row in rows)
    return {
        "source": source,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "included_frame_count": included,
        "excluded_frame_count": len(rows) - included,
    }


def global_flight_intervals_from_rows(
    rows: list[dict[str, Any]], timing: dromia_timebase.VideoTimebase
) -> list[dict[str, Any]]:
    """Return observed airborne intervals, censoring unknown/activity boundaries."""

    active = [row for row in rows if row.get("activity_window_included", True)]
    states: list[bool | None] = []
    for row in active:
        left, right = row.get("left_contact"), row.get("right_contact")
        if left is False and right is False:
            states.append(True)
        elif left is True or right is True:
            states.append(False)
        else:
            states.append(None)
    frames = [int(row["frame_idx"]) for row in active]
    intervals = []
    for start, end in boolean_intervals([state is True for state in states]):
        censored_start = start == 0 or states[start - 1] is None
        censored_end = end == len(states) - 1 or states[end + 1] is None
        missing_frames = any(
            b != a + 1
            for a, b in zip(
                frames[max(0, start - 1) : end + 1],
                frames[max(0, start - 1) + 1 : end + 2],
                strict=False,
            )
        )
        censored_end = censored_end or missing_frames
        duration = dromia_timebase.interval_duration_s(timing, frames[start], frames[end])
        intervals.append(
            {
                "start_frame": frames[start],
                "end_frame": frames[end],
                "duration_frames": end - start + 1,
                "duration_s": duration,
                "duration_ms": to_ms(duration),
                "preceding_takeoff_frame": frames[start - 1] if start > 0 else None,
                "complete": not censored_start and not censored_end,
                "censored_start": censored_start,
                "censored_end": censored_end,
            }
        )
    return intervals


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


def analyze_run_pose(
    run_dir: Path,
    pose_path: Path,
    *,
    runner_id: int | None = None,
) -> dict[str, str]:
    """Rerun gait analysis from an automatic or CVAT-reviewed pose artifact."""

    run = run_dir.resolve()
    manifest = dromia_dto.RunManifest.model_validate_json((run / "manifest.json").read_text())
    base = np.load(run / manifest.artifacts["pose_npz"])
    pose_data = np.load(pose_path)
    pose_key = next(
        (
            key
            for key in (
                "reviewed_keypoints_xy",
                "first_pass_keypoints_xy",
                "temporal_keypoints_xy",
                "posterior_keypoints_xy",
                "pose_xy",
            )
            if key in pose_data
        ),
        None,
    )
    if pose_key is None:
        raise ValueError(
            "Pose NPZ needs reviewed_keypoints_xy, first_pass_keypoints_xy, temporal_keypoints_xy, "
            "posterior_keypoints_xy, or pose_xy"
        )
    frames = np.asarray(pose_data.get("frame_indices", base["frame_indices"]), dtype=np.int32)
    ids = np.asarray(pose_data.get("object_ids", base["object_ids"]), dtype=np.int32)
    pose = np.asarray(pose_data[pose_key], dtype=np.float32)
    base_frames = {int(value): idx for idx, value in enumerate(base["frame_indices"])}
    base_ids = {int(value): idx for idx, value in enumerate(base["object_ids"])}
    frame_select = [base_frames[int(value)] for value in frames]
    id_select = [base_ids[int(value)] for value in ids]
    bboxes = base["bboxes_xyxy"][np.ix_(frame_select, id_select)]
    if runner_id is not None:
        matches = np.where(ids == runner_id)[0]
        if not len(matches):
            raise ValueError(f"Runner {runner_id} is not present in {pose_path}")
        pose = pose[:, matches]
        bboxes = bboxes[:, matches]
        ids = ids[matches]
    cfg = dromia_config.DromiaConfig.model_validate_json(
        (run / "config.json").read_text(encoding="utf-8")
    ).gait_analysis
    assignments = load_assignments_from_cache(run, frames, ids)
    timing = dromia_timebase.inspect_video(
        run / manifest.source_video.path,
        frames,
        fps_override=cfg.fps_override,
        capture_fps_override=cfg.capture_fps_override,
    )
    (run / "timebase.json").write_text(
        json.dumps(timing.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    calibration = dromia_calibration.load_ground_calibration(run)
    if calibration is not None and calibration.video_sha256 != timing.video_sha256:
        calibration = calibration.model_copy(
            update={"valid": False, "invalid_reason": "video_fingerprint_mismatch"}
        )
    payload = analyze_gait(
        frame_indices=frames,
        object_ids=ids,
        pose_xy=pose,
        bboxes_xyxy=bboxes,
        shoe_assignments=assignments,
        fps=timing.source_fps,
        cfg=cfg,
        source_pose=pose_path.name,
        fps_is_assumed=False,
        timebase=timing,
        calibration=calibration,
    )
    suffix = "" if runner_id is None else f"_runner_{runner_id}"
    return write_gait_artifacts(
        payload,
        run_dir=run,
        video_path=run / manifest.source_video.path,
        frame_indices=frames,
        object_ids=ids,
        pose_xy=pose,
        shoe_assignments=assignments,
        fps=min(timing.source_fps, 30.0),
        draw_video=cfg.draw_debug_video,
        suffix=suffix,
    )


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
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    masks = {(x.frame_idx, x.runner_id, x.side): x.mask for x in assignments if x.mask is not None}
    rows = {
        (int(rid), int(row["frame_idx"])): row
        for rid, runner in ((int(k), v) for k, v in payload["runners"].items())
        for row in runner["frames"]
    }
    events = {
        (int(k), e["landing_frame"]): f"LANDING {e['side'][0].upper()} {e['strike_type']}"
        for k, v in payload["runners"].items()
        for e in v["events"]
        if e["landing_frame"] is not None
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
                alignment_key = (int(key), event["knee_alignment_frame"])
                alignment_label = f"KNEE ALIGNMENT {event['side'][0].upper()}"
                current = events.get(alignment_key)
                events[alignment_key] = (
                    alignment_label if current is None else f"{current} / {alignment_label}"
                )
    try:
        for t, frame_idx_raw in enumerate(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx_raw))
            ok, image = capture.read()
            if not ok:
                continue
            frame_idx = int(frame_idx_raw)
            if ground_slope is not None and ground_intercept is not None:
                start_y = round(float(ground_intercept))
                end_y = round(float(ground_slope) * (width - 1) + float(ground_intercept))
                cv2.line(image, (0, start_y), (width - 1, end_y), (255, 255, 0), 3)
            for o, rid_raw in enumerate(object_ids):
                rid = int(rid_raw)
                row = rows[(rid, frame_idx)]
                if ground_slope is None or ground_intercept is None:
                    runner_ground = payload["runners"][str(rid)]
                    lines_by_side = runner_ground.get("ground_lines", {})
                    ground_by_side = runner_ground["ground_y_by_side"]
                    for side, color in (("left", (0, 220, 255)), ("right", (255, 160, 20))):
                        line = lines_by_side.get(side, {})
                        if line.get("slope") is not None and line.get("intercept") is not None:
                            start_y = round(float(line["intercept"]))
                            end_y = round(
                                float(line["slope"]) * (width - 1) + float(line["intercept"])
                            )
                            cv2.line(image, (0, start_y), (width - 1, end_y), color, 2)
                        else:
                            ground = row.get(f"{side}_ground_y")
                            if ground is None:
                                ground = ground_by_side.get(side)
                            if ground is None:
                                continue
                            cv2.line(
                                image,
                                (0, round(ground)),
                                (width - 1, round(ground)),
                                color,
                                2,
                            )
                for side, color in (("left", (0, 220, 255)), ("right", (255, 160, 20))):
                    mask = masks.get((frame_idx, rid, side))
                    if mask is not None and mask.shape == image.shape[:2]:
                        image[np.asarray(mask) > 0] = color
                draw_pose_geometry(image, pose[t, o])
                pose_x = (
                    int(np.nanmin(pose[t, o, :, 0])) if np.isfinite(pose[t, o, :, 0]).any() else 10
                )
                x = min(max(pose_x, 5), max(width - 720, 5))
                y = int(np.nanmin(pose[t, o, :, 1])) if np.isfinite(pose[t, o, :, 1]).any() else 30
                label = (
                    events.get((rid, frame_idx), "")
                    if row.get("measurement_window_included", True)
                    else "OUTSIDE CALIBRATION WINDOW"
                )
                lines = [
                    f"runner {rid} {label}",
                    format_side_overlay("L", "left", row),
                    format_side_overlay("R", "right", row),
                    f"torso={row['torso_posture']} {fmt(row['torso_lean_deg'])}",
                ]
                text_y = max(min(y - 12, height - 75), 18)
                panel = image.copy()
                cv2.rectangle(panel, (x - 5, text_y - 15), (width - 5, text_y + 65), (0, 0, 0), -1)
                cv2.addWeighted(panel, 0.55, image, 0.45, 0, image)
                for i, text in enumerate(lines):
                    cv2.putText(
                        image,
                        text,
                        (x, text_y + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            writer.write(image)
    finally:
        capture.release()
        writer.release()


def draw_pose_geometry(image: np.ndarray, p: np.ndarray) -> None:
    for a, b in ((5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 12)):
        if finite(p[a], p[b]):
            cv2.line(
                image,
                tuple(np.rint(p[a]).astype(int)),
                tuple(np.rint(p[b]).astype(int)),
                (60, 255, 60),
                2,
            )


def lower_curve(mask: np.ndarray | None, max_points: int = 80) -> np.ndarray:
    if mask is None:
        return np.empty((0, 2), np.float32)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 1:
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == component).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if not len(xs):
        return np.empty((0, 2), np.float32)
    curve = np.asarray([(x, np.max(ys[xs == x])) for x in np.unique(xs)], np.float32)
    return (
        curve[np.linspace(0, len(curve) - 1, max_points).astype(int)]
        if len(curve) > max_points
        else curve
    )


def classify_strike(item: ShoeFrame, direction: str) -> tuple[str, float | None]:
    if item.contact_position is not None:
        position = item.contact_position
        return strike_label(position), position
    curve = np.asarray(item.curve, np.float32)
    if len(curve) < 2 or item.ground_y is None:
        return "not_observed", None
    tolerance = max((item.clearance_px or 0.0) + 1.5, 1.5)
    contact = curve[curve[:, 1] >= item.ground_y - tolerance]
    if not len(contact):
        return "not_observed", None
    xmin, xmax = float(curve[:, 0].min()), float(curve[:, 0].max())
    x = float(contact[:, 0].mean())
    position = (x - xmin) / max(xmax - xmin, 1.0)
    if direction == "left":
        position = 1.0 - position
    return strike_label(position), position


def contact_region(
    curve: np.ndarray,
    ground_y: float,
    tolerance: float,
    direction: str,
) -> tuple[float | None, float]:
    """Return travel-oriented shoe position and fraction of the near-ground curve."""

    if len(curve) < 2:
        return None, 0.0
    contact = curve[curve[:, 1] >= ground_y - tolerance]
    fraction = float(len(contact) / len(curve))
    if not len(contact):
        return None, fraction
    xmin, xmax = float(curve[:, 0].min()), float(curve[:, 0].max())
    position = (float(contact[:, 0].mean()) - xmin) / max(xmax - xmin, 1.0)
    if direction == "left":
        position = 1.0 - position
    return position, fraction


def strike_label(position: float) -> str:
    # TODO: Replace thirds with thresholds calibrated against labelled shoe-mask contacts.
    return "heel" if position <= 0.34 else "forefoot" if position >= 0.66 else "midfoot"


def boolean_intervals(values: list[bool]) -> list[tuple[int, int]]:
    output = []
    start = None
    for i, value in enumerate([*values, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            output.append((start, i - 1))
            start = None
    return output


def contiguous_contact_groups(track: list[ShoeFrame], max_gap: int) -> list[list[ShoeFrame]]:
    groups = []
    current = []
    for item in track:
        if not item.contact:
            continue
        if current and item.frame_idx - current[-1].frame_idx > max_gap + 1:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    return groups


def runner_direction(pose: np.ndarray) -> str:
    left = pose[:, 11, 0]
    right = pose[:, 12, 0]
    hip = np.where(np.isfinite(left) & np.isfinite(right), (left + right) * 0.5, np.nan)
    finite_values = hip[np.isfinite(hip)]
    # TODO: Use a calibrated facing-direction classifier for nearly stationary/cropped runners.
    return "right" if len(finite_values) < 2 or finite_values[-1] >= finite_values[0] else "left"


def runner_ground_y_by_side(runner_id: int, frames: list[ShoeFrame]) -> dict[str, float | None]:
    return {
        side: (
            float(np.median(values))
            if (
                values := [
                    item.ground_y
                    for item in frames
                    if item.runner_id == runner_id
                    and item.side == side
                    and item.ground_y is not None
                ]
            )
            else None
        )
        for side in SIDES
    }


def runner_ground_lines(
    runner_id: int, frames: list[ShoeFrame]
) -> dict[str, dict[str, float | int | str | None]]:
    output: dict[str, dict[str, float | int | str | None]] = {}
    for side in SIDES:
        fitted = next(
            (
                item
                for item in frames
                if item.runner_id == runner_id
                and item.side == side
                and item.ground_line_slope is not None
                and item.ground_line_intercept is not None
            ),
            None,
        )
        if fitted is None:
            continue
        output[side] = {
            "slope": fitted.ground_line_slope,
            "angle_deg": float(np.degrees(np.arctan(fitted.ground_line_slope))),
            "intercept": fitted.ground_line_intercept,
            "r2": fitted.ground_line_r2,
            "quality": fitted.ground_line_quality,
            "candidate_count": fitted.ground_line_candidate_count,
            "bin_count": fitted.ground_line_bin_count,
        }
    return output


def add_stance_phases(
    event: dict[str, Any],
    rows: list[dict[str, Any]],
    timing: dromia_timebase.VideoTimebase,
) -> None:
    landing = event.get("landing_frame")
    takeoff = event.get("takeoff_frame")
    event.update(
        {
            "knee_alignment_frame": None,
            "braking_time_s": None,
            "braking_time_ms": None,
            "flexion_braking_time_ms": None,
            "impulse_propulsion_time_ms": None,
            "propulsion_time_s": None,
            "propulsion_time_ms": None,
            "knee_alignment_separation_norm": None,
            "phase_quality": "not_available",
        }
    )
    if landing is None or takeoff is None:
        event["phase_quality"] = "censored_contact_endpoint"
        return
    candidates = []
    for row in rows:
        frame = int(row["frame_idx"])
        separation = row.get("knee_horizontal_separation_px")
        height = row.get("bbox_height_px")
        if landing <= frame <= takeoff and separation is not None and height:
            candidates.append((float(separation) / float(height), frame))
    if not candidates:
        event["phase_quality"] = "pose_not_available"
        return
    separation, frame = min(candidates)
    if frame <= int(landing) or frame >= int(takeoff):
        event["phase_quality"] = "endpoint_alignment_withheld"
        event["knee_alignment_frame"] = frame
        event["knee_alignment_separation_norm"] = separation
        return
    braking = dromia_timebase.elapsed_s(timing, int(landing), frame)
    propulsion = dromia_timebase.interval_duration_s(timing, frame, int(takeoff))
    event.update(
        {
            "knee_alignment_frame": frame,
            "braking_time_s": braking,
            "braking_time_ms": to_ms(braking),
            "flexion_braking_time_ms": to_ms(braking),
            "impulse_propulsion_time_ms": to_ms(propulsion),
            "propulsion_time_s": propulsion,
            "propulsion_time_ms": to_ms(propulsion),
            "knee_alignment_separation_norm": separation,
            "phase_quality": "valid",
        }
    )


def add_event_quality(event: dict[str, Any], cfg: dromia_config.GaitAnalysisConfig) -> None:
    """Attach explicit gates without discarding inspectable low-quality events."""

    warnings: list[str] = []
    endpoint_quality = str(event.get("endpoint_quality", "unknown"))
    duration = event.get("contact_time_s")
    inferred_fraction = float(event.get("inferred_contact_fraction") or 0.0)
    endpoint_is_valid = endpoint_quality in {"observed", "human_reviewed"}
    if not endpoint_is_valid:
        warnings.append("contact_endpoint_censored")
    if duration is None:
        warnings.append("contact_duration_unavailable")
    elif not cfg.min_event_contact_seconds <= float(duration) <= cfg.max_event_contact_seconds:
        warnings.append("contact_duration_outside_plausible_range")
    if inferred_fraction > cfg.max_inferred_contact_fraction:
        warnings.append("excess_inferred_contact")
    if event.get("phase_quality") != "valid":
        warnings.append(str(event.get("phase_quality", "phase_not_available")))
    event["duration_quality"] = (
        "not_available"
        if duration is None
        else "plausible"
        if cfg.min_event_contact_seconds <= float(duration) <= cfg.max_event_contact_seconds
        else "outside_plausible_range"
    )
    event["confidence_quality"] = "uncalibrated_engineering_score"
    event["event_quality"] = (
        "valid" if not warnings else "censored" if not endpoint_is_valid else "low_quality"
    )
    event["quality_warnings"] = warnings
    if event["event_quality"] != "valid" and event.get("phase_quality") == "valid":
        for field in (
            "braking_time_s",
            "braking_time_ms",
            "flexion_braking_time_ms",
            "impulse_propulsion_time_ms",
            "propulsion_time_s",
            "propulsion_time_ms",
        ):
            event[field] = None
        event["phase_quality"] = "withheld_low_quality_event"


def assign_event_ids(events: list[dict[str, Any]]) -> None:
    counts = {side: 0 for side in SIDES}
    for event in events:
        side = str(event["side"])
        event["event_id"] = f"{side}:{counts[side]}"
        counts[side] += 1


def cadence_metrics(
    events: list[dict[str, Any]],
    timing: dromia_timebase.VideoTimebase,
    cfg: dromia_config.GaitAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Count observed step transitions; use complete alternating cycles for cadence."""
    contacts = sorted(
        [item for item in events if item.get("landing_frame") is not None],
        key=lambda item: int(item["landing_frame"]),
    )

    def usable(first: dict[str, Any], second: dict[str, Any]) -> bool:
        return first["side"] != second["side"] and second["landing_frame"] > first["landing_frame"]

    steps = [
        {
            "start_frame": first["landing_frame"],
            "end_frame": second["landing_frame"],
            "duration_ms": to_ms(
                dromia_timebase.elapsed_s(timing, first["landing_frame"], second["landing_frame"])
            ),
        }
        for first, second in zip(contacts, contacts[1:], strict=False)
        if usable(first, second)
    ]
    cycles = []
    for index, first in enumerate(contacts):
        following = next(
            (i for i in range(index + 1, len(contacts)) if contacts[i]["side"] == first["side"]),
            None,
        )
        second = contacts[following] if following is not None else None
        complete = (
            following == index + 2
            and usable(first, contacts[index + 1])
            and usable(contacts[index + 1], second)
        )
        cycles.append(
            {
                "start_event_id": first.get("event_id"),
                "end_event_id": None if second is None else second.get("event_id"),
                "side": first["side"],
                "start_frame": first["landing_frame"],
                "end_frame": None if second is None else second["landing_frame"],
                "two_step_time_ms": to_ms(
                    dromia_timebase.elapsed_s(
                        timing, first["landing_frame"], second["landing_frame"]
                    )
                )
                if complete
                else None,
                "complete": bool(complete),
                "censored_end": second is None,
                "quality": "complete_alternating_cycle"
                if complete
                else "censored_end"
                if second is None
                else "non_alternating_contacts",
            }
        )
    two_step = median_or_none([x["two_step_time_ms"] for x in cycles if x["complete"]])
    median_step = median_or_none([x["duration_ms"] for x in steps])
    raw = 120000.0 / two_step if two_step else 60000.0 / median_step if median_step else None
    minimum = cfg.cadence_min_spm if cfg is not None else 100.0
    maximum = cfg.cadence_max_spm if cfg is not None else 260.0
    plausible = raw is not None and minimum <= raw <= maximum
    return {
        "definition": "120000_over_median_two_step_ms_else_60000_over_median_step_ms",
        "units": "steps_per_minute",
        "cadence_spm": raw if plausible else None,
        "unfiltered_cadence_spm": raw,
        "plausible_range_spm": [minimum, maximum],
        "plausible": plausible,
        "step_count": len(steps) if steps else None,
        "step_count_definition": "usable_alternating_initial_contact_transitions",
        "step_intervals": steps,
        "step_interval_count": len(steps),
        "contact_count": len(contacts),
        "two_step_time_ms": two_step,
        "two_step_intervals": cycles,
        "two_step_interval_count": sum(x["complete"] for x in cycles),
        "two_step_censored_interval_count": sum(x["censored_end"] for x in cycles),
        "is_extrapolated_from_two_contacts": len(contacts) == 2 and len(steps) == 1,
        "quality": "not_available"
        if raw is None
        else "outside_plausible_range"
        if not plausible
        else "complete_alternating_cycles"
        if two_step
        else "extrapolated_two_contacts"
        if len(contacts) == 2
        else "single_alternating_interval"
        if len(steps) == 1
        else "multiple_step_intervals",
        "window_start_frame": steps[0]["start_frame"] if steps else None,
        "window_end_frame": steps[-1]["end_frame"] if steps else None,
    }


def migrate_metric_fields(value: Any) -> None:
    """Remove retired schema families when recalibrating cached earlier artifacts."""
    if isinstance(value, list):
        for item in value:
            migrate_metric_fields(item)
    elif isinstance(value, dict):
        for key in list(value):
            item = value[key]
            if (
                any(
                    name in key
                    for name in (
                        "foot_placement",
                        "knee_flexion",
                        "hip_leg",
                        "midflight",
                        "knee_alignment_is_approximate",
                    )
                )
                or key == "mean_flight_time_s"
            ):
                value.pop(key)
                continue
            renamed = key.replace("tibia_floor_angle_deg", "tibia_horizontal_angle_deg")
            if key in {"flight_time_s", "flight_time_ms", "flight_frames"}:
                renamed = "same_foot_" + key
            if renamed != key:
                value[renamed] = value.pop(key)
            migrate_metric_fields(item)


def add_public_event_fields(event: dict[str, Any]) -> None:
    """Canonical units; internal seconds and strike evidence remain auditable."""
    event["flexion_braking_time_ms"] = to_ms(event.get("braking_time_s"))
    event["impulse_propulsion_time_ms"] = to_ms(event.get("propulsion_time_s"))
    event["foot_strike"] = {"heel": "RFS", "midfoot": "MFS", "forefoot": "FFS"}.get(
        event.get("strike_type")
    )


def associate_global_flights(events: list[dict[str, Any]], flights: list[dict[str, Any]]) -> None:
    for event in events:
        event["global_flight_time_ms"] = None
        event["global_flight_quality"] = "censored_or_unknown"
    for flight in flights:
        preceding = [
            event
            for event in events
            if event.get("takeoff_frame") is not None
            and event["takeoff_frame"] == flight["preceding_takeoff_frame"]
        ]
        flight["preceding_event_ids"] = [event.get("event_id") for event in preceding]
        flight["global_flight_time_ms"] = flight["duration_ms"] if flight["complete"] else None
        for event in preceding:
            event["global_flight_time_ms"] = flight["global_flight_time_ms"]
            event["global_flight_quality"] = (
                "complete" if flight["complete"] else "censored_or_unknown"
            )


def add_spatial_public_fields(spatial: dict[str, Any], events: list[dict[str, Any]]) -> None:
    for kind in ("step", "stride"):
        value = spatial.get(f"mean_{kind}_length_m")
        spatial[f"mean_{kind}_length_cm"] = None if value is None else 100.0 * value
        for event in events:
            event[f"{kind}_length_cm"] = None
        for record in spatial.get(f"{kind}s", []):
            value = record.get("longitudinal_distance_m")
            record[f"{kind}_length_cm"] = None if value is None else 100.0 * value
            for event in events:
                if (
                    event.get("landing_frame") == record["end_frame"]
                    and event["side"] == record["end_side"]
                ):
                    event[f"{kind}_length_cm"] = record[f"{kind}_length_cm"]


def global_contact_metrics(
    rows: list[dict[str, Any]], timing: dromia_timebase.VideoTimebase
) -> dict[str, Any]:
    """Summarize intervals where at least one observed shoe is in contact."""

    all_rows = rows
    rows = [row for row in rows if row.get("activity_window_included", True)]
    states: list[bool | None] = []
    for row in rows:
        left, right = row.get("left_contact"), row.get("right_contact")
        row["global_contact"] = (
            True
            if left is True or right is True
            else (False if left is False and right is False else None)
        )
        if left is True or right is True:
            states.append(True)
        elif left is False and right is False:
            states.append(False)
        else:
            states.append(None)
    intervals = []
    for start, end in boolean_intervals([state is True for state in states]):
        censored_start = start == 0 or states[start - 1] is None
        censored_end = end == len(states) - 1 or states[end + 1] is None
        complete = not censored_start and not censored_end
        duration = dromia_timebase.interval_duration_s(
            timing, int(rows[start]["frame_idx"]), int(rows[end]["frame_idx"])
        )
        intervals.append(
            {
                "start_frame": int(rows[start]["frame_idx"]),
                "end_frame": int(rows[end]["frame_idx"]),
                "duration_frames": end - start + 1,
                "duration_s": duration,
                "duration_ms": to_ms(duration),
                "complete": complete,
                "censored_start": censored_start,
                "censored_end": censored_end,
            }
        )
    complete_durations = [item["duration_s"] for item in intervals if item["complete"]]
    observed = [state for state in states if state is not None]
    contact_frames = sum(state is True for state in observed)
    return {
        "definition": "union_of_left_and_right_observed_shoe_contact",
        "units": {"duration_ms": "milliseconds", "duration_s": "seconds", "duty_factor": "ratio"},
        "quality": (
            "complete_intervals_available"
            if complete_durations
            else "censored_intervals_only"
            if intervals
            else "no_contact_observed"
            if observed
            else "not_observed"
        ),
        "intervals": intervals,
        "complete_interval_count": len(complete_durations),
        "censored_interval_count": len(intervals) - len(complete_durations),
        "total_complete_contact_time_s": float(sum(complete_durations)),
        "mean_global_contact_time_s": mean_or_none(complete_durations),
        "mean_global_contact_time_ms": to_ms(mean_or_none(complete_durations)),
        "median_global_contact_time_ms": to_ms(median_or_none(complete_durations)),
        "total_complete_contact_time_ms": to_ms(float(sum(complete_durations))),
        "median_global_contact_time_s": median_or_none(complete_durations),
        "contact_duty_factor": contact_frames / len(observed) if observed else None,
        "duty_factor_definition": (
            "observed_global_contact_frames_divided_by_observed_active_frames"
        ),
        "duty_factor_validation_status": "not_evaluated_formula_unconfirmed",
        "observed_frame_count": len(observed),
        "unknown_frame_count": len(states) - len(observed),
        "active_frame_count": len(states),
        "inactive_excluded_frame_count": len(all_rows) - len(rows),
    }


def spatial_distance_metrics(
    events: list[dict[str, Any]],
    calibration: dromia_calibration.GroundCalibration | None,
) -> dict[str, Any]:
    if calibration is None or not calibration.valid:
        result = {
            "available": False,
            "reason": "ground_calibration_missing_or_invalid",
            "steps": [],
            "strides": [],
            "mean_step_length_m": None,
            "mean_stride_length_m": None,
            "mean_step_length_m_by_side": {side: None for side in SIDES},
            "mean_stride_length_m_by_side": {side: None for side in SIDES},
        }
        add_spatial_public_fields(result, events)
        return result
    contacts = sorted(
        [item for item in events if item.get("landing_contact_point_m") is not None],
        key=lambda item: int(item["landing_frame"]),
    )
    steps = []
    strides = []
    for first, second in zip(contacts, contacts[1:], strict=False):
        if first["side"] == second["side"]:
            continue
        steps.append(distance_record(first, second, calibration=calibration))
    for side in SIDES:
        same_side = [item for item in contacts if item["side"] == side]
        strides.extend(
            distance_record(first, second, calibration=calibration)
            for first, second in zip(same_side, same_side[1:], strict=False)
        )
    result = {
        "available": True,
        "reason": None,
        "steps": steps,
        "strides": strides,
        "mean_step_length_m": mean_or_none([item["longitudinal_distance_m"] for item in steps]),
        "mean_stride_length_m": mean_or_none([item["longitudinal_distance_m"] for item in strides]),
        "mean_step_length_m_by_side": {
            side: mean_or_none(
                [item["longitudinal_distance_m"] for item in steps if item["end_side"] == side]
            )
            for side in SIDES
        },
        "mean_stride_length_m_by_side": {
            side: mean_or_none(
                [item["longitudinal_distance_m"] for item in strides if item["end_side"] == side]
            )
            for side in SIDES
        },
    }
    add_spatial_public_fields(result, events)
    return result


def asymmetry_metrics(
    events: list[dict[str, Any]],
    distances: dict[str, Any],
    cfg: dromia_config.GaitAnalysisConfig,
) -> dict[str, Any]:
    specifications = {
        "contact_time_s": (events, "contact_time_s"),
        "braking_time_s": (events, "braking_time_s"),
        "propulsion_time_s": (events, "propulsion_time_s"),
        "same_foot_flight_time_s": (events, "same_foot_flight_time_s"),
        "landing_knee_angle_deg": (events, "landing_knee_angle_deg"),
        "landing_tibia_horizontal_angle_deg": (events, "landing_tibia_horizontal_angle_deg"),
        "landing_foot_tibia_angle_deg": (events, "landing_foot_tibia_angle_deg"),
    }
    metrics = {
        name: asymmetry_record(
            [
                float(item[field])
                for item in source
                if item.get("side") == "left"
                and item.get(field) is not None
                and item.get("event_quality", "valid") == "valid"
            ],
            [
                float(item[field])
                for item in source
                if item.get("side") == "right"
                and item.get(field) is not None
                and item.get("event_quality", "valid") == "valid"
            ],
            cfg.asymmetry_review_threshold_percent,
        )
        for name, (source, field) in specifications.items()
    }
    for name, key in (
        ("step_length_m", "mean_step_length_m_by_side"),
        ("stride_length_m", "mean_stride_length_m_by_side"),
    ):
        values = distances.get(key, {})
        metrics[name] = asymmetry_record(
            [] if values.get("left") is None else [float(values["left"])],
            [] if values.get("right") is None else [float(values["right"])],
            cfg.asymmetry_review_threshold_percent,
        )
    return {
        "definition": "100_abs_left_minus_right_over_bilateral_mean",
        "review_threshold_percent": cfg.asymmetry_review_threshold_percent,
        "diagnostic": False,
        "metrics": metrics,
        "review_recommended": any(item["review_recommended"] for item in metrics.values()),
    }


def asymmetry_record(
    left_values: list[float], right_values: list[float], threshold_percent: float
) -> dict[str, Any]:
    left = mean_or_none(left_values)
    right = mean_or_none(right_values)
    if left is None or right is None:
        return {
            "left_value": left,
            "right_value": right,
            "left_n": len(left_values),
            "right_n": len(right_values),
            "absolute_difference": None,
            "symmetry_index_percent": None,
            "quality": "insufficient_evidence",
            "review_recommended": False,
        }
    difference = abs(left - right)
    denominator = (abs(left) + abs(right)) / 2.0
    index = None if denominator <= EPS else 100.0 * difference / denominator
    return {
        "left_value": left,
        "right_value": right,
        "left_n": len(left_values),
        "right_n": len(right_values),
        "absolute_difference": difference,
        "symmetry_index_percent": index,
        "quality": "valid" if min(len(left_values), len(right_values)) >= 2 else "limited_sample",
        "review_recommended": bool(index is not None and index >= threshold_percent),
    }


def distance_record(
    first: dict[str, Any],
    second: dict[str, Any],
    calibration: dromia_calibration.GroundCalibration | None = None,
) -> dict[str, Any]:
    a = np.asarray(first["landing_contact_point_m"], np.float64)
    b = np.asarray(second["landing_contact_point_m"], np.float64)
    axis = 0
    if calibration is not None and calibration.travel_axis == "world_y":
        axis = 1
    elif abs(b[1] - a[1]) > abs(b[0] - a[0]):
        axis = 1
    return {
        "start_frame": int(first["landing_frame"]),
        "end_frame": int(second["landing_frame"]),
        "start_side": first["side"],
        "end_side": second["side"],
        "distance_m": float(np.linalg.norm(b - a)),
        "longitudinal_distance_m": float(abs(b[axis] - a[axis])),
    }


def project_point(
    calibration: dromia_calibration.GroundCalibration | None,
    point: tuple[float, float] | list[float] | None,
    *,
    require_inside: bool = True,
    margin_m: float = 0.05,
) -> tuple[float, float] | None:
    if calibration is None or point is None:
        return None
    mapped = calibration.project(point)
    if mapped is None:
        return None
    if require_inside:
        longitudinal = float(calibration.longitudinal_m)
        transverse = float(calibration.transverse_m)
        x_w, y_w = float(mapped[0]), float(mapped[1])
        if not (
            -margin_m <= x_w <= longitudinal + margin_m
            and -margin_m <= y_w <= transverse + margin_m
        ):
            return None
    return mapped


def calibration_direction(
    calibration: dromia_calibration.GroundCalibration | None,
) -> str | None:
    if calibration is None or not calibration.valid:
        return None
    return "right" if calibration.travel_direction == "left_to_right" else "left"


def robust_outsole_axis(curve: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """Fit a directed outsole line after rejecting upper/jittering curve points."""

    points = np.asarray(curve, np.float64)
    if len(points) < 6 or np.ptp(points[:, 0]) < EPS:
        return None, None
    keep = np.ones(len(points), dtype=bool)
    coefficients = None
    for _ in range(4):
        if np.count_nonzero(keep) < 4:
            return None, None
        coefficients = np.polyfit(points[keep, 0], points[keep, 1], 1)
        residual = points[:, 1] - np.polyval(coefficients, points[:, 0])
        scale = max(
            float(np.median(np.abs(residual[keep] - np.median(residual[keep])))) * 1.4826, 0.5
        )
        keep = np.abs(residual) <= 2.5 * scale
    assert coefficients is not None
    residual = points[keep, 1] - np.polyval(coefficients, points[keep, 0])
    axis = np.asarray([1.0, coefficients[0]], np.float64)
    axis /= max(float(np.linalg.norm(axis)), EPS)
    return axis, float(np.sqrt(np.mean(residual**2)))


def summarize(
    events: list[dict[str, Any]],
    flights: list[dict[str, Any]],
    cadence: dict[str, Any],
    distances: dict[str, Any],
    global_contact: dict[str, Any],
) -> dict[str, Any]:
    valid_events = [x for x in events if x.get("event_quality", "valid") == "valid"]
    contact = [x["contact_time_s"] for x in valid_events if x.get("contact_time_s") is not None]
    flight = [x["duration_s"] for x in flights if x.get("complete", True)]
    braking = [x["braking_time_s"] for x in valid_events if x.get("braking_time_s") is not None]
    propulsion = [
        x["propulsion_time_s"] for x in valid_events if x.get("propulsion_time_s") is not None
    ]
    return {
        "contact_count": sum(x.get("landing_frame") is not None for x in events),
        "complete_contact_count": len(contact),
        "valid_event_count": len(valid_events),
        "low_quality_or_censored_event_count": len(events) - len(valid_events),
        "flight_count": len(flights),
        "mean_contact_time_s": mean_or_none(contact),
        "median_contact_time_s": median_or_none(contact),
        "std_contact_time_s": std_or_none(contact),
        "total_complete_global_contact_time_s": global_contact["total_complete_contact_time_s"],
        "mean_global_contact_time_s": global_contact["mean_global_contact_time_s"],
        "median_global_contact_time_s": global_contact["median_global_contact_time_s"],
        "contact_duty_factor": global_contact["contact_duty_factor"],
        "mean_global_flight_time_s": mean_or_none(flight),
        "mean_global_flight_time_ms": to_ms(mean_or_none(flight)),
        "mean_same_foot_flight_time_ms": mean_or_none(
            [
                x["same_foot_flight_time_ms"]
                for x in valid_events
                if x.get("same_foot_flight_time_ms") is not None
            ]
        ),
        "mean_braking_time_s": mean_or_none(braking),
        "mean_propulsion_time_s": mean_or_none(propulsion),
        "cadence_spm": cadence["cadence_spm"],
        "step_count": cadence.get("step_count"),
        "two_step_time_ms": cadence.get("two_step_time_ms"),
        "mean_contact_time_ms": to_ms(mean_or_none(contact)),
        "median_contact_time_ms": to_ms(median_or_none(contact)),
        "std_contact_time_ms": to_ms(std_or_none(contact)),
        "mean_flexion_braking_time_ms": to_ms(mean_or_none(braking)),
        "mean_impulse_propulsion_time_ms": to_ms(mean_or_none(propulsion)),
        "mean_step_length_cm": distances.get("mean_step_length_cm"),
        "mean_stride_length_cm": distances.get("mean_stride_length_cm"),
        "mean_step_length_m": distances["mean_step_length_m"],
        "mean_stride_length_m": distances["mean_stride_length_m"],
    }


def synthetic_timebase(frames: np.ndarray, fps: float) -> dromia_timebase.VideoTimebase:
    indices = [int(value) for value in frames]
    return dromia_timebase.VideoTimebase(
        video_path="synthetic_or_unspecified",
        video_sha256="unknown",
        source_fps=float(fps),
        source_frame_count=(max(indices) + 1) if indices else 0,
        metadata_frame_count=(max(indices) + 1) if indices else 0,
        decoded_frame_count=(max(indices) + 1) if indices else 0,
        frame_indices=indices,
        timestamps_s=[value / float(fps) for value in indices],
        frame_durations_s=[1.0 / float(fps) for _ in indices],
        frame_duration_s=1.0 / float(fps),
        media_timestamps_s=[value / float(fps) for value in indices],
        media_frame_durations_s=[1.0 / float(fps) for _ in indices],
        media_frame_duration_s=1.0 / float(fps),
        timing_source="caller_supplied_fps",
        real_world_fps=float(fps),
    )


def to_ms(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def std_or_none(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    return (a + b) * 0.5 if finite(a, b) else None


def finite(*points: np.ndarray) -> bool:
    return all(np.isfinite(x).all() for x in points)


def vector_angle_deg(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or not finite(a, b):
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < EPS:
        return None
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)) / denom, -1, 1))))


def axis_angle_deg(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    value = vector_angle_deg(a, b)
    return None if value is None else min(value, 180.0 - value)


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    if not finite(a, b, c):
        return None
    u = a - b
    v = c - b
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < EPS:
        return None
    return float(np.degrees(np.arccos(np.clip(float(np.dot(u, v)) / denom, -1, 1))))


def torso_lean_deg(torso: np.ndarray | None, direction: str) -> float | None:
    if torso is None or not finite(torso) or np.linalg.norm(torso) < EPS:
        return None
    lean = float(np.degrees(np.arctan2(float(torso[0]), float(-torso[1]))))
    return lean if direction == "right" else -lean


def posture(lean: float | None, tolerance: float) -> str:
    if lean is None:
        return "unknown"
    return "front" if lean > tolerance else "back" if lean < -tolerance else "straight"


def numeric_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([np.nan if row.get(key) is None else row[key] for row in rows], np.float32)


def fmt(value: Any) -> str:
    return "--" if value is None else f"{float(value):.1f}deg"


def format_side_overlay(label: str, side: str, row: dict[str, Any]) -> str:
    return (
        f"{label} c={row[f'{side}_contact']} "
        f"k={fmt(row[f'{side}_knee_angle_deg'])} "
        f"tf={fmt(row[f'{side}_tibia_horizontal_angle_deg'])} "
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute gait metrics for an DromIA run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pose-npz", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = dromia_dto.RunManifest.model_validate_json((run / "manifest.json").read_text())
    default_pose = manifest.artifacts.get("first_pass_pose_npz")
    if default_pose is None:
        default_pose = manifest.artifacts.get("temporal_biomechanics_npz")
    if default_pose is None:
        default_pose = manifest.artifacts["posterior_npz"]
    pose_path = args.pose_npz or run / default_pose
    artifacts = analyze_run_pose(run, pose_path)
    print(json.dumps(artifacts, indent=2))
    return 0


def load_assignments_from_cache(
    run: Path, frames: np.ndarray, ids: np.ndarray
) -> list[dromia_dto.ShoeAssignment]:
    rows = json.loads((run / "shoes" / "shoe_assignments.json").read_text())
    row_map = {(x["frame_idx"], x["runner_id"], x["side"]): x for x in rows}
    output = []
    manifest = dromia_dto.RunManifest.model_validate_json((run / "manifest.json").read_text())
    for path in sam31_cache.frame_paths(run / manifest.sam_cache_dir):
        sam = sam31_cache.load_frame(path)
        shoes = {x.obj_id: x.mask for x in sam.shoes}
        for rid in ids:
            for side in SIDES:
                row = row_map.get((sam.frame_idx, int(rid), side))
                shoe_id = None if row is None else row["shoe_obj_id"]
                output.append(
                    dromia_dto.ShoeAssignment(
                        frame_idx=sam.frame_idx,
                        runner_id=int(rid),
                        side=side,
                        shoe_obj_id=shoe_id,
                        score=0.0 if row is None else row["score"],
                        mask=shoes.get(shoe_id),
                    )
                )
    return output


def read_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps if fps > 0 else 30.0


if __name__ == "__main__":
    raise SystemExit(main())
