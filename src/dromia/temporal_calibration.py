"""Post-hoc real-world timing calibration for completed DromIA runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from dromia import config as dromia_config
from dromia import gait_contract
from dromia import timebase as dromia_timebase
from dromia.gait import analysis as gait_analysis

STANDARD_CAPTURE_FPS = (30.0, 50.0, 60.0, 100.0, 120.0, 200.0, 240.0)


def estimate_capture_fps(
    video_path: Path,
    playback_fps: float,
    *,
    run_dir: Path | None = None,
    current_capture_fps: float | None = None,
) -> dict[str, Any]:
    """Suggest capture FPS from metadata/name evidence and existing gait intervals."""

    video = video_path.expanduser().resolve()
    intent = dromia_timebase.quicktime_full_frame_rate_playback_intent(video)
    filename_match = re.search(
        r"(?:hsr|hfr|slowmo|slow[_ -]?motion|fps)[_ -]?(\d{2,3})|"
        r"(\d{2,3})[_ -]?(?:fps|hz)",
        video.stem,
        flags=re.IGNORECASE,
    )
    filename_fps = (
        float(next(value for value in filename_match.groups() if value))
        if filename_match is not None
        else None
    )
    timing = None
    persisted_capture_fps = current_capture_fps
    media_cadences: list[float] = []
    if run_dir is not None:
        run = run_dir.expanduser().resolve()
        timebase_path = run / "timebase.json"
        if timebase_path.is_file():
            timing = dromia_timebase.VideoTimebase.model_validate_json(timebase_path.read_text())
            if timing.temporal_calibration_source != "media_timeline_assumed_real_time":
                persisted_capture_fps = timing.real_world_fps
            intent = (
                timing.quicktime_full_frame_rate_playback_intent
                if timing.quicktime_full_frame_rate_playback_intent is not None
                else intent
            )
        gait_path = run / "gait" / "gait_analysis.json"
        if gait_path.is_file():
            payload = json.loads(gait_path.read_text())
            current_scale = float(timing.media_to_real_time_scale) if timing is not None else 1.0
            for runner in payload.get("runners", {}).values():
                cadence = runner.get("cadence", {})
                value = cadence.get("unfiltered_cadence_spm", cadence.get("cadence_spm"))
                if value is not None and math.isfinite(float(value)):
                    media_cadences.append(float(value) * current_scale)

    candidates = {value for value in STANDARD_CAPTURE_FPS if value >= playback_fps * 0.95}
    candidates.add(float(playback_fps))
    if filename_fps is not None:
        candidates.add(filename_fps)
    if persisted_capture_fps is not None:
        candidates.add(float(persisted_capture_fps))
    scored = []
    usable = [value for value in media_cadences if 5.0 <= value <= 300.0]
    for capture_fps in sorted(candidates):
        corrected = [value * capture_fps / playback_fps for value in usable]
        evidence_scores = [
            math.exp(-0.5 * ((value - 175.0) / 25.0) ** 2) if 100.0 <= value <= 260.0 else 0.0
            for value in corrected
        ]
        support = sum(score >= 0.25 for score in evidence_scores)
        score = sum(sorted(evidence_scores, reverse=True)[:6])
        if intent is False and capture_fps <= playback_fps * 1.05:
            score *= 0.1
        if intent is True and abs(capture_fps - playback_fps) < 0.01:
            score += 2.0
        elif intent is not False and abs(capture_fps - playback_fps) < 0.01:
            score += 0.5
        if filename_fps is not None and abs(capture_fps - filename_fps) < 0.01:
            score += 3.0
        if persisted_capture_fps is not None and abs(capture_fps - persisted_capture_fps) < 0.01:
            score += 4.0
        scored.append(
            {
                "capture_fps": capture_fps,
                "score": score,
                "supporting_runner_count": support,
                "corrected_cadence_median_spm": (
                    float(
                        np.median(
                            [
                                value
                                for value, score in zip(corrected, evidence_scores, strict=True)
                                if score >= 0.25
                            ]
                        )
                    )
                    if support
                    else None
                ),
            }
        )
    scored.sort(key=lambda item: (item["score"], item["capture_fps"]), reverse=True)
    best = scored[0]
    second_score = scored[1]["score"] if len(scored) > 1 else 0.0
    margin = (best["score"] - second_score) / max(best["score"], 1e-9)
    if (
        persisted_capture_fps is not None
        and abs(best["capture_fps"] - persisted_capture_fps) < 0.01
    ):
        confidence = "high"
    elif filename_fps is not None and best["capture_fps"] == filename_fps:
        confidence = "high"
    elif intent is True and abs(best["capture_fps"] - playback_fps) < 0.01:
        confidence = "high"
    elif best["supporting_runner_count"] >= 3 and margin >= 0.2:
        confidence = "high"
    elif best["supporting_runner_count"] >= 2 or intent is True:
        confidence = "medium"
    else:
        confidence = "low"
    evidence = []
    if intent is False:
        evidence.append("Apple metadata marks the movie for slow-motion playback")
    elif intent is True:
        evidence.append("Apple metadata requests full-frame-rate playback")
    if filename_fps is not None:
        evidence.append(f"Filename indicates {filename_fps:g} FPS")
    if persisted_capture_fps is not None:
        evidence.append(f"Existing reviewed timing uses {persisted_capture_fps:g} FPS")
    if usable:
        evidence.append(f"Scored {len(usable)} detected runner cadence interval sets")
    if not evidence:
        evidence.append("No slow-motion metadata or gait evidence; playback FPS assumed")
    exact_rate_unresolved = (
        intent is False and filename_fps is None and persisted_capture_fps is None and not usable
    )
    if exact_rate_unresolved:
        evidence.append("The file does not distinguish the exact capture rate")
    return {
        "suggested_capture_fps": best["capture_fps"],
        "confidence": confidence,
        "auto_fill_recommended": confidence in {"high", "medium"} and not exact_rate_unresolved,
        "exact_rate_unresolved": exact_rate_unresolved,
        "requires_review": True,
        "playback_fps": float(playback_fps),
        "quicktime_full_frame_rate_playback_intent": intent,
        "evidence": evidence,
        "candidates": scored[:4],
    }


def calibrate_timebase(
    timing: dromia_timebase.VideoTimebase, capture_fps: float
) -> dromia_timebase.VideoTimebase:
    """Return an idempotently calibrated copy while preserving the media timeline."""

    if capture_fps <= 0:
        raise ValueError("Capture FPS must be positive")
    count = len(timing.frame_indices)
    old_scale = float(timing.media_to_real_time_scale or 1.0)
    media_timestamps = (
        timing.media_timestamps_s
        if len(timing.media_timestamps_s) == count
        else [value / old_scale for value in timing.timestamps_s]
    )
    media_durations = (
        timing.media_frame_durations_s
        if len(timing.media_frame_durations_s) == count
        else [value / old_scale for value in timing.frame_durations_s]
        if len(timing.frame_durations_s) == count
        else [float(timing.frame_duration_s) / old_scale] * count
    )
    scale = float(timing.source_fps) / float(capture_fps)
    real_timestamps = [value * scale for value in media_timestamps]
    real_durations = [value * scale for value in media_durations]
    media_duration = (
        float(timing.media_frame_duration_s)
        if timing.media_frame_duration_s is not None
        else float(timing.frame_duration_s) / old_scale
    )
    intent = timing.quicktime_full_frame_rate_playback_intent
    if intent is None and timing.video_path not in {"", "synthetic_or_unspecified"}:
        intent = dromia_timebase.quicktime_full_frame_rate_playback_intent(Path(timing.video_path))
    return timing.model_copy(
        update={
            "schema_version": 3,
            "timestamps_s": real_timestamps,
            "frame_durations_s": real_durations,
            "frame_duration_s": media_duration * scale,
            "media_timestamps_s": media_timestamps,
            "media_frame_durations_s": media_durations,
            "media_frame_duration_s": media_duration,
            "real_world_fps": float(capture_fps),
            "media_to_real_time_scale": scale,
            "temporal_calibration_source": "posthoc_capture_fps_override",
            "real_world_timing_resolved": True,
            "quicktime_full_frame_rate_playback_intent": intent,
        }
    )


def recalibrate_gait_payload(
    payload: dict[str, Any],
    timing: dromia_timebase.VideoTimebase,
    cfg: dromia_config.GaitAnalysisConfig,
) -> dict[str, Any]:
    """Recompute only timestamp-derived values from existing frames and events."""

    gait_analysis.migrate_metric_fields(payload)
    for runner in payload.get("runners", {}).values():
        runner["source_fps"] = float(timing.source_fps)
        runner["real_world_fps"] = float(timing.real_world_fps or timing.source_fps)
        rows = runner.get("frames", [])
        events = runner.get("events", [])
        for event in events:
            landing = event.get("landing_frame")
            takeoff = event.get("takeoff_frame")
            contact = (
                dromia_timebase.interval_duration_s(timing, int(landing), int(takeoff))
                if landing is not None and takeoff is not None
                else None
            )
            if landing is not None and takeoff is not None:
                event["contact_frames"] = int(takeoff) - int(landing) + 1
            event["contact_time_s"] = contact
            event["contact_time_ms"] = gait_analysis.to_ms(contact)
            gait_analysis.add_stance_phases(event, rows, timing)
        gait_analysis.recompute_event_flights(events, timing, rows)
        flights = gait_analysis.global_flight_intervals_from_rows(rows, timing)
        gait_analysis.associate_global_flights(events, flights)
        for event in events:
            gait_analysis.add_event_quality(event, cfg)
            gait_analysis.add_public_event_fields(event)
        global_contact = gait_analysis.global_contact_metrics(rows, timing)
        cadence = gait_analysis.cadence_metrics(events, timing, cfg)
        spatial = runner.get("spatial", {})
        spatial.setdefault("mean_step_length_m", None)
        spatial.setdefault("mean_stride_length_m", None)
        gait_analysis.add_spatial_public_fields(spatial, events)
        runner["flight_intervals"] = flights
        runner["global_contact"] = global_contact
        runner["cadence"] = cadence
        runner["asymmetry"] = gait_analysis.asymmetry_metrics(events, spatial, cfg)
        runner["summary"] = gait_analysis.summarize(
            events, flights, cadence, spatial, global_contact
        )
    payload["schema_version"] = gait_contract.SCHEMA_VERSION
    payload["source_fps"] = float(timing.source_fps)
    payload["fps"] = float(timing.source_fps)
    payload["real_world_fps"] = float(timing.real_world_fps or timing.source_fps)
    payload["timebase"] = timing.model_dump(mode="json")
    limitations = payload.setdefault("limitations", [])
    note = "Time-derived metrics use post-hoc real-world capture-FPS calibration."
    if note not in limitations:
        limitations.append(note)
    return payload


def recalibrate_run(run_dir: Path, capture_fps: float, *, backup: bool = True) -> dict[str, Any]:
    """Update timing and gait artifacts without rerunning segmentation or pose inference."""

    run = run_dir.expanduser().resolve()
    timebase_path = run / "timebase.json"
    manifest_path = run / "manifest.json"
    gait_dir = run / "gait"
    if not timebase_path.is_file() or not manifest_path.is_file() or not gait_dir.is_dir():
        raise FileNotFoundError("Run needs manifest.json, timebase.json, and gait artifacts")
    timing = calibrate_timebase(
        dromia_timebase.VideoTimebase.model_validate_json(timebase_path.read_text()), capture_fps
    )
    manifest = json.loads(manifest_path.read_text())
    gait_config = manifest.setdefault("config", {}).setdefault("gait_analysis", {})
    gait_config["capture_fps_override"] = float(capture_fps)
    cfg = dromia_config.GaitAnalysisConfig.model_validate(gait_config)
    json_paths = sorted(gait_dir.glob("gait_analysis*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No gait_analysis JSON artifacts in {gait_dir}")

    backup_dir = None
    if backup:
        backup_dir = (
            gait_dir / "temporal_calibration_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        backup_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(timebase_path, backup_dir / "timebase.json")
        shutil.copy2(manifest_path, backup_dir / "manifest.json")
        for path in gait_dir.glob("gait_analysis*.json"):
            shutil.copy2(path, backup_dir / path.name)
        for path in gait_dir.glob("gait_events*.csv"):
            shutil.copy2(path, backup_dir / path.name)
        for path in gait_dir.glob("gait_analysis*.npz"):
            shutil.copy2(path, backup_dir / path.name)

    outputs: dict[str, str] = {}
    for json_path in json_paths:
        suffix = json_path.stem.removeprefix("gait_analysis")
        payload = recalibrate_gait_payload(json.loads(json_path.read_text()), timing, cfg)
        outputs.update(
            gait_analysis.write_gait_artifacts(
                payload,
                run_dir=run,
                draw_video=False,
                suffix=suffix,
            )
        )
    timebase_path.write_text(json.dumps(timing.model_dump(mode="json"), indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "run_dir": str(run),
        "playback_fps": timing.source_fps,
        "capture_fps": timing.real_world_fps,
        "duration_scale": timing.media_to_real_time_scale,
        "cadence_scale": 1.0 / timing.media_to_real_time_scale,
        "backup_dir": None if backup_dir is None else str(backup_dir),
        "artifacts": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recalibrate completed DromIA time metrics without rerunning inference"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--capture-fps", type=float, required=True)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            recalibrate_run(args.run_dir, args.capture_fps, backup=not args.no_backup), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
