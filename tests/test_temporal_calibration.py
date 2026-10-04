from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dromia import gait_contract, temporal_calibration
from dromia.gait import analysis as gait_analysis


def test_recalibrates_existing_metrics_without_touching_debug_video(tmp_path: Path) -> None:
    run = tmp_path / "run"
    gait = run / "gait"
    gait.mkdir(parents=True)
    timing = gait_analysis.synthetic_timebase(np.arange(86), 30.0)
    (run / "timebase.json").write_text(timing.model_dump_json(indent=2))
    (run / "manifest.json").write_text(
        json.dumps({"config": {"gait_analysis": {}}, "artifacts": {}})
    )
    rows = [
        {
            "frame_idx": frame,
            "runner_id": 7,
            "left_contact": False,
            "right_contact": False,
            "knee_horizontal_separation_px": 10.0,
            "bbox_height_px": 100.0,
        }
        for frame in range(86)
    ]
    events = [
        {"runner_id": 7, "side": "left", "landing_frame": 0, "takeoff_frame": 0},
        {"runner_id": 7, "side": "right", "landing_frame": 85, "takeoff_frame": 85},
    ]
    payload = {
        "schema_version": gait_contract.SCHEMA_VERSION,
        "fps": 30.0,
        "timebase": timing.model_dump(mode="json"),
        "limitations": [],
        "runners": {
            "7": {
                "runner_id": 7,
                "events": events,
                "frames": rows,
                "flight_intervals": [],
                "global_contact": {},
                "cadence": {"cadence_spm": 60.0 / (85.0 / 30.0)},
                "spatial": {},
                "asymmetry": {},
                "summary": {},
            }
        },
    }
    gait_analysis.write_gait_artifacts(payload, run_dir=run, draw_video=False)
    debug_video = gait / "gait_debug.mp4"
    debug_video.write_bytes(b"slow-motion-video-is-unchanged")

    result = temporal_calibration.recalibrate_run(run, 240.0)
    calibrated = json.loads((gait / "gait_analysis.json").read_text())
    runner = calibrated["runners"]["7"]
    saved_timing = json.loads((run / "timebase.json").read_text())

    assert result["duration_scale"] == 0.125
    assert result["cadence_scale"] == 8.0
    assert Path(result["backup_dir"]).is_dir()
    assert debug_video.read_bytes() == b"slow-motion-video-is-unchanged"
    assert calibrated["schema_version"] == 13
    assert runner["cadence"]["cadence_spm"] == pytest.approx(60.0 / (85.0 / 240.0))
    assert runner["events"][0]["contact_time_s"] == pytest.approx(1.0 / 240.0)
    assert saved_timing["source_fps"] == 30.0
    assert saved_timing["real_world_fps"] == 240.0
    assert saved_timing["media_timestamps_s"][-1] == pytest.approx(85.0 / 30.0)
    assert saved_timing["timestamps_s"][-1] == pytest.approx(85.0 / 240.0)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["config"]["gait_analysis"]["capture_fps_override"] == 240.0

    temporal_calibration.recalibrate_run(run, 240.0, backup=False)
    repeated = json.loads((gait / "gait_analysis.json").read_text())
    assert repeated["runners"]["7"]["cadence"]["cadence_spm"] == runner["cadence"]["cadence_spm"]


def test_estimator_uses_gait_evidence_to_suggest_240_fps(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    gait = run / "gait"
    gait.mkdir(parents=True)
    timing = gait_analysis.synthetic_timebase(np.arange(10), 30.0)
    (run / "timebase.json").write_text(timing.model_dump_json())
    cadences = [21.176, 22.222, 21.951, 22.5, 38.297]
    (gait / "gait_analysis.json").write_text(
        json.dumps(
            {
                "runners": {
                    str(index): {"cadence": {"cadence_spm": value}}
                    for index, value in enumerate(cadences)
                }
            }
        )
    )
    monkeypatch.setattr(
        temporal_calibration.dromia_timebase,
        "quicktime_full_frame_rate_playback_intent",
        lambda _path: False,
    )

    estimate = temporal_calibration.estimate_capture_fps(tmp_path / "4.MOV", 30.0, run_dir=run)

    assert estimate["suggested_capture_fps"] == 240.0
    assert estimate["confidence"] == "high"
    assert estimate["auto_fill_recommended"] is True
    assert estimate["requires_review"] is True
    assert estimate["candidates"][0]["supporting_runner_count"] == 4


def test_estimator_prefers_explicit_fps_in_filename(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        temporal_calibration.dromia_timebase,
        "quicktime_full_frame_rate_playback_intent",
        lambda _path: None,
    )

    estimate = temporal_calibration.estimate_capture_fps(
        tmp_path / "VID_20241201_HSR_240.mov", 30.0
    )

    assert estimate["suggested_capture_fps"] == 240.0
    assert estimate["confidence"] == "high"
    assert estimate["auto_fill_recommended"] is True


def test_estimator_assumes_playback_rate_without_slow_motion_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        temporal_calibration.dromia_timebase,
        "quicktime_full_frame_rate_playback_intent",
        lambda _path: None,
    )

    estimate = temporal_calibration.estimate_capture_fps(tmp_path / "ordinary.mp4", 30.0)

    assert estimate["suggested_capture_fps"] == 30.0
    assert estimate["confidence"] == "low"
    assert estimate["auto_fill_recommended"] is False


def test_estimator_does_not_auto_fill_ambiguous_apple_slow_motion(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        temporal_calibration.dromia_timebase,
        "quicktime_full_frame_rate_playback_intent",
        lambda _path: False,
    )

    estimate = temporal_calibration.estimate_capture_fps(tmp_path / "slow.mov", 30.0)

    assert estimate["exact_rate_unresolved"] is True
    assert estimate["auto_fill_recommended"] is False
    assert "does not distinguish" in " ".join(estimate["evidence"])


def test_estimator_preserves_existing_reviewed_capture_fps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        temporal_calibration.dromia_timebase,
        "quicktime_full_frame_rate_playback_intent",
        lambda _path: False,
    )

    estimate = temporal_calibration.estimate_capture_fps(
        tmp_path / "slow.mov", 30.0, current_capture_fps=120.0
    )

    assert estimate["suggested_capture_fps"] == 120.0
    assert estimate["confidence"] == "high"
    assert estimate["auto_fill_recommended"] is True
