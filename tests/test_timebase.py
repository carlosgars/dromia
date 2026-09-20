from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

from dromia import timebase as dromia_timebase
from dromia.gait import analysis as gait_analysis


def atom(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def test_parses_apple_full_frame_rate_playback_intent() -> None:
    key = b"com.apple.quicktime.full-frame-rate-playback-intent"
    keys = atom(b"keys", b"\0\0\0\0" + struct.pack(">I", 1) + atom(b"mdta", key))
    data = atom(b"data", struct.pack(">IIQ", 21, 0, 0))
    values = atom(b"ilst", atom(struct.pack(">I", 1), data))
    metadata = atom(b"meta", atom(b"hdlr", b"metadata-handler") + keys + values)

    assert dromia_timebase.parse_full_frame_rate_intent(metadata) is False


def test_24_inclusive_frames_at_240_fps_are_one_tenth_second() -> None:
    timing = gait_analysis.synthetic_timebase(np.arange(24, dtype=np.int32), 240.0)

    assert dromia_timebase.interval_duration_s(timing, 0, 23) == 0.1


def test_elapsed_time_uses_source_frame_indices_across_gaps() -> None:
    timing = gait_analysis.synthetic_timebase(np.asarray([0, 4, 8], np.int32), 240.0)

    assert dromia_timebase.elapsed_s(timing, 0, 8) == round(8 / 240, 12)


def test_interval_uses_end_frames_own_duration_for_variable_rate() -> None:
    timing = dromia_timebase.VideoTimebase(
        video_path="variable.mp4",
        video_sha256="hash",
        source_fps=50.0,
        source_frame_count=3,
        frame_indices=[0, 1, 2],
        timestamps_s=[0.0, 0.01, 0.03],
        frame_durations_s=[0.01, 0.02, 0.015],
        frame_duration_s=0.015,
        timing_source="test_pts",
        variable_frame_rate=True,
    )

    assert dromia_timebase.interval_duration_s(timing, 1, 2) == 0.035


def test_inspect_video_persists_decoded_pts_and_detects_variable_rate(monkeypatch) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.index = -1

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FPS:
                return 30.0
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 4.0
            if property_id == cv2.CAP_PROP_POS_MSEC:
                return [0.0, 10.0, 30.0][self.index]
            return 0.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            self.index += 1
            return (True, np.zeros((2, 2, 3), np.uint8)) if self.index < 3 else (False, None)

        def release(self) -> None:
            return None

    monkeypatch.setattr(dromia_timebase.cv2, "VideoCapture", lambda _path: FakeCapture())
    monkeypatch.setattr(dromia_timebase, "file_sha256", lambda _path: "hash")

    timing = dromia_timebase.inspect_video(Path("variable.mp4"), [0, 1, 2])

    assert timing.schema_version == 3
    assert timing.timing_source == "opencv_decoded_pts"
    assert timing.source_frame_count == 4
    assert timing.metadata_frame_count == 4
    assert timing.decoded_frame_count == 3
    assert timing.frame_count_mismatch is True
    assert timing.unanalyzed_source_frames == [3]
    np.testing.assert_allclose(timing.timestamps_s, [0.0, 0.01, 0.03])
    np.testing.assert_allclose(timing.frame_durations_s, [0.01, 0.02, 0.015])
    assert timing.variable_frame_rate is True


def test_capture_fps_converts_playback_timestamps_to_real_world_time(monkeypatch) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.index = -1

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FPS:
                return 30.0
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 3.0
            if property_id == cv2.CAP_PROP_POS_MSEC:
                return [0.0, 33.333333, 66.666667][self.index]
            return 0.0

        def read(self):
            self.index += 1
            return (True, np.zeros((2, 2, 3), np.uint8)) if self.index < 3 else (False, None)

        def release(self) -> None:
            return None

    monkeypatch.setattr(dromia_timebase.cv2, "VideoCapture", lambda _path: FakeCapture())
    monkeypatch.setattr(dromia_timebase, "file_sha256", lambda _path: "hash")
    monkeypatch.setattr(
        dromia_timebase, "quicktime_full_frame_rate_playback_intent", lambda _path: False
    )

    timing = dromia_timebase.inspect_video(Path("slow.mov"), [0, 1, 2], capture_fps_override=240.0)

    assert timing.source_fps == 30.0
    assert timing.real_world_fps == 240.0
    assert timing.media_to_real_time_scale == 0.125
    np.testing.assert_allclose(timing.media_timestamps_s, [0.0, 1 / 30, 2 / 30])
    np.testing.assert_allclose(timing.timestamps_s, [0.0, 1 / 240, 2 / 240])
    assert timing.temporal_calibration_source == "explicit_capture_fps_override"


def test_slow_motion_metadata_requires_capture_fps(monkeypatch) -> None:
    monkeypatch.setattr(
        dromia_timebase, "quicktime_full_frame_rate_playback_intent", lambda _path: False
    )
    monkeypatch.setattr(dromia_timebase.cv2, "VideoCapture", lambda _path: object())

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, property_id):
            return 30.0 if property_id == cv2.CAP_PROP_FPS else 2.0

        def read(self):
            return (False, None)

        def release(self):
            return None

    monkeypatch.setattr(dromia_timebase.cv2, "VideoCapture", lambda _path: FakeCapture())
    with np.testing.assert_raises_regex(ValueError, "capture FPS is unknown"):
        dromia_timebase.inspect_video(Path("slow.mov"), [0, 1])
