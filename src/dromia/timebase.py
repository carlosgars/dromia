"""Video timing metadata and frame-alignment validation.

DromIA keeps source-frame indices throughout the pipeline.  This module turns
those indices into seconds without silently substituting an analysis frame rate.
"""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dromia import artifacts


class VideoTimebase(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 3
    video_path: str
    video_sha256: str
    source_fps: float = Field(gt=0.0)
    source_frame_count: int = Field(ge=0)
    metadata_frame_count: int | None = Field(default=None, ge=0)
    decoded_frame_count: int | None = Field(default=None, ge=0)
    frame_count_mismatch: bool = False
    frame_indices: list[int]
    timestamps_s: list[float]
    frame_durations_s: list[float] = Field(default_factory=list)
    frame_duration_s: float = Field(gt=0.0)
    media_timestamps_s: list[float] = Field(default_factory=list)
    media_frame_durations_s: list[float] = Field(default_factory=list)
    media_frame_duration_s: float | None = Field(default=None, gt=0.0)
    timing_source: str
    real_world_fps: float | None = Field(default=None, gt=0.0)
    media_to_real_time_scale: float = Field(default=1.0, gt=0.0)
    temporal_calibration_source: str = "media_timeline_assumed_real_time"
    real_world_timing_resolved: bool = True
    quicktime_full_frame_rate_playback_intent: bool | None = None
    variable_frame_rate: bool = False
    fps_override: float | None = None
    missing_source_frames: list[int] = Field(default_factory=list)
    unanalyzed_source_frames: list[int] = Field(default_factory=list)
    duplicate_frame_indices: list[int] = Field(default_factory=list)

    def timestamp_map(self) -> dict[int, float]:
        return dict(zip(self.frame_indices, self.timestamps_s, strict=True))

    def duration_map(self) -> dict[int, float]:
        if len(self.frame_durations_s) == len(self.frame_indices):
            return dict(zip(self.frame_indices, self.frame_durations_s, strict=True))
        return {frame: self.frame_duration_s for frame in self.frame_indices}


def inspect_video(
    path: Path,
    frame_indices: list[int] | np.ndarray,
    *,
    fps_override: float | None = None,
    capture_fps_override: float | None = None,
) -> VideoTimebase:
    """Map source frames to media time and calibrated real-world acquisition time."""

    video = path.expanduser().resolve()
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video for timing inspection: {video}")
        metadata_fps = float(capture.get(cv2.CAP_PROP_FPS))
        metadata_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded_timestamps: list[float] = []
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            decoded_timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
    finally:
        capture.release()
    pts_are_usable = usable_decoded_timestamps(decoded_timestamps)
    decoded_count = len(decoded_timestamps)
    source_count = metadata_count or decoded_count
    if fps_override is not None:
        fps = float(fps_override)
        media_timestamps = [index / fps for index in range(source_count)]
        timing_source = "explicit_fps_override"
    elif pts_are_usable:
        media_timestamps = normalize_timestamps(decoded_timestamps)
        media_durations = timestamp_durations(media_timestamps, metadata_fps)
        fps = (
            float(metadata_fps)
            if np.isfinite(metadata_fps) and metadata_fps > 0
            else 1.0 / float(np.median(media_durations))
        )
        timing_source = "opencv_decoded_pts"
    else:
        fps = float(metadata_fps)
        media_timestamps = [index / fps for index in range(source_count)] if fps > 0 else []
        timing_source = "video_metadata_cfr_fallback"
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Video has no valid FPS metadata and no override was supplied: {video}")
    media_durations = timestamp_durations(media_timestamps, fps)
    full_rate_intent = quicktime_full_frame_rate_playback_intent(video)
    if full_rate_intent is False and capture_fps_override is None:
        raise ValueError(
            "Apple slow-motion metadata is present, but the real-world capture FPS is unknown. "
            "Supply capture_fps_override (for example 120 or 240); playback FPS is not sufficient."
        )
    real_world_fps = float(capture_fps_override or fps)
    time_scale = fps / real_world_fps
    real_world_timestamps = [value * time_scale for value in media_timestamps]
    real_world_durations = [value * time_scale for value in media_durations]
    timing_resolved = True
    calibration_source = (
        "explicit_capture_fps_override"
        if capture_fps_override is not None
        else "media_timeline_assumed_real_time"
    )

    indices = [int(value) for value in frame_indices]
    duplicates = sorted({value for value in indices if indices.count(value) > 1})
    invalid = [value for value in indices if value < 0 or (source_count and value >= source_count)]
    if invalid:
        raise ValueError(f"Analyzed frame indices are outside the source video: {invalid[:10]}")
    undecoded = [value for value in indices if value >= len(real_world_timestamps)]
    if undecoded:
        raise ValueError(f"Analyzed frames have no decoded timestamp: {undecoded[:10]}")
    unique = sorted(set(indices))
    missing = sorted(set(range(unique[0], unique[-1] + 1)).difference(unique)) if unique else []
    duration = (
        float(np.median(real_world_durations)) if real_world_durations else 1.0 / real_world_fps
    )
    media_duration = float(np.median(media_durations)) if media_durations else 1.0 / fps
    selected_durations = [real_world_durations[value] for value in indices]
    selected_media_durations = [media_durations[value] for value in indices]
    variable = variable_frame_rate(media_durations)
    return VideoTimebase(
        video_path=str(video),
        video_sha256=file_sha256(video),
        source_fps=fps,
        source_frame_count=source_count,
        metadata_frame_count=metadata_count,
        decoded_frame_count=decoded_count,
        frame_count_mismatch=bool(
            metadata_count and decoded_count and metadata_count != decoded_count
        ),
        frame_indices=indices,
        timestamps_s=[real_world_timestamps[value] for value in indices],
        frame_durations_s=selected_durations,
        frame_duration_s=duration,
        media_timestamps_s=[media_timestamps[value] for value in indices],
        media_frame_durations_s=selected_media_durations,
        media_frame_duration_s=media_duration,
        timing_source=timing_source,
        real_world_fps=real_world_fps,
        media_to_real_time_scale=time_scale,
        temporal_calibration_source=calibration_source,
        real_world_timing_resolved=timing_resolved,
        quicktime_full_frame_rate_playback_intent=full_rate_intent,
        variable_frame_rate=variable,
        fps_override=fps_override,
        missing_source_frames=missing,
        unanalyzed_source_frames=(
            sorted(set(range(source_count)).difference(unique)) if source_count else []
        ),
        duplicate_frame_indices=duplicates,
    )


def quicktime_full_frame_rate_playback_intent(path: Path) -> bool | None:
    """Read Apple's slow-motion playback-intent flag from a MOV/MP4 `mdta` atom."""

    try:
        with path.open("rb") as handle:
            for atom_type, payload_start, payload_size in iter_file_atoms(handle):
                if atom_type != b"moov":
                    continue
                handle.seek(payload_start)
                return parse_full_frame_rate_intent(handle.read(payload_size))
    except (OSError, ValueError, struct.error):
        return None
    return None


def iter_file_atoms(handle):
    handle.seek(0, 2)
    end = handle.tell()
    position = 0
    while position + 8 <= end:
        handle.seek(position)
        header = handle.read(16)
        size, atom_type = struct.unpack(">I4s", header[:8])
        header_size = 8
        if size == 1:
            size = struct.unpack(">Q", header[8:16])[0]
            header_size = 16
        elif size == 0:
            size = end - position
        if size < header_size or position + size > end:
            raise ValueError("Invalid QuickTime atom size")
        yield atom_type, position + header_size, size - header_size
        position += size


def parse_full_frame_rate_intent(moov_payload: bytes) -> bool | None:
    target = "com.apple.quicktime.full-frame-rate-playback-intent"
    for atom_type, payload in child_atoms(moov_payload):
        if atom_type != b"meta":
            continue
        children = list(child_atoms(payload))
        if not children and len(payload) >= 4:
            children = list(child_atoms(payload[4:]))
        keys_payload = next((value for kind, value in children if kind == b"keys"), None)
        values_payload = next((value for kind, value in children if kind == b"ilst"), None)
        if keys_payload is None or values_payload is None or len(keys_payload) < 8:
            continue
        count = struct.unpack(">I", keys_payload[4:8])[0]
        keys: list[str] = []
        position = 8
        for _ in range(count):
            if position + 8 > len(keys_payload):
                break
            size = struct.unpack(">I", keys_payload[position : position + 4])[0]
            if size < 8 or position + size > len(keys_payload):
                break
            keys.append(keys_payload[position + 8 : position + size].decode("utf-8", "replace"))
            position += size
        if target not in keys:
            continue
        wanted_index = keys.index(target) + 1
        for value_type, value_payload in child_atoms(values_payload):
            if int.from_bytes(value_type, "big") != wanted_index:
                continue
            data_payload = next(
                (value for kind, value in child_atoms(value_payload) if kind == b"data"), None
            )
            if data_payload is None or len(data_payload) < 9:
                return None
            return bool(int.from_bytes(data_payload[8:], "big"))
    return None


def child_atoms(data: bytes):
    position = 0
    while position + 8 <= len(data):
        size, atom_type = struct.unpack(">I4s", data[position : position + 8])
        header_size = 8
        if size == 1:
            if position + 16 > len(data):
                return
            size = struct.unpack(">Q", data[position + 8 : position + 16])[0]
            header_size = 16
        elif size == 0:
            size = len(data) - position
        if size < header_size or position + size > len(data):
            return
        yield atom_type, data[position + header_size : position + size]
        position += size


def interval_duration_s(timebase: VideoTimebase, start_frame: int, end_frame: int) -> float:
    """Inclusive duration of a frame interval."""

    lookup = timebase.timestamp_map()
    durations = timebase.duration_map()
    if start_frame not in lookup or end_frame not in lookup:
        raise ValueError(f"Frames {start_frame}..{end_frame} are not in the analyzed timebase")
    return round(max(lookup[end_frame] - lookup[start_frame] + durations[end_frame], 0.0), 12)


def elapsed_s(timebase: VideoTimebase, start_frame: int, end_frame: int) -> float:
    """Elapsed time between two frame timestamps (non-inclusive)."""

    lookup = timebase.timestamp_map()
    if start_frame not in lookup or end_frame not in lookup:
        raise ValueError(f"Frames {start_frame} and {end_frame} are not in the analyzed timebase")
    return round(max(lookup[end_frame] - lookup[start_frame], 0.0), 12)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    return artifacts.file_sha256(path, chunk_size)


def usable_decoded_timestamps(values: list[float]) -> bool:
    if len(values) < 2 or not np.isfinite(values).all():
        return False
    differences = np.diff(np.asarray(values, np.float64))
    return bool(np.all(differences > 0))


def normalize_timestamps(values: list[float]) -> list[float]:
    origin = float(values[0])
    return [float(value - origin) for value in values]


def timestamp_durations(timestamps: list[float], nominal_fps: float) -> list[float]:
    if len(timestamps) >= 2:
        differences = np.diff(np.asarray(timestamps, np.float64))
        positive = differences[differences > 0]
        fallback = (
            float(np.median(positive))
            if len(positive)
            else 1.0 / nominal_fps
            if nominal_fps > 0
            else 1.0 / 30.0
        )
        return [*map(float, differences), fallback]
    fallback = 1.0 / nominal_fps if nominal_fps > 0 else 1.0 / 30.0
    return [fallback] * len(timestamps)


def variable_frame_rate(durations: list[float], tolerance: float = 0.01) -> bool:
    if len(durations) < 3:
        return False
    values = np.asarray(durations[:-1], np.float64)
    median = float(np.median(values))
    return bool(median > 0 and np.max(np.abs(values - median)) / median > tolerance)
