"""Local-window CoTracker adapter for expert correction propagation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from dromia import config

JOINT_NAMES = {13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"}


class PointSeed(BaseModel):
    runner_id: int
    keypoint_id: int
    keypoint_name: str
    seed_frame_idx: int
    seed_time_idx: int
    xy: tuple[float, float]
    confidence: float
    mask_distance_norm: float


@dataclass(slots=True)
class TrackResult:
    xy: np.ndarray
    visibility: np.ndarray
    input_size_wh: tuple[int, int]
    inference_seconds: float


def track_points(
    video_path: Path,
    frame_indices: list[int],
    seeds: list[PointSeed],
    cfg: config.CoTrackerConfig,
) -> TrackResult:
    """Track points inside a caller-bounded window using only pinned local files."""

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    positions = sampled_time_positions(len(frame_indices), seeds, cfg.temporal_stride)
    sampled_indices = [frame_indices[position] for position in positions]
    lookup = {position: index for index, position in enumerate(positions)}
    local_seeds = [
        seed.model_copy(update={"seed_time_idx": lookup[seed.seed_time_idx]}) for seed in seeds
    ]
    frames, scale_xy = read_resized_frames(video_path, sampled_indices, cfg.input_max_width)
    input_h, input_w = frames.shape[1:3]
    device = _device(cfg.device, torch)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    queries = torch.tensor(
        [
            [seed.seed_time_idx, seed.xy[0] * scale_xy[0], seed.xy[1] * scale_xy[1]]
            for seed in local_seeds
        ],
        dtype=torch.float32,
        device=device,
    )[None]
    model = _load_model(torch, device)
    started = time.perf_counter()
    with torch.inference_mode():
        predicted_xy, predicted_visibility = model(
            video,
            queries=queries,
            backward_tracking=cfg.backward_tracking,
        )
    if str(device) == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    sampled_xy = predicted_xy[0].detach().cpu().numpy().astype(np.float32).transpose(1, 0, 2)
    sampled_visibility = predicted_visibility[0].detach().cpu().numpy().astype(bool).transpose(1, 0)
    sampled_xy[..., 0] /= scale_xy[0]
    sampled_xy[..., 1] /= scale_xy[1]
    xy, visibility = interpolate_sampled_tracks(
        sampled_xy, sampled_visibility, positions, len(frame_indices)
    )
    return TrackResult(xy, visibility, (input_w, input_h), float(elapsed))


def _load_model(torch: object, device: object) -> object:
    root = config.REPO_ROOT / ".vendor" / "co-tracker"
    checkpoint = config.REPO_ROOT / "models" / "checkpoints" / "scaled_offline.pth"
    if not root.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("CoTracker is missing; run `uv run dromia models install`")
    model = torch.hub.load(str(root), "cotracker3_offline", source="local", pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.model.load_state_dict(state)
    return model.to(device).eval()


def _device(requested: str, torch: object) -> object:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(requested)


def sampled_time_positions(
    frame_count: int, seeds: list[PointSeed], temporal_stride: int
) -> list[int]:
    if frame_count <= 0:
        return []
    positions = set(range(0, frame_count, max(int(temporal_stride), 1)))
    positions.add(frame_count - 1)
    positions.update(seed.seed_time_idx for seed in seeds)
    return sorted(position for position in positions if 0 <= position < frame_count)


def interpolate_sampled_tracks(
    sampled_xy: np.ndarray,
    sampled_visibility: np.ndarray,
    positions: list[int],
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(positions) == frame_count:
        return sampled_xy, sampled_visibility
    source = np.asarray(positions, dtype=np.float32)
    target = np.arange(frame_count, dtype=np.float32)
    xy = np.empty((sampled_xy.shape[0], frame_count, 2), dtype=np.float32)
    visibility = np.empty((sampled_visibility.shape[0], frame_count), dtype=bool)
    for track in range(sampled_xy.shape[0]):
        for axis in range(2):
            xy[track, :, axis] = np.interp(target, source, sampled_xy[track, :, axis])
        visibility[track] = (
            np.interp(target, source, sampled_visibility[track].astype(np.float32)) >= 0.5
        )
    return xy, visibility


def read_resized_frames(
    video_path: Path, frame_indices: list[int], max_width: int
) -> tuple[np.ndarray, tuple[float, float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_w = min(source_w, max_width)
    target_h = max(2, int(round(source_h * target_w / max(source_w, 1))))
    target_h += target_h % 2
    frames: list[np.ndarray] = []
    try:
        for frame_idx in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_idx}")
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    scale = (target_w / source_w, target_h / source_h)
    return np.asarray(frames, dtype=np.uint8), scale
