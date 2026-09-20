"""Minimal CoTracker adapter used only for bounded pose repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from dromia import config


@dataclass(slots=True)
class PointTrackSeed:
    runner_id: int
    keypoint_id: int
    keypoint_name: str
    seed_frame_idx: int
    xy: np.ndarray
    local_point_type: str
    source_confidence: float
    inside_sam_mask: bool
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PointTrackResult:
    xy: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    tracker_name: str = "cotracker3"
    tracker_config: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class CoTracker3PointTracker:
    """Load the pinned local model; never download code or weights at runtime."""

    def __init__(self) -> None:
        self._model: object | None = None
        self._device: str | None = None

    def is_available(self) -> bool:
        return (
            config.REPO_ROOT.joinpath(".vendor", "co-tracker", "hubconf.py").is_file()
            and config.REPO_ROOT.joinpath("models", "checkpoints", "scaled_offline.pth").is_file()
        )

    def track_points(
        self,
        *,
        video_path: Path,
        frame_indices: list[int],
        seeds: list[PointTrackSeed],
        cfg: object,
    ) -> PointTrackResult:
        import torch

        frames = _read_frames(video_path, frame_indices)
        device = _device(getattr(cfg, "tracker_device", "auto"), torch)
        height, width = frames.shape[1:3]
        max_width = int(getattr(cfg, "tracker_input_max_width", 640))
        scale = min(1.0, max_width / max(width, 1))
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            frames = np.stack([cv2.resize(frame, size) for frame in frames])
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
        frame_to_t = {frame_idx: index for index, frame_idx in enumerate(frame_indices)}
        queries = torch.tensor(
            [
                [frame_to_t.get(seed.seed_frame_idx, 0), *np.asarray(seed.xy) * scale]
                for seed in seeds
            ],
            dtype=torch.float32,
            device=device,
        )[None]
        if self._model is None or self._device != str(device):
            root = config.REPO_ROOT / ".vendor" / "co-tracker"
            checkpoint = config.REPO_ROOT / "models" / "checkpoints" / "scaled_offline.pth"
            model = torch.hub.load(
                str(root), "cotracker3_offline", source="local", pretrained=False
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.model.load_state_dict(state)
            self._model = model.to(device).eval()
            self._device = str(device)
        with torch.no_grad():
            tracks, visibility = self._model(video, queries=queries)
        xy = np.transpose(tracks[0].detach().cpu().numpy(), (1, 0, 2)) / scale
        visible = visibility[0].detach().cpu().numpy()
        if visible.ndim == 3:
            visible = visible[..., 0]
        visible = np.transpose(visible, (1, 0)).astype(bool)
        return PointTrackResult(
            xy=xy.astype(np.float32),
            visibility=visible,
            confidence=visible.astype(np.float32),
            tracker_config={"device": str(device), "mode": "bounded_offline"},
        )


def _read_frames(path: Path, frame_indices: list[int]) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    try:
        for frame_idx in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_idx}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.stack(frames)


def _device(requested: str, torch: object) -> object:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
