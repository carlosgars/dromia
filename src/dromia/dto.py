"""Small Pydantic DTOs shared by pipeline stages."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ArrayModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SamDetection(ArrayModel):
    obj_id: int
    frame_idx: int
    label: str
    score: float
    bbox_xyxy: np.ndarray
    # Large portable caches may supply an on-demand object implementing
    # ``__array__`` instead of retaining every full-resolution mask in RAM.
    mask: Any


class SamFrame(ArrayModel):
    frame_idx: int
    detections: list[SamDetection]

    @property
    def runners(self) -> list[SamDetection]:
        return [
            item for item in self.detections if normalized_label(item.label) == "runner running"
        ]

    @property
    def shoes(self) -> list[SamDetection]:
        return [item for item in self.detections if "shoe" in item.label.casefold()]


class RunnerTrackDecision(BaseModel):
    obj_id: int
    accepted: bool
    reason: str
    frame_count: int
    first_frame: int
    last_frame: int
    mean_score: float
    displacement_px: float
    displacement_fraction: float
    track_fraction: float


class PoseObservation(ArrayModel):
    frame_idx: int
    obj_id: int
    keypoints_xy: np.ndarray
    confidence: np.ndarray
    heatmaps: np.ndarray | None
    heatmap_metadata: dict[str, Any] = Field(default_factory=dict)
    crop_xyxy: np.ndarray
    bbox_xyxy: np.ndarray
    heatmap_confidence: np.ndarray | None = None
    presence_probability: np.ndarray | None = None
    visibility_probability: np.ndarray | None = None
    normalized_localization_error: np.ndarray | None = None


class ShoeAssignment(ArrayModel):
    frame_idx: int
    runner_id: int
    side: str
    shoe_obj_id: int | None
    score: float
    mask: Any = None


class PosteriorFrame(ArrayModel):
    frame_idx: int
    obj_id: int
    keypoints_xy: np.ndarray
    peak_probability: np.ndarray
    entropy: np.ndarray


class SourceVideo(BaseModel):
    name: str
    sha256: str
    path: str


class RunManifestV1(BaseModel):
    schema_version: Literal[1] = 1
    run_id: str
    source_video: SourceVideo
    frame_count: int
    accepted_runner_ids: list[int]
    runner_decisions: list[RunnerTrackDecision]
    config_fingerprint: str
    model_fingerprints: dict[str, str | None]
    timing: dict[str, Any]
    stage_durations_s: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)

    @property
    def input_video(self) -> str:
        return self.source_video.path

    @property
    def sam_cache_dir(self) -> str:
        path = self.artifacts.get("sam_evidence")
        if path is None:
            raise ValueError("Run manifest has no SAM evidence")
        return path


RunManifest = RunManifestV1


def normalized_label(label: str) -> str:
    return " ".join(label.casefold().split())
