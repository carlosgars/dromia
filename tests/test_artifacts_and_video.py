from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dromia import artifacts, video


def test_run_layout_keeps_artifacts_inside_run(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "run_id": "test",
        "created_at": "2026-01-01T00:00:00Z",
        "source_video": {
            "path": "source.mp4",
            "sha256": "0" * 64,
            "size_bytes": 0,
        },
        "input_video": "source.mp4",
        "config_path": "config.json",
        "config_sha256": "0" * 64,
        "models": [],
        "accepted_runner_ids": [],
        "sam_cache_dir": "sam",
        "timing": {},
        "stage_durations_s": {},
        "artifacts": {"pose_npz": "pose/pose.npz"},
    }
    # RunLayout itself is deliberately independent of manifest validation here.
    layout = artifacts.RunLayout.open(tmp_path)
    destination = tmp_path / "state.json"
    artifacts.atomic_write_json(destination, manifest)
    assert json.loads(destination.read_text()) == manifest

    class PortableManifest:
        artifacts = {"pose_npz": "pose/pose.npz"}

    assert layout.artifact("pose_npz", PortableManifest()) == tmp_path / "pose" / "pose.npz"
    PortableManifest.artifacts = {"pose_npz": "../outside.npz"}
    with pytest.raises(ValueError, match="escapes run directory"):
        layout.artifact("pose_npz", PortableManifest())


def test_ordered_frames_decodes_forward_without_seeking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [np.full((1, 1, 3), value, dtype=np.uint8) for value in range(6)]

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            self.position = 0

        def isOpened(self) -> bool:
            return True

        def grab(self) -> bool:
            self.position += 1
            return self.position <= len(frames)

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.position >= len(frames):
                return False, None
            frame = frames[self.position]
            self.position += 1
            return True, frame

        def release(self) -> None:
            return None

    monkeypatch.setattr(video.cv2, "VideoCapture", FakeCapture)

    decoded = list(video.ordered_frames(tmp_path / "video.mp4", [0, 2, 5]))
    assert [index for index, _frame in decoded] == [0, 2, 5]
    assert [int(frame[0, 0, 0]) for _index, frame in decoded] == [0, 2, 5]


def test_ordered_frames_rejects_duplicate_or_unsorted_indices(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        list(video.ordered_frames(tmp_path / "video.mp4", [1, 1]))
    with pytest.raises(ValueError, match="strictly increasing"):
        list(video.ordered_frames(tmp_path / "video.mp4", [2, 1]))
