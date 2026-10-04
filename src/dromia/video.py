"""Shared OpenCV video inspection and ordered frame decoding."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def inspect(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return VideoInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps > 0 else 30.0,
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def ordered_frames(path: Path, indices: Sequence[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield sorted source frames using forward decoding instead of random seeks."""

    requested = [int(value) for value in indices]
    if requested != sorted(set(requested)):
        raise ValueError("Frame indices must be strictly increasing")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {path}")
    try:
        current = 0
        for target in requested:
            while current < target:
                if not capture.grab():
                    return
                current += 1
            ok, frame = capture.read()
            if not ok:
                return
            yield target, frame
            current += 1
    finally:
        capture.release()
