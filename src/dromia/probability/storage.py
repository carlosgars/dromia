"""Compact storage for native posterior heatmap evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

LOWER_BODY_JOINT_IDS = np.arange(11, 17, dtype=np.int16)


def save_compact_posterior(
    path: Path,
    maps: np.ndarray,
    *,
    crop_xyxy: np.ndarray,
    bbox_xyxy: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Save only lower-body maps as float16; this is fast and sufficient downstream."""

    values = np.asarray(maps, dtype=np.float32)[LOWER_BODY_JOINT_IDS]
    np.savez_compressed(
        path,
        posterior=values.astype(np.float16),
        joint_ids=LOWER_BODY_JOINT_IDS,
        crop_xyxy=np.asarray(crop_xyxy, dtype=np.float32),
        bbox_xyxy=np.asarray(bbox_xyxy, dtype=np.float32),
        heatmap_metadata=np.asarray(json.dumps(metadata), dtype="<U2048"),
        storage_format=np.asarray("compact_lower_body_float16_v1"),
    )


def load_posterior_joint(data: Any, joint_id: int) -> np.ndarray | None:
    maps = np.asarray(data["posterior"], dtype=np.float32)
    joint_ids = np.asarray(data["joint_ids"], dtype=np.int16)
    positions = np.flatnonzero(joint_ids == joint_id)
    return maps[int(positions[0])] if len(positions) else None
