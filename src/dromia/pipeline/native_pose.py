"""Bounded-memory staging for native PMPose evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from dromia import dto


class NativePoseStore:
    """Keep large heatmaps on disk while retaining lightweight observations in memory."""

    def __init__(self, run_dir: Path, keys: list[tuple[int, int]]) -> None:
        self._keys = sorted(set(keys))
        self._positions = {key: index for index, key in enumerate(self._keys)}
        self._observations: list[dto.PoseObservation | None] = [None] * len(self._keys)
        self._present = np.zeros(len(self._keys), dtype=bool)
        self._staging_path = run_dir / "pose" / ".pmpose_native_heatmaps.tmp.npy"
        self._export_path = run_dir / "pose" / ".pmpose_native_heatmaps.float16.tmp.npy"
        self._heatmaps: np.memmap | None = None

    def add(self, observation: dto.PoseObservation) -> None:
        key = (observation.frame_idx, observation.obj_id)
        index = self._positions[key]
        heatmaps = observation.heatmaps
        native = (
            heatmaps is not None
            and observation.heatmap_metadata.get("heatmap_space") == "pmpose_affine_heatmap"
        )
        if native:
            values = np.asarray(heatmaps, dtype=np.float32)
            if self._heatmaps is None:
                self._heatmaps = open_memmap(
                    self._staging_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(self._keys), *values.shape),
                )
            if values.shape != self._heatmaps.shape[1:]:
                raise ValueError(
                    "PMPose native heatmaps must share one shape, got "
                    f"{values.shape} and {self._heatmaps.shape[1:]}"
                )
            self._heatmaps[index] = values
            self._present[index] = True
        self._observations[index] = observation.model_copy(update={"heatmaps": None})

    def get(self, frame_idx: int, obj_id: int) -> dto.PoseObservation | None:
        index = self._positions.get((frame_idx, obj_id))
        if index is None or self._observations[index] is None:
            return None
        heatmaps = (
            self._heatmaps[index]
            if self._present[index] and self._heatmaps is not None
            else None
        )
        return self._observations[index].model_copy(update={"heatmaps": heatmaps})

    def finalize(self, run_dir: Path) -> Path | None:
        if self._heatmaps is None or not np.any(self._present):
            self.close()
            return None
        selected = np.flatnonzero(self._present)
        # Compact rare missing observations one row at a time, avoiding fancy-index copies.
        for output_index, source_index in enumerate(selected):
            if output_index != source_index:
                self._heatmaps[output_index] = self._heatmaps[source_index]
        self._heatmaps.flush()
        observations = [self._observations[index] for index in selected]
        if any(item is None for item in observations):
            raise RuntimeError("Native PMPose store contains an incomplete observation")
        items = [item for item in observations if item is not None]
        joint_count = self._heatmaps.shape[1]

        def scalar_rows(name: str) -> np.ndarray:
            rows = []
            for item in items:
                value = getattr(item, name)
                rows.append(
                    np.full(joint_count, np.nan, dtype=np.float32)
                    if value is None
                    else np.asarray(value, dtype=np.float32).reshape(joint_count)
                )
            return np.stack(rows)

        path = run_dir / "pose" / "pmpose_native_outputs.npz"
        temporary = path.with_suffix(".npz.tmp")
        export_heatmaps = open_memmap(
            self._export_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(items), *self._heatmaps.shape[1:]),
        )
        for index in range(len(items)):
            export_heatmaps[index] = self._heatmaps[index]
        export_heatmaps.flush()
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                frame_indices=np.asarray([item.frame_idx for item in items], dtype=np.int32),
                object_ids=np.asarray([item.obj_id for item in items], dtype=np.int32),
                decoded_keypoints_xy=np.stack([item.keypoints_xy for item in items]).astype(
                    np.float32
                ),
                crop_xyxy=np.stack([item.crop_xyxy for item in items]).astype(np.float32),
                bbox_xyxy=np.stack([item.bbox_xyxy for item in items]).astype(np.float32),
                oks_confidence=np.stack([item.confidence for item in items]).astype(np.float32),
                heatmap_confidence=scalar_rows("heatmap_confidence"),
                presence_probability=scalar_rows("presence_probability"),
                visibility_probability=scalar_rows("visibility_probability"),
                normalized_localization_error=scalar_rows("normalized_localization_error"),
                native_heatmaps=export_heatmaps,
                heatmap_metadata=np.asarray(
                    [json.dumps(item.heatmap_metadata) for item in items], dtype="<U2048"
                ),
                storage_format=np.asarray("pmpose_native_float16_v1"),
            )
        temporary.replace(path)
        del export_heatmaps
        self.close()
        return path

    def close(self) -> None:
        if self._heatmaps is not None:
            self._heatmaps.flush()
            self._heatmaps = None
        self._staging_path.unlink(missing_ok=True)
        self._export_path.unlink(missing_ok=True)
