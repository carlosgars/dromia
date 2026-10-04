"""SAM3.1 MLX artifact cache."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from dromia import artifacts
from dromia import config as dromia_config
from dromia import dto as dromia_dto
from dromia.models import mlx_runner as sam31_mlx_runner


def ensure_sam_cache(
    video_path: Path,
    *,
    cfg: dromia_config.DromiaConfig,
    run_dir: Path,
) -> Path:
    cache_root = run_dir / "evidence" / "segmentation"
    manifest_path = cache_root / "manifest.json"
    if cache_is_valid(manifest_path, video_path, cfg=cfg):
        return cache_root

    cache_root.mkdir(parents=True, exist_ok=True)
    video_out = run_dir / "videos" / "sam_masks.mp4"
    video_out.parent.mkdir(parents=True, exist_ok=True)
    model_root = cfg.repo_root / "models" / "downloads" / "sam3.1-bf16"
    if not model_root.is_dir():
        raise FileNotFoundError("SAM model is missing; run `uv run dromia models install`")
    sam31_mlx_runner.run_video(
        sam31_mlx_runner.MlxSamRunConfig(
            source=video_path.resolve(),
            output=video_out.resolve(),
            artifact_root=cache_root.resolve(),
            prompts=cfg.sam.prompts,
            model=str(model_root),
            model_identity=cfg.sam.model_id,
            model_sha256=cfg.sam.model_sha256,
            revision=cfg.sam.revision,
            threshold=cfg.sam.threshold,
            resolution=cfg.sam.resolution,
            detect_every=cfg.sam.detect_every,
            memory_every=cfg.sam.memory_every,
            memory_mode=cfg.sam.memory_mode,
        )
    )
    return cache_root


def cache_is_valid(
    manifest_path: Path, video_path: Path, *, cfg: dromia_config.DromiaConfig
) -> bool:
    return validate_cache(manifest_path.parent, video_path, cfg=cfg).valid


class CacheValidation:
    def __init__(self, valid: bool, reason: str | None = None) -> None:
        self.valid = valid
        self.reason = reason


def validate_cache(
    cache_root: Path, video_path: Path, *, cfg: dromia_config.DromiaConfig
) -> CacheValidation:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        return CacheValidation(False, "manifest.json is missing")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return CacheValidation(False, "manifest.json is not valid JSON")
    frames = frame_paths(cache_root)
    if not frames:
        return CacheValidation(False, "cache contains no frame artifacts")
    if data.get("format") == "dromia_sam31_mlx_v2":
        expected = {
            "model": cfg.sam.model_id,
            "model_revision": cfg.sam.revision,
            "model_sha256": cfg.sam.model_sha256,
            "prompts": list(cfg.sam.prompts),
            "threshold": float(cfg.sam.threshold),
            "inference_resolution": int(cfg.sam.resolution),
            "detect_every": int(cfg.sam.detect_every),
            "memory_every": int(cfg.sam.memory_every),
            "memory_mode": cfg.sam.memory_mode,
        }
        mismatch = manifest_mismatch(data, expected)
        if mismatch is not None:
            return CacheValidation(False, mismatch)
        return validate_portable_frames(data, frames, cache_root, video_path)

    return CacheValidation(False, "unsupported cache format")


def manifest_mismatch(data: dict[str, object], expected: dict[str, object]) -> str | None:
    for field, value in expected.items():
        if data.get(field) != value:
            return f"{field} does not match DromIA configuration"
    return None


def validate_portable_frames(
    data: dict[str, object], frames: list[Path], cache_root: Path, video_path: Path
) -> CacheValidation:
    if data.get("video_sha256") != file_sha256(video_path):
        return CacheValidation(False, "video SHA-256 does not match")
    if data.get("complete") is not True:
        return CacheValidation(False, "cache is not marked complete")
    frames_written = int(data.get("frames_written", -1))
    if frames_written != len(frames):
        return CacheValidation(False, "frame artifact count does not match manifest")
    indices = [frame_index(path) for path in frames]
    if indices != list(range(frames_written)):
        return CacheValidation(False, "frame artifacts are not contiguous from zero")
    checksum_error = verify_checksums(cache_root)
    if checksum_error is not None:
        return CacheValidation(False, checksum_error)
    return CacheValidation(True)


def frame_paths(cache_root: Path) -> list[Path]:
    return sorted((cache_root / "frames").glob("frame_*.npz"))


def included_runner_ids(cache_root: Path) -> set[int]:
    """Return runner IDs deliberately retained by a remapped clip cache."""

    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        return set()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        values = payload.get("included_runner_ids", [])
        return {int(value) for value in values}
    except (json.JSONDecodeError, TypeError, ValueError):
        return set()


def frame_index(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("frame_"))
    except ValueError as exc:
        raise ValueError(f"Invalid SAM frame artifact name: {path.name}") from exc


class LazyNpzMask:
    """Array-compatible view of one mask stored in a portable frame artifact."""

    __slots__ = ("path", "index", "shape")

    def __init__(self, path: Path, index: int, shape: tuple[int, int]) -> None:
        self.path = path.resolve()
        self.index = index
        self.shape = shape

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        array = _cached_frame_masks(str(self.path))[self.index]
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy:
            array = array.copy()
        return array


@lru_cache(maxsize=4)
def _cached_frame_masks(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["masks"], dtype=np.uint8)


def load_frame(
    path: Path,
    *,
    lazy_masks: bool = False,
    mask_shape: tuple[int, int] | None = None,
) -> dromia_dto.SamFrame:
    with np.load(path, allow_pickle=False) as data:
        frame_idx = int(data["frame_index"])
        boxes = np.asarray(data["boxes"], dtype=np.float32)
        masks = None if lazy_masks else np.asarray(data["masks"], dtype=np.uint8)
        scores = np.asarray(data["scores"], dtype=np.float32)
        labels = [str(item) for item in data["labels"].tolist()]
        track_ids = np.asarray(data["track_ids"], dtype=np.int32)
    if lazy_masks and mask_shape is None:
        raise ValueError("mask_shape is required when loading masks lazily")
    detections: list[dromia_dto.SamDetection] = []
    for idx, label in enumerate(labels):
        mask = (
            LazyNpzMask(path, idx, mask_shape)
            if lazy_masks and mask_shape is not None
            else np.asarray(masks[idx], dtype=np.uint8)
        )
        detections.append(
            dromia_dto.SamDetection(
                obj_id=int(track_ids[idx]),
                frame_idx=frame_idx,
                label=label,
                score=float(scores[idx]),
                bbox_xyxy=np.asarray(boxes[idx], dtype=np.float32),
                mask=mask,
            )
        )
    return dromia_dto.SamFrame(frame_idx=frame_idx, detections=detections)


def file_sha256(path: Path) -> str:
    return artifacts.file_sha256(path)


def verify_checksums(cache_root: Path) -> str | None:
    checksum_path = cache_root / "checksums.sha256"
    if not checksum_path.exists():
        return None
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            return f"checksums.sha256 line {line_number} is malformed"
        path = cache_root / relative
        if not path.is_file():
            return f"checksummed file is missing: {relative}"
        if file_sha256(path) != expected:
            return f"checksum mismatch: {relative}"
    return None
