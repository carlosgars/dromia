from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dromia import config
from dromia.models import cache


def test_v2_cache_validates_content_and_checksums(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    root = write_cache(tmp_path / "cache", video)
    assert cache.validate_cache(root, video, cfg=config.DromiaConfig()).valid
    with (root / "frames" / "frame_000000.npz").open("ab") as handle:
        handle.write(b"corrupt")
    result = cache.validate_cache(root, video, cfg=config.DromiaConfig())
    assert not result.valid
    assert result.reason == "checksum mismatch: frames/frame_000000.npz"


def test_cache_rejects_legacy_contract(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    root = tmp_path / "cache"
    write_frame(root / "frames" / "frame_000000.npz")
    (root / "manifest.json").write_text(json.dumps({"format": "dromia_sam31_mlx_v1"}))
    result = cache.validate_cache(root, video, cfg=config.DromiaConfig())
    assert not result.valid
    assert result.reason == "unsupported cache format"


def test_masks_can_be_loaded_lazily(tmp_path: Path) -> None:
    path = tmp_path / "frame_000000.npz"
    write_frame(path)
    frame = cache.load_frame(path, lazy_masks=True, mask_shape=(4, 5))
    assert isinstance(frame.detections[0].mask, cache.LazyNpzMask)
    assert np.asarray(frame.detections[0].mask).shape == (4, 5)


def write_cache(root: Path, video: Path) -> Path:
    frame = root / "frames" / "frame_000000.npz"
    write_frame(frame)
    cfg = config.DromiaConfig().sam
    payload = {
        "format": "dromia_sam31_mlx_v2",
        "model": cfg.model_id,
        "model_revision": cfg.revision,
        "model_sha256": cfg.model_sha256,
        "prompts": list(cfg.prompts),
        "threshold": cfg.threshold,
        "inference_resolution": cfg.resolution,
        "detect_every": cfg.detect_every,
        "memory_every": cfg.memory_every,
        "memory_mode": cfg.memory_mode,
        "video_sha256": cache.file_sha256(video),
        "complete": True,
        "frames_written": 1,
    }
    root.mkdir(exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(payload))
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    (root / "checksums.sha256").write_text(f"{digest}  frames/frame_000000.npz\n")
    return root


def write_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_index=np.asarray(0),
        boxes=np.asarray([[0, 0, 5, 4]], dtype=np.float32),
        masks=np.ones((1, 4, 5), dtype=np.uint8),
        scores=np.asarray([0.9], dtype=np.float32),
        labels=np.asarray(["Runner running"]),
        track_ids=np.asarray([1], dtype=np.int32),
    )
