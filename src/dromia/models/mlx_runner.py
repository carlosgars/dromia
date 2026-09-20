"""Minimal SAM3.1 MLX video runner."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict


class MlxSamRunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: Path
    output: Path
    artifact_root: Path
    prompts: tuple[str, ...]
    model: str
    model_identity: str | None = None
    model_sha256: str | None = None
    revision: str
    threshold: float = 0.4
    resolution: int = 1008
    detect_every: int = 10
    memory_every: int = 3
    memory_mode: str = "off"
    opacity: float = 0.5
    codec: str = "mp4v"
    write_preview: bool = True


def run_video(cfg: MlxSamRunConfig) -> dict[str, object]:
    import cv2
    import mlx.core as mx
    from mlx_vlm.generate import wired_limit
    from mlx_vlm.models.sam3.generate import Sam3Predictor, SimpleTracker
    from mlx_vlm.models.sam3_1.generate import (
        _detect_with_backbone,
        _get_backbone_features,
        _init_tracker_memory,
        _propagate_tracker,
    )
    from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
    from mlx_vlm.utils import get_model_path, load_model

    if not cfg.source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {cfg.source}")

    model_path = get_model_path(cfg.model, revision=cfg.revision)
    model = load_model(model_path)
    processor = Sam31Processor.from_pretrained(str(model_path))
    predictor = Sam3Predictor(model, processor, score_threshold=cfg.threshold)

    capture = cv2.VideoCapture(str(cfg.source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {cfg.source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if cfg.write_preview:
        # High-speed source FPS values can produce an MP4 timebase denominator above 65535.
        # The preview is diagnostic only; cache frames retain the original frame indices.
        preview_fps = min(max(round(fps), 1), 60)
        candidate = cv2.VideoWriter(
            str(cfg.output),
            cv2.VideoWriter_fourcc(*cfg.codec),
            preview_fps,
            (width, height),
        )
        if candidate.isOpened():
            writer = candidate
        else:
            candidate.release()
            print(f"Warning: could not create optional SAM preview video: {cfg.output}")

    memory_bank = []
    tracked_labels: list[str] = []
    previous_areas = np.zeros((0,), dtype=np.float32)
    n_objects = 0
    propagation_count = 0
    id_tracker = SimpleTracker()
    prompt_list = list(cfg.prompts)
    multiplex_count = int(model.config.tracker_config.multiplex_count)
    frames_written = 0
    started = time.perf_counter()

    try:
        with wired_limit(model):
            for frame_index in range(total_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs = processor.preprocess_image(rgb_frame)
                backbone = _get_backbone_features(model, mx.array(inputs["pixel_values"]))
                image_size = (width, height)
                should_detect = (
                    cfg.memory_mode == "off"
                    or not memory_bank
                    or n_objects == 0
                    or frame_index % cfg.detect_every == 0
                )
                if should_detect:
                    result = _detect_with_backbone(
                        predictor,
                        backbone,
                        prompt_list,
                        image_size,
                        cfg.threshold,
                        encoder_cache={},
                    )
                    result = limit_multiplex_result(result, multiplex_count)
                    if len(result.scores):
                        n_objects = len(result.scores)
                        tracked_labels = list(result.labels or prompt_list)
                        previous_areas = result.masks.reshape(n_objects, -1).mean(axis=1)
                        propagation_count = 0
                        if cfg.memory_mode != "off":
                            memory_bank = _init_tracker_memory(model, backbone, list(result.masks))
                    else:
                        memory_bank = []
                        n_objects = 0
                        tracked_labels = []
                        previous_areas = np.zeros((0,), dtype=np.float32)
                else:
                    result, updated_memory = _propagate_tracker(
                        model,
                        backbone,
                        memory_bank,
                        n_objects,
                        image_size,
                    )
                    if propagation_is_plausible(result.masks, previous_areas):
                        result.labels = tracked_labels[: len(result.scores)]
                        previous_areas = result.masks.reshape(n_objects, -1).mean(axis=1)
                        propagation_count += 1
                        if propagation_count % cfg.memory_every == 0:
                            memory_bank = updated_memory
                    else:
                        result = _detect_with_backbone(
                            predictor,
                            backbone,
                            prompt_list,
                            image_size,
                            cfg.threshold,
                            encoder_cache={},
                        )
                        result = limit_multiplex_result(result, multiplex_count)

                result = id_tracker.update(result)
                write_frame_artifact(cfg.artifact_root, frame_index=frame_index, result=result)
                if writer is not None:
                    writer.write(annotate_masks(frame, result, opacity=cfg.opacity))
                frames_written += 1
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(
                    f"\rSAM3.1 frame {frame_index}/{total_frames - 1} "
                    f"{frames_written / elapsed:.2f} fps",
                    end="",
                    flush=True,
                )
    finally:
        print()
        capture.release()
        if writer is not None:
            writer.release()

    write_manifest(
        cfg,
        fps=fps,
        resolution=(width, height),
        frames_written=frames_written,
        preview_written=writer is not None,
    )
    return {"frames_written": frames_written, "artifact_root": str(cfg.artifact_root.resolve())}


def write_frame_artifact(root: Path, *, frame_index: int, result: object) -> None:
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    track_ids = (
        np.asarray(result.track_ids, dtype=np.int32)
        if getattr(result, "track_ids", None) is not None
        else np.arange(len(result.scores), dtype=np.int32)
    )
    np.savez_compressed(
        frames_dir / f"frame_{frame_index:06d}.npz",
        frame_index=np.asarray(frame_index, dtype=np.int32),
        boxes=np.asarray(result.boxes, dtype=np.float32),
        masks=np.asarray(result.masks, dtype=np.uint8),
        scores=np.asarray(result.scores, dtype=np.float32),
        labels=np.asarray(result.labels or [], dtype="<U64"),
        track_ids=track_ids,
    )


def write_manifest(
    cfg: MlxSamRunConfig,
    *,
    fps: float,
    resolution: tuple[int, int],
    frames_written: int,
    preview_written: bool = True,
) -> None:
    cfg.artifact_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "dromia_sam31_mlx_v2",
        "source_filename": cfg.source.name,
        "video_sha256": file_sha256(cfg.source),
        "model": cfg.model_identity or cfg.model,
        "model_revision": cfg.revision,
        "model_sha256": cfg.model_sha256,
        "prompts": list(cfg.prompts),
        "threshold": cfg.threshold,
        "inference_resolution": cfg.resolution,
        "detect_every": cfg.detect_every,
        "memory_every": cfg.memory_every,
        "memory_mode": cfg.memory_mode,
        "frame_range": [0, frames_written],
        "frames_written": frames_written,
        "complete": True,
        "preview_written": preview_written,
        "fps": fps,
        "video_resolution": list(resolution),
        "frame_files": "frames/frame_XXXXXX.npz",
    }
    (cfg.artifact_root / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotate_masks(frame: np.ndarray, result: object, *, opacity: float) -> np.ndarray:
    import cv2

    output = frame.copy()
    overlay = frame.copy()
    labels = result.labels or ["object"] * len(result.scores)
    track_ids = result.track_ids if result.track_ids is not None else np.arange(len(result.scores))
    for idx, _score in enumerate(result.scores):
        label = labels[idx] if idx < len(labels) else "object"
        mask = np.asarray(result.masks[idx], dtype=np.uint8)
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        color = (30, 190, 255) if "shoe" in label.casefold() else (80, 220, 80)
        overlay[mask > 0] = color
        ys, xs = np.where(mask > 0)
        if len(xs):
            prefix = "S" if "shoe" in label.casefold() else "R"
            cv2.putText(
                output,
                f"{prefix}#{int(track_ids[idx])}",
                (int(xs.min()), max(12, int(ys.min()) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.addWeighted(overlay, opacity, output, 1.0 - opacity, 0, output)
    return output


def propagation_is_plausible(
    masks: np.ndarray,
    previous_areas: np.ndarray,
    *,
    max_frame_fraction: float = 0.35,
    max_growth: float = 12.0,
) -> bool:
    if len(masks) != len(previous_areas) or not len(masks):
        return False
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    allowed = np.minimum(max_frame_fraction, previous_areas * max_growth + 0.01)
    return bool(np.all(areas > 0) and np.all(areas <= allowed))


def limit_multiplex_result(result: object, multiplex_count: int) -> object:
    if len(result.scores) <= multiplex_count:
        return result
    keep = np.argsort(result.scores)[::-1][:multiplex_count]
    result.boxes = result.boxes[keep]
    result.masks = result.masks[keep]
    result.scores = result.scores[keep]
    result.labels = [result.labels[i] for i in keep]
    return result
