"""DromIA adapter for PMPose-B from BBoxMaskPose."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from dromia import config as dromia_config
from dromia import dto as dromia_dto
from dromia.models import crops as pose_crops


class PMPoseRunner:
    def __init__(self, cfg: dromia_config.PoseConfig) -> None:
        self.cfg = cfg
        self._model: Any | None = None
        self._inference_topdown: Any | None = None
        self._torch: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        root = self.cfg.pmpose_root.expanduser().resolve()
        checkpoint = self.cfg.pmpose_checkpoint_path.expanduser().resolve()
        config_path = root / "mmpose" / "configs" / "ProbMaskPose" / "PMPose-b-1.0.0.py"
        metainfo_path = (
            root / "mmpose" / "configs" / "_base_" / "datasets" / "merged_COCO_AIC_MPII_mergable.py"
        )
        for required in (root, checkpoint, config_path, metainfo_path):
            if not required.exists():
                raise FileNotFoundError(f"Required PMPose artifact does not exist: {required}")

        value = str(root)
        if value not in sys.path:
            sys.path.insert(0, value)

        import torch
        from mmengine import Config
        from mmpose.apis import inference_topdown, init_model

        config = Config.fromfile(str(config_path))
        for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
            loader = config.get(loader_name)
            if loader is not None:
                loader.dataset.metainfo.from_file = str(metainfo_path)
        config.model.test_cfg.output_heatmaps = True
        self._model = init_model(config, str(checkpoint), device=self.cfg.device)
        self._inference_topdown = inference_topdown
        self._torch = torch

    def predict(
        self,
        frame_bgr: np.ndarray,
        runner: dromia_dto.SamDetection,
    ) -> tuple[dromia_dto.PoseObservation, pose_crops.MaskedCrop]:
        prediction, crop = self.predict_diagnostics(frame_bgr, runner)
        return (
            dromia_dto.PoseObservation(
                frame_idx=runner.frame_idx,
                obj_id=runner.obj_id,
                keypoints_xy=prediction.keypoints_xy[:17],
                confidence=prediction.oks_confidence[:17],
                heatmaps=prediction.heatmaps[:17],
                heatmap_metadata={
                    "processor_size": [256, 192],
                    "heatmap_space": "pmpose_affine_heatmap",
                    "heatmap_values": "probability",
                    "input_padding": 1.25,
                    "exact_affine_registration": self.cfg.exact_pmpose_heatmap_registration,
                    "input_center_xy": prediction.input_center_xy.tolist(),
                    "input_scale_xy": prediction.input_scale_xy.tolist(),
                    "input_size_xy": prediction.input_size_xy.tolist(),
                    "model_variant": self.cfg.variant,
                },
                crop_xyxy=crop.crop_xyxy,
                bbox_xyxy=runner.bbox_xyxy,
                heatmap_confidence=prediction.heatmap_confidence[:17],
                presence_probability=prediction.presence_probability[:17],
                visibility_probability=prediction.visibility_probability[:17],
                normalized_localization_error=prediction.normalized_error[:17],
            ),
            crop,
        )

    def predict_diagnostics(
        self,
        frame_bgr: np.ndarray,
        runner: dromia_dto.SamDetection,
    ) -> tuple[PMPoseDiagnostics, pose_crops.MaskedCrop]:
        """Return every PMPose scalar head together with its spatial maps."""
        self.load()
        assert self._model is not None and self._inference_topdown is not None
        crop = pose_crops.build_masked_crop(
            frame_bgr,
            runner.bbox_xyxy,
            runner.mask,
            padding=self.cfg.crop_padding,
            blur_kernel=self.cfg.blur_kernel,
        )
        mask = pose_crops.prepare_binary_mask(runner.mask, frame_shape=frame_bgr.shape[:2])
        results = self._inference_topdown(
            self._model,
            frame_bgr,
            bboxes=np.asarray([runner.bbox_xyxy], dtype=np.float32),
            masks=np.asarray([mask], dtype=np.uint8),
            bbox_format="xyxy",
        )
        if not results:
            raise RuntimeError(
                f"PMPose produced no result for frame {runner.frame_idx}, runner {runner.obj_id}"
            )
        prediction = extract_pmpose_diagnostics(results[0])
        return prediction, crop

    def close(self) -> None:
        self._model = None
        self._inference_topdown = None
        if self._torch is not None and self.cfg.device == "mps":
            self._torch.mps.empty_cache()
        self._torch = None


@dataclass(frozen=True, slots=True)
class PMPoseDiagnostics:
    keypoints_xy: np.ndarray
    oks_confidence: np.ndarray
    heatmap_confidence: np.ndarray
    presence_probability: np.ndarray
    visibility_probability: np.ndarray
    normalized_error: np.ndarray
    heatmaps: np.ndarray
    input_center_xy: np.ndarray
    input_scale_xy: np.ndarray
    input_size_xy: np.ndarray

    @property
    def visibility_binary(self) -> np.ndarray:
        return (self.visibility_probability >= 0.5).astype(np.uint8)


def extract_pmpose_diagnostics(result: Any) -> PMPoseDiagnostics:
    instances = result.pred_instances
    keypoints = np.asarray(instances.keypoints, dtype=np.float32).reshape(-1, 2)
    count = keypoints.shape[0]
    oks_confidence = scalar_field(instances, "keypoint_scores", count)
    heatmap_confidence = scalar_field(instances, "keypoints_conf", count)
    presence = scalar_field(instances, "keypoints_probs", count)
    visibility = scalar_field(instances, "keypoints_visible", count)
    error = scalar_field(instances, "keypoints_error", count, clip=False)
    heatmaps_raw = result.pred_fields.heatmaps
    if hasattr(heatmaps_raw, "detach"):
        heatmaps_raw = heatmaps_raw.detach().cpu().numpy()
    heatmaps = np.asarray(heatmaps_raw, dtype=np.float32)
    if heatmaps.ndim != 3:
        raise ValueError(f"Expected PMPose heatmaps [K,H,W], got {heatmaps.shape}")
    metainfo = result.metainfo
    input_center = np.asarray(metainfo["input_center"], dtype=np.float32).reshape(-1, 2)[0]
    input_scale = np.asarray(metainfo["input_scale"], dtype=np.float32).reshape(-1, 2)[0]
    input_size = np.asarray(metainfo["input_size"], dtype=np.float32).reshape(2)
    return PMPoseDiagnostics(
        keypoints_xy=keypoints,
        oks_confidence=oks_confidence,
        heatmap_confidence=heatmap_confidence,
        presence_probability=presence,
        visibility_probability=visibility,
        normalized_error=error,
        heatmaps=heatmaps,
        input_center_xy=input_center,
        input_scale_xy=input_scale,
        input_size_xy=input_size,
    )


def scalar_field(instances: Any, name: str, count: int, *, clip: bool = True) -> np.ndarray:
    value = getattr(instances, name, np.zeros(count, dtype=np.float32))
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != count:
        raise ValueError(f"Expected PMPose {name} to contain {count} values, got {array.size}")
    return np.clip(array, 0.0, 1.0) if clip else array
