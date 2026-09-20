"""Runner crop preparation."""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict


class MaskedCrop(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_bgr: np.ndarray
    bbox_xyxy: np.ndarray
    crop_xyxy: np.ndarray
    mask_crop: np.ndarray
    local_to_global: np.ndarray
    global_to_local: np.ndarray

    def points_local_to_global(self, points_xy: np.ndarray) -> np.ndarray:
        return transform_points(points_xy, self.local_to_global)

    def points_global_to_local(self, points_xy: np.ndarray) -> np.ndarray:
        return transform_points(points_xy, self.global_to_local)


def build_masked_crop(
    frame_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    mask: np.ndarray,
    *,
    padding: float,
    blur_kernel: int,
) -> MaskedCrop:
    frame = np.asarray(frame_bgr)
    frame_h, frame_w = frame.shape[:2]
    bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)
    x0, y0, x1, y1 = bbox.tolist()
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    box_w = max(float(x1 - x0), 1.0) * float(padding)
    box_h = max(float(y1 - y0), 1.0) * float(padding)
    crop_x0 = int(np.floor(max(0.0, center_x - box_w * 0.5)))
    crop_y0 = int(np.floor(max(0.0, center_y - box_h * 0.5)))
    crop_x1 = int(np.ceil(min(float(frame_w), center_x + box_w * 0.5)))
    crop_y1 = int(np.ceil(min(float(frame_h), center_y + box_h * 0.5)))
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, frame_w, frame_h

    image_crop = frame[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    full_mask = prepare_binary_mask(mask, frame_shape=(frame_h, frame_w))
    mask_crop = full_mask[crop_y0:crop_y1, crop_x0:crop_x1]
    foreground = mask_crop > 0
    kernel = valid_blur_kernel(blur_kernel)
    background = cv2.GaussianBlur(image_crop, (kernel, kernel), 0)
    background[foreground] = image_crop[foreground]

    local_to_global = np.asarray(
        [[1.0, 0.0, float(crop_x0)], [0.0, 1.0, float(crop_y0)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    global_to_local = np.asarray(
        [[1.0, 0.0, -float(crop_x0)], [0.0, 1.0, -float(crop_y0)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return MaskedCrop(
        image_bgr=background,
        bbox_xyxy=bbox,
        crop_xyxy=np.asarray([crop_x0, crop_y0, crop_x1, crop_y1], dtype=np.float32),
        mask_crop=mask_crop,
        local_to_global=local_to_global,
        global_to_local=global_to_local,
    )


def transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.size == 0:
        return points.reshape((-1, 2)).astype(np.float32)
    valid = np.isfinite(points).all(axis=1)
    result = np.full(points.shape, np.nan, dtype=np.float32)
    homo = np.concatenate([points[valid], np.ones((int(valid.sum()), 1), dtype=np.float32)], axis=1)
    mapped = homo @ np.asarray(matrix, dtype=np.float32).T
    result[valid] = mapped[:, :2]
    return result


def prepare_binary_mask(mask: np.ndarray, *, frame_shape: tuple[int, int]) -> np.ndarray:
    frame_h, frame_w = frame_shape
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.shape[:2] != (frame_h, frame_w):
        binary = cv2.resize(binary, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
    return binary


def valid_blur_kernel(kernel: int) -> int:
    value = max(int(kernel), 3)
    return value if value % 2 == 1 else value + 1
