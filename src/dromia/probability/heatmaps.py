"""Exact PMPose heatmap registration."""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-8


def normalized_heatmaps_to_crop_probability_maps_exact(
    heatmaps: np.ndarray,
    *,
    crop_xyxy: np.ndarray,
    input_center_xy: np.ndarray,
    input_scale_xy: np.ndarray,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Register PMPose probabilities through its exact source-image affine."""

    arr = np.asarray(heatmaps, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError("heatmaps must be [K,H,W]")
    crop = np.asarray(crop_xyxy, dtype=np.float32).reshape(4)
    center = np.asarray(input_center_xy, dtype=np.float32).reshape(2)
    scale = np.asarray(input_scale_xy, dtype=np.float32).reshape(2)
    if not np.isfinite(np.concatenate((crop, center, scale))).all() or np.any(scale <= 0):
        raise ValueError("PMPose affine metadata must be finite with positive scale")

    crop_x0, crop_y0, crop_x1, crop_y1 = crop
    crop_w = max(int(round(float(crop_x1 - crop_x0))), 1)
    crop_h = max(int(round(float(crop_y1 - crop_y0))), 1)
    hm_h, hm_w = arr.shape[1:]
    top_left = center - 0.5 * scale
    peak_flat = np.argmax(arr.reshape(arr.shape[0], -1), axis=1)
    peak_hm = np.stack((peak_flat % hm_w, peak_flat // hm_w), axis=-1).astype(np.float32)
    peak_global = np.empty_like(peak_hm)
    peak_global[:, 0] = peak_hm[:, 0] * scale[0] / max(hm_w - 1, 1) + top_left[0]
    peak_global[:, 1] = peak_hm[:, 1] * scale[1] / max(hm_h - 1, 1) + top_left[1]

    yy, xx = np.mgrid[:crop_h, :crop_w].astype(np.float32)
    map_x = (xx + crop_x0 - top_left[0]) * max(hm_w - 1, 1) / scale[0]
    map_y = (yy + crop_y0 - top_left[1]) * max(hm_h - 1, 1) / scale[1]
    registered = np.empty((arr.shape[0], crop_h, crop_w), dtype=np.float32)
    for joint_id, heatmap in enumerate(arr):
        registered[joint_id] = cv2.remap(
            np.clip(heatmap, 0.0, None),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
    support = registered.sum(axis=(1, 2)) > EPS
    temp = max(float(temperature), 1e-3)
    if abs(temp - 1.0) > 1e-6:
        registered = np.power(np.clip(registered, EPS, None), 1.0 / temp)
    peak_inside = (
        (peak_global[:, 0] >= crop_x0)
        & (peak_global[:, 0] < crop_x1)
        & (peak_global[:, 1] >= crop_y0)
        & (peak_global[:, 1] < crop_y1)
    )
    usable = support & peak_inside & np.isfinite(peak_global).all(axis=1)
    maps = normalize_maps(registered)
    maps[~usable] = 1.0 / float(crop_h * crop_w)
    return maps.astype(np.float32), peak_global.astype(np.float32), usable


def native_heatmap_peaks_to_global(
    heatmaps: np.ndarray,
    *,
    input_center_xy: np.ndarray,
    input_scale_xy: np.ndarray,
) -> np.ndarray:
    """Decode native-grid maxima through PMPose's source-image affine."""

    arr = np.asarray(heatmaps, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError("heatmaps must be [K,H,W]")
    center = np.asarray(input_center_xy, dtype=np.float32).reshape(2)
    scale = np.asarray(input_scale_xy, dtype=np.float32).reshape(2)
    if not np.isfinite(np.concatenate((center, scale))).all() or np.any(scale <= 0):
        raise ValueError("PMPose affine metadata must be finite with positive scale")
    hm_h, hm_w = arr.shape[1:]
    peak_flat = np.argmax(arr.reshape(arr.shape[0], -1), axis=1)
    peaks = np.stack((peak_flat % hm_w, peak_flat // hm_w), axis=-1).astype(np.float32)
    denominator = np.asarray([max(hm_w - 1, 1), max(hm_h - 1, 1)], dtype=np.float32)
    return peaks / denominator * scale + center - 0.5 * scale


def normalize_maps(maps: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(maps, dtype=np.float32), 0.0, None)
    if arr.ndim != 3:
        raise ValueError("maps must be [K,H,W]")
    total = arr.sum(axis=(1, 2), keepdims=True)
    empty = total <= EPS
    normalized = arr / np.where(empty, 1.0, total)
    if np.any(empty):
        normalized[empty[:, 0, 0]] = 1.0 / max(arr.shape[1] * arr.shape[2], 1)
    return normalized.astype(np.float32)
