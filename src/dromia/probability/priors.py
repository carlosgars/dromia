"""SAM-derived probability priors."""

from __future__ import annotations

import cv2
import numpy as np

ANKLE_IDS = (15, 16)


def runner_mask_prior(
    mask: np.ndarray, *, grid_shape: tuple[int, int], sigma_px: float
) -> np.ndarray:
    binary = resize_binary(mask, grid_shape)
    return mask_distance_prior(binary, outside_sigma=sigma_px)


def shoe_mask_priors(
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
    *,
    grid_shape: tuple[int, int],
    num_keypoints: int,
    sigma_px: float,
    neutral_weight: float,
) -> np.ndarray:
    maps = np.ones((num_keypoints, grid_shape[0], grid_shape[1]), dtype=np.float32)
    for keypoint_id, shoe_mask in ((15, left_mask), (16, right_mask)):
        if keypoint_id >= num_keypoints or shoe_mask is None:
            continue
        binary = resize_binary(shoe_mask, grid_shape)
        if not np.any(binary):
            continue
        prior = mask_distance_prior(binary, outside_sigma=sigma_px)
        maps[keypoint_id] = np.clip((1.0 - neutral_weight) + neutral_weight * prior, 1e-4, 1.0)
    return maps


def mask_distance_prior(mask: np.ndarray, *, outside_sigma: float) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not np.any(binary):
        return np.ones(binary.shape, dtype=np.float32)
    distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5).astype(np.float32)
    sigma = max(float(outside_sigma), 1e-3)
    prior = np.exp(-(distance**2) / (2.0 * sigma * sigma)).astype(np.float32)
    prior[binary > 0] = 1.0
    return np.clip(prior, 1e-4, 1.0).astype(np.float32)


def resize_binary(mask: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.shape != grid_shape:
        binary = cv2.resize(binary, (grid_shape[1], grid_shape[0]), interpolation=cv2.INTER_NEAREST)
    return binary.astype(np.uint8)
