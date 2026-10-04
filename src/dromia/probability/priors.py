"""SAM-derived probability priors."""

from __future__ import annotations

import cv2
import numpy as np


def mask_distance_prior(mask: np.ndarray, *, outside_sigma: float) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not np.any(binary):
        return np.ones(binary.shape, dtype=np.float32)
    distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5).astype(np.float32)
    sigma = max(float(outside_sigma), 1e-3)
    prior = np.exp(-(distance**2) / (2.0 * sigma * sigma)).astype(np.float32)
    prior[binary > 0] = 1.0
    return np.clip(prior, 1e-4, 1.0).astype(np.float32)
