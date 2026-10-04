"""Decode the fixed MAP posterior used by DromIA."""

from __future__ import annotations

import numpy as np

from dromia.probability import heatmaps as heatmap_probability

EPS = 1e-8


def decode_maps(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = heatmap_probability.normalize_maps(maps)
    keypoints = np.empty((probs.shape[0], 2), dtype=np.float32)
    peaks = np.empty(probs.shape[0], dtype=np.float32)
    entropy = np.empty(probs.shape[0], dtype=np.float32)
    for keypoint_id, prob in enumerate(probs):
        peak_y, peak_x = np.unravel_index(int(np.argmax(prob)), prob.shape)
        peaks[keypoint_id] = float(prob[peak_y, peak_x])
        clipped = np.clip(prob, EPS, 1.0)
        entropy[keypoint_id] = float(-np.sum(clipped * np.log(clipped)))
        keypoints[keypoint_id] = [float(peak_x), float(peak_y)]
    return keypoints, peaks, entropy
