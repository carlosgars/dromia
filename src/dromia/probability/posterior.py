"""Log-space posterior fusion."""

from __future__ import annotations

import cv2
import numpy as np

from dromia.probability import heatmaps as heatmap_probability

EPS = 1e-8


def fuse_posterior(
    likelihood_maps: np.ndarray,
    evidence_maps: dict[str, np.ndarray],
    weights: dict[str, float],
    *,
    decode_method: str,
    likelihood_reliability: float = 1.0,
    evidence_reliabilities: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    likelihood = heatmap_probability.normalize_maps(likelihood_maps)
    likelihood = reliable_distribution(likelihood, likelihood_reliability)
    logp = np.log(np.clip(likelihood, EPS, None))
    for name, evidence in evidence_maps.items():
        weight = float(weights.get(name, 1.0))
        if abs(weight) <= 1e-12:
            continue
        prepared = prepare_evidence(evidence, likelihood.shape)
        reliability = float((evidence_reliabilities or {}).get(name, 1.0))
        prepared = reliable_factor(prepared, reliability)
        logp += weight * np.log(np.clip(prepared, EPS, None))
    logp -= np.max(logp, axis=(1, 2), keepdims=True)
    posterior = np.exp(logp).astype(np.float32)
    posterior = heatmap_probability.normalize_maps(posterior)
    points, peak, entropy = decode_maps(posterior, method=decode_method)
    return posterior, points, peak, entropy


def reliable_distribution(distribution: np.ndarray, reliability: float) -> np.ndarray:
    """Mix a spatial distribution with uniform uncertainty."""

    probability = heatmap_probability.normalize_maps(distribution)
    value = float(np.clip(reliability, 0.0, 1.0))
    uniform = np.full_like(probability, 1.0 / (probability.shape[1] * probability.shape[2]))
    return heatmap_probability.normalize_maps(value * probability + (1.0 - value) * uniform)


def reliable_factor(factor: np.ndarray, reliability: float) -> np.ndarray:
    """Move a factor toward the neutral multiplicative factor as trust falls."""

    value = float(np.clip(reliability, 0.0, 1.0))
    return (value * np.asarray(factor, dtype=np.float32) + (1.0 - value)).astype(np.float32)


def decode_maps(maps: np.ndarray, *, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = heatmap_probability.normalize_maps(maps)
    keypoints = np.empty((probs.shape[0], 2), dtype=np.float32)
    peaks = np.empty(probs.shape[0], dtype=np.float32)
    entropy = np.empty(probs.shape[0], dtype=np.float32)
    yy, xx = np.mgrid[: probs.shape[1], : probs.shape[2]].astype(np.float32)
    for keypoint_id, prob in enumerate(probs):
        peak_y, peak_x = np.unravel_index(int(np.argmax(prob)), prob.shape)
        peaks[keypoint_id] = float(prob[peak_y, peak_x])
        clipped = np.clip(prob, EPS, 1.0)
        entropy[keypoint_id] = float(-np.sum(clipped * np.log(clipped)))
        if method == "expectation":
            keypoints[keypoint_id] = [float(np.sum(prob * xx)), float(np.sum(prob * yy))]
        elif method == "map":
            keypoints[keypoint_id] = [float(peak_x), float(peak_y)]
        else:
            raise ValueError(f"Unsupported decode method: {method}")
    return keypoints, peaks, entropy


def select_evidence_supported_points(
    raw_points: np.ndarray,
    posterior_points: np.ndarray,
    posterior_maps: np.ndarray,
    *,
    min_peak_ratio: float,
    replace_invalid: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relocate a raw point only when fused evidence decisively prefers its peak."""
    raw = np.asarray(raw_points, dtype=np.float32)
    candidate = np.asarray(posterior_points, dtype=np.float32)
    maps = heatmap_probability.normalize_maps(posterior_maps)
    selected = raw.copy()
    ratios = np.ones(raw.shape[0], dtype=np.float32)
    valid = np.isfinite(raw).all(axis=1)
    inside = (
        valid
        & (raw[:, 0] >= 0)
        & (raw[:, 0] <= maps.shape[2] - 1)
        & (raw[:, 1] >= 0)
        & (raw[:, 1] <= maps.shape[1] - 1)
    )
    rounded = np.rint(np.nan_to_num(raw, nan=0.0)).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, maps.shape[2] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, maps.shape[1] - 1)
    at_raw = maps[np.arange(maps.shape[0]), rounded[:, 1], rounded[:, 0]]
    peak = np.max(maps, axis=(1, 2))
    ratios[inside] = peak[inside] / np.maximum(at_raw[inside], EPS)
    ratios[~valid] = np.inf
    relocate = ((~valid) & bool(replace_invalid)) | (inside & (ratios >= float(min_peak_ratio)))
    selected[relocate] = candidate[relocate]
    return selected, relocate, ratios


def prepare_evidence(evidence: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(evidence, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[None, :, :], target_shape[0], axis=0)
    if arr.ndim != 3:
        raise ValueError("evidence must be [H,W] or [K,H,W]")
    if arr.shape[0] == 1 and target_shape[0] > 1:
        arr = np.repeat(arr, target_shape[0], axis=0)
    if arr.shape[0] != target_shape[0]:
        raise ValueError("evidence keypoint count does not match likelihood")
    if arr.shape[1:] != target_shape[1:]:
        resized = np.empty(target_shape, dtype=np.float32)
        for keypoint_id in range(arr.shape[0]):
            resized[keypoint_id] = cv2.resize(
                arr[keypoint_id],
                (target_shape[2], target_shape[1]),
                interpolation=cv2.INTER_LINEAR,
            )
        arr = resized
    return np.clip(arr, 1e-4, 1.0).astype(np.float32)
