"""Human-conditioned posterior updates for reviewed lower-body keypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dromia.review import cvat as cvat_annotation

EPS = 1e-8
ALGORITHM_VERSION = "human_conditioning_v9"
BONES = ((11, 13), (13, 15), (12, 14), (14, 16))
BILATERAL_PAIRS = ((11, 12), (13, 14), (15, 16))
BILATERAL_PAIR_NAMES = ("hips", "knees", "ankles")
BILATERAL_PAIR_INDEX = {
    joint_id: pair_idx for pair_idx, pair in enumerate(BILATERAL_PAIRS) for joint_id in pair
}
BILATERAL_JOINT = {left: right for left, right in BILATERAL_PAIRS} | {
    right: left for left, right in BILATERAL_PAIRS
}
# Refinement is intentionally lower-body-only even though CVAT also displays
# the derived neck and shoulder landmarks for orientation.
REFINEMENT_JOINT_IDS = (11, 12, 13, 14, 15, 16)
NEIGHBORS = {
    11: ((13, (11, 13)),),
    12: ((14, (12, 14)),),
    13: ((11, (11, 13)), (15, (13, 15))),
    14: ((12, (12, 14)), (16, (14, 16))),
    15: ((13, (13, 15)),),
    16: ((14, (14, 16)),),
}
CONTRALATERAL_JOINT = {13: 14, 14: 13, 15: 16, 16: 15}


class HumanConditioningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_radius: int = 24
    strict_local_window: bool = True
    correction_min_px: float = 0.5
    tracker_weight: float = 2.0
    transport_weight: float = 0.8
    motion_weight: float = 0.6
    geometry_weight: float = 0.7
    contralateral_exclusion_weight: float = 1.5
    duplicate_joint_distance_norm: float = 0.02
    tracker_sigma_norm: float = 0.025
    transport_sigma_norm: float = 0.05
    geometry_sigma_relative: float = 0.15
    min_geometry_sigma_norm: float = 0.015
    min_motion_sigma_norm: float = 0.015
    max_tracker_anchors_per_joint: int = 6
    smoother_weight: float = Field(default=12.0, ge=0.0)
    smoother_conditioning_sigma_norm: float = Field(default=0.008, gt=0.0)
    identity_swap_enabled: bool = True
    identity_position_reliability: float = Field(default=1.0, ge=0.0)
    identity_heatmap_reliability: float = Field(default=1.0, ge=0.0)
    identity_temporal_reliability: float = Field(default=3.0, ge=0.0)
    identity_position_sigma_norm: float = Field(default=0.05, gt=0.0)
    identity_heatmap_sigma_norm: float = Field(default=0.025, gt=0.0)
    identity_switch_probability: float = Field(default=0.002, gt=0.0, lt=0.5)
    identity_initial_swap_probability: float = Field(default=0.01, gt=0.0, lt=0.5)
    identity_swap_probability_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    identity_min_pair_count: int = Field(default=1, ge=1, le=len(BILATERAL_PAIRS))
    smoother_process_sigma_norm: float = Field(default=0.004, gt=0.0)
    smoother_tracker_sigma_norm: float = Field(default=0.01, gt=0.0)
    smoother_tracker_drift_sigma_norm_per_frame: float = Field(default=0.001, ge=0.0)
    smoother_ground_truth_sigma_norm: float = Field(default=0.001, gt=0.0)
    enforce_constant_limb_length: bool = True
    limb_constraint_iterations: int = Field(default=16, ge=1, le=100)
    limb_constraint_tolerance_px: float = Field(default=0.05, ge=0.0)


class TrustedObservation(BaseModel):
    runner_id: int
    joint_id: int
    frame_idx: int
    time_idx: int
    xy: tuple[float, float]
    source: str
    confidence: float = 1.0
    posterior_delta_px: float


class RunnerPrior(BaseModel):
    runner_id: int
    bone_length_norm: dict[str, float] = Field(default_factory=dict)
    bone_sigma_norm: dict[str, float] = Field(default_factory=dict)
    bone_length_px: dict[str, float] = Field(default_factory=dict)
    bone_sigma_px: dict[str, float] = Field(default_factory=dict)
    trusted_bone_samples: dict[str, int] = Field(default_factory=dict)
    machine_bone_samples: dict[str, int] = Field(default_factory=dict)
    fallback_bones: list[str] = Field(default_factory=list)
    motion_sigma_norm: dict[str, float] = Field(default_factory=dict)


class ConditioningDiagnostics(BaseModel):
    algorithm_version: str = ALGORITHM_VERSION
    scope: str
    runner_ids: list[int]
    trusted_observation_count: int
    implicit_human_correction_count: int
    tracker_observation_count: int
    tracker_used_count: int
    transport_used_count: int
    changed_keypoint_count: int
    changed_frames: list[int]
    identity_swap_frame_count: int = 0
    identity_swap_frames: dict[str, list[int]] = Field(default_factory=dict)
    identity_pair_swap_frames: dict[str, dict[str, list[int]]] = Field(default_factory=dict)
    identity_mean_swap_probability: float = 0.0
    config: HumanConditioningConfig
    priors: list[RunnerPrior]


@dataclass(slots=True)
class ConditioningResult:
    state: cvat_annotation.AnnotationState
    confidence: np.ndarray
    source: np.ndarray
    trusted: np.ndarray
    applied: np.ndarray
    source_anchor: np.ndarray
    identity_swapped: np.ndarray
    identity_swap_probability: np.ndarray
    identity_evidence_pair_count: np.ndarray
    identity_pair_swapped: np.ndarray
    identity_pair_swap_probability: np.ndarray
    identity_pair_evidence: np.ndarray
    smoothed_identity_centers: np.ndarray
    diagnostics: ConditioningDiagnostics


@dataclass(slots=True)
class IdentityAssignmentResult:
    """Smoothed model-channel assignment for each runner and frame.

    ``swapped[t, o]`` is the aggregate of the pair-specific assignments.
    Human observations define the identity tracks and are exact
    coordinate observations even when the corresponding raw model channel is
    inferred to be swapped.
    """

    swapped: np.ndarray
    swap_probability: np.ndarray
    evidence_pair_count: np.ndarray
    pair_swapped: np.ndarray
    pair_swap_probability: np.ndarray
    pair_evidence: np.ndarray
    smoothed_centers: np.ndarray


def trusted_observations(
    bundle: cvat_annotation.CvatBundle,
    state: cvat_annotation.AnnotationState,
    *,
    previous_proposal: np.ndarray | None = None,
    cfg: HumanConditioningConfig | None = None,
) -> list[TrustedObservation]:
    config = cfg or HumanConditioningConfig()
    with np.load(bundle.preannotations_npz, allow_pickle=False) as data:
        posterior = np.asarray(data["posterior_keypoints_xy"], dtype=np.float32)
    observations: list[TrustedObservation] = []
    for t, frame_idx in enumerate(bundle.frame_indices):
        for obj_idx, runner_id in enumerate(bundle.runner_ids):
            status = str(state.review_status[t, obj_idx])
            for local_idx, joint_id in enumerate(cvat_annotation.JOINT_IDS):
                point = state.points_xy[t, obj_idx, local_idx]
                original = posterior[t, obj_idx, local_idx]
                if state.visibility[t, obj_idx, local_idx] == 0 or not np.isfinite(point).all():
                    continue
                delta = (
                    float(np.linalg.norm(point - original))
                    if np.isfinite(original).all()
                    else float("inf")
                )
                prior_proposal = bool(
                    previous_proposal is not None and previous_proposal[t, obj_idx, local_idx]
                )
                source = None
                if state.keypoint_ground_truth[t, obj_idx, local_idx]:
                    source = "HUMAN_FRAME_GROUND_TRUTH"
                elif status == "ACCEPTED":
                    source = "HUMAN_ACCEPTED"
                elif status == "CORRECTED" and delta > config.correction_min_px:
                    source = "HUMAN_CORRECTED"
                elif (
                    status == "UNREVIEWED"
                    and delta > config.correction_min_px
                    and not prior_proposal
                ):
                    source = "HUMAN_CORRECTED_IMPLICIT"
                if source is None:
                    continue
                observations.append(
                    TrustedObservation(
                        runner_id=runner_id,
                        joint_id=joint_id,
                        frame_idx=frame_idx,
                        time_idx=t,
                        xy=(float(point[0]), float(point[1])),
                        source=source,
                        posterior_delta_px=delta,
                    )
                )
    return observations


def select_tracker_observations(
    observations: list[TrustedObservation],
    cfg: HumanConditioningConfig,
) -> list[TrustedObservation]:
    selected: list[TrustedObservation] = []
    keys = sorted({(item.runner_id, item.joint_id) for item in observations})
    for key in keys:
        candidates = [item for item in observations if (item.runner_id, item.joint_id) == key]
        corrected = [item for item in candidates if item.source != "HUMAN_ACCEPTED"]
        accepted = [item for item in candidates if item.source == "HUMAN_ACCEPTED"]
        keep = corrected[: cfg.max_tracker_anchors_per_joint]
        remaining = cfg.max_tracker_anchors_per_joint - len(keep)
        if remaining > 0 and accepted:
            indices = (
                np.linspace(0, len(accepted) - 1, min(remaining, len(accepted))).round().astype(int)
            )
            keep.extend(accepted[index] for index in np.unique(indices))
        selected.extend(keep)
    return sorted(selected, key=lambda item: (item.runner_id, item.joint_id, item.time_idx))


def infer_identity_assignments(
    *,
    run_dir: Path,
    bundle: cvat_annotation.CvatBundle,
    state: cvat_annotation.AnnotationState,
    posterior: np.ndarray,
    observations: list[TrustedObservation],
    tracker_observations: list[TrustedObservation],
    tracker_xy: np.ndarray,
    tracker_visibility: np.ndarray,
    bboxes_xyxy: np.ndarray,
    cfg: HumanConditioningConfig,
) -> IdentityAssignmentResult:
    """Infer anatomical-to-model channel assignments with a switching smoother.

    Each bilateral pair has its own normal/swapped latent state.  Its continuous
    anatomical trajectory is estimated first with a linear-Gaussian RTS smoother
    whose measurements are CoTracker tracks and whose near-zero-noise observations
    are human ground truth.  The two channel hypotheses are then scored against
    that trajectory using MAP-point distance and calibrated heatmap support.  A
    forward-backward/Viterbi pass gives the smoothed pair assignment over the full
    sequence.  Human coordinates remain immutable; at a reviewed frame they inform
    the channel association instead of being removed from inference.
    """

    shape = posterior.shape[:2]
    pair_shape = (*shape, len(BILATERAL_PAIRS))
    pair_swapped = np.zeros(pair_shape, dtype=bool)
    pair_swap_probability = np.zeros(pair_shape, dtype=np.float32)
    pair_evidence = np.zeros(pair_shape, dtype=bool)
    smoothed_centers = np.full(
        (*shape, len(cvat_annotation.JOINT_IDS), 2), np.nan, dtype=np.float32
    )

    local = {joint_id: idx for idx, joint_id in enumerate(cvat_annotation.JOINT_IDS)}
    for obj_idx, runner_id in enumerate(bundle.runner_ids):
        centers = smooth_identity_centers(
            runner_id=runner_id,
            frame_count=len(bundle.frame_indices),
            observations=observations,
            tracker_observations=tracker_observations,
            tracker_xy=tracker_xy,
            tracker_visibility=tracker_visibility,
            bboxes_xyxy=bboxes_xyxy[:, obj_idx],
            cfg=cfg,
        )
        smoothed_centers[:, obj_idx] = centers
        if not cfg.identity_swap_enabled:
            continue

        emissions = np.zeros((len(bundle.frame_indices), len(BILATERAL_PAIRS), 2), dtype=np.float64)
        for t, frame_idx in enumerate(bundle.frame_indices):
            bbox_h = bbox_height(bboxes_xyxy[t, obj_idx])
            if bbox_h <= 0:
                continue
            position_sigma = max(cfg.identity_position_sigma_norm * bbox_h, 1.0)
            heatmap_sigma = max(cfg.identity_heatmap_sigma_norm * bbox_h, 1.0)
            for pair_idx, (left_id, right_id) in enumerate(BILATERAL_PAIRS):
                left_idx, right_idx = local[left_id], local[right_id]
                left_center, right_center = centers[t, left_idx], centers[t, right_idx]
                left_point = posterior[t, obj_idx, left_idx]
                right_point = posterior[t, obj_idx, right_idx]
                if not np.isfinite([left_center, right_center, left_point, right_point]).all():
                    continue
                normal_distance = float(
                    np.sum((left_point - left_center) ** 2)
                    + np.sum((right_point - right_center) ** 2)
                )
                swapped_distance = float(
                    np.sum((right_point - left_center) ** 2)
                    + np.sum((left_point - right_center) ** 2)
                )
                scale = 2.0 * position_sigma * position_sigma
                emissions[t, pair_idx, 0] -= (
                    cfg.identity_position_reliability * normal_distance / scale
                )
                emissions[t, pair_idx, 1] -= (
                    cfg.identity_position_reliability * swapped_distance / scale
                )

                left_map = load_posterior_map(run_dir, frame_idx, runner_id, left_id)
                right_map = load_posterior_map(run_dir, frame_idx, runner_id, right_id)
                if left_map is not None and right_map is not None:
                    left_on_left = heatmap_support(*left_map, left_center, heatmap_sigma)
                    right_on_right = heatmap_support(*right_map, right_center, heatmap_sigma)
                    right_on_left = heatmap_support(*right_map, left_center, heatmap_sigma)
                    left_on_right = heatmap_support(*left_map, right_center, heatmap_sigma)
                    emissions[t, pair_idx, 0] += cfg.identity_heatmap_reliability * (
                        np.log(max(left_on_left, EPS)) + np.log(max(right_on_right, EPS))
                    )
                    emissions[t, pair_idx, 1] += cfg.identity_heatmap_reliability * (
                        np.log(max(right_on_left, EPS)) + np.log(max(left_on_right, EPS))
                    )
                pair_evidence[t, obj_idx, pair_idx] = True

        for pair_idx in range(len(BILATERAL_PAIRS)):
            eligible = pair_evidence[:, obj_idx, pair_idx]
            for segment in contiguous_true_segments(eligible):
                path, probability = decode_identity_hmm(emissions[segment, pair_idx], cfg)
                selected = (path == 1) & (probability >= cfg.identity_swap_probability_threshold)
                pair_swapped[segment, obj_idx, pair_idx] = selected
                pair_swap_probability[segment, obj_idx, pair_idx] = probability.astype(np.float32)
    return summarize_pair_assignments(
        pair_swapped,
        pair_swap_probability,
        pair_evidence,
        smoothed_centers,
    )


def summarize_pair_assignments(
    pair_swapped: np.ndarray,
    pair_swap_probability: np.ndarray,
    pair_evidence: np.ndarray,
    smoothed_centers: np.ndarray,
) -> IdentityAssignmentResult:
    """Build frame-level summaries from pair assignments."""

    swapped = np.any(pair_swapped, axis=-1)
    swap_probability = np.max(pair_swap_probability, axis=-1)
    evidence_pair_count = np.sum(pair_evidence, axis=-1, dtype=np.int16)
    return IdentityAssignmentResult(
        swapped=swapped,
        swap_probability=swap_probability,
        evidence_pair_count=evidence_pair_count,
        pair_swapped=pair_swapped,
        pair_swap_probability=pair_swap_probability,
        pair_evidence=pair_evidence,
        smoothed_centers=smoothed_centers,
    )


def smooth_identity_centers(
    *,
    runner_id: int,
    frame_count: int,
    observations: list[TrustedObservation],
    tracker_observations: list[TrustedObservation],
    tracker_xy: np.ndarray,
    tracker_visibility: np.ndarray,
    bboxes_xyxy: np.ndarray,
    cfg: HumanConditioningConfig,
) -> np.ndarray:
    """Return anatomical joint trajectories from a linear-Gaussian RTS smoother.

    CoTracker outputs are ordinary noisy measurements. Human observations use a
    separate, near-zero measurement variance and are restored exactly after the
    smoothing pass. Multiple tracks seeded for the same joint are robustly merged
    at each frame before filtering.
    """

    centers = np.full((frame_count, len(cvat_annotation.JOINT_IDS), 2), np.nan, dtype=np.float32)
    heights = np.asarray([bbox_height(value) for value in bboxes_xyxy], dtype=np.float32)
    finite_heights = heights[heights > 0]
    scale = float(np.median(finite_heights)) if finite_heights.size else 1.0
    tracker_variance = (cfg.smoother_tracker_sigma_norm * scale) ** 2
    ground_truth_variance = (cfg.smoother_ground_truth_sigma_norm * scale) ** 2
    process_variance = (cfg.smoother_process_sigma_norm * scale) ** 2

    for local_idx, joint_id in enumerate(cvat_annotation.JOINT_IDS):
        human = [
            item
            for item in observations
            if item.runner_id == runner_id and item.joint_id == joint_id
        ]
        track_rows = [
            (row_idx, item)
            for row_idx, item in enumerate(tracker_observations)
            if item.runner_id == runner_id and item.joint_id == joint_id
        ]
        if not human and not track_rows:
            continue
        if not track_rows:
            for item in human:
                if 0 <= item.time_idx < frame_count:
                    centers[item.time_idx, local_idx] = item.xy
            continue
        measurements: list[list[tuple[np.ndarray, float]]] = [[] for _ in range(frame_count)]
        for t in range(frame_count):
            for row_idx, anchor in track_rows:
                if (
                    row_idx >= len(tracker_xy)
                    or t >= tracker_xy.shape[1]
                    or anchor.time_idx >= tracker_xy.shape[1]
                    or not tracker_visibility[row_idx, t]
                    or not np.isfinite(tracker_xy[row_idx, t]).all()
                    or not np.isfinite(tracker_xy[row_idx, anchor.time_idx]).all()
                ):
                    continue
                anchor_offset = (
                    np.asarray(anchor.xy, dtype=np.float32) - tracker_xy[row_idx, anchor.time_idx]
                )
                aligned = tracker_xy[row_idx, t] + anchor_offset
                distance = abs(t - anchor.time_idx)
                sigma_norm = (
                    cfg.smoother_tracker_sigma_norm
                    + cfg.smoother_tracker_drift_sigma_norm_per_frame * distance
                )
                measurements[t].append((aligned, (sigma_norm * scale) ** 2))
        for item in human:
            if 0 <= item.time_idx < frame_count:
                measurements[item.time_idx].append(
                    (np.asarray(item.xy, dtype=np.float32), ground_truth_variance)
                )
        if not any(measurements):
            continue
        smoothed = rts_constant_velocity_smoother(
            measurements,
            process_variance=max(process_variance, EPS),
            initial_position_variance=max(tracker_variance, 1.0),
        )
        for item in human:
            if 0 <= item.time_idx < frame_count:
                smoothed[item.time_idx] = item.xy
        centers[:, local_idx] = smoothed
    return centers


def rts_constant_velocity_smoother(
    measurements: list[list[tuple[np.ndarray, float]]],
    *,
    process_variance: float,
    initial_position_variance: float,
) -> np.ndarray:
    """Linear constant-velocity Kalman filter followed by an RTS backward pass."""

    count = len(measurements)
    output = np.full((count, 2), np.nan, dtype=np.float32)
    observed = [idx for idx, values in enumerate(measurements) if values]
    if not observed:
        return output
    first, last = 0, count - 1
    transition = np.asarray(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    observation = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    process = process_variance * np.asarray(
        [
            [0.25, 0, 0.5, 0],
            [0, 0.25, 0, 0.5],
            [0.5, 0, 1, 0],
            [0, 0.5, 0, 1],
        ],
        dtype=np.float64,
    )
    identity = np.eye(4, dtype=np.float64)
    initial_xy = np.asarray(measurements[observed[0]][0][0], dtype=np.float64)
    filtered_mean = np.zeros((count, 4), dtype=np.float64)
    predicted_mean = np.zeros((count, 4), dtype=np.float64)
    filtered_covariance = np.zeros((count, 4, 4), dtype=np.float64)
    predicted_covariance = np.zeros((count, 4, 4), dtype=np.float64)
    mean = np.asarray([initial_xy[0], initial_xy[1], 0.0, 0.0], dtype=np.float64)
    covariance = np.diag(
        [
            initial_position_variance,
            initial_position_variance,
            4.0 * initial_position_variance,
            4.0 * initial_position_variance,
        ]
    )
    for t in range(first, last + 1):
        if t > first:
            mean = transition @ mean
            covariance = transition @ covariance @ transition.T + process
        predicted_mean[t] = mean
        predicted_covariance[t] = covariance
        for value, variance in measurements[t]:
            noise = max(float(variance), EPS) * np.eye(2, dtype=np.float64)
            innovation_covariance = observation @ covariance @ observation.T + noise
            gain = covariance @ observation.T @ np.linalg.pinv(innovation_covariance)
            mean = mean + gain @ (np.asarray(value, dtype=np.float64) - observation @ mean)
            covariance = (identity - gain @ observation) @ covariance
            covariance = 0.5 * (covariance + covariance.T)
        filtered_mean[t] = mean
        filtered_covariance[t] = covariance

    smoothed_mean = filtered_mean.copy()
    smoothed_covariance = filtered_covariance.copy()
    for t in range(last - 1, first - 1, -1):
        gain = filtered_covariance[t] @ transition.T @ np.linalg.pinv(predicted_covariance[t + 1])
        smoothed_mean[t] += gain @ (smoothed_mean[t + 1] - predicted_mean[t + 1])
        smoothed_covariance[t] += (
            gain @ (smoothed_covariance[t + 1] - predicted_covariance[t + 1]) @ gain.T
        )
    output[first : last + 1] = smoothed_mean[first : last + 1, :2].astype(np.float32)
    return output


def heatmap_support(
    probability: np.ndarray,
    crop_xyxy: np.ndarray,
    center: np.ndarray,
    sigma_px: float,
) -> float:
    """Return locally averaged probability density around an image point."""

    yy, xx = np.mgrid[: probability.shape[0], : probability.shape[1]].astype(np.float32)
    local_center = np.asarray(center, dtype=np.float32) - crop_xyxy[:2]
    kernel = gaussian_grid(xx, yy, local_center, sigma_px)
    return float(np.sum(probability * kernel) / max(float(kernel.sum()), EPS))


def contiguous_true_segments(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    return [part for part in np.split(indices, boundaries) if len(part)]


def decode_identity_hmm(
    emissions: np.ndarray,
    cfg: HumanConditioningConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Viterbi assignment and smoothed P(swapped) for a 2-state HMM."""

    count = len(emissions)
    if count == 0:
        return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float64)
    switch = cfg.identity_switch_probability
    initial_swap = cfg.identity_initial_swap_probability
    log_transition = cfg.identity_temporal_reliability * np.log(
        np.asarray([[1.0 - switch, switch], [switch, 1.0 - switch]], dtype=np.float64)
    )
    log_initial = np.log(np.asarray([1.0 - initial_swap, initial_swap], dtype=np.float64))
    normalized_emissions = emissions - np.max(emissions, axis=1, keepdims=True)

    viterbi = np.empty((count, 2), dtype=np.float64)
    backpointer = np.zeros((count, 2), dtype=np.int8)
    viterbi[0] = log_initial + normalized_emissions[0]
    for t in range(1, count):
        candidates = viterbi[t - 1][:, None] + log_transition
        backpointer[t] = np.argmax(candidates, axis=0)
        viterbi[t] = normalized_emissions[t] + np.max(candidates, axis=0)
    path = np.empty(count, dtype=np.int8)
    path[-1] = int(np.argmax(viterbi[-1]))
    for t in range(count - 1, 0, -1):
        path[t - 1] = backpointer[t, path[t]]

    forward = np.empty((count, 2), dtype=np.float64)
    backward = np.zeros((count, 2), dtype=np.float64)
    forward[0] = log_initial + normalized_emissions[0]
    for t in range(1, count):
        forward[t] = normalized_emissions[t] + logsumexp_axis0(
            forward[t - 1][:, None] + log_transition
        )
    for t in range(count - 2, -1, -1):
        backward[t] = logsumexp_axis1(
            log_transition + (normalized_emissions[t + 1] + backward[t + 1])[None, :]
        )
    marginal = forward + backward
    marginal -= logsumexp_axis0(marginal.T)[:, None]
    return path, np.exp(marginal[:, 1])


def logsumexp_axis0(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=0)
    return maximum + np.log(np.sum(np.exp(values - maximum), axis=0))


def logsumexp_axis1(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    return maximum + np.log(np.sum(np.exp(values - maximum[:, None]), axis=1))


def connected_native_anomaly_mask(
    posterior: np.ndarray,
    bboxes_xyxy: np.ndarray,
    *,
    obj_idx: int,
    local_idx: int,
    anchor_time_idx: int,
    reliable_run: int = 3,
) -> np.ndarray:
    """Limit a manual correction to its connected native-pose anomaly.

    At most the first two consecutive reliable native frames are traversed; the
    third closes the segment. This prevents a fixed-radius window from changing
    a later portion of the clip whose unconditioned pose has recovered.
    """

    count = posterior.shape[0]
    allowed = np.zeros(count, dtype=bool)
    allowed[anchor_time_idx] = True
    point = posterior[:, obj_idx, local_idx]
    box = bboxes_xyxy[:, obj_idx]
    height = np.maximum(box[:, 3] - box[:, 1], 1.0)
    pad = 0.05 * height
    finite = np.all(np.isfinite(point), axis=-1)
    inside = (
        finite
        & (point[:, 0] >= box[:, 0] - pad)
        & (point[:, 0] <= box[:, 2] + pad)
        & (point[:, 1] >= box[:, 1] - pad)
        & (point[:, 1] <= box[:, 3] + pad)
    )
    velocity = np.full(count, np.inf, dtype=np.float32)
    if count > 1:
        velocity[1:] = np.linalg.norm(point[1:] - point[:-1], axis=-1) / height[1:]
        velocity[0] = velocity[1]
    reliable = inside & (velocity <= 0.18)

    for direction in (-1, 1):
        consecutive = 0
        t = anchor_time_idx + direction
        while 0 <= t < count:
            consecutive = consecutive + 1 if reliable[t] else 0
            if consecutive >= reliable_run:
                break
            allowed[t] = True
            t += direction
    return allowed


def condition_pose(
    *,
    run_dir: Path,
    bundle: cvat_annotation.CvatBundle,
    state: cvat_annotation.AnnotationState,
    observations: list[TrustedObservation],
    tracker_observations: list[TrustedObservation],
    tracker_xy: np.ndarray,
    tracker_visibility: np.ndarray,
    bboxes_xyxy: np.ndarray,
    cfg: HumanConditioningConfig,
    propagation_observations: list[TrustedObservation] | None = None,
) -> ConditioningResult:
    with np.load(bundle.preannotations_npz, allow_pickle=False) as data:
        posterior = data["posterior_keypoints_xy"].astype(np.float32)
        peak = data["posterior_peak_probability"].astype(np.float32)
        entropy = data["posterior_entropy"].astype(np.float32)
    output = cvat_annotation.AnnotationState(
        points_xy=state.points_xy.copy(),
        visibility=state.visibility.copy(),
        review_status=state.review_status.copy(),
        frame_ground_truth=state.frame_ground_truth.copy(),
        keypoint_ground_truth=state.keypoint_ground_truth.copy(),
    )
    confidence = quality_scores(peak, entropy)
    source = np.full(posterior.shape[:-1], "POSTERIOR", dtype="<U32")
    trusted = np.zeros(posterior.shape[:-1], dtype=bool)
    source_anchor = np.full(posterior.shape[:-1], -1, dtype=np.int32)
    applied = np.zeros(posterior.shape[:-1], dtype=bool)
    obs_lookup = {(item.runner_id, item.joint_id, item.time_idx): item for item in observations}
    tracker_lookup = {
        (item.runner_id, item.joint_id, item.time_idx): idx
        for idx, item in enumerate(tracker_observations)
    }
    refinement_observations = (
        observations if propagation_observations is None else propagation_observations
    )
    propagation_mode = propagation_observations is not None
    apply_trusted_observations(
        bundle=bundle,
        observations=obs_lookup,
        output=output,
        confidence=confidence,
        source=source,
        trusted=trusted,
    )

    identity = infer_identity_assignments(
        run_dir=run_dir,
        bundle=bundle,
        state=state,
        posterior=posterior,
        observations=observations,
        tracker_observations=tracker_observations,
        tracker_xy=tracker_xy,
        tracker_visibility=tracker_visibility,
        bboxes_xyxy=bboxes_xyxy,
        cfg=cfg,
    )
    priors = fit_runner_priors(run_dir, bundle, posterior, observations, bboxes_xyxy, peak, entropy)
    prior_lookup = {prior.runner_id: prior for prior in priors}
    tracker_used = 0
    transport_used = 0
    for obj_idx, runner_id in enumerate(bundle.runner_ids):
        runner_prior = prior_lookup[runner_id]
        for joint_id in REFINEMENT_JOINT_IDS:
            local_idx = cvat_annotation.JOINT_IDS.index(joint_id)
            anchors = [
                item
                for item in refinement_observations
                if item.runner_id == runner_id and item.joint_id == joint_id
            ]
            if not anchors:
                continue
            allowed_by_anchor = (
                {
                    item.time_idx: connected_native_anomaly_mask(
                        posterior,
                        bboxes_xyxy,
                        obj_idx=obj_idx,
                        local_idx=local_idx,
                        anchor_time_idx=item.time_idx,
                    )
                    for item in anchors
                }
                if propagation_mode
                else {}
            )
            for t in order_by_anchor_distance(len(bundle.frame_indices), anchors):
                if trusted[t, obj_idx, local_idx]:
                    continue
                anchor_index, anchor = min(
                    enumerate(anchors),
                    key=lambda item: abs(t - item[1].time_idx),
                )
                distance = abs(t - anchor.time_idx)
                if propagation_mode and not allowed_by_anchor[anchor.time_idx][t]:
                    continue
                if str(state.review_status[t, obj_idx]) != "UNREVIEWED":
                    continue
                smoother_center = identity.smoothed_centers[t, obj_idx, local_idx]
                smoother_ok = bool(np.isfinite(smoother_center).all())
                if distance > cfg.window_radius and (cfg.strict_local_window or not smoother_ok):
                    continue
                bbox_h = bbox_height(bboxes_xyxy[t, obj_idx])
                if bbox_h <= 0:
                    continue
                joint_swapped = identity_joint_swapped(identity, t, obj_idx, joint_id)
                model_joint_id = BILATERAL_JOINT[joint_id] if joint_swapped else joint_id
                model_local_idx = cvat_annotation.JOINT_IDS.index(model_joint_id)
                track_idx = tracker_lookup.get((runner_id, joint_id, anchor.time_idx))
                tracker_ok = bool(
                    track_idx is not None
                    and track_idx < len(tracker_xy)
                    and np.isfinite(tracker_xy[track_idx, t]).all()
                    and tracker_visibility[track_idx, t]
                )
                # A manual correction must not be extrapolated by the Bayesian
                # smoother after the local visual track has ended. This was the
                # source of the frame-149/150 knee drift in the runner-0 fixture.
                if propagation_mode and t != anchor.time_idx and not tracker_ok:
                    continue
                if smoother_ok:
                    center = smoother_center
                    weight = cfg.smoother_weight
                    sigma = cfg.smoother_conditioning_sigma_norm * bbox_h
                    point_source = "BAYESIAN_SMOOTHER_PRIOR"
                    tracker_used += 1
                elif tracker_ok:
                    center = tracker_xy[track_idx, t]
                    weight = cfg.tracker_weight
                    sigma = cfg.tracker_sigma_norm * bbox_h
                    point_source = "TRACKER_PRIOR"
                    tracker_used += 1
                else:
                    center = transported_center(
                        anchor,
                        posterior,
                        t,
                        obj_idx,
                        local_idx,
                        target_local_idx=model_local_idx,
                    )
                    if center is None:
                        continue
                    weight = cfg.transport_weight
                    sigma = cfg.transport_sigma_norm * bbox_h
                    point_source = "TEMPORAL_PRIOR"
                    transport_used += 1
                decay = max(0.15, 1.0 - distance / (cfg.window_radius + 1.0))
                near_t = t + 1 if t < anchor.time_idx else t - 1
                motion_center = transported_from_neighbor(
                    posterior,
                    output.points_xy,
                    bboxes_xyxy,
                    t,
                    near_t,
                    obj_idx,
                    local_idx,
                    current_posterior_local_idx=model_local_idx,
                    previous_posterior_local_idx=(
                        cvat_annotation.JOINT_IDS.index(BILATERAL_JOINT[joint_id])
                        if identity_joint_swapped(identity, near_t, obj_idx, joint_id)
                        else local_idx
                    ),
                )
                conditioned, map_confidence = condition_frame_map(
                    run_dir=run_dir,
                    frame_idx=bundle.frame_indices[t],
                    runner_id=runner_id,
                    joint_id=joint_id,
                    base_joint_id=model_joint_id,
                    center=np.asarray(center, dtype=np.float32),
                    dynamic_weight=weight * decay,
                    dynamic_sigma_px=max(sigma, 1.0),
                    neighbor_points=output.points_xy[t, obj_idx],
                    prior=runner_prior,
                    bbox_h=bbox_h,
                    motion_center=motion_center,
                    motion_sigma_norm=runner_prior.motion_sigma_norm.get(str(joint_id)),
                    cfg=cfg,
                )
                if conditioned is None:
                    continue
                output.points_xy[t, obj_idx, local_idx] = conditioned
                confidence[t, obj_idx, local_idx] = map_confidence
                changed = bool(
                    np.linalg.norm(conditioned - posterior[t, obj_idx, local_idx])
                    > cfg.correction_min_px
                )
                if changed:
                    applied[t, obj_idx, local_idx] = True
                    source[t, obj_idx, local_idx] = (
                        "LEFT_RIGHT_SWAP_PRIOR" if joint_swapped else point_source
                    )
                    source_anchor[t, obj_idx, local_idx] = anchor_index

        if cfg.enforce_constant_limb_length:
            active_constraint_frames = np.zeros(len(bundle.frame_indices), dtype=bool)
            movable_constraint_points = np.zeros_like(trusted[:, obj_idx])
            for item in refinement_observations:
                if item.runner_id != runner_id:
                    continue
                start = max(0, item.time_idx - cfg.window_radius)
                end = min(len(bundle.frame_indices), item.time_idx + cfg.window_radius + 1)
                active_constraint_frames[start:end] = True
                local_idx = cvat_annotation.JOINT_IDS.index(item.joint_id)
                movable_constraint_points[start:end, local_idx] = True
            constraint_changed = enforce_constant_limb_lengths(
                output.points_xy[:, obj_idx],
                trusted[:, obj_idx],
                confidence[:, obj_idx],
                runner_prior,
                cfg,
                active_frames=active_constraint_frames,
                movable_points=movable_constraint_points,
            )
            applied[:, obj_idx] |= constraint_changed
            swapped_constraint = constraint_changed & (
                np.char.find(source[:, obj_idx], "SWAP") >= 0
            )
            source[:, obj_idx][constraint_changed] = "LIMB_LENGTH_CONSTRAINT"
            source[:, obj_idx][swapped_constraint] = "CONSTRAINED_SWAP_PRIOR"

    changed_frames = sorted(
        {bundle.frame_indices[t] for t in np.where(np.any(applied, axis=(1, 2)))[0].tolist()}
    )
    diagnostics = ConditioningDiagnostics(
        scope=bundle.scope,
        runner_ids=bundle.runner_ids,
        trusted_observation_count=len(observations),
        implicit_human_correction_count=sum(
            item.source.endswith("IMPLICIT") for item in observations
        ),
        tracker_observation_count=len(tracker_observations),
        tracker_used_count=tracker_used,
        transport_used_count=transport_used,
        changed_keypoint_count=int(np.count_nonzero(applied)),
        changed_frames=changed_frames,
        identity_swap_frame_count=int(np.count_nonzero(identity.swapped)),
        identity_swap_frames={
            str(runner_id): [
                bundle.frame_indices[t]
                for t in np.flatnonzero(identity.swapped[:, obj_idx]).tolist()
            ]
            for obj_idx, runner_id in enumerate(bundle.runner_ids)
        },
        identity_pair_swap_frames={
            str(runner_id): {
                pair_name: [
                    bundle.frame_indices[t]
                    for t in np.flatnonzero(identity.pair_swapped[:, obj_idx, pair_idx]).tolist()
                ]
                for pair_idx, pair_name in enumerate(BILATERAL_PAIR_NAMES)
            }
            for obj_idx, runner_id in enumerate(bundle.runner_ids)
        },
        identity_mean_swap_probability=(
            float(np.mean(identity.pair_swap_probability[identity.pair_evidence]))
            if np.any(identity.pair_evidence)
            else 0.0
        ),
        config=cfg,
        priors=priors,
    )
    return ConditioningResult(
        state=output,
        confidence=confidence,
        source=source,
        trusted=trusted,
        applied=applied,
        source_anchor=source_anchor,
        identity_swapped=identity.swapped,
        identity_swap_probability=identity.swap_probability,
        identity_evidence_pair_count=identity.evidence_pair_count,
        identity_pair_swapped=identity.pair_swapped,
        identity_pair_swap_probability=identity.pair_swap_probability,
        identity_pair_evidence=identity.pair_evidence,
        smoothed_identity_centers=identity.smoothed_centers,
        diagnostics=diagnostics,
    )


def apply_trusted_observations(
    *,
    bundle: cvat_annotation.CvatBundle,
    observations: dict[tuple[int, int, int], TrustedObservation],
    output: cvat_annotation.AnnotationState,
    confidence: np.ndarray,
    source: np.ndarray,
    trusted: np.ndarray,
) -> None:
    """Copy immutable expert evidence into the working lower-body pose."""

    for obj_idx, runner_id in enumerate(bundle.runner_ids):
        for local_idx, joint_id in enumerate(cvat_annotation.JOINT_IDS):
            for time_idx in range(len(bundle.frame_indices)):
                observation = observations.get((runner_id, joint_id, time_idx))
                if observation is None:
                    continue
                output.points_xy[time_idx, obj_idx, local_idx] = observation.xy
                output.visibility[time_idx, obj_idx, local_idx] = max(
                    output.visibility[time_idx, obj_idx, local_idx], 2
                )
                confidence[time_idx, obj_idx, local_idx] = 1.0
                source[time_idx, obj_idx, local_idx] = observation.source
                trusted[time_idx, obj_idx, local_idx] = True


def identity_joint_swapped(
    identity: IdentityAssignmentResult,
    t: int,
    obj_idx: int,
    joint_id: int,
) -> bool:
    pair_idx = BILATERAL_PAIR_INDEX.get(joint_id)
    return bool(pair_idx is not None and identity.pair_swapped[t, obj_idx, pair_idx])


def enforce_constant_limb_lengths(
    points_xy: np.ndarray,
    trusted: np.ndarray,
    confidence: np.ndarray,
    prior: RunnerPrior,
    cfg: HumanConditioningConfig,
    *,
    active_frames: np.ndarray | None = None,
    movable_points: np.ndarray | None = None,
) -> np.ndarray:
    """Project each non-human pose onto fixed thigh and shank lengths.

    This is a position-based equality-constraint projection. Human observations
    have zero mobility. When both endpoints are inferred, the lower-confidence
    endpoint moves more. Repeated projections make the two constraints in each
    articulated leg converge jointly.
    """

    before = points_xy.copy()
    local = {joint_id: idx for idx, joint_id in enumerate(cvat_annotation.JOINT_IDS)}
    for t in range(len(points_xy)):
        if active_frames is not None and not bool(active_frames[t]):
            continue
        for _ in range(cfg.limb_constraint_iterations):
            maximum_error = 0.0
            for a, b in BONES:
                target = prior.bone_length_px.get(bone_name(a, b))
                if target is None or target <= 0:
                    continue
                a_idx, b_idx = local[a], local[b]
                pa, pb = points_xy[t, a_idx], points_xy[t, b_idx]
                if not np.isfinite([pa, pb]).all():
                    continue
                delta = pb - pa
                distance = float(np.linalg.norm(delta))
                if distance <= EPS:
                    continue
                error = distance - target
                maximum_error = max(maximum_error, abs(error))
                fixed_a = bool(
                    trusted[t, a_idx]
                    or (movable_points is not None and not movable_points[t, a_idx])
                )
                fixed_b = bool(
                    trusted[t, b_idx]
                    or (movable_points is not None and not movable_points[t, b_idx])
                )
                if fixed_a and fixed_b:
                    continue
                if fixed_a:
                    share_a, share_b = 0.0, 1.0
                elif fixed_b:
                    share_a, share_b = 1.0, 0.0
                else:
                    mobility_a = 1.0 / max(float(confidence[t, a_idx]), 0.05)
                    mobility_b = 1.0 / max(float(confidence[t, b_idx]), 0.05)
                    total = mobility_a + mobility_b
                    share_a, share_b = mobility_a / total, mobility_b / total
                correction = error * delta / distance
                points_xy[t, a_idx] += share_a * correction
                points_xy[t, b_idx] -= share_b * correction
            if maximum_error <= cfg.limb_constraint_tolerance_px:
                break
    displacement = np.linalg.norm(points_xy - before, axis=-1)
    return np.isfinite(displacement) & (displacement > cfg.limb_constraint_tolerance_px) & ~trusted


def fit_runner_priors(
    run_dir: Path,
    bundle: cvat_annotation.CvatBundle,
    posterior: np.ndarray,
    observations: list[TrustedObservation],
    bboxes_xyxy: np.ndarray,
    peak: np.ndarray,
    entropy: np.ndarray,
) -> list[RunnerPrior]:
    quality = quality_scores(peak, entropy)
    machine_trust = machine_trust_mask(run_dir, bundle, quality)
    local = {joint_id: idx for idx, joint_id in enumerate(cvat_annotation.JOINT_IDS)}
    trusted_lookup = {
        (item.runner_id, item.time_idx, item.joint_id): np.asarray(item.xy) for item in observations
    }
    priors: list[RunnerPrior] = []
    for obj_idx, runner_id in enumerate(bundle.runner_ids):
        lengths: dict[str, float] = {}
        sigmas: dict[str, float] = {}
        pixel_lengths: dict[str, float] = {}
        pixel_sigmas: dict[str, float] = {}
        counts: dict[str, int] = {}
        machine_counts: dict[str, int] = {}
        fallback_bones: list[str] = []
        for a, b in BONES:
            name = bone_name(a, b)
            values: list[float] = []
            human_values: list[float] = []
            pixel_values: list[float] = []
            human_pixel_values: list[float] = []
            for t in range(len(bundle.frame_indices)):
                bbox_h = bbox_height(bboxes_xyxy[t, obj_idx])
                if bbox_h <= 0:
                    continue
                pa = posterior[t, obj_idx, local[a]]
                pb = posterior[t, obj_idx, local[b]]
                if (
                    np.isfinite(pa).all()
                    and np.isfinite(pb).all()
                    and machine_trust[t, obj_idx, local[a]]
                    and machine_trust[t, obj_idx, local[b]]
                ):
                    pixel_length = float(np.linalg.norm(pa - pb))
                    values.append(pixel_length / bbox_h)
                    pixel_values.append(pixel_length)
                ha = trusted_lookup.get((runner_id, t, a))
                hb = trusted_lookup.get((runner_id, t, b))
                if ha is not None and hb is not None:
                    human_pixel_length = float(np.linalg.norm(ha - hb))
                    human_values.append(human_pixel_length / bbox_h)
                    human_pixel_values.append(human_pixel_length)
            machine_counts[name] = len(values)
            if len(values) < 3 and not human_values:
                values = fallback_bone_lengths(
                    posterior[:, obj_idx], bboxes_xyxy[:, obj_idx], local[a], local[b]
                )
                fallback_bones.append(name)
            samples = np.asarray([*values, *human_values, *human_values], dtype=np.float32)
            if not samples.size:
                continue
            median = float(np.median(samples))
            mad = float(1.4826 * np.median(np.abs(samples - median)))
            lengths[name] = median
            sigmas[name] = max(mad, median * 0.08, 0.01)
            pixel_samples = np.asarray(
                human_pixel_values if human_pixel_values else pixel_values,
                dtype=np.float32,
            )
            if pixel_samples.size:
                pixel_median = float(np.median(pixel_samples))
                pixel_mad = float(1.4826 * np.median(np.abs(pixel_samples - pixel_median)))
                pixel_lengths[name] = pixel_median
                pixel_sigmas[name] = max(pixel_mad, 0.5)
            counts[name] = len(human_values)
        motion_sigmas: dict[str, float] = {}
        for local_idx, joint_id in enumerate(cvat_annotation.JOINT_IDS):
            relative = runner_relative_points(
                posterior[:, obj_idx, local_idx], bboxes_xyxy[:, obj_idx]
            )
            values: list[float] = []
            for t in range(1, len(relative) - 1):
                if np.isfinite(relative[t - 1 : t + 2]).all() and np.all(
                    machine_trust[t - 1 : t + 2, obj_idx, local_idx]
                ):
                    values.append(
                        float(np.linalg.norm(relative[t + 1] - 2 * relative[t] + relative[t - 1]))
                    )
            if values:
                samples = np.asarray(values, dtype=np.float32)
                median = float(np.median(samples))
                mad = float(1.4826 * np.median(np.abs(samples - median)))
                motion_sigmas[str(joint_id)] = max(median + mad, 0.015)
        priors.append(
            RunnerPrior(
                runner_id=runner_id,
                bone_length_norm=lengths,
                bone_sigma_norm=sigmas,
                bone_length_px=pixel_lengths,
                bone_sigma_px=pixel_sigmas,
                trusted_bone_samples=counts,
                machine_bone_samples=machine_counts,
                fallback_bones=fallback_bones,
                motion_sigma_norm=motion_sigmas,
            )
        )
    return priors


def condition_frame_map(
    *,
    run_dir: Path,
    frame_idx: int,
    runner_id: int,
    joint_id: int,
    base_joint_id: int | None = None,
    center: np.ndarray,
    dynamic_weight: float,
    dynamic_sigma_px: float,
    neighbor_points: np.ndarray,
    prior: RunnerPrior,
    bbox_h: float,
    motion_center: np.ndarray | None,
    motion_sigma_norm: float | None,
    cfg: HumanConditioningConfig,
) -> tuple[np.ndarray | None, float]:
    loaded = load_posterior_map(
        run_dir,
        frame_idx,
        runner_id,
        joint_id if base_joint_id is None else base_joint_id,
    )
    if loaded is None:
        return center.astype(np.float32), max(
            0.5, min(0.95, dynamic_weight / max(cfg.tracker_weight, 1e-6))
        )
    base, crop_xyxy = loaded
    yy, xx = np.mgrid[: base.shape[0], : base.shape[1]].astype(np.float32)
    local_center = center - crop_xyxy[:2]
    dynamic = gaussian_grid(xx, yy, local_center, dynamic_sigma_px)
    logp = np.log(np.clip(base, EPS, None)) + dynamic_weight * np.log(np.clip(dynamic, EPS, None))
    if motion_center is not None:
        local_motion = motion_center - crop_xyxy[:2]
        motion_sigma = (
            max(motion_sigma_norm or cfg.min_motion_sigma_norm, cfg.min_motion_sigma_norm) * bbox_h
        )
        motion = gaussian_grid(xx, yy, local_motion, max(motion_sigma, 1.0))
        logp += cfg.motion_weight * np.log(np.clip(motion, EPS, None))
    geometry = geometry_grid(
        xx,
        yy,
        crop_xyxy,
        joint_id,
        neighbor_points,
        prior,
        bbox_h,
        cfg,
    )
    if geometry is not None:
        logp += cfg.geometry_weight * np.log(np.clip(geometry, EPS, None))
    exclusion = contralateral_exclusion_grid(
        xx,
        yy,
        crop_xyxy,
        joint_id,
        neighbor_points,
        bbox_h,
        cfg,
    )
    if exclusion is not None:
        logp += cfg.contralateral_exclusion_weight * np.log(np.clip(exclusion, EPS, None))
    logp -= float(np.max(logp))
    probability = np.exp(logp).astype(np.float32)
    probability /= max(float(probability.sum()), EPS)
    y, x = np.unravel_index(int(np.argmax(probability)), probability.shape)
    point = np.asarray([x + crop_xyxy[0], y + crop_xyxy[1]], dtype=np.float32)
    entropy = float(-np.sum(probability * np.log(np.clip(probability, EPS, 1.0))))
    quality = float(np.clip(1.0 - entropy / max(np.log(probability.size), 1.0), 0.0, 1.0))
    return point, quality


def geometry_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    crop_xyxy: np.ndarray,
    joint_id: int,
    neighbor_points: np.ndarray,
    prior: RunnerPrior,
    bbox_h: float,
    cfg: HumanConditioningConfig,
) -> np.ndarray | None:
    local = {joint: idx for idx, joint in enumerate(cvat_annotation.JOINT_IDS)}
    evidence: np.ndarray | None = None
    for neighbor_id, pair in NEIGHBORS[joint_id]:
        point = neighbor_points[local[neighbor_id]]
        name = bone_name(*pair)
        target_norm = prior.bone_length_norm.get(name)
        if target_norm is None or not np.isfinite(point).all():
            continue
        neighbor = point - crop_xyxy[:2]
        distance = np.sqrt((xx - neighbor[0]) ** 2 + (yy - neighbor[1]) ** 2)
        target = target_norm * bbox_h
        sigma_norm = max(
            prior.bone_sigma_norm.get(name, target_norm * cfg.geometry_sigma_relative),
            cfg.min_geometry_sigma_norm,
        )
        current = np.exp(-0.5 * ((distance - target) / max(sigma_norm * bbox_h, 1.0)) ** 2)
        evidence = current if evidence is None else evidence * current
    return evidence


def contralateral_exclusion_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    crop_xyxy: np.ndarray,
    joint_id: int,
    neighbor_points: np.ndarray,
    bbox_h: float,
    cfg: HumanConditioningConfig,
) -> np.ndarray | None:
    opposite_id = CONTRALATERAL_JOINT.get(joint_id)
    if opposite_id is None:
        return None
    local_idx = cvat_annotation.JOINT_IDS.index(opposite_id)
    opposite = neighbor_points[local_idx]
    if not np.isfinite(opposite).all() or bbox_h <= 0:
        return None
    center = opposite - crop_xyxy[:2]
    distance_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    radius = max(cfg.duplicate_joint_distance_norm * bbox_h, 1.0)
    return 1.0 - np.exp(-0.5 * distance_sq / (radius * radius))


def load_posterior_map(
    run_dir: Path,
    frame_idx: int,
    runner_id: int,
    joint_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    path = (
        run_dir / "posterior" / "posteriors" / f"frame_{frame_idx:06d}_runner_{runner_id:04d}.npz"
    )
    if not path.is_file():
        return None
    maps, joint_ids, crop_xyxy = load_posterior_file(path)
    positions = np.flatnonzero(joint_ids == joint_id)
    if not len(positions):
        return None
    probability = maps[int(positions[0])].copy()
    probability /= max(float(probability.sum()), EPS)
    return probability, crop_xyxy


@lru_cache(maxsize=64)
def load_posterior_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one immutable frame/runner posterior and close its NPZ immediately."""

    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["posterior"], dtype=np.float32),
            np.asarray(data["joint_ids"], dtype=np.int16),
            np.asarray(data["crop_xyxy"], dtype=np.float32),
        )


def transported_center(
    anchor: TrustedObservation,
    posterior: np.ndarray,
    t: int,
    obj_idx: int,
    local_idx: int,
    *,
    target_local_idx: int | None = None,
) -> np.ndarray | None:
    source = posterior[anchor.time_idx, obj_idx, local_idx]
    target = posterior[t, obj_idx, target_local_idx if target_local_idx is not None else local_idx]
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        return None
    return np.asarray(anchor.xy, dtype=np.float32) + target - source


def transported_from_neighbor(
    posterior: np.ndarray,
    conditioned: np.ndarray,
    bboxes_xyxy: np.ndarray,
    t: int,
    near_t: int,
    obj_idx: int,
    local_idx: int,
    *,
    current_posterior_local_idx: int | None = None,
    previous_posterior_local_idx: int | None = None,
) -> np.ndarray | None:
    if near_t < 0 or near_t >= len(posterior):
        return None
    current = posterior[
        t,
        obj_idx,
        current_posterior_local_idx if current_posterior_local_idx is not None else local_idx,
    ]
    previous = posterior[
        near_t,
        obj_idx,
        previous_posterior_local_idx if previous_posterior_local_idx is not None else local_idx,
    ]
    previous_conditioned = conditioned[near_t, obj_idx, local_idx]
    previous_h = bbox_height(bboxes_xyxy[near_t, obj_idx])
    current_h = bbox_height(bboxes_xyxy[t, obj_idx])
    if (
        previous_h <= 0
        or current_h <= 0
        or not np.isfinite([current, previous, previous_conditioned]).all()
    ):
        return None
    correction_norm = (previous_conditioned - previous) / previous_h
    return current + correction_norm * current_h


def order_by_anchor_distance(length: int, anchors: list[TrustedObservation]) -> list[int]:
    return sorted(range(length), key=lambda t: min(abs(t - item.time_idx) for item in anchors))


def quality_scores(peak: np.ndarray, entropy: np.ndarray) -> np.ndarray:
    peak = np.asarray(peak, dtype=np.float32)
    entropy = np.asarray(entropy, dtype=np.float32)
    output = np.zeros_like(peak)
    for obj_idx in range(peak.shape[1]):
        for joint_idx in range(peak.shape[2]):
            values = peak[:, obj_idx, joint_idx]
            finite = values[np.isfinite(values)]
            scale = float(np.quantile(finite, 0.9)) if finite.size else 1.0
            peak_score = np.clip(values / max(scale, EPS), 0.0, 1.0)
            entropy_values = entropy[:, obj_idx, joint_idx]
            finite_entropy = entropy_values[np.isfinite(entropy_values)]
            if finite_entropy.size:
                low, high = np.quantile(finite_entropy, [0.1, 0.9])
                entropy_score = 1.0 - np.clip(
                    (entropy_values - low) / max(high - low, EPS), 0.0, 1.0
                )
            else:
                entropy_score = np.zeros_like(entropy_values)
            output[:, obj_idx, joint_idx] = 0.5 * peak_score + 0.5 * entropy_score
    return output.astype(np.float32)


def machine_trust_mask(
    run_dir: Path,
    bundle: cvat_annotation.CvatBundle,
    quality: np.ndarray,
) -> np.ndarray:
    _ = run_dir, bundle
    return quality >= 0.2


def fallback_bone_lengths(
    points: np.ndarray,
    bboxes: np.ndarray,
    a: int,
    b: int,
) -> list[float]:
    values: list[float] = []
    for pose, bbox in zip(points, bboxes, strict=True):
        height = bbox_height(bbox)
        if height > 0 and np.isfinite(pose[a]).all() and np.isfinite(pose[b]).all():
            values.append(float(np.linalg.norm(pose[a] - pose[b]) / height))
    if len(values) < 4:
        return values
    samples = np.asarray(values, dtype=np.float32)
    median = float(np.median(samples))
    mad = max(float(1.4826 * np.median(np.abs(samples - median))), 0.01)
    return samples[np.abs(samples - median) <= 3.0 * mad].astype(float).tolist()


def runner_relative_points(points: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    output = np.full_like(points, np.nan, dtype=np.float32)
    for t, (point, bbox) in enumerate(zip(points, bboxes, strict=True)):
        height = bbox_height(bbox)
        if height <= 0 or not np.isfinite(point).all():
            continue
        center = np.asarray(
            [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32
        )
        output[t] = (point - center) / height
    return output


def gaussian_grid(xx: np.ndarray, yy: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / max(sigma * sigma, EPS))


def bbox_height(bbox: np.ndarray) -> float:
    values = np.asarray(bbox, dtype=np.float32)
    if values.shape != (4,) or not np.isfinite(values).all():
        return 0.0
    return max(float(values[3] - values[1]), 0.0)


def bone_name(a: int, b: int) -> str:
    return f"{min(a, b)}_{max(a, b)}"


def save_result(
    run_dir: Path,
    bundle: cvat_annotation.CvatBundle,
    result: ConditioningResult,
) -> dict[str, str]:
    run_dir.joinpath("annotations").mkdir(parents=True, exist_ok=True)
    suffix = "" if bundle.runner_id is None else f"_runner_{bundle.runner_id}"
    pose_path = run_dir / "annotations" / f"human_conditioned_pose{suffix}.npz"
    diagnostics_path = run_dir / "annotations" / f"human_conditioning_diagnostics{suffix}.json"
    manifest = cvat_annotation.load_manifest(run_dir)
    with np.load(
        cvat_annotation.required_artifact(run_dir, manifest, "posterior_npz"),
        allow_pickle=False,
    ) as posterior_data:
        posterior_frames = np.asarray(posterior_data["frame_indices"], dtype=np.int32)
        posterior_ids = np.asarray(posterior_data["object_ids"], dtype=np.int32)
        canonical_xy = np.asarray(posterior_data["posterior_keypoints_xy"], dtype=np.float32)
        peak_probability = np.asarray(
            posterior_data["posterior_peak_probability"], dtype=np.float32
        )
        entropy = np.asarray(posterior_data["posterior_entropy"], dtype=np.float32)
    frame_lookup = {int(value): idx for idx, value in enumerate(posterior_frames)}
    runner_lookup = {int(value): idx for idx, value in enumerate(posterior_ids)}
    frame_selection = [frame_lookup[value] for value in bundle.frame_indices]
    runner_selection = [runner_lookup[value] for value in bundle.runner_ids]
    first_pass_path = manifest.artifacts.get("first_pass_pose_npz")
    if first_pass_path and (run_dir / first_pass_path).is_file():
        with np.load(run_dir / first_pass_path, allow_pickle=False) as first_pass_data:
            canonical_xy = np.asarray(first_pass_data["first_pass_keypoints_xy"], dtype=np.float32)
    posterior_full = canonical_xy[np.ix_(frame_selection, runner_selection)].astype(np.float32)
    conditioned_full = posterior_full.copy()
    confidence_full = quality_scores(
        peak_probability[np.ix_(frame_selection, runner_selection)],
        entropy[np.ix_(frame_selection, runner_selection)],
    )
    source_full = np.full(posterior_full.shape[:-1], "POSTERIOR", dtype="<U32")
    trusted_full = np.zeros(posterior_full.shape[:-1], dtype=bool)
    applied_full = np.zeros(posterior_full.shape[:-1], dtype=bool)
    for local_idx, joint_id in enumerate(cvat_annotation.JOINT_IDS):
        if joint_id == cvat_annotation.NECK_JOINT_ID:
            continue
        conditioned_full[:, :, joint_id] = result.state.points_xy[:, :, local_idx]
        confidence_full[:, :, joint_id] = result.confidence[:, :, local_idx]
        source_full[:, :, joint_id] = result.source[:, :, local_idx]
        trusted_full[:, :, joint_id] = result.trusted[:, :, local_idx]
        applied_full[:, :, joint_id] = result.applied[:, :, local_idx]
    np.savez_compressed(
        pose_path,
        frame_indices=np.asarray(bundle.frame_indices, dtype=np.int32),
        object_ids=np.asarray(bundle.runner_ids, dtype=np.int32),
        posterior_keypoints_xy=posterior_full,
        human_conditioned_keypoints_xy=conditioned_full,
        human_conditioned_confidence=confidence_full,
        source=source_full,
        trusted_observation=trusted_full,
        frame_ground_truth=result.state.frame_ground_truth,
        keypoint_ground_truth=result.state.keypoint_ground_truth,
        prior_applied=applied_full,
        source_anchor_index=result.source_anchor,
        identity_swapped=result.identity_swapped,
        identity_swap_probability=result.identity_swap_probability,
        identity_evidence_pair_count=result.identity_evidence_pair_count,
        identity_pair_names=np.asarray(BILATERAL_PAIR_NAMES, dtype="<U8"),
        identity_pair_swapped=result.identity_pair_swapped,
        identity_pair_swap_probability=result.identity_pair_swap_probability,
        identity_pair_evidence=result.identity_pair_evidence,
        smoothed_identity_centers=result.smoothed_identity_centers,
    )
    diagnostics_path.write_text(
        json.dumps(result.diagnostics.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return {
        "human_conditioned_pose_npz": str(pose_path.resolve()),
        "human_conditioning_diagnostics_json": str(diagnostics_path.resolve()),
    }
