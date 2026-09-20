"""Probabilistic temporal identity and knee-biomechanics decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dromia import config as dromia_config

EPS = 1e-9
LOWER_JOINTS = (13, 14, 15, 16)
KNEE_TRIPLETS = ((11, 13, 15), (12, 14, 16))
BONES = ((11, 13), (13, 15), (12, 14), (14, 16))
STATE_BITS = ((0, 0), (1, 0), (0, 1), (1, 1))
STATE_NAMES = ("normal", "knees_swapped", "ankles_swapped", "knees_and_ankles_swapped")


@dataclass(frozen=True, slots=True)
class RunnerPriors:
    bone_log_mean: np.ndarray
    bone_log_sigma: np.ndarray
    bend_direction: np.ndarray


@dataclass(frozen=True, slots=True)
class TemporalBiomechanicsResult:
    corrected_xy: np.ndarray
    state_path: np.ndarray
    state_probability: np.ndarray
    state_names: tuple[str, ...]
    chosen_position_log_prior: np.ndarray
    chosen_angle_log_prior: np.ndarray
    chosen_bend_continuity_log_prior: np.ndarray
    chosen_bend_direction_log_prior: np.ndarray
    chosen_limb_length_log_prior: np.ndarray
    priors: list[RunnerPriors]


def decode_temporal_biomechanics(
    points_xy: np.ndarray,
    bboxes_xyxy: np.ndarray,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> TemporalBiomechanicsResult:
    points = np.asarray(points_xy, dtype=np.float32)
    bboxes = np.asarray(bboxes_xyxy, dtype=np.float32)
    if points.ndim != 4 or points.shape[2:] != (17, 2):
        raise ValueError("points_xy must be [T,O,17,2]")
    if bboxes.shape != points.shape[:2] + (4,):
        raise ValueError("bboxes_xyxy must be [T,O,4]")
    frame_count, object_count = points.shape[:2]
    corrected = points.copy()
    state_path = np.zeros((frame_count, object_count), dtype=np.int8)
    state_probability = np.zeros((frame_count, object_count, len(STATE_BITS)), dtype=np.float32)
    components = [np.zeros((frame_count, object_count), dtype=np.float32) for _ in range(5)]
    priors: list[RunnerPriors] = []
    for obj_idx in range(object_count):
        runner_points = points[:, obj_idx]
        scales = bbox_heights(bboxes[:, obj_idx])
        prior = fit_runner_priors(runner_points, scales, cfg)
        priors.append(prior)
        decoded = decode_runner(runner_points, scales, prior, cfg)
        path, probability, selected_components = decoded
        state_path[:, obj_idx] = path
        state_probability[:, obj_idx] = probability
        for t, state_idx in enumerate(path):
            corrected[t, obj_idx] = apply_state(runner_points[t], int(state_idx))
        for output, values in zip(components, selected_components, strict=True):
            output[:, obj_idx] = values
    return TemporalBiomechanicsResult(
        corrected_xy=corrected,
        state_path=state_path,
        state_probability=state_probability,
        state_names=STATE_NAMES,
        chosen_position_log_prior=components[0],
        chosen_angle_log_prior=components[1],
        chosen_bend_continuity_log_prior=components[2],
        chosen_bend_direction_log_prior=components[3],
        chosen_limb_length_log_prior=components[4],
        priors=priors,
    )


def decode_framewise_biomechanics(
    points_xy: np.ndarray,
    bboxes_xyxy: np.ndarray,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> TemporalBiomechanicsResult:
    """Apply the temporal decoder's unary evidence independently in each frame.

    Runner-level anatomical priors are fitted exactly as in the temporal decoder.
    State selection uses no transition score, neighbouring-frame evidence, initial
    state prior, Viterbi pass, or forward--backward pass.  The existing swap
    acceptance threshold is applied to a softmax over the four unary scores.
    """

    points = np.asarray(points_xy, dtype=np.float32)
    bboxes = np.asarray(bboxes_xyxy, dtype=np.float32)
    if points.ndim != 4 or points.shape[2:] != (17, 2):
        raise ValueError("points_xy must be [T,O,17,2]")
    if bboxes.shape != points.shape[:2] + (4,):
        raise ValueError("bboxes_xyxy must be [T,O,4]")
    frame_count, object_count = points.shape[:2]
    corrected = points.copy()
    state_path = np.zeros((frame_count, object_count), dtype=np.int8)
    state_probability = np.zeros((frame_count, object_count, len(STATE_BITS)), dtype=np.float32)
    components = [np.zeros((frame_count, object_count), dtype=np.float32) for _ in range(5)]
    priors: list[RunnerPriors] = []
    for obj_idx in range(object_count):
        runner_points = points[:, obj_idx]
        scales = bbox_heights(bboxes[:, obj_idx])
        prior = fit_runner_priors(runner_points, scales, cfg)
        priors.append(prior)
        path, probability, selected_components = decode_runner_framewise(
            runner_points, scales, prior, cfg
        )
        state_path[:, obj_idx] = path
        state_probability[:, obj_idx] = probability
        for t, state_idx in enumerate(path):
            corrected[t, obj_idx] = apply_state(runner_points[t], int(state_idx))
        for output, values in zip(components, selected_components, strict=True):
            output[:, obj_idx] = values
    return TemporalBiomechanicsResult(
        corrected_xy=corrected,
        state_path=state_path,
        state_probability=state_probability,
        state_names=STATE_NAMES,
        chosen_position_log_prior=components[0],
        chosen_angle_log_prior=components[1],
        chosen_bend_continuity_log_prior=components[2],
        chosen_bend_direction_log_prior=components[3],
        chosen_limb_length_log_prior=components[4],
        priors=priors,
    )


def decode_runner(
    points: np.ndarray,
    scales: np.ndarray,
    priors: RunnerPriors,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    count = len(points)
    state_count = len(STATE_BITS)
    poses, unary, limb_component, direction_component = runner_unary_evidence(
        points, scales, priors, cfg
    )
    transitions = np.zeros((count, state_count, state_count), dtype=np.float64)
    position_component = np.zeros_like(transitions)
    angle_component = np.zeros_like(transitions)
    bend_component = np.zeros_like(transitions)
    for t in range(1, count):
        for previous in range(state_count):
            for current in range(state_count):
                position_component[t, previous, current] = position_log_prior(
                    poses[t - 1, previous], poses[t, current], scales[t], cfg
                )
                angle_component[t, previous, current] = angle_log_prior(
                    poses[t - 1, previous], poses[t, current], cfg
                )
                bend_component[t, previous, current] = bend_continuity_log_prior(
                    poses[t - 1, previous], poses[t, current], cfg
                )
                transitions[t, previous, current] = (
                    position_component[t, previous, current]
                    + angle_component[t, previous, current]
                    + bend_component[t, previous, current]
                    + switch_log_prior(previous, current, cfg)
                )
    path, probability = decode_hmm(unary, transitions, cfg)
    selected_probability = probability[np.arange(count), path]
    uncertain_swap = (path != 0) & (selected_probability < cfg.posterior_swap_threshold)
    path[uncertain_swap] = 0
    selected_position = np.zeros(count, dtype=np.float32)
    selected_angle = np.zeros(count, dtype=np.float32)
    selected_bend = np.zeros(count, dtype=np.float32)
    for t in range(1, count):
        previous, current = int(path[t - 1]), int(path[t])
        selected_position[t] = position_component[t, previous, current]
        selected_angle[t] = angle_component[t, previous, current]
        selected_bend[t] = bend_component[t, previous, current]
    selected_direction = direction_component[np.arange(count), path].astype(np.float32)
    selected_limb = limb_component[np.arange(count), path].astype(np.float32)
    return (
        path,
        probability,
        (
            selected_position,
            selected_angle,
            selected_bend,
            selected_direction,
            selected_limb,
        ),
    )


def decode_runner_framewise(
    points: np.ndarray,
    scales: np.ndarray,
    priors: RunnerPriors,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Select the best anatomical assignment from unary evidence in each frame."""

    count = len(points)
    if count == 0:
        return (
            np.empty(0, dtype=np.int8),
            np.empty((0, len(STATE_BITS)), dtype=np.float32),
            tuple(np.empty(0, dtype=np.float32) for _ in range(5)),
        )
    _poses, unary, limb_component, direction_component = runner_unary_evidence(
        points, scales, priors, cfg
    )
    log_probability = unary - logsumexp(unary, axis=1)[:, None]
    probability = np.exp(log_probability).astype(np.float32)
    path = np.argmax(unary, axis=1).astype(np.int8)
    selected_probability = probability[np.arange(count), path]
    uncertain_swap = (path != 0) & (selected_probability < cfg.posterior_swap_threshold)
    path[uncertain_swap] = 0
    zeros = np.zeros(count, dtype=np.float32)
    selected_direction = direction_component[np.arange(count), path].astype(np.float32)
    selected_limb = limb_component[np.arange(count), path].astype(np.float32)
    return (
        path,
        probability,
        (zeros.copy(), zeros.copy(), zeros.copy(), selected_direction, selected_limb),
    )


def runner_unary_evidence(
    points: np.ndarray,
    scales: np.ndarray,
    priors: RunnerPriors,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the four candidate poses and the shared frame-level score terms."""

    count = len(points)
    state_count = len(STATE_BITS)
    poses = np.asarray(
        [[apply_state(points[t], state) for state in range(state_count)] for t in range(count)],
        dtype=np.float32,
    )
    unary = np.zeros((count, state_count), dtype=np.float64)
    limb_component = np.zeros_like(unary)
    direction_component = np.zeros_like(unary)
    for t in range(count):
        for state in range(state_count):
            limb_component[t, state] = limb_length_log_prior(
                poses[t, state], scales[t], priors, cfg
            )
            direction_component[t, state] = bend_direction_log_prior(poses[t, state], priors, cfg)
            unary[t, state] = (
                limb_component[t, state]
                + direction_component[t, state]
                + model_identity_log_prior(state, cfg)
            )
    return poses, unary, limb_component, direction_component


def decode_hmm(
    unary: np.ndarray,
    transitions: np.ndarray,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    count, state_count = unary.shape
    if count == 0:
        return np.empty(0, dtype=np.int8), np.empty((0, state_count), dtype=np.float32)
    initial_swap = cfg.initial_swap_probability
    initial = np.asarray(
        [
            (1.0 - initial_swap) ** 2,
            initial_swap * (1.0 - initial_swap),
            initial_swap * (1.0 - initial_swap),
            initial_swap**2,
        ],
        dtype=np.float64,
    )
    log_initial = np.log(np.clip(initial, EPS, None))
    viterbi = np.empty((count, state_count), dtype=np.float64)
    backpointer = np.zeros((count, state_count), dtype=np.int8)
    viterbi[0] = log_initial + unary[0]
    for t in range(1, count):
        candidates = viterbi[t - 1][:, None] + transitions[t]
        backpointer[t] = np.argmax(candidates, axis=0)
        viterbi[t] = unary[t] + np.max(candidates, axis=0)
    path = np.zeros(count, dtype=np.int8)
    path[-1] = int(np.argmax(viterbi[-1]))
    for t in range(count - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]

    forward = np.empty_like(viterbi)
    forward[0] = log_initial + unary[0]
    for t in range(1, count):
        forward[t] = unary[t] + logsumexp(forward[t - 1][:, None] + transitions[t], axis=0)
    backward = np.zeros_like(viterbi)
    for t in range(count - 2, -1, -1):
        backward[t] = logsumexp(
            transitions[t + 1] + unary[t + 1][None, :] + backward[t + 1][None, :],
            axis=1,
        )
    log_marginal = forward + backward
    log_marginal -= logsumexp(log_marginal, axis=1)[:, None]
    return path, np.exp(log_marginal).astype(np.float32)


def apply_state(points: np.ndarray, state: int) -> np.ndarray:
    result = np.asarray(points, dtype=np.float32).copy()
    knee_swap, ankle_swap = STATE_BITS[state]
    if knee_swap:
        result[[13, 14]] = result[[14, 13]]
    if ankle_swap:
        result[[15, 16]] = result[[16, 15]]
    return result


def apply_state_path(points_xy: np.ndarray, state_path: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    path = np.asarray(state_path, dtype=np.int8)
    if path.shape != points.shape[:2]:
        raise ValueError("state_path must be [T,O]")
    corrected = points.copy()
    for t in range(points.shape[0]):
        for obj_idx in range(points.shape[1]):
            corrected[t, obj_idx] = apply_state(points[t, obj_idx], int(path[t, obj_idx]))
    return corrected


def restore_collapsed_pairs_from_model_candidates(
    corrected_posterior: np.ndarray,
    corrected_model: np.ndarray,
    bboxes_xyxy: np.ndarray,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the alternate model mode when posterior evidence collapses a bilateral pair."""
    posterior = np.asarray(corrected_posterior, dtype=np.float32).copy()
    model = np.asarray(corrected_model, dtype=np.float32)
    bboxes = np.asarray(bboxes_xyxy, dtype=np.float32)
    restored = np.zeros(posterior.shape[:3], dtype=bool)
    for t in range(posterior.shape[0]):
        for obj_idx in range(posterior.shape[1]):
            scale = max(float(bboxes[t, obj_idx, 3] - bboxes[t, obj_idx, 1]), 1.0)
            for left, right in ((13, 14), (15, 16)):
                post_pair = posterior[t, obj_idx, [left, right]]
                model_pair = model[t, obj_idx, [left, right]]
                if not np.isfinite(post_pair).all() or not np.isfinite(model_pair).all():
                    continue
                post_separation = float(np.linalg.norm(post_pair[0] - post_pair[1]) / scale)
                model_separation = float(np.linalg.norm(model_pair[0] - model_pair[1]) / scale)
                if (
                    post_separation < cfg.duplicate_pair_distance_norm
                    and model_separation >= cfg.raw_candidate_min_separation_norm
                ):
                    posterior[t, obj_idx, [left, right]] = model_pair
                    restored[t, obj_idx, [left, right]] = True
    return posterior, restored


def fit_runner_priors(
    points: np.ndarray,
    scales: np.ndarray,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> RunnerPriors:
    logs = [[] for _ in BONES]
    bends = [[], []]
    for t, pose in enumerate(points):
        scale = max(float(scales[t]), 1.0)
        for bone_idx, (a, b) in enumerate(BONES):
            if np.isfinite(pose[[a, b]]).all():
                length = float(np.linalg.norm(pose[a] - pose[b]) / scale)
                if length > EPS:
                    logs[bone_idx].append(np.log(length))
        for side, triplet in enumerate(KNEE_TRIPLETS):
            value = signed_bend(pose, triplet)
            if np.isfinite(value) and abs(value) > 0.15:
                bends[side].append(value)
    means = np.zeros(len(BONES), dtype=np.float32)
    sigmas = np.full(len(BONES), cfg.limb_log_sigma_min, dtype=np.float32)
    for idx, values in enumerate(logs):
        if not values:
            continue
        array = np.asarray(values, dtype=np.float32)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        means[idx] = median
        sigmas[idx] = max(1.4826 * mad, cfg.limb_log_sigma_min)
    direction = np.ones(2, dtype=np.float32)
    for side, values in enumerate(bends):
        if values:
            direction[side] = 1.0 if float(np.median(values)) >= 0.0 else -1.0
    return RunnerPriors(means, sigmas, direction)


def position_log_prior(
    previous: np.ndarray,
    current: np.ndarray,
    scale: float,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> float:
    valid = np.isfinite(previous[list(LOWER_JOINTS)]).all(axis=1) & np.isfinite(
        current[list(LOWER_JOINTS)]
    ).all(axis=1)
    if not np.any(valid):
        return 0.0
    delta = current[list(LOWER_JOINTS)][valid] - previous[list(LOWER_JOINTS)][valid]
    z = np.linalg.norm(delta, axis=1) / max(float(scale) * cfg.position_step_sigma_norm, EPS)
    return float(-0.5 * cfg.position_weight * np.sum(z**2))


def angle_log_prior(
    previous: np.ndarray,
    current: np.ndarray,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> float:
    sigma = np.deg2rad(cfg.angle_step_sigma_degrees)
    score = 0.0
    for triplet in KNEE_TRIPLETS:
        old = flexion_angle(previous, triplet)
        new = flexion_angle(current, triplet)
        if np.isfinite(old) and np.isfinite(new):
            score -= 0.5 * cfg.angle_weight * ((new - old) / sigma) ** 2
    return float(score)


def bend_continuity_log_prior(
    previous: np.ndarray,
    current: np.ndarray,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> float:
    score = 0.0
    for triplet in KNEE_TRIPLETS:
        old = signed_bend(previous, triplet)
        new = signed_bend(current, triplet)
        if np.isfinite(old) and np.isfinite(new):
            score -= 0.5 * cfg.bend_continuity_weight * ((new - old) / cfg.bend_step_sigma) ** 2
    return float(score)


def bend_direction_log_prior(
    pose: np.ndarray,
    priors: RunnerPriors,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> float:
    score = 0.0
    for side, triplet in enumerate(KNEE_TRIPLETS):
        bend = signed_bend(pose, triplet)
        if not np.isfinite(bend):
            continue
        wrong_way = max(0.0, -float(priors.bend_direction[side]) * bend)
        score -= 0.5 * cfg.bend_direction_weight * (wrong_way / cfg.bend_direction_sigma) ** 2
    return float(score)


def limb_length_log_prior(
    pose: np.ndarray,
    scale: float,
    priors: RunnerPriors,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> float:
    score = 0.0
    for idx, (a, b) in enumerate(BONES):
        if not np.isfinite(pose[[a, b]]).all():
            continue
        length = float(np.linalg.norm(pose[a] - pose[b]) / max(float(scale), 1.0))
        if length <= EPS:
            continue
        z = (np.log(length) - float(priors.bone_log_mean[idx])) / float(priors.bone_log_sigma[idx])
        score -= 0.5 * cfg.limb_length_weight * z * z
    return float(score)


def model_identity_log_prior(state: int, cfg: dromia_config.TemporalBiomechanicsConfig) -> float:
    swaps = sum(STATE_BITS[state])
    p = cfg.model_identity_probability
    return float(cfg.identity_weight * (swaps * np.log(1.0 - p) + (2 - swaps) * np.log(p)))


def switch_log_prior(
    previous: int,
    current: int,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> float:
    previous_bits = STATE_BITS[previous]
    current_bits = STATE_BITS[current]
    switches = sum(a != b for a, b in zip(previous_bits, current_bits, strict=True))
    stays = 2 - switches
    p = cfg.identity_switch_probability
    return float(switches * np.log(p) + stays * np.log(1.0 - p))


def flexion_angle(pose: np.ndarray, triplet: tuple[int, int, int]) -> float:
    hip, knee, ankle = triplet
    if not np.isfinite(pose[[hip, knee, ankle]]).all():
        return float("nan")
    thigh = pose[hip] - pose[knee]
    shank = pose[ankle] - pose[knee]
    denominator = max(float(np.linalg.norm(thigh) * np.linalg.norm(shank)), EPS)
    return float(np.arccos(np.clip(float(np.dot(thigh, shank)) / denominator, -1.0, 1.0)))


def signed_bend(pose: np.ndarray, triplet: tuple[int, int, int]) -> float:
    hip, knee, ankle = triplet
    if not np.isfinite(pose[[hip, knee, ankle]]).all():
        return float("nan")
    thigh = pose[hip] - pose[knee]
    shank = pose[ankle] - pose[knee]
    denominator = max(float(np.linalg.norm(thigh) * np.linalg.norm(shank)), EPS)
    return float((thigh[0] * shank[1] - thigh[1] * shank[0]) / denominator)


def bbox_heights(bboxes: np.ndarray) -> np.ndarray:
    arr = np.asarray(bboxes, dtype=np.float32)
    return np.maximum(arr[:, 3] - arr[:, 1], 1.0)


def logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def diagnostics_payload(
    result: TemporalBiomechanicsResult,
    object_ids: list[int],
    frame_indices: np.ndarray,
    cfg: dromia_config.TemporalBiomechanicsConfig,
) -> dict[str, object]:
    runners = {}
    for obj_idx, obj_id in enumerate(object_ids):
        states = result.state_path[:, obj_idx]
        runners[str(obj_id)] = {
            name: frame_indices[states == state].astype(int).tolist()
            for state, name in enumerate(result.state_names)
            if state > 0
        }
        prior = result.priors[obj_idx]
        runners[str(obj_id)]["learned_priors"] = {
            "bone_log_mean": prior.bone_log_mean.astype(float).tolist(),
            "bone_log_sigma": prior.bone_log_sigma.astype(float).tolist(),
            "bend_direction": prior.bend_direction.astype(float).tolist(),
        }
    return {
        "algorithm": "temporal_biomechanics_v1",
        "state_names": list(result.state_names),
        "config": cfg.model_dump(mode="json"),
        "runners": runners,
    }
