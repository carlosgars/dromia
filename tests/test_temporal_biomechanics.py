from __future__ import annotations

import numpy as np

from dromia import config as dromia_config
from dromia.pipeline import temporal_biomechanics


def test_temporal_decoder_recovers_short_bilateral_swap() -> None:
    points = synthetic_sequence()
    original = points.copy()
    points[3:5, 0, [13, 14]] = points[3:5, 0, [14, 13]]
    points[3:5, 0, [15, 16]] = points[3:5, 0, [16, 15]]
    bboxes = np.tile(np.asarray([0, 0, 200, 200], dtype=np.float32), (8, 1, 1))

    result = temporal_biomechanics.decode_temporal_biomechanics(
        points,
        bboxes,
        dromia_config.TemporalBiomechanicsConfig(),
    )

    assert result.state_path[:, 0].tolist() == [0, 0, 0, 3, 3, 0, 0, 0]
    np.testing.assert_allclose(result.corrected_xy[:, 0, 13:17], original[:, 0, 13:17])
    assert np.all(result.state_probability[3:5, 0, 3] > 0.5)


def test_framewise_decoder_is_invariant_to_frame_order() -> None:
    points = synthetic_sequence()
    points[3:5, 0, [13, 14]] = points[3:5, 0, [14, 13]]
    points[3:5, 0, [15, 16]] = points[3:5, 0, [16, 15]]
    bboxes = np.tile(np.asarray([0, 0, 200, 200], dtype=np.float32), (8, 1, 1))
    cfg = dromia_config.TemporalBiomechanicsConfig()

    forward = temporal_biomechanics.decode_framewise_biomechanics(points, bboxes, cfg)
    reverse = temporal_biomechanics.decode_framewise_biomechanics(points[::-1], bboxes[::-1], cfg)

    np.testing.assert_array_equal(forward.state_path, reverse.state_path[::-1])
    np.testing.assert_allclose(
        forward.state_probability,
        reverse.state_probability[::-1],
    )
    np.testing.assert_allclose(
        np.sum(forward.state_probability, axis=-1),
        1.0,
    )


def test_wrong_way_knee_bend_has_lower_prior_probability() -> None:
    points = synthetic_sequence()
    scales = np.full(8, 200.0, dtype=np.float32)
    cfg = dromia_config.TemporalBiomechanicsConfig()
    priors = temporal_biomechanics.fit_runner_priors(points[:, 0], scales, cfg)
    valid = points[0, 0]
    wrong = valid.copy()
    wrong[15] = [100.0, 140.0]

    valid_score = temporal_biomechanics.bend_direction_log_prior(valid, priors, cfg)
    wrong_score = temporal_biomechanics.bend_direction_log_prior(wrong, priors, cfg)

    assert wrong_score < valid_score


def test_collapsed_posterior_pair_uses_separated_model_candidates() -> None:
    model = synthetic_sequence()
    posterior = model.copy()
    posterior[3, 0, 16] = posterior[3, 0, 15]
    bboxes = np.tile(np.asarray([0, 0, 200, 200], dtype=np.float32), (8, 1, 1))

    restored, flags = temporal_biomechanics.restore_collapsed_pairs_from_model_candidates(
        posterior,
        model,
        bboxes,
        dromia_config.TemporalBiomechanicsConfig(),
    )

    np.testing.assert_allclose(restored[3, 0, [15, 16]], model[3, 0, [15, 16]])
    assert flags[3, 0, 15] and flags[3, 0, 16]


def synthetic_sequence() -> np.ndarray:
    points = np.full((8, 1, 17, 2), np.nan, dtype=np.float32)
    for t in range(8):
        shift = float(t * 2)
        points[t, 0, 11] = [60 + shift, 50]
        points[t, 0, 13] = [68 + shift, 95]
        points[t, 0, 15] = [52 + shift, 140]
        points[t, 0, 12] = [140 + shift, 50]
        points[t, 0, 14] = [132 + shift, 95]
        points[t, 0, 16] = [148 + shift, 140]
    return points
