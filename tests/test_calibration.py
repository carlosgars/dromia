from __future__ import annotations

import numpy as np

from dromia import calibration as dromia_calibration


def test_ground_calibration_maps_four_by_six_rectangle() -> None:
    calibration = dromia_calibration.fit_ground_calibration(
        video_sha256="video",
        image_points_xy=[(100, 100), (500, 100), (500, 700), (100, 700)],
        validation_error_m=0.0,
    )

    assert calibration.valid is True
    assert np.allclose(calibration.project((300, 400)), (2.0, 3.0), atol=1e-6)


def test_validation_error_disables_metric_distance() -> None:
    calibration = dromia_calibration.fit_ground_calibration(
        video_sha256="video",
        image_points_xy=[(0, 0), (4, 0), (4, 6), (0, 6)],
        validation_error_m=0.051,
    )

    assert calibration.valid is False
    assert calibration.invalid_reason == "validation_error_exceeds_0.05_m"
    assert calibration.project((2, 3)) is None


def test_missing_independent_validation_disables_metric_distance() -> None:
    calibration = dromia_calibration.fit_ground_calibration(
        video_sha256="video",
        image_points_xy=[(0, 0), (4, 0), (4, 6), (0, 6)],
    )

    assert calibration.valid is False
    assert calibration.invalid_reason == "independent_validation_distance_missing"
