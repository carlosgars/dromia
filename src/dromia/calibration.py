"""Ground-plane calibration for metric contact and distance measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GroundCalibration(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    video_sha256: str
    image_points_xy: list[tuple[float, float]]
    world_points_xy_m: list[tuple[float, float]]
    longitudinal_m: float = Field(default=4.0, gt=0.0)
    transverse_m: float = Field(default=6.0, gt=0.0)
    travel_axis: str = "world_x"
    travel_direction: str = "left_to_right"
    homography: list[list[float]]
    fit_residual_m: float = Field(ge=0.0)
    validation_error_m: float | None = Field(default=None, ge=0.0)
    valid: bool = True
    invalid_reason: str | None = None

    @model_validator(mode="after")
    def validate_shapes(self) -> GroundCalibration:
        if len(self.image_points_xy) != 4 or len(self.world_points_xy_m) != 4:
            raise ValueError("Ground calibration requires exactly four image/world points")
        if np.asarray(self.homography).shape != (3, 3):
            raise ValueError("homography must be 3x3")
        return self

    def project(self, point_xy: tuple[float, float] | np.ndarray) -> tuple[float, float] | None:
        if not self.valid:
            return None
        point = np.asarray(point_xy, np.float64).reshape(1, 1, 2)
        mapped = cv2.perspectiveTransform(point, np.asarray(self.homography, np.float64))[0, 0]
        return (float(mapped[0]), float(mapped[1])) if np.isfinite(mapped).all() else None

    def fingerprint(self) -> str:
        body = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()


def fit_ground_calibration(
    *,
    video_sha256: str,
    image_points_xy: list[tuple[float, float]],
    longitudinal_m: float = 4.0,
    transverse_m: float = 6.0,
    validation_error_m: float | None = None,
    travel_direction: str = "left_to_right",
    max_validation_error_m: float = 0.05,
) -> GroundCalibration:
    image = np.asarray(image_points_xy, np.float32)
    if image.shape != (4, 2) or not np.isfinite(image).all():
        raise ValueError("image_points_xy must contain four finite points")
    world = np.asarray(
        [(0.0, 0.0), (longitudinal_m, 0.0), (longitudinal_m, transverse_m), (0.0, transverse_m)],
        np.float32,
    )
    homography = cv2.getPerspectiveTransform(image, world)
    projected = cv2.perspectiveTransform(image.reshape(1, 4, 2), homography)[0]
    residual = float(np.sqrt(np.mean(np.sum((projected - world) ** 2, axis=1))))
    determinant = float(np.linalg.det(homography))
    if travel_direction not in {"left_to_right", "right_to_left"}:
        raise ValueError("travel_direction must be left_to_right or right_to_left")
    valid = bool(np.isfinite(homography).all() and abs(determinant) > 1e-12)
    reason = None if valid else "degenerate_homography"
    if validation_error_m is None:
        valid = False
        reason = "independent_validation_distance_missing"
    elif validation_error_m > max_validation_error_m:
        valid = False
        reason = "validation_error_exceeds_0.05_m"
    return GroundCalibration(
        video_sha256=video_sha256,
        image_points_xy=[tuple(map(float, row)) for row in image],
        world_points_xy_m=[tuple(map(float, row)) for row in world],
        longitudinal_m=longitudinal_m,
        transverse_m=transverse_m,
        travel_direction=travel_direction,
        homography=homography.tolist(),
        fit_residual_m=residual,
        validation_error_m=validation_error_m,
        valid=valid,
        invalid_reason=reason,
    )


def load_ground_calibration(run_dir: Path) -> GroundCalibration | None:
    path = run_dir / "calibration" / "ground_calibration.json"
    return GroundCalibration.model_validate_json(path.read_text()) if path.exists() else None


def save_ground_calibration(run_dir: Path, calibration: GroundCalibration) -> Path:
    path = run_dir / "calibration" / "ground_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(calibration.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate an DromIA ground rectangle")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--points",
        type=float,
        nargs=8,
        metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
        required=True,
    )
    parser.add_argument("--longitudinal-m", type=float, default=4.0)
    parser.add_argument("--transverse-m", type=float, default=6.0)
    parser.add_argument("--validation-error-m", type=float)
    parser.add_argument(
        "--travel-direction", choices=("left_to_right", "right_to_left"), default="left_to_right"
    )
    args = parser.parse_args()
    run = args.run_dir.resolve()
    timebase = json.loads((run / "timebase.json").read_text())
    points = list(zip(args.points[::2], args.points[1::2], strict=True))
    calibration = fit_ground_calibration(
        video_sha256=timebase["video_sha256"],
        image_points_xy=points,
        longitudinal_m=args.longitudinal_m,
        transverse_m=args.transverse_m,
        validation_error_m=args.validation_error_m,
        travel_direction=args.travel_direction,
    )
    path = save_ground_calibration(run, calibration)
    print(json.dumps({"artifact": str(path), **calibration.model_dump(mode="json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
