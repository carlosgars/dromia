"""Frozen configuration for the paper implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSE_VARIANT = "pmpose-b"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class SamConfig(FrozenModel):
    model_id: str = "mlx-community/sam3.1-bf16"
    revision: str = "a992e302ea9b0f03f41dfd93414a4fd0e818f65b"
    model_sha256: str = "a1b1c19dcc9bdd68438bcd74433fadc90740e73c37a1f386872672d134879c42"
    prompts: tuple[str, str] = ("Runner running", "Running shoe")
    threshold: float = 0.4
    resolution: int = 1008
    detect_every: int = 10
    memory_every: int = 3
    memory_mode: Literal["off"] = "off"


class RunnerFilterConfig(FrozenModel):
    min_track_frames: int = 5
    min_track_fraction: float = 0.45
    min_displacement_fraction: float = 0.01
    min_mean_score: float = 0.55


class PoseConfig(FrozenModel):
    variant: Literal["pmpose-b"] = "pmpose-b"
    pmpose_root: Path = REPO_ROOT / ".vendor" / "BBoxMaskPose"
    pmpose_checkpoint_path: Path = (
        REPO_ROOT / "models" / "checkpoints" / "PMPose-b-1.0.0-inference.pth"
    )
    device: str = "mps"
    crop_padding: float = 1.0
    blur_kernel: int = 81
    heatmap_temperature: float = 1.0
    exact_pmpose_heatmap_registration: bool = True


class ShoeTrackAssignmentConfig(FrozenModel):
    runner_min_overlap_margin: float = 0.02
    runner_min_supporting_frames: int = 3
    side_min_distance_margin_norm: float = 0.01
    side_min_supporting_frames: int = 3


class ShoeRefinementConfig(FrozenModel):
    compatibility_strength: float = 0.65
    sigma_bbox_height_fraction: float = 0.04
    min_peak_ratio_for_relocation: float = 5.0
    min_compatibility_ratio_for_relocation: float = 1.25


class CalibrationConfig(FrozenModel):
    temperature_scale: float = 1.0


class AutoPoseRepairConfig(FrozenModel):
    joint_ids: tuple[int, ...] = (5, 6, 11, 12, 13, 14, 15, 16)
    per_joint_trigger_ids: tuple[int, ...] = (13, 14, 15, 16)
    mixed_pose_core_joint_ids: tuple[int, ...] = (5, 6, 11, 12)
    mixed_pose_min_core_jumps: int = 2
    final_step_joint_ids: tuple[int, ...] = (13, 14, 15, 16)
    bbox_padding_height_fraction: float = 0.05
    severe_jump_threshold_norm: float = 0.08
    weak_jump_threshold_norm: float = 0.05
    final_max_step_norm: float = 0.08
    bilateral_collapse_max_separation_norm: float = 0.025
    bilateral_neighbor_min_separation_norm: float = 0.08
    bilateral_neighbor_window: int = 2
    stable_anchor_frames: int = 3
    bilateral_anchor_assignment_margin_norm: float = 0.03
    stable_quality_median_fraction: float = 0.75
    heatmap_alignment_threshold_px: float = 24.0
    boundary_frames: int = 8
    tracker_device: str = "auto"
    tracker_input_max_width: int = 640
    tracker_visibility_threshold: float = 0.5
    tracker_boundary_agreement_norm: float = 0.12
    tracker_endpoint_agreement_norm: float = 0.05
    tracker_invisible_patience: int = 2
    interpolation_max_gap: int = 24
    draw_comparison_video: bool = False
    draw_diagnostic_plots: bool = False


class CoTrackerConfig(FrozenModel):
    """Pinned local-window tracker settings for expert correction propagation."""

    model_name: str = "cotracker3_offline"
    repository: str = "facebookresearch/co-tracker:82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
    device: str = "mps"
    input_max_width: int = 640
    temporal_stride: int = 1
    backward_tracking: bool = True


class TemporalBiomechanicsConfig(FrozenModel):
    position_step_sigma_norm: float = 0.035
    angle_step_sigma_degrees: float = 22.0
    bend_step_sigma: float = 0.28
    bend_direction_sigma: float = 0.22
    limb_log_sigma_min: float = 0.08
    identity_switch_probability: float = 0.01
    initial_swap_probability: float = 0.01
    model_identity_probability: float = 0.72
    position_weight: float = 1.0
    angle_weight: float = 0.8
    bend_continuity_weight: float = 0.7
    bend_direction_weight: float = 1.0
    limb_length_weight: float = 0.0
    identity_weight: float = 0.35
    posterior_swap_threshold: float = 0.5
    duplicate_pair_distance_norm: float = 0.025
    raw_candidate_min_separation_norm: float = 0.08


class GaitAnalysisConfig(FrozenModel):
    fps_override: float | None = Field(default=None, gt=0.0)
    capture_fps_override: float | None = Field(default=None, gt=0.0)
    cadence_min_spm: float = 100.0
    cadence_max_spm: float = 260.0
    ground_percentile: float = 94.0
    ground_line_candidate_quantile: float = 94.0
    ground_line_bin_count: int = 12
    ground_line_min_candidates_per_bin: int = 8
    ground_line_min_r2: float = 0.50
    ground_line_min_total_rise_px: float = 3.0
    ground_line_max_abs_angle_deg: float = 2.0
    ground_line_frame_edge_margin_fraction: float = 0.05
    ground_line_min_peak_prominence_bbox_fraction: float = 0.04
    ground_line_peak_window_seconds: float = 0.05
    contact_refine_window_seconds: float = 0.075
    contact_refine_turning_prominence_bbox_fraction: float = 0.004
    contact_refine_landing_clearance_bbox_fraction: float = 0.015
    contact_refine_takeoff_lift_bbox_fraction: float = 0.01
    contact_refine_persistence_frames: int = 3
    ground_step_smoothing_seconds: float = 0.04
    ground_step_min_separation_seconds: float = 0.25
    ground_step_min_prominence_bbox_fraction: float = 0.08
    ground_step_max_interpolation_gap_seconds: float = 0.10
    contact_tolerance_bbox_fraction: float = 0.002
    min_contact_tolerance_px: float = 0.5
    min_contact_curve_fraction: float = 0.04
    extremity_contact_tolerance_bbox_fraction: float = 0.01
    min_extremity_contact_tolerance_px: float = 3.0
    min_extremity_contact_curve_fraction: float = 0.15
    ground_penetration_tolerance_bbox_fraction: float = 0.015
    min_ground_penetration_tolerance_px: float = 4.0
    max_contact_gap_seconds: float = 0.10
    max_contact_gap_frames: int = 1
    min_contact_seconds: float = 0.04
    min_contact_frames: int = 1
    min_event_contact_seconds: float = 0.08
    max_event_contact_seconds: float = 0.50
    max_inferred_contact_fraction: float = 0.85
    torso_straight_tolerance_deg: float = 5.0
    foot_axis_min_length_px: float = 12.0
    foot_axis_max_rmse_px: float = 2.5
    foot_axis_temporal_window_frames: int = 2
    foot_axis_min_temporal_samples: int = 3
    foot_axis_max_temporal_delta_deg: float = 10.0
    calibration_max_validation_error_m: float = 0.05
    asymmetry_review_threshold_percent: float = 10.0
    draw_debug_video: bool = True


class DromiaConfig(FrozenModel):
    repo_root: Path = REPO_ROOT
    cache_dir: Path = REPO_ROOT / "cache"
    runs_dir: Path = REPO_ROOT / "runs"
    random_seed: int = 0
    sam: SamConfig = Field(default_factory=SamConfig)
    runners: RunnerFilterConfig = Field(default_factory=RunnerFilterConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    shoe_track_assignment: ShoeTrackAssignmentConfig = Field(
        default_factory=ShoeTrackAssignmentConfig
    )
    shoe_refinement: ShoeRefinementConfig = Field(default_factory=ShoeRefinementConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    auto_pose_repair: AutoPoseRepairConfig = Field(default_factory=AutoPoseRepairConfig)
    cotracker: CoTrackerConfig = Field(default_factory=CoTrackerConfig)
    temporal_biomechanics: TemporalBiomechanicsConfig = Field(
        default_factory=TemporalBiomechanicsConfig
    )
    gait_analysis: GaitAnalysisConfig = Field(default_factory=GaitAnalysisConfig)
