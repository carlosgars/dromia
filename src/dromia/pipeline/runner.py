"""Dromia end-to-end pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from dromia import config as dromia_config
from dromia import dto as dromia_dto
from dromia import provenance as dromia_provenance
from dromia import timebase as dromia_timebase
from dromia.gait import analysis as postprocess_gait_analysis
from dromia.models import cache as sam31_cache
from dromia.models import crops as pose_crops
from dromia.models import pmpose as pmpose_pose
from dromia.pipeline import auto_pose_repair as postprocess_auto_pose_repair
from dromia.pipeline import shoes as shoe_assignment
from dromia.pipeline import temporal_biomechanics as postprocess_temporal_biomechanics
from dromia.probability import heatmaps as heatmap_probability
from dromia.probability import posterior as posterior_probability
from dromia.probability import priors as prior_probability
from dromia.probability import storage as heatmap_storage


@dataclass(slots=True)
class HeatmapDiagnostics:
    used_heatmap_count: int = 0
    missing_heatmap_count: int = 0
    alignment_errors_px: list[float] = field(default_factory=list)
    posterior_relocation_events: list[dict[str, object]] = field(default_factory=list)
    usable_heatmap_joint_count: int = 0
    neutral_heatmap_joint_count: int = 0
    neutral_heatmap_events: list[dict[str, object]] = field(default_factory=list)

    def record_used(self, errors: np.ndarray) -> None:
        self.used_heatmap_count += 1
        finite = np.asarray(errors, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        self.alignment_errors_px.extend(float(item) for item in finite)

    def record_joint_evidence(
        self,
        *,
        frame_idx: int,
        obj_id: int,
        usable: np.ndarray,
        reason: str,
        decoder_errors_px: np.ndarray | None = None,
        heatmap_present: bool = True,
    ) -> None:
        mask = np.asarray(usable, dtype=bool)
        if heatmap_present:
            self.used_heatmap_count += 1
        self.usable_heatmap_joint_count += int(mask.sum())
        self.neutral_heatmap_joint_count += int((~mask).sum())
        if decoder_errors_px is not None:
            finite = np.asarray(decoder_errors_px, dtype=np.float32)
            self.alignment_errors_px.extend(float(item) for item in finite[np.isfinite(finite)])
        if np.any(~mask):
            self.neutral_heatmap_events.append(
                {
                    "frame_idx": int(frame_idx),
                    "obj_id": int(obj_id),
                    "reason": reason,
                    "joint_ids": [int(item) for item in np.flatnonzero(~mask)],
                }
            )

    def record_posterior_relocations(
        self,
        *,
        frame_idx: int,
        obj_id: int,
        joint_ids: np.ndarray,
        peak_ratios: np.ndarray,
    ) -> None:
        self.posterior_relocation_events.append(
            {
                "frame_idx": int(frame_idx),
                "obj_id": int(obj_id),
                "joints": [
                    {"joint_id": int(joint_id), "peak_ratio": float(peak_ratios[joint_id])}
                    for joint_id in joint_ids
                ],
            }
        )

    def to_dict(self) -> dict[str, object]:
        errors = np.asarray(self.alignment_errors_px, dtype=np.float32)
        return {
            "diagnostics_schema_version": 2,
            "used_heatmap_count": self.used_heatmap_count,
            "missing_heatmap_count": self.missing_heatmap_count,
            "alignment_error_mean_px": float(np.mean(errors)) if errors.size else None,
            "alignment_error_max_px": float(np.max(errors)) if errors.size else None,
            "usable_heatmap_joint_count": self.usable_heatmap_joint_count,
            "neutral_heatmap_joint_count": self.neutral_heatmap_joint_count,
            "neutral_heatmap_events": self.neutral_heatmap_events,
            "posterior_relocation_count": sum(
                len(event["joints"]) for event in self.posterior_relocation_events
            ),
            "posterior_relocation_events": self.posterior_relocation_events,
        }


@dataclass(slots=True)
class PoseStageResult:
    raw_xy: np.ndarray
    raw_conf: np.ndarray
    identity_corrected_xy: np.ndarray
    identity_result: postprocess_temporal_biomechanics.TemporalBiomechanicsResult
    shoe_candidate_xy: np.ndarray
    shoe_refinement_accepted: np.ndarray
    posterior_xy: np.ndarray
    posterior_peak: np.ndarray
    posterior_entropy: np.ndarray
    bboxes_xyxy: np.ndarray
    heatmap_alignment_error_px: np.ndarray
    heatmap_confidence: np.ndarray
    presence_probability: np.ndarray
    visibility_probability: np.ndarray
    normalized_localization_error: np.ndarray
    native_outputs_npz: Path | None
    shoe_assignments: list[dromia_dto.ShoeAssignment]
    shoe_assignment_artifacts: dict[str, str]


def run(
    video_path: Path,
    cfg: dromia_config.DromiaConfig | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
    resume_run_dir: Path | None = None,
) -> dromia_dto.RunManifestV1:
    config = cfg or dromia_config.DromiaConfig()
    set_random_seeds(config.random_seed)
    stage_started = time.perf_counter()
    current_stage = "initialization"
    stage_durations: dict[str, float] = {}

    def progress(stage: str, fraction: float) -> None:
        nonlocal stage_started, current_stage
        now = time.perf_counter()
        stage_durations[current_stage] = stage_durations.get(current_stage, 0.0) + (
            now - stage_started
        )
        current_stage = stage
        stage_started = now
        report_progress(progress_callback, stage, fraction)

    video = video_path.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video does not exist: {video}")
    if resume_run_dir is None:
        run_dir = make_run_dir(video, config.runs_dir, pose_model=config.pose.variant)
    else:
        run_dir = resume_run_dir.expanduser().resolve()
        runs_root = config.runs_dir.expanduser().resolve()
        if run_dir != runs_root and runs_root not in run_dir.parents:
            raise ValueError("Resume directory must be inside the configured runs directory")
        run_dir.mkdir(parents=True, exist_ok=True)
    make_output_dirs(run_dir)
    input_dir = run_dir / "input"
    input_dir.mkdir(exist_ok=True)
    run_video = input_dir / video.name
    if not run_video.exists():
        shutil.copy2(video, run_video)

    progress("segmentation", 0.05)
    sam_cache = sam31_cache.ensure_sam_cache(
        video,
        cfg=config,
        run_dir=run_dir,
    )
    frame_paths = sam31_cache.frame_paths(sam_cache)
    frame_width = read_width(video)
    frame_height = read_height(video)
    lazy_masks = len(frame_paths) > 2_000
    sam_frames = [
        sam31_cache.load_frame(
            path,
            lazy_masks=lazy_masks,
            mask_shape=(frame_height, frame_width),
        )
        for path in frame_paths
    ]
    frame_indices = [frame.frame_idx for frame in sam_frames]
    timing = dromia_timebase.inspect_video(
        video,
        frame_indices,
        fps_override=config.gait_analysis.fps_override,
        capture_fps_override=config.gait_analysis.capture_fps_override,
    )
    fps = timing.source_fps
    portable_timing = timing.model_copy(
        update={"video_path": run_video.relative_to(run_dir).as_posix()}
    )
    timebase_path = run_dir / "timebase.json"
    timebase_path.write_text(
        json.dumps(portable_timing.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    accepted_ids, decisions = select_real_runner_ids(
        sam_frames,
        frame_width=frame_width,
        cfg=config.runners,
        explicitly_included_ids=sam31_cache.included_runner_ids(sam_cache),
    )
    object_ids = sorted(accepted_ids)

    progress("pose", 0.25)
    diagnostics = HeatmapDiagnostics()
    stage_result = run_pose_stages(
        video,
        run_dir=run_dir,
        sam_frames=sam_frames,
        object_ids=object_ids,
        source_cache=str(sam_cache.resolve()),
        cfg=config,
        diagnostics=diagnostics,
    )
    quality = posterior_quality(stage_result.posterior_peak, stage_result.posterior_entropy)

    pose_npz = run_dir / "pose" / "pose_observations.npz"
    np.savez_compressed(
        pose_npz,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        object_ids=np.asarray(object_ids, dtype=np.int32),
        raw_keypoints_xy=stage_result.raw_xy,
        raw_confidence=stage_result.raw_conf,
        heatmap_confidence=stage_result.heatmap_confidence,
        presence_probability=stage_result.presence_probability,
        visibility_probability=stage_result.visibility_probability,
        normalized_localization_error=stage_result.normalized_localization_error,
        bboxes_xyxy=stage_result.bboxes_xyxy,
        timestamps_s=np.asarray(timing.timestamps_s, dtype=np.float64),
        media_timestamps_s=np.asarray(timing.media_timestamps_s, dtype=np.float64),
    )
    posterior_npz = run_dir / "posterior" / "posterior_pose.npz"
    np.savez_compressed(
        posterior_npz,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        object_ids=np.asarray(object_ids, dtype=np.int32),
        posterior_keypoints_xy=stage_result.posterior_xy,
        posterior_peak_probability=stage_result.posterior_peak,
        posterior_entropy=stage_result.posterior_entropy,
    )
    repair_result = None
    pre_hmm_repair_result = None
    post_hmm_repair_result = None
    repair_tracker_warnings: list[str] = []
    if config.auto_pose_repair.enabled:
        # Identity decoding must see the untouched PMPose channels. Repairing a
        # channel before its anatomical side is known can erase the evidence the
        # HMM needs to perform a bilateral swap.
        pre_hmm_repair_result = postprocess_auto_pose_repair.unchanged_repair_result(
            stage_result.posterior_xy
        )
    # Identity has already been decoded from untouched PMPose channels before
    # track-level shoe assignment and optional ankle refinement.
    temporal_result = stage_result.identity_result
    temporal_xy = stage_result.posterior_xy.copy()
    temporal_restored_candidates = np.zeros(temporal_xy.shape[:3], dtype=bool)
    hmm_xy = temporal_xy.copy()
    if config.auto_pose_repair.enabled:
        post_started = time.perf_counter()
        # Registration maps were relabelled with the identity state before
        # their quality was calculated, so applying the state again would undo it.
        hmm_quality = quality
        interval_flags, interval_reasons = postprocess_auto_pose_repair.detect_unstable_intervals(
            hmm_xy,
            stage_result.bboxes_xyxy,
            hmm_quality,
            config.auto_pose_repair,
        )
        interval_flags, interval_reasons = (
            postprocess_auto_pose_repair.coalesce_incompatible_bilateral_intervals(
                hmm_xy,
                stage_result.bboxes_xyxy,
                interval_flags,
                interval_reasons,
                config.auto_pose_repair,
            )
        )
        post_probe = postprocess_auto_pose_repair.repair_pose(
            hmm_xy,
            stage_result.bboxes_xyxy,
            None,
            config.auto_pose_repair,
            allow_interpolation=False,
            detected_flags=interval_flags,
            detected_reasons=interval_reasons,
        )
        post_unresolved = post_probe.flagged & (post_probe.method == 0)
        post_tracker_candidates = None
        post_tracker_visibility = None
        if config.auto_pose_repair.tracker_enabled and np.any(post_unresolved):
            post_tracker_candidates, post_tracker_visibility, tracker_warnings = (
                postprocess_auto_pose_repair.run_bounded_cotracker(
                    video_path=video,
                    frame_indices=frame_indices,
                    object_ids=object_ids,
                    pose_xy=hmm_xy,
                    bboxes_xyxy=stage_result.bboxes_xyxy,
                    unresolved=post_unresolved,
                    segments=post_probe.segments,
                    cfg=config.auto_pose_repair,
                )
            )
            repair_tracker_warnings.extend(tracker_warnings)
        post_hmm_repair_result = postprocess_auto_pose_repair.repair_pose(
            hmm_xy,
            stage_result.bboxes_xyxy,
            None,
            config.auto_pose_repair,
            tracker_candidate_xy=post_tracker_candidates,
            tracker_visibility=post_tracker_visibility,
            detected_flags=post_probe.flagged,
            detected_reasons=post_probe.reasons,
        )
        post_hmm_repair_result.runtime_seconds = time.perf_counter() - post_started
        if config.auto_pose_repair.final_step_cap_enabled:
            post_hmm_repair_result = postprocess_auto_pose_repair.enforce_final_step_bound(
                post_hmm_repair_result,
                stage_result.bboxes_xyxy,
                config.auto_pose_repair,
            )
        temporal_xy = post_hmm_repair_result.keypoints_xy
        repair_result = postprocess_auto_pose_repair.merge_repair_results(
            pre_hmm_repair_result,
            post_hmm_repair_result,
            original_xy=stage_result.posterior_xy,
        )

    coordinate_source = build_coordinate_sources(
        identity_corrected_xy=stage_result.identity_corrected_xy,
        identity_state=temporal_result.state_path,
        shoe_refinement_accepted=stage_result.shoe_refinement_accepted,
        repaired_xy=temporal_xy,
        repair_result=repair_result,
    )
    lineage_npz = run_dir / "pose" / "pose_lineage.npz"
    np.savez_compressed(
        lineage_npz,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        object_ids=np.asarray(object_ids, dtype=np.int32),
        pmpose_decoded_xy=stage_result.raw_xy,
        identity_corrected_xy=stage_result.identity_corrected_xy,
        shoe_candidate_xy=stage_result.shoe_candidate_xy,
        shoe_refinement_accepted=stage_result.shoe_refinement_accepted,
        pre_repair_selected_xy=hmm_xy,
        repaired_xy=temporal_xy,
        coordinate_source=coordinate_source,
        identity_state=temporal_result.state_path,
    )

    temporal_npz = run_dir / "postprocess" / "temporal_biomechanics_pose.npz"
    np.savez_compressed(
        temporal_npz,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        object_ids=np.asarray(object_ids, dtype=np.int32),
        original_keypoints_xy=stage_result.posterior_xy,
        pmpose_decoded_xy=stage_result.raw_xy,
        identity_corrected_xy=stage_result.identity_corrected_xy,
        shoe_candidate_xy=stage_result.shoe_candidate_xy,
        shoe_refinement_accepted=stage_result.shoe_refinement_accepted,
        pre_repair_selected_xy=hmm_xy,
        repaired_xy=temporal_xy,
        coordinate_source=coordinate_source,
        temporal_keypoints_xy=hmm_xy,
        post_hmm_repaired_keypoints_xy=temporal_xy,
        identity_state=temporal_result.state_path,
        identity_state_probability=temporal_result.state_probability,
        state_names=np.asarray(temporal_result.state_names),
        position_log_prior=temporal_result.chosen_position_log_prior,
        angle_log_prior=temporal_result.chosen_angle_log_prior,
        bend_continuity_log_prior=temporal_result.chosen_bend_continuity_log_prior,
        bend_direction_log_prior=temporal_result.chosen_bend_direction_log_prior,
        limb_length_log_prior=temporal_result.chosen_limb_length_log_prior,
        restored_model_candidate=temporal_restored_candidates,
    )
    temporal_diagnostics = run_dir / "postprocess" / "temporal_biomechanics_diagnostics.json"
    temporal_diagnostics.write_text(
        json.dumps(
            postprocess_temporal_biomechanics.diagnostics_payload(
                temporal_result,
                object_ids,
                np.asarray(frame_indices, dtype=np.int32),
                config.temporal_biomechanics,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    first_pass_artifacts: dict[str, str] = {}
    first_pass_xy = temporal_xy
    if repair_result is not None:
        repair_result.keypoints_xy = first_pass_xy
        first_pass_artifacts = postprocess_auto_pose_repair.write_artifacts(
            repair_result,
            run_dir=run_dir,
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            object_ids=np.asarray(object_ids, dtype=np.int32),
            original_xy=stage_result.posterior_xy,
            video_path=video,
            sam_frames=sam_frames,
            fps=min(fps, 30.0),
            cfg=config.auto_pose_repair,
            raw_confidence=stage_result.raw_conf,
            alignment_error_px=stage_result.heatmap_alignment_error_px,
            bboxes_xyxy=stage_result.bboxes_xyxy,
            pre_hmm_result=pre_hmm_repair_result,
            post_hmm_result=post_hmm_repair_result,
            tracker_warnings=repair_tracker_warnings,
        )
    progress("gait_analysis", 0.72)
    if config.gait_analysis.enabled:
        progress("gait_analysis", 0.82)
        gait_payload = postprocess_gait_analysis.analyze_gait(
            frame_indices=frame_indices,
            object_ids=object_ids,
            pose_xy=first_pass_xy,
            bboxes_xyxy=stage_result.bboxes_xyxy,
            shoe_assignments=stage_result.shoe_assignments,
            fps=timing.source_fps,
            cfg=config.gait_analysis,
            source_pose="first_pass_pose",
            fps_is_assumed=False,
            timebase=timing,
        )
        gait_artifacts = postprocess_gait_analysis.write_gait_artifacts(
            gait_payload,
            run_dir=run_dir,
            video_path=video,
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            object_ids=np.asarray(object_ids, dtype=np.int32),
            pose_xy=first_pass_xy,
            shoe_assignments=stage_result.shoe_assignments,
            fps=min(fps, 30.0),
            draw_video=config.gait_analysis.draw_debug_video,
        )
    else:
        gait_artifacts = {}

    config_payload = config.model_dump(mode="json")
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    config_fingerprint = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifacts = {
        "config_json": str(config_path.resolve()),
        "sam_evidence": str(sam_cache.resolve()),
        "pose_npz": str(pose_npz.resolve()),
        "pose_lineage_npz": str(lineage_npz.resolve()),
        "timebase_json": str(timebase_path.resolve()),
        "posterior_npz": str(posterior_npz.resolve()),
        "temporal_biomechanics_npz": str(temporal_npz.resolve()),
        "temporal_biomechanics_diagnostics_json": str(temporal_diagnostics.resolve()),
        **first_pass_artifacts,
        "heatmap_diagnostics_json": str(
            (run_dir / "posterior" / "heatmap_diagnostics.json").resolve()
        ),
        "shoe_assignments_json": str((run_dir / "shoes" / "shoe_assignments.json").resolve()),
        "shoe_assignments_csv": str((run_dir / "shoes" / "shoe_assignments.csv").resolve()),
        "shoe_assignments_npz": str((run_dir / "shoes" / "shoe_assignments.npz").resolve()),
        **stage_result.shoe_assignment_artifacts,
        **gait_artifacts,
        **(
            {"pmpose_native_outputs_npz": str(stage_result.native_outputs_npz.resolve())}
            if stage_result.native_outputs_npz is not None
            else {}
        ),
    }
    manifest = dromia_dto.RunManifestV1(
        run_id=run_dir.name,
        source_video=dromia_dto.SourceVideo(
            name=video.name,
            sha256=timing.video_sha256,
            path=run_video.relative_to(run_dir).as_posix(),
        ),
        frame_count=len(frame_indices),
        accepted_runner_ids=object_ids,
        runner_decisions=decisions,
        config_fingerprint=config_fingerprint,
        model_fingerprints={
            "sam": config.sam.model_sha256,
            "pmpose": dromia_provenance.checkpoint_sha256(config.pose.pmpose_checkpoint_path),
            "cotracker": dromia_provenance.COTRACKER_CHECKPOINT_SHA256,
        },
        timing=portable_timing.model_dump(mode="json"),
        artifacts=relative_artifacts(artifacts, run_dir),
    )
    write_heatmap_diagnostics(diagnostics, run_dir)
    write_manifest_and_readme(manifest, run_dir)
    progress("finalizing", 0.98)
    stage_durations[current_stage] = stage_durations.get(current_stage, 0.0) + (
        time.perf_counter() - stage_started
    )
    provenance_path = dromia_provenance.save_run_provenance(
        run_dir,
        dromia_provenance.build_run_provenance(
            cfg=config,
            video_sha256=timing.video_sha256,
            stage_durations_s=stage_durations,
        ),
    )
    manifest.stage_durations_s = stage_durations
    manifest.artifacts["provenance_json"] = provenance_path.relative_to(run_dir).as_posix()
    write_manifest_and_readme(manifest, run_dir)
    report_progress(progress_callback, "complete", 1.0)
    return manifest


def report_progress(
    callback: Callable[[str, float], None] | None, stage: str, fraction: float
) -> None:
    if callback is not None:
        callback(stage, fraction)


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def run_pose_stages(
    video: Path,
    *,
    run_dir: Path,
    sam_frames: list[dromia_dto.SamFrame],
    object_ids: list[int],
    source_cache: str,
    cfg: dromia_config.DromiaConfig,
    diagnostics: HeatmapDiagnostics,
) -> PoseStageResult:
    object_to_index = {obj_id: idx for idx, obj_id in enumerate(object_ids)}
    shape = (len(sam_frames), len(object_ids), 17)
    raw_xy = np.full((*shape, 2), np.nan, dtype=np.float32)
    raw_conf = np.zeros(shape, dtype=np.float32)
    heatmap_confidence = np.full(shape, np.nan, dtype=np.float32)
    presence_probability = np.full(shape, np.nan, dtype=np.float32)
    visibility_probability = np.full(shape, np.nan, dtype=np.float32)
    normalized_localization_error = np.full(shape, np.nan, dtype=np.float32)
    shoe_candidate_xy = np.full((*shape, 2), np.nan, dtype=np.float32)
    shoe_refinement_accepted = np.zeros(shape, dtype=bool)
    posterior_xy = np.full((*shape, 2), np.nan, dtype=np.float32)
    posterior_peak = np.zeros(shape, dtype=np.float32)
    posterior_entropy = np.zeros(shape, dtype=np.float32)
    alignment_error = np.full(shape, np.nan, dtype=np.float32)
    bboxes = np.full((len(sam_frames), len(object_ids), 4), np.nan, dtype=np.float32)
    pose_runner = create_pose_runner(cfg.pose)
    posterior_dir = run_dir / "posterior" / "posteriors"
    posterior_dir.mkdir(parents=True, exist_ok=True)
    observations: dict[tuple[int, int], dromia_dto.PoseObservation] = {}

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    try:
        for t, sam_frame in enumerate(sam_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, sam_frame.frame_idx)
            ok, frame = capture.read()
            if not ok:
                continue
            runners = [r for r in sam_frame.runners if r.obj_id in object_to_index]
            for runner in runners:
                obj_idx = object_to_index[runner.obj_id]
                observation, _crop = pose_runner.predict(frame, runner)
                observations[(sam_frame.frame_idx, runner.obj_id)] = observation
                raw_xy[t, obj_idx] = observation.keypoints_xy
                raw_conf[t, obj_idx] = observation.confidence
                copy_optional_joint_values(
                    heatmap_confidence[t, obj_idx], observation.heatmap_confidence
                )
                copy_optional_joint_values(
                    presence_probability[t, obj_idx], observation.presence_probability
                )
                copy_optional_joint_values(
                    visibility_probability[t, obj_idx], observation.visibility_probability
                )
                copy_optional_joint_values(
                    normalized_localization_error[t, obj_idx],
                    observation.normalized_localization_error,
                )
                bboxes[t, obj_idx] = observation.bbox_xyxy
    finally:
        capture.release()
        pose_runner.close()

    native_outputs_npz = write_native_pmpose_outputs(observations, run_dir)
    identity_result = postprocess_temporal_biomechanics.decode_temporal_biomechanics(
        raw_xy,
        bboxes,
        cfg.temporal_biomechanics,
    )
    if not cfg.temporal_biomechanics.enabled:
        normal_probability = np.zeros_like(identity_result.state_probability)
        normal_probability[..., 0] = 1.0
        identity_result = replace(
            identity_result,
            corrected_xy=raw_xy.copy(),
            state_path=np.zeros_like(identity_result.state_path),
            state_probability=normal_probability,
        )
    identity_xy = identity_result.corrected_xy
    pose_by_frame_runner = {
        (sam_frame.frame_idx, obj_id): identity_xy[t, object_to_index[obj_id]]
        for t, sam_frame in enumerate(sam_frames)
        for obj_id in object_ids
    }
    track_assignments = shoe_assignment.assign_shoe_tracks(
        frames=sam_frames,
        runner_ids=object_ids,
        pose_by_frame_runner=pose_by_frame_runner,
        cfg=cfg.shoe_track_assignment,
        source_pose_configuration="pmpose_temporal_identity",
        source_cache=source_cache,
        tracker_provenance=shoe_tracker_provenance(Path(source_cache)),
    )
    stable_assignments = track_assignments.downstream_assignments
    shoe_assignment_artifacts = shoe_assignment.write_track_assignment_artifacts(
        track_assignments,
        run_dir,
    )
    write_shoe_assignments(stable_assignments, run_dir)
    assignment_lookup = shoe_assignment.assignments_by_frame_runner_side(stable_assignments)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    try:
        for t, sam_frame in enumerate(sam_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, sam_frame.frame_idx)
            ok, frame = capture.read()
            if not ok:
                continue
            runners = [r for r in sam_frame.runners if r.obj_id in object_to_index]
            for runner in runners:
                key = (sam_frame.frame_idx, runner.obj_id)
                observation = observations.get(key)
                if observation is None:
                    continue
                obj_idx = object_to_index[runner.obj_id]
                crop = pose_crops.build_masked_crop(
                    frame,
                    runner.bbox_xyxy,
                    runner.mask,
                    padding=cfg.pose.crop_padding,
                    blur_kernel=cfg.pose.blur_kernel,
                )
                alignment_error[t, obj_idx] = heatmap_alignment_errors_for_observation(
                    observation,
                    grid_shape=crop.image_bgr.shape[:2],
                    crop=crop,
                    temperature=cfg.pose.heatmap_temperature * cfg.calibration.temperature_scale,
                )
                posterior = build_shoe_ankle_refinement(
                    observation,
                    crop,
                    runner,
                    assignment_lookup,
                    canonical_xy=identity_xy[t, obj_idx],
                    identity_state=int(identity_result.state_path[t, obj_idx]),
                    cfg=cfg,
                    diagnostics=diagnostics,
                )
                (
                    posterior_maps,
                    points_global,
                    candidates_global,
                    accepted,
                    peak,
                    entropy,
                    probability_maps,
                ) = posterior
                posterior_xy[t, obj_idx] = points_global
                shoe_candidate_xy[t, obj_idx] = candidates_global
                shoe_refinement_accepted[t, obj_idx] = accepted
                posterior_peak[t, obj_idx] = peak
                posterior_entropy[t, obj_idx] = entropy
                heatmap_storage.save_compact_posterior(
                    posterior_dir
                    / f"frame_{observation.frame_idx:06d}_runner_{observation.obj_id:04d}.npz",
                    posterior_maps,
                    crop_xyxy=observation.crop_xyxy,
                    bbox_xyxy=observation.bbox_xyxy,
                    metadata=observation.heatmap_metadata,
                )
    finally:
        capture.release()
    return PoseStageResult(
        raw_xy=raw_xy,
        raw_conf=raw_conf,
        identity_corrected_xy=identity_xy,
        identity_result=identity_result,
        shoe_candidate_xy=shoe_candidate_xy,
        shoe_refinement_accepted=shoe_refinement_accepted,
        posterior_xy=posterior_xy,
        posterior_peak=posterior_peak,
        posterior_entropy=posterior_entropy,
        bboxes_xyxy=bboxes,
        heatmap_alignment_error_px=alignment_error,
        heatmap_confidence=heatmap_confidence,
        presence_probability=presence_probability,
        visibility_probability=visibility_probability,
        normalized_localization_error=normalized_localization_error,
        native_outputs_npz=native_outputs_npz,
        shoe_assignments=stable_assignments,
        shoe_assignment_artifacts=shoe_assignment_artifacts,
    )


def build_shoe_ankle_refinement(
    observation: dromia_dto.PoseObservation,
    crop: pose_crops.MaskedCrop,
    runner: dromia_dto.SamDetection,
    assignments: dict[tuple[int, int, str], dromia_dto.ShoeAssignment],
    *,
    canonical_xy: np.ndarray,
    identity_state: int,
    cfg: dromia_config.DromiaConfig,
    diagnostics: HeatmapDiagnostics,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Register PMPose evidence and optionally select shoe-supported ankles only."""

    probability_maps = probability_maps_for_observation(
        observation,
        grid_shape=crop.image_bgr.shape[:2],
        crop=crop,
        temperature=cfg.pose.heatmap_temperature * cfg.calibration.temperature_scale,
        diagnostics=diagnostics,
    )
    # The heatmap channels must undergo the same identity relabelling as the
    # canonical coordinates before an anatomical shoe side can condition them.
    anatomical_maps = postprocess_temporal_biomechanics.apply_state(
        probability_maps,
        identity_state,
    )
    posterior_maps = anatomical_maps.copy()
    selected_global = np.asarray(canonical_xy, dtype=np.float32).copy()
    candidate_global = np.full_like(selected_global, np.nan)
    accepted = np.zeros(selected_global.shape[0], dtype=bool)
    exact_pmpose = uses_exact_pmpose_registration(observation)
    canonical_local = crop.points_global_to_local(selected_global)
    height = max(float(runner.bbox_xyxy[3] - runner.bbox_xyxy[1]), 1.0)
    for joint_id, side in ((15, "left"), (16, "right")):
        assignment = assignments.get((observation.frame_idx, observation.obj_id, side))
        likelihood = anatomical_maps[joint_id]
        if (
            not exact_pmpose
            or observation.heatmaps is None
            or metadata_pmpose_affine(observation) is None
            or assignment is None
            or assignment.mask is None
            or not np.isfinite(canonical_local[joint_id]).all()
            or float(np.ptp(likelihood)) <= 1e-12
        ):
            continue
        shoe_mask = crop_full_mask(assignment.mask, crop.crop_xyxy)
        if not np.any(shoe_mask):
            continue
        compatibility_core = prior_probability.mask_distance_prior(
            shoe_mask,
            outside_sigma=cfg.shoe_refinement.sigma_bbox_height_fraction * height,
        )
        strength = cfg.shoe_refinement.compatibility_strength
        compatibility = (1.0 - strength) + strength * compatibility_core
        candidate_map = heatmap_probability.normalize_maps((likelihood * compatibility)[None, ...])[
            0
        ]
        candidate_local, _candidate_peak, _candidate_entropy = posterior_probability.decode_maps(
            candidate_map[None, ...],
            method=cfg.posterior.decode_method,
        )
        candidate_global[joint_id] = crop.points_local_to_global(candidate_local)[0]
        raw_x, raw_y = np.rint(canonical_local[joint_id]).astype(np.int32)
        if not (0 <= raw_x < candidate_map.shape[1] and 0 <= raw_y < candidate_map.shape[0]):
            continue
        peak_ratio = float(
            np.max(candidate_map)
            / max(float(candidate_map[raw_y, raw_x]), posterior_probability.EPS)
        )
        candidate_x, candidate_y = np.rint(candidate_local[0]).astype(np.int32)
        compatibility_ratio = float(
            compatibility[candidate_y, candidate_x]
            / max(float(compatibility[raw_y, raw_x]), posterior_probability.EPS)
        )
        if (
            cfg.shoe_refinement.enabled
            and peak_ratio >= cfg.shoe_refinement.min_peak_ratio_for_relocation
            and compatibility_ratio >= cfg.shoe_refinement.min_compatibility_ratio_for_relocation
        ):
            selected_global[joint_id] = candidate_global[joint_id]
            posterior_maps[joint_id] = candidate_map
            accepted[joint_id] = True
            ratios = np.ones(selected_global.shape[0], dtype=np.float32)
            ratios[joint_id] = peak_ratio
            diagnostics.record_posterior_relocations(
                frame_idx=observation.frame_idx,
                obj_id=observation.obj_id,
                joint_ids=np.asarray([joint_id], dtype=np.int32),
                peak_ratios=ratios,
            )
    _decoded, peak, entropy = posterior_probability.decode_maps(
        posterior_maps,
        method=cfg.posterior.decode_method,
    )
    return (
        posterior_maps,
        selected_global,
        candidate_global,
        accepted,
        peak,
        entropy,
        anatomical_maps,
    )


def probability_maps_for_observation(
    observation: dromia_dto.PoseObservation,
    *,
    grid_shape: tuple[int, int],
    crop: pose_crops.MaskedCrop,
    temperature: float = 1.0,
    diagnostics: HeatmapDiagnostics | None = None,
) -> np.ndarray:
    joint_count = len(observation.keypoints_xy)
    if observation.heatmaps is None:
        if diagnostics is not None:
            diagnostics.missing_heatmap_count += 1
            diagnostics.record_joint_evidence(
                frame_idx=observation.frame_idx,
                obj_id=observation.obj_id,
                usable=np.zeros(joint_count, dtype=bool),
                reason="missing_heatmap",
                heatmap_present=False,
            )
        return uniform_maps(joint_count, grid_shape)
    affine = metadata_pmpose_affine(observation)
    if affine is None:
        if diagnostics is not None:
            diagnostics.record_joint_evidence(
                frame_idx=observation.frame_idx,
                obj_id=observation.obj_id,
                usable=np.zeros(joint_count, dtype=bool),
                reason="missing_affine_metadata",
            )
        return uniform_maps(joint_count, grid_shape)
    center, scale = affine
    probability_maps, peak_global, usable = (
        heatmap_probability.normalized_heatmaps_to_crop_probability_maps_exact(
            observation.heatmaps,
            crop_xyxy=crop.crop_xyxy,
            input_center_xy=center,
            input_scale_xy=scale,
            temperature=temperature,
        )
    )
    errors = np.linalg.norm(peak_global - observation.keypoints_xy, axis=1)
    if diagnostics is not None:
        diagnostics.record_joint_evidence(
            frame_idx=observation.frame_idx,
            obj_id=observation.obj_id,
            usable=usable,
            reason="peak_outside_crop_or_empty",
            decoder_errors_px=errors,
        )
    return probability_maps


def heatmap_alignment_errors_for_observation(
    observation: dromia_dto.PoseObservation,
    *,
    grid_shape: tuple[int, int],
    crop: pose_crops.MaskedCrop,
    temperature: float = 1.0,
) -> np.ndarray:
    """Measure each decoded coordinate against its own heatmap maximum."""

    if observation.heatmaps is None:
        return np.full(len(observation.keypoints_xy), np.nan, dtype=np.float32)
    affine = metadata_pmpose_affine(observation)
    if affine is None:
        return np.full(len(observation.keypoints_xy), np.nan, dtype=np.float32)
    center, scale = affine
    decoded = heatmap_probability.native_heatmap_peaks_to_global(
        observation.heatmaps,
        input_center_xy=center,
        input_scale_xy=scale,
    )
    return np.linalg.norm(decoded - observation.keypoints_xy, axis=1).astype(np.float32)


def create_pose_runner(cfg: dromia_config.PoseConfig):
    return pmpose_pose.PMPoseRunner(cfg)


def uniform_maps(joint_count: int, grid_shape: tuple[int, int]) -> np.ndarray:
    height, width = grid_shape
    return np.full(
        (joint_count, height, width),
        1.0 / max(height * width, 1),
        dtype=np.float32,
    )


def select_real_runner_ids(
    frames: list[dromia_dto.SamFrame],
    *,
    frame_width: int,
    cfg: dromia_config.RunnerFilterConfig,
    explicitly_included_ids: set[int] | None = None,
) -> tuple[set[int], list[dromia_dto.RunnerTrackDecision]]:
    included = explicitly_included_ids or set()
    tracks: dict[int, list[dromia_dto.SamDetection]] = {}
    for frame in frames:
        for runner in frame.runners:
            tracks.setdefault(runner.obj_id, []).append(runner)
    accepted: set[int] = set()
    decisions: list[dromia_dto.RunnerTrackDecision] = []
    for obj_id, detections in sorted(tracks.items()):
        ordered = sorted(detections, key=lambda item: item.frame_idx)
        centers = np.asarray([mask_or_bbox_center_x(item) for item in ordered], dtype=np.float32)
        mean_score = float(np.mean([item.score for item in ordered]))
        displacement_px = float(centers[-1] - centers[0]) if len(centers) else 0.0
        displacement_fraction = abs(displacement_px) / max(float(frame_width), 1.0)
        track_fraction = len(ordered) / max(len(frames), 1)
        reasons: list[str] = []
        if len(ordered) < cfg.min_track_frames:
            reasons.append("short_track")
        # A remapped batch cache records the runner IDs deliberately retained
        # during clip construction. Their persistence is relative to their own
        # passage, not to the concatenated pack duration.
        if track_fraction < cfg.min_track_fraction and obj_id not in included:
            reasons.append("low_persistence")
        if displacement_fraction < cfg.min_displacement_fraction:
            reasons.append("low_motion")
        if mean_score < cfg.min_mean_score:
            reasons.append("low_score")
        if reasons:
            accepted_track = False
        else:
            accepted_track = True
            accepted.add(obj_id)
        decisions.append(
            dromia_dto.RunnerTrackDecision(
                obj_id=obj_id,
                accepted=accepted_track,
                reason="accepted" if accepted_track else ",".join(reasons),
                frame_count=len(ordered),
                first_frame=int(ordered[0].frame_idx),
                last_frame=int(ordered[-1].frame_idx),
                mean_score=mean_score,
                displacement_px=displacement_px,
                displacement_fraction=displacement_fraction,
                track_fraction=track_fraction,
            )
        )
    return accepted, decisions


def copy_optional_joint_values(target: np.ndarray, values: np.ndarray | None) -> None:
    if values is None:
        return
    source = np.asarray(values, dtype=np.float32).reshape(-1)
    target[: min(len(target), len(source))] = source[: len(target)]


def build_coordinate_sources(
    *,
    identity_corrected_xy: np.ndarray,
    identity_state: np.ndarray,
    shoe_refinement_accepted: np.ndarray,
    repaired_xy: np.ndarray,
    repair_result: postprocess_auto_pose_repair.RepairResult | None,
) -> np.ndarray:
    """Build per-coordinate lineage without conflating identity and position."""

    identity = np.asarray(identity_corrected_xy, dtype=np.float32)
    source = np.full(identity.shape[:3], "unavailable", dtype="<U24")
    source[np.isfinite(identity).all(axis=-1)] = "pmpose"
    for state, (knee_swapped, ankle_swapped) in enumerate(
        postprocess_temporal_biomechanics.STATE_BITS
    ):
        selected = np.asarray(identity_state) == state
        if knee_swapped:
            for joint_id in (13, 14):
                valid = selected & np.isfinite(identity[..., joint_id, :]).all(axis=-1)
                source[..., joint_id][valid] = "identity_relabelled"
        if ankle_swapped:
            for joint_id in (15, 16):
                valid = selected & np.isfinite(identity[..., joint_id, :]).all(axis=-1)
                source[..., joint_id][valid] = "identity_relabelled"
    source[np.asarray(shoe_refinement_accepted, dtype=bool)] = "shoe_refined"
    if repair_result is not None:
        source[repair_result.method == postprocess_auto_pose_repair.METHOD_TRACKER] = (
            "cotracker_repaired"
        )
        source[repair_result.method == postprocess_auto_pose_repair.METHOD_INTERPOLATION] = (
            "interpolated"
        )
    source[~np.isfinite(np.asarray(repaired_xy)).all(axis=-1)] = "unavailable"
    return source


def shoe_tracker_provenance(cache_dir: Path) -> str:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return ""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = {
        key: payload.get(key)
        for key in (
            "format",
            "model",
            "model_revision",
            "detect_every",
            "memory_every",
            "memory_mode",
        )
    }
    return json.dumps(selected, sort_keys=True, separators=(",", ":"))


def write_native_pmpose_outputs(
    observations: dict[tuple[int, int], dromia_dto.PoseObservation],
    run_dir: Path,
) -> Path | None:
    """Persist PMPose's native distributions and scalar heads without resampling."""

    selected = [
        observation
        for _key, observation in sorted(observations.items())
        if observation.heatmaps is not None
        and observation.heatmap_metadata.get("heatmap_space") == "pmpose_affine_heatmap"
    ]
    if not selected:
        return None
    shapes = {np.asarray(item.heatmaps).shape for item in selected}
    if len(shapes) != 1:
        raise ValueError(f"PMPose native heatmaps must share one shape, got {sorted(shapes)}")

    joint_count = int(np.asarray(selected[0].heatmaps).shape[0])

    def scalar_rows(name: str) -> np.ndarray:
        rows = []
        for item in selected:
            value = getattr(item, name)
            if value is None:
                rows.append(np.full(joint_count, np.nan, dtype=np.float32))
            else:
                rows.append(np.asarray(value, dtype=np.float32).reshape(joint_count))
        return np.stack(rows)

    path = run_dir / "pose" / "pmpose_native_outputs.npz"
    np.savez_compressed(
        path,
        frame_indices=np.asarray([item.frame_idx for item in selected], dtype=np.int32),
        object_ids=np.asarray([item.obj_id for item in selected], dtype=np.int32),
        decoded_keypoints_xy=np.stack([item.keypoints_xy for item in selected]).astype(np.float32),
        crop_xyxy=np.stack([item.crop_xyxy for item in selected]).astype(np.float32),
        bbox_xyxy=np.stack([item.bbox_xyxy for item in selected]).astype(np.float32),
        oks_confidence=np.stack([item.confidence for item in selected]).astype(np.float32),
        heatmap_confidence=scalar_rows("heatmap_confidence"),
        presence_probability=scalar_rows("presence_probability"),
        visibility_probability=scalar_rows("visibility_probability"),
        normalized_localization_error=scalar_rows("normalized_localization_error"),
        native_heatmaps=np.stack(
            [np.asarray(item.heatmaps, dtype=np.float16) for item in selected]
        ),
        heatmap_metadata=np.asarray(
            [json.dumps(item.heatmap_metadata) for item in selected], dtype="<U2048"
        ),
        storage_format=np.asarray("pmpose_native_float16_v1"),
    )
    return path


def crop_full_mask(mask: np.ndarray, crop_xyxy: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = np.asarray(crop_xyxy, dtype=np.int32)
    return (np.asarray(mask) > 0).astype(np.uint8)[y0:y1, x0:x1]


def posterior_quality(peak: np.ndarray, entropy: np.ndarray) -> np.ndarray:
    if peak.size == 0:
        return np.zeros_like(peak, dtype=np.float32)
    max_entropy = np.log(192 * 256)
    sharpness = np.clip(1.0 - entropy / max_entropy, 0.0, 1.0)
    peak_scale = np.maximum(np.nanmax(peak), 1e-6)
    return np.clip(0.5 * peak / peak_scale + 0.5 * sharpness, 0.0, 1.0).astype(np.float32)


def pose_video_confidence(points_xy: np.ndarray, quality: np.ndarray) -> np.ndarray:
    finite = np.isfinite(points_xy).all(axis=-1)
    confidence = np.asarray(quality, dtype=np.float32).copy()
    confidence[finite] = np.maximum(confidence[finite], 0.25)
    confidence[~finite] = 0.0
    return confidence


def mask_or_bbox_center_x(detection: dromia_dto.SamDetection) -> float:
    bbox = np.asarray(detection.bbox_xyxy, dtype=np.float32)
    if not isinstance(detection.mask, sam31_cache.LazyNpzMask):
        _ys, xs = np.where(np.asarray(detection.mask) > 0)
        if len(xs):
            return float(xs.mean())
    return float((bbox[0] + bbox[2]) * 0.5)


def read_width(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        capture.release()


def read_height(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()


def read_fps(video: Path) -> float:
    capture = cv2.VideoCapture(str(video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 30.0
    finally:
        capture.release()


def make_run_dir(video: Path, runs_dir: Path, *, pose_model: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = runs_dir / f"{stamp}_{video.stem}_{pose_model}"
    run_dir = base
    suffix = 2
    while run_dir.exists():
        run_dir = Path(f"{base}_{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_output_dirs(run_dir: Path) -> None:
    for name in ("pose", "posterior", "shoes", "postprocess", "tracking", "plots", "videos"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def relative_artifacts(artifacts: dict[str, str], run_dir: Path) -> dict[str, str]:
    """Return only run-local paths, expressed relative to the run root."""

    root = run_dir.resolve()
    portable: dict[str, str] = {}
    for name, raw_path in artifacts.items():
        try:
            portable[name] = Path(raw_path).resolve().relative_to(root).as_posix()
        except ValueError:
            # External caches are inputs, never part of the portable artifact contract.
            continue
    return portable


def write_manifest_and_readme(manifest: dromia_dto.RunManifestV1, run_dir: Path) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    lines = [
        "# DromIA Run",
        "",
        f"- input video: `{manifest.source_video.name}`",
        f"- frames: `{manifest.frame_count}`",
        f"- accepted runners: `{manifest.accepted_runner_ids}`",
        "",
        "## Artifacts",
        "",
    ]
    for name, path in manifest.artifacts.items():
        lines.append(f"- `{name}`: `{path}`")
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def metadata_pmpose_affine(
    observation: dromia_dto.PoseObservation,
) -> tuple[np.ndarray, np.ndarray] | None:
    metadata = observation.heatmap_metadata
    try:
        center = np.asarray(metadata["input_center_xy"], dtype=np.float32).reshape(2)
        scale = np.asarray(metadata["input_scale_xy"], dtype=np.float32).reshape(2)
        input_size = np.asarray(metadata["input_size_xy"], dtype=np.float32).reshape(2)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not np.isfinite(np.concatenate((center, scale, input_size))).all()
        or np.any(scale <= 0)
        or np.any(input_size <= 0)
    ):
        return None
    return center, scale


def uses_exact_pmpose_registration(observation: dromia_dto.PoseObservation) -> bool:
    metadata = observation.heatmap_metadata
    return metadata.get("heatmap_space") == "pmpose_affine_heatmap" and bool(
        metadata.get("exact_affine_registration", True)
    )


def write_heatmap_diagnostics(diagnostics: HeatmapDiagnostics, run_dir: Path) -> None:
    path = run_dir / "posterior" / "heatmap_diagnostics.json"
    path.write_text(json.dumps(diagnostics.to_dict(), indent=2), encoding="utf-8")


def write_shoe_assignments(assignments: list[dromia_dto.ShoeAssignment], run_dir: Path) -> None:
    shoes_dir = run_dir / "shoes"
    shoes_dir.mkdir(parents=True, exist_ok=True)
    rows = [shoe_assignment_row(item) for item in assignments]
    (shoes_dir / "shoe_assignments.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (shoes_dir / "shoe_assignments.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame_idx",
                "runner_id",
                "side",
                "shoe_obj_id",
                "score",
                "center_x",
                "center_y",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        shoes_dir / "shoe_assignments.npz",
        frame_idx=np.asarray([item["frame_idx"] for item in rows], dtype=np.int32),
        runner_id=np.asarray([item["runner_id"] for item in rows], dtype=np.int32),
        side=np.asarray([item["side"] for item in rows], dtype="<U8"),
        shoe_obj_id=np.asarray(
            [-1 if item["shoe_obj_id"] is None else item["shoe_obj_id"] for item in rows],
            dtype=np.int32,
        ),
        score=np.asarray([item["score"] for item in rows], dtype=np.float32),
        center_xy=np.asarray(
            [[item["center_x"], item["center_y"]] for item in rows], dtype=np.float32
        ),
    )


def shoe_assignment_row(item: dromia_dto.ShoeAssignment) -> dict[str, object]:
    center = shoe_assignment.mask_center(item.mask) if item.mask is not None else None
    return {
        "frame_idx": int(item.frame_idx),
        "runner_id": int(item.runner_id),
        "side": item.side,
        "shoe_obj_id": item.shoe_obj_id,
        "score": float(item.score),
        "center_x": float(center[0]) if center is not None else None,
        "center_y": float(center[1]) if center is not None else None,
    }
