from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from dromia import config, dto
from dromia.pipeline import runner


class FakePMPoseRunner:
    def __init__(self, _cfg: config.PoseConfig) -> None:
        pass

    def predict(self, _frame: np.ndarray, detection: dto.SamDetection):
        keypoints = np.stack(
            [np.asarray([18.0 + joint, 18.0 + 2 * joint]) for joint in range(17)]
        ).astype(np.float32)
        heatmaps = np.zeros((17, 64, 48), dtype=np.float32)
        heatmaps[:, 32, 24] = 1.0
        observation = dto.PoseObservation(
            frame_idx=detection.frame_idx,
            obj_id=detection.obj_id,
            keypoints_xy=keypoints,
            confidence=np.full(17, 0.9, dtype=np.float32),
            heatmaps=heatmaps,
            heatmap_metadata={
                "heatmap_space": "pmpose_affine_heatmap",
                "heatmap_values": "probability",
                "exact_affine_registration": True,
                "input_center_xy": [40.0, 40.0],
                "input_scale_xy": [80.0, 80.0],
                "input_size_xy": [192, 256],
            },
            crop_xyxy=np.asarray([0, 0, 80, 80], dtype=np.float32),
            bbox_xyxy=detection.bbox_xyxy,
        )
        return observation, None

    def close(self) -> None:
        pass


def test_mocked_pipeline_writes_portable_v1_manifest(tmp_path, monkeypatch) -> None:
    video = tmp_path / "tiny.mp4"
    write_video(video)
    cache = tmp_path / "sam"
    write_sam_cache(cache)
    monkeypatch.setattr(runner.sam31_cache, "ensure_sam_cache", lambda *_args, **_kwargs: cache)
    monkeypatch.setattr(runner, "create_pose_runner", FakePMPoseRunner)

    cfg = config.DromiaConfig(
        runs_dir=tmp_path / "runs",
        runners=config.RunnerFilterConfig(
            min_track_frames=1,
            min_track_fraction=0.0,
            min_displacement_fraction=0.0,
            min_mean_score=0.0,
        ),
        gait_analysis=config.GaitAnalysisConfig(draw_debug_video=False),
    )
    manifest = runner.run(video, cfg)
    run_dir = cfg.runs_dir / manifest.run_id
    saved = json.loads((run_dir / "manifest.json").read_text())

    assert saved["schema_version"] == 1
    assert saved["source_video"]["path"].startswith("input/")
    assert saved["accepted_runner_ids"] == [1]
    assert all(not Path(path).is_absolute() for path in saved["artifacts"].values())
    assert str(tmp_path) not in json.dumps(saved)
    assert (run_dir / saved["artifacts"]["pose_lineage_npz"]).is_file()
    assert (run_dir / saved["artifacts"]["first_pass_pose_npz"]).is_file()
    assert (run_dir / "provenance.json").is_file()


def write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (80, 80))
    for value in (0, 40, 80):
        writer.write(np.full((80, 80, 3), value, dtype=np.uint8))
    writer.release()


def write_sam_cache(root: Path) -> None:
    frames = root / "frames"
    frames.mkdir(parents=True)
    for frame_idx in range(3):
        mask = np.zeros((1, 80, 80), dtype=np.uint8)
        mask[:, 5:75, 5:75] = 1
        np.savez_compressed(
            frames / f"frame_{frame_idx:06d}.npz",
            frame_index=np.asarray(frame_idx),
            boxes=np.asarray([[5, 5, 75, 75]], dtype=np.float32),
            masks=mask,
            scores=np.asarray([0.99], dtype=np.float32),
            labels=np.asarray(["Runner running"]),
            track_ids=np.asarray([1], dtype=np.int32),
        )
