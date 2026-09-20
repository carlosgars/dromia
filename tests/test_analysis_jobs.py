from __future__ import annotations

import threading
from pathlib import Path

from dromia.review import analysis_jobs


def test_recovery_requeues_interrupted_work_without_changing_run(tmp_path: Path) -> None:
    manager = analysis_jobs.AnalysisManager(tmp_path / "runs", start_worker=False)
    run = manager.runs_dir / "partial"
    run.mkdir()
    record = analysis_jobs.AnalysisRecord(
        analysis_id="a" * 32,
        name="interrupted",
        view_name="sagittal",
        input_video=str(tmp_path / "video.mp4"),
        status="running",
        stage="pose",
        progress=0.4,
        run_dir=str(run),
    )
    manager.save(record)
    manager.recover_pending()
    recovered = manager.get(record.analysis_id)
    assert recovered.status == "queued"
    assert recovered.stage == "recovering_after_bridge_restart"
    assert recovered.run_dir == str(run)
    assert manager.pending.get_nowait() == record.analysis_id


def test_concurrent_start_enqueues_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        analysis_jobs.dromia_timebase,
        "quicktime_full_frame_rate_playback_intent",
        lambda _path: True,
    )
    manager = analysis_jobs.AnalysisManager(tmp_path / "runs", start_worker=False)
    record = manager.create(
        tmp_path / "video.mp4",
        name="ready",
        view_name="sagittal",
        source_fps=30,
        source_frame_count=10,
        enqueue=False,
    )
    outcomes: list[str] = []

    def start() -> None:
        try:
            manager.start(record.analysis_id)
            outcomes.append("started")
        except ValueError:
            outcomes.append("rejected")

    threads = [threading.Thread(target=start) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["rejected", "started"]
    assert manager.pending.qsize() == 1


def test_capture_fps_is_required_for_slow_motion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        analysis_jobs.dromia_timebase,
        "quicktime_full_frame_rate_playback_intent",
        lambda _path: False,
    )
    manager = analysis_jobs.AnalysisManager(tmp_path / "runs", start_worker=False)
    record = manager.create(
        tmp_path / "slow.mov",
        name="slow",
        view_name="sagittal",
        source_fps=30,
        source_frame_count=300,
        enqueue=False,
    )
    try:
        manager.start(record.analysis_id)
    except ValueError as exc:
        assert "capture FPS" in str(exc)
    else:
        raise AssertionError("slow-motion input was accepted without capture FPS")
    assert manager.start(record.analysis_id, capture_fps=240).capture_fps == 240
