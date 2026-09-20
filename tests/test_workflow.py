from __future__ import annotations

import json

from dromia.review import workflow


def test_metrics_context_fingerprint_includes_uncommitted_metric_code(
    tmp_path, monkeypatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text("{}")
    source = tmp_path / "metric.py"
    source.write_text("first")
    monkeypatch.setattr(workflow, "METRIC_CODE_PATHS", (source,))
    monkeypatch.setattr(workflow.dromia_config, "REPO_ROOT", tmp_path)
    first = workflow.metrics_context_fingerprint(run)

    source.write_text("second")
    second = workflow.metrics_context_fingerprint(run)

    assert first != second


def test_metrics_context_ignores_unrelated_files(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    gait = run / "gait"
    gait.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")
    monkeypatch.setattr(workflow, "METRIC_CODE_PATHS", ())

    first = workflow.metrics_context_fingerprint(run, runner_id=0)
    (gait / "notes.json").write_text('{"note": "not a metric input"}')
    second = workflow.metrics_context_fingerprint(run, runner_id=0)

    assert first == second


def test_metrics_context_uses_stable_timebase_fields(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text("{}")
    monkeypatch.setattr(workflow, "METRIC_CODE_PATHS", ())
    timebase = {
        "schema_version": 3,
        "video_sha256": "video-a",
        "source_fps": 240.0,
        "source_frame_count": 500,
        "frame_indices": [0, 1, 2],
        "timestamps_s": [0.0, 0.1, 0.2],
    }
    (run / "timebase.json").write_text(json.dumps(timebase))
    first = workflow.metrics_context_fingerprint(run, runner_id=0)

    timebase["frame_indices"] = [100, 101]
    timebase["timestamps_s"] = [1.0, 1.1]
    (run / "timebase.json").write_text(json.dumps(timebase))
    second = workflow.metrics_context_fingerprint(run, runner_id=0)
    assert first == second

    timebase["video_sha256"] = "video-b"
    (run / "timebase.json").write_text(json.dumps(timebase))
    third = workflow.metrics_context_fingerprint(run, runner_id=0)
    assert third != second
