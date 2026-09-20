import json

from dromia.review import cvat, service


def test_api_v1_routes_are_stable():
    task_id = 42
    assert service.API == "/api/v1"
    assert service.TASK_CONTEXT.fullmatch(f"{service.API}/tasks/{task_id}/context")
    assert service.TASK_UPDATE.fullmatch(f"{service.API}/tasks/{task_id}/pose/update")
    assert service.TASK_METRICS.fullmatch(f"{service.API}/tasks/{task_id}/metrics")
    assert service.TASK_METRICS.fullmatch(f"{service.API}/tasks/{task_id}/metrics/generate")
    assert service.TASK_VIDEO.fullmatch(f"{service.API}/tasks/{task_id}/videos/reviewed")
    assert service.TASK_EXPORT.fullmatch(f"{service.API}/tasks/{task_id}/export")


def test_task_lookup_uses_only_versioned_registry(tmp_path):
    run = tmp_path / "run-1"
    registry_path = run / "annotations" / "cvat" / "tasks.json"
    registry_path.parent.mkdir(parents=True)
    record = cvat.CvatTaskRecord(
        task_id=42,
        task_url="http://localhost:8080/tasks/42",
        host="http://localhost:8080",
        run_dir=str(run),
        input_video="input/video.mp4",
    )
    registry = cvat.CvatTaskRegistry(
        project_id=7,
        project_url="http://localhost:8080/projects/7",
        run_dir=str(run),
        full_runner=record,
        runners={},
    )
    registry_path.write_text(json.dumps(registry.model_dump(mode="json")))

    found = service.find_task_context(tmp_path, 42)
    assert found is not None
    assert found[0] == run
    assert found[1] == record
