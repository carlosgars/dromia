"""Versioned local HTTP API used by the DromIA CVAT interface."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
import urllib.parse
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from dromia import config
from dromia.review import analysis_jobs, cvat, exports, reviewed_pose, workflow

API = "/api/v1"
ANALYSIS = re.compile(rf"^{API}/analyses/([a-f0-9]+)$")
ANALYSIS_START = re.compile(rf"^{API}/analyses/([a-f0-9]+)/start$")
TASK_CONTEXT = re.compile(rf"^{API}/tasks/(\d+)/context$")
TASK_UPDATE = re.compile(rf"^{API}/tasks/(\d+)/pose/update$")
TASK_METRICS = re.compile(rf"^{API}/tasks/(\d+)/metrics(?:/generate)?$")
TASK_VIDEO = re.compile(rf"^{API}/tasks/(\d+)/videos/(reviewed|comparison|gait)$")
TASK_EXPORT = re.compile(rf"^{API}/tasks/(\d+)/export$")
LOCKS: dict[int, threading.Lock] = {}


def find_task_context(runs_dir: Path, task_id: int) -> tuple[Path, cvat.CvatTaskRecord] | None:
    for path in sorted(runs_dir.glob("*/annotations/cvat/tasks.json"), reverse=True):
        try:
            registry = cvat.CvatTaskRegistry.model_validate_json(path.read_text())
            records = cvat.registry_records(registry)
        except (OSError, ValueError):
            continue
        for record in records:
            if record.task_id == task_id:
                return path.parents[2], record
    return None


class Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], runs_dir: Path) -> None:
        super().__init__(address, Handler)
        self.runs_dir = runs_dir.expanduser().resolve()
        self.manager = analysis_jobs.AnalysisManager(self.runs_dir)


class Handler(BaseHTTPRequestHandler):
    server: Server

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if path == f"{API}/health":
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "api_version": 1,
                    "cvat_credentials_configured": bool(
                        os.getenv("CVAT_USERNAME") and os.getenv("CVAT_PASSWORD")
                    ),
                },
            )
            return
        match = ANALYSIS.fullmatch(path)
        if match:
            return self._analysis(match.group(1))
        match = TASK_CONTEXT.fullmatch(path)
        if match:
            return self._context(int(match.group(1)))
        match = TASK_METRICS.fullmatch(path)
        if match:
            return self._metrics(int(match.group(1)))
        match = TASK_VIDEO.fullmatch(path)
        if match:
            return self._video(int(match.group(1)), match.group(2))
        match = TASK_EXPORT.fullmatch(path)
        if match:
            return self._export(int(match.group(1)))
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if path == f"{API}/analyses":
            return self._create_analysis()
        match = ANALYSIS_START.fullmatch(path)
        if match:
            try:
                capture = self.headers.get("X-Dromia-Capture-FPS")
                record = self.server.manager.start(
                    match.group(1), capture_fps=float(capture) if capture else None
                )
                return self._json(HTTPStatus.ACCEPTED, record.model_dump(mode="json"))
            except (FileNotFoundError, ValueError) as exc:
                return self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        match = TASK_UPDATE.fullmatch(path)
        if match:
            return self._update_pose(int(match.group(1)))
        match = TASK_METRICS.fullmatch(path)
        if match and path.endswith("/generate"):
            return self._generate_metrics(int(match.group(1)))
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _create_analysis(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        filename = Path(
            urllib.parse.unquote(self.headers.get("X-Dromia-Filename", "video.mp4"))
        ).name
        if length <= 0 or Path(filename).suffix.lower() not in {".mp4", ".mov", ".m4v", ".avi"}:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_video_upload"})
        if length + 64 * 1024 * 1024 > shutil.disk_usage(self.server.manager.upload_dir).free:
            return self._json(HTTPStatus.INSUFFICIENT_STORAGE, {"error": "insufficient_storage"})
        target = self.server.manager.upload_dir / f"{uuid.uuid4().hex}_{filename}"
        with target.open("wb") as handle:
            remaining = length
            while remaining:
                block = self.rfile.read(min(1024 * 1024, remaining))
                if not block:
                    target.unlink(missing_ok=True)
                    return self._json(HTTPStatus.BAD_REQUEST, {"error": "incomplete_upload"})
                handle.write(block)
                remaining -= len(block)
        video = cv2.VideoCapture(str(target))
        try:
            count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(video.get(cv2.CAP_PROP_FPS))
            valid = video.isOpened() and count > 0 and fps > 0 and video.read()[0]
        finally:
            video.release()
        if not valid:
            target.unlink(missing_ok=True)
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_video"})
        capture = self.headers.get("X-Dromia-Capture-FPS")
        record = self.server.manager.create(
            target,
            name=urllib.parse.unquote(self.headers.get("X-Dromia-Name", Path(filename).stem)),
            view_name="sagittal",
            source_fps=fps,
            source_frame_count=count,
            capture_fps=float(capture) if capture else None,
            enqueue=self.headers.get("X-Dromia-Auto-Start", "true").lower() != "false",
        )
        self._json(HTTPStatus.ACCEPTED, record.model_dump(mode="json"))

    def _analysis(self, analysis_id: str) -> None:
        try:
            record = self.server.manager.get(analysis_id)
            self._json(HTTPStatus.OK, record.model_dump(mode="json"))
        except FileNotFoundError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})

    def _task(self, task_id: int) -> tuple[Path, cvat.CvatTaskRecord] | None:
        result = find_task_context(self.server.runs_dir, task_id)
        if result is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown_task"})
        return result

    def _context(self, task_id: int) -> None:
        context = self._task(task_id)
        if context is None:
            return
        run, record = context
        state = workflow.invalidate_if_context_changed(
            run,
            workflow.load_workflow(
                run, task_id=task_id, scope=record.scope, runner_id=record.runner_id
            ),
        )
        self._json(HTTPStatus.OK, state.model_dump(mode="json"))

    def _update_pose(self, task_id: int) -> None:
        context = self._task(task_id)
        if context is None:
            return
        run, _record = context
        lock = LOCKS.setdefault(task_id, threading.Lock())
        if not lock.acquire(blocking=False):
            return self._json(HTTPStatus.CONFLICT, {"error": "task_update_in_progress"})
        try:
            result = reviewed_pose.sync_run(run, cvat.CvatConnection.from_environment(), task_id)
            self._json(HTTPStatus.OK, result.model_dump(mode="json"))
        except Exception as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        finally:
            lock.release()

    def _generate_metrics(self, task_id: int) -> None:
        context = self._task(task_id)
        if context is None:
            return
        try:
            result = reviewed_pose.generate_metrics_run(context[0], task_id)
            self._json(HTTPStatus.OK, result.model_dump(mode="json"))
        except Exception as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})

    def _metrics(self, task_id: int) -> None:
        context = self._task(task_id)
        if context is None:
            return
        run, record = context
        state = workflow.invalidate_if_context_changed(
            run,
            workflow.load_workflow(
                run, task_id=task_id, scope=record.scope, runner_id=record.runner_id
            ),
        )
        if not workflow.metrics_are_current(state, run):
            return self._json(
                HTTPStatus.CONFLICT,
                {"error": "metrics_stale", "workflow": state.model_dump(mode="json")},
            )
        suffix = "" if record.runner_id is None else f"_runner_{record.runner_id}"
        path = run / "gait" / f"gait_analysis{suffix}.json"
        self._file(path, "application/json")

    def _video(self, task_id: int, kind: str) -> None:
        context = self._task(task_id)
        if context is None:
            return
        run, record = context
        suffix = "" if record.runner_id is None else f"_runner_{record.runner_id}"
        stem = {
            "reviewed": "reviewed_pose",
            "comparison": "posterior_vs_reviewed",
            "gait": "gait_debug",
        }[kind]
        path = (run / "gait" if kind == "gait" else run / "videos") / f"{stem}{suffix}.mp4"
        self._file(path, "video/mp4")

    def _export(self, task_id: int) -> None:
        context = self._task(task_id)
        if context is None:
            return
        artifact = exports.build_portable_export(context[0])["portable_zip"]
        self._file(Path(artifact), "application/zip", attachment=True)

    def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str, *, attachment: bool = False) -> None:
        if not path.is_file():
            return self._json(HTTPStatus.NOT_FOUND, {"error": "artifact_not_generated"})
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(body)

    def _cors(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin in {"http://localhost:8080", "http://127.0.0.1:8080"}:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Dromia-Filename, X-Dromia-Capture-FPS, "
            "X-Dromia-Name, X-Dromia-Auto-Start",
        )

    def log_message(self, format: str, *args: object) -> None:
        print(f"DromIA API: {format % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the DromIA CVAT bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs-dir", type=Path, default=config.REPO_ROOT / "runs")
    args = parser.parse_args()
    server = Server((args.host, args.port), args.runs_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
