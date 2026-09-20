from __future__ import annotations

from contextlib import nullcontext

import pytest

from dromia import stack


class _Response:
    status = 200

    def read(self) -> bytes:
        return b'{"status": "ok", "cvat_credentials_configured": true}'

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_health_uses_cvat_traefik_host(monkeypatch) -> None:
    urls: list[str] = []

    def urlopen(url: str, timeout: int) -> _Response:
        urls.append(url)
        return _Response()

    monkeypatch.setattr(stack.urllib.request, "urlopen", urlopen)

    result = stack.health()

    assert result["bridge"] == {
        "status": "ok",
        "cvat_credentials_configured": True,
    }
    assert result["cvat"] == {"status": "ok", "http_status": 200}
    assert urls == [
        "http://127.0.0.1:8765/api/v1/health",
        "http://localhost:8080/api/server/about",
    ]


def test_up_reuses_healthy_unmanaged_bridge(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(stack, "require_credentials", lambda: None)
    monkeypatch.setattr(stack, "cvat_root", lambda: tmp_path)
    monkeypatch.setattr(stack.dromia_config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(stack.subprocess, "run", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(stack.subprocess, "Popen", lambda *_args, **_kwargs: _fail_duplicate())
    monkeypatch.setattr(
        stack,
        "health",
        lambda: {
            "bridge": {"status": "ok", "cvat_credentials_configured": True},
            "cvat": {"status": "ok"},
        },
    )
    monkeypatch.setattr(
        stack,
        "wait_for_health",
        lambda pid: {"status": "running", "bridge_pid": pid},
    )

    result = stack.up()

    assert result == {"status": "running", "bridge_pid": None}


def _fail_duplicate() -> None:
    raise AssertionError("a second DromIA bridge must not be started")


def test_up_rejects_uncredentialed_unmanaged_bridge(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(stack, "require_credentials", lambda: None)
    monkeypatch.setattr(stack, "cvat_root", lambda: tmp_path)
    monkeypatch.setattr(stack.dromia_config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(stack.subprocess, "run", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        stack,
        "health",
        lambda: {
            "bridge": {"status": "ok", "cvat_credentials_configured": False},
            "cvat": {"status": "ok"},
        },
    )

    with pytest.raises(RuntimeError, match="no confirmed CVAT credentials"):
        stack.up()
