"""One-command launcher for the local CVAT + native MPS bridge stack."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from dromia import config as dromia_config


def cvat_root() -> Path:
    return (
        Path(os.getenv("DROMIA_CVAT_ROOT", str(dromia_config.REPO_ROOT.parent / "dromia-cvat")))
        .expanduser()
        .resolve()
    )


def compose_command(*args: str) -> list[str]:
    root = cvat_root()
    return [
        "docker",
        "compose",
        "-f",
        str(root / "docker-compose.yml"),
        "-f",
        str(root / "docker-compose.dromia.yml"),
        *args,
    ]


def up() -> dict[str, object]:
    require_credentials()
    subprocess.run(compose_command("up", "-d", "--build"), cwd=cvat_root(), check=True)
    logs = dromia_config.REPO_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    pid_path = logs / "dromia-ui.pid"
    if pid_path.exists() and process_alive(int(pid_path.read_text())):
        pid: int | None = int(pid_path.read_text())
    elif (bridge := health()["bridge"]).get("status") == "ok":
        # The bridge may have been started manually or survived a stale PID file.
        # Only reuse it when it confirms that it inherited CVAT credentials. Older
        # or uncredentialed bridges can be healthy while every pose update fails.
        if bridge.get("cvat_credentials_configured") is not True:
            raise RuntimeError(
                "An existing DromIA bridge on port 8765 has no confirmed CVAT credentials. "
                "Stop that bridge, then run dromia-stack up again from this shell."
            )
        pid_path.unlink(missing_ok=True)
        pid = None
    else:
        log_handle = (logs / "dromia-ui.log").open("ab")
        process = subprocess.Popen(
            [sys.executable, "-m", "dromia.review.service"],
            cwd=dromia_config.REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid = process.pid
        pid_path.write_text(str(pid))
    return wait_for_health(pid)


def down() -> dict[str, object]:
    pid_path = dromia_config.REPO_ROOT / "logs" / "dromia-ui.pid"
    stopped_pid = None
    if pid_path.exists():
        pid = int(pid_path.read_text())
        if process_alive(pid):
            os.kill(pid, signal.SIGTERM)
            stopped_pid = pid
        pid_path.unlink(missing_ok=True)
    subprocess.run(compose_command("down"), cwd=cvat_root(), check=True)
    return {"status": "stopped", "bridge_pid": stopped_pid}


def health() -> dict[str, object]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/api/v1/health", timeout=2) as response:
            bridge = json.loads(response.read())
    except Exception as exc:
        bridge = {"status": "unavailable", "error": str(exc)}
    try:
        # CVAT's Traefik configuration routes requests by the ``localhost`` host.
        # Using 127.0.0.1 reaches Traefik but does not match the CVAT router.
        with urllib.request.urlopen(
            "http://localhost:8080/api/server/about", timeout=2
        ) as response:
            cvat = {"status": "ok", "http_status": response.status}
    except Exception as exc:
        cvat = {"status": "unavailable", "error": str(exc)}
    return {"bridge": bridge, "cvat": cvat}


def wait_for_health(pid: int | None) -> dict[str, object]:
    for _ in range(30):
        state = health()
        if state["bridge"].get("status") == "ok" and state["cvat"].get("status") == "ok":
            return {"status": "running", "bridge_pid": pid, **state}
        time.sleep(1)
    raise RuntimeError(f"DromIA stack did not become healthy: {health()}")


def require_credentials() -> None:
    if not os.getenv("CVAT_USERNAME") or not os.getenv("CVAT_PASSWORD"):
        raise RuntimeError("Set CVAT_USERNAME and CVAT_PASSWORD before starting DromIA")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the local DromIA/CVAT stack")
    parser.add_argument("command", choices=("up", "down", "health"))
    args = parser.parse_args()
    result = up() if args.command == "up" else down() if args.command == "down" else health()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
