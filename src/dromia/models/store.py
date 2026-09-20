"""Install and verify the pinned external model runtime."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "models" / "manifest.toml"
VENDOR = ROOT / ".vendor"
CHECKPOINTS = ROOT / "models" / "checkpoints"
DOWNLOADS = ROOT / "models" / "downloads"


def load_manifest() -> dict[str, Any]:
    with MANIFEST.open("rb") as handle:
        return tomllib.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def clone_pinned(name: str, repository: str, revision: str) -> Path:
    destination = VENDOR / name
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repository, str(destination)], check=True)
    subprocess.run(["git", "fetch", "origin", revision], cwd=destination, check=True)
    subprocess.run(["git", "checkout", "--detach", revision], cwd=destination, check=True)
    return destination


def install(*, accept_licenses: bool) -> dict[str, Any]:
    if not accept_licenses:
        raise ValueError(
            "Model installation requires --accept-licenses after reviewing THIRD_PARTY_NOTICES.md"
        )
    manifest = load_manifest()
    sam = manifest["sam"]
    sam_root = DOWNLOADS / "sam3.1-bf16"
    snapshot_download(
        repo_id=sam["model_id"],
        revision=sam["revision"],
        local_dir=sam_root,
    )
    pmpose = manifest["pmpose"]
    pmpose_root = clone_pinned("BBoxMaskPose", pmpose["repository"], pmpose["revision"])
    patch = ROOT / "patches" / "bboxmaskpose-macos.patch"
    check = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=pmpose_root,
        check=False,
        capture_output=True,
    )
    if check.returncode:
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=pmpose_root, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=pmpose_root, check=True)
    cotracker = manifest["cotracker"]
    clone_pinned("co-tracker", cotracker["repository"], cotracker["revision"])
    for item in (pmpose, cotracker):
        target = CHECKPOINTS / item["filename"]
        if not target.exists() or sha256(target) != item["sha256"]:
            download(item["url"], target)
    return verify()


def verify() -> dict[str, Any]:
    manifest = load_manifest()
    checks: dict[str, dict[str, Any]] = {}
    files = {
        "sam": DOWNLOADS / "sam3.1-bf16" / manifest["sam"]["filename"],
        "pmpose": CHECKPOINTS / manifest["pmpose"]["filename"],
        "cotracker": CHECKPOINTS / manifest["cotracker"]["filename"],
    }
    for name, path in files.items():
        expected = manifest[name]["sha256"]
        actual = sha256(path) if path.is_file() else None
        checks[name] = {
            "path": str(path.relative_to(ROOT)),
            "present": path.is_file(),
            "sha256": actual,
            "expected_sha256": expected,
            "valid": actual == expected,
        }
    for name, directory in {
        "pmpose_source": VENDOR / "BBoxMaskPose",
        "cotracker_source": VENDOR / "co-tracker",
    }.items():
        model_name = name.removesuffix("_source")
        expected = manifest[model_name]["revision"]
        result = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
            )
            if directory.is_dir()
            else None
        )
        actual = result.stdout.strip() if result is not None and result.returncode == 0 else None
        checks[name] = {
            "revision": actual,
            "expected_revision": expected,
            "valid": actual == expected,
        }
    payload = {
        "schema_version": 1,
        "valid": all(item["valid"] for item in checks.values()),
        "checks": checks,
    }
    if not payload["valid"]:
        raise RuntimeError(json.dumps(payload, indent=2))
    return payload
