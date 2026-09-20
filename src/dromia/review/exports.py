"""Create a privacy-scoped portable DromIA result archive."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

ALLOWED_FILES = (
    "manifest.json",
    "timebase.json",
    "config.json",
    "provenance.json",
)
ALLOWED_TREES = ("gait", "calibration", "videos")
DENIED_PARTS = {"input", "annotations", "evidence", "pose", "posterior", "tracking"}


def build_portable_export(run_dir: Path) -> dict[str, str]:
    run = run_dir.expanduser().resolve()
    destination = run / "exports" / f"{run.name}-results.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = [run / name for name in ALLOWED_FILES if (run / name).is_file()]
    for tree in ALLOWED_TREES:
        root = run / tree
        if root.is_dir():
            candidates.extend(path for path in root.rglob("*") if path.is_file())
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(candidates)):
            relative = path.relative_to(run)
            if any(part in DENIED_PARTS for part in relative.parts):
                continue
            if path == run / "manifest.json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["source_video"].pop("path", None)
                payload["artifacts"] = {
                    name: value
                    for name, value in payload.get("artifacts", {}).items()
                    if not any(part in DENIED_PARTS for part in Path(value).parts)
                }
                archive.writestr(relative.as_posix(), json.dumps(payload, indent=2))
            else:
                archive.write(path, relative.as_posix())
    return {"portable_zip": str(destination)}
