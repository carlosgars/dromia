from __future__ import annotations

import json
import zipfile
from pathlib import Path

from dromia.review import exports


def test_export_omits_private_video_annotations_and_absolute_paths(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "input").mkdir(parents=True)
    (run / "annotations").mkdir()
    (run / "gait").mkdir()
    (run / "input" / "private.mp4").write_bytes(b"private")
    (run / "annotations" / "pose.npz").write_bytes(b"private")
    (run / "gait" / "metrics.json").write_text('{"cadence": 180}')
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "source_video": {
                    "name": "private.mp4",
                    "sha256": "abc",
                    "path": "input/private.mp4",
                },
                "artifacts": {"metrics": "gait/metrics.json", "pose": "annotations/pose.npz"},
            }
        )
    )
    archive = Path(exports.build_portable_export(run)["portable_zip"])
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        manifest = handle.read("manifest.json").decode()
    assert names == {"manifest.json", "gait/metrics.json"}
    assert "path" not in manifest
    assert str(tmp_path) not in manifest
