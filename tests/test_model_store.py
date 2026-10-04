import hashlib
import json

import pytest

from dromia import cli
from dromia.models import store


def test_install_requires_explicit_license_acceptance():
    with pytest.raises(ValueError, match="accept-licenses"):
        store.install(accept_licenses=False)


def test_sha256_reads_file_contents(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"dromia")
    assert store.sha256(artifact) == hashlib.sha256(b"dromia").hexdigest()


def test_verify_reports_missing_or_altered_artifacts(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[sam]
filename = "sam.bin"
sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[pmpose]
filename = "pmpose.pth"
sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
revision = "pmpose-revision"

[cotracker]
filename = "cotracker.pth"
sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
revision = "cotracker-revision"
""".strip()
        + "\n"
    )
    downloads = tmp_path / "downloads"
    checkpoints = tmp_path / "checkpoints"
    (downloads / "sam3.1-bf16").mkdir(parents=True)
    checkpoints.mkdir()
    (downloads / "sam3.1-bf16" / "sam.bin").write_bytes(b"altered")
    monkeypatch.setattr(store, "ROOT", tmp_path)
    monkeypatch.setattr(store, "MANIFEST", manifest)
    monkeypatch.setattr(store, "VENDOR", tmp_path / "vendor")
    monkeypatch.setattr(store, "CHECKPOINTS", checkpoints)
    monkeypatch.setattr(store, "DOWNLOADS", downloads)

    with pytest.raises(RuntimeError) as raised:
        store.verify()

    report = json.loads(str(raised.value))
    assert report["valid"] is False
    assert report["checks"]["sam"]["present"] is True
    assert report["checks"]["sam"]["valid"] is False
    assert report["checks"]["pmpose"]["present"] is False
    assert report["checks"]["cotracker"]["present"] is False


def test_model_verification_failure_has_no_traceback(monkeypatch, capsys):
    def fail_verification():
        raise RuntimeError("missing")

    monkeypatch.setattr(store, "verify", fail_verification)

    assert cli.main(["models", "verify"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "missing\n"
