# DromIA

DromIA is a local Apple Silicon research application for expert-guided 2D running gait
analysis in crowded sagittal video. It implements the system described in the accompanying
paper; it is not a diagnostic or training-prescription tool.

```bash
uv sync --extra review
uv run dromia models install --accept-licenses
uv run dromia run /path/to/video.mp4 --capture-fps 240
uv run dromia stack up
```

Model weights are downloaded from their original publishers, verified by SHA-256, and are
not stored in this repository. Their licenses include GPL-3.0, CC BY-NC 4.0, and the custom
SAM license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The DromIA source license is pending review. Do not redistribute this repository until a
`LICENSE` file has been added.

Paper: manuscript forthcoming; see [docs/method.md](docs/method.md) for the exact
paper-to-code correspondence and `CITATION.cff` for the software citation.
