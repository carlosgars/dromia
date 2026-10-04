# Publication license review

Publication is intentionally blocked until counsel or the project owner resolves the combined
effect of the model and dependency terms and chooses DromIA's source license.

| Component | Use | Terms | Redistribution in DromIA |
| --- | --- | --- | --- |
| SAM 3.1 / MLX checkpoint | segmentation | Meta SAM license | no code or weights redistributed |
| BBoxMaskPose / PMPose-B | pose inference | GPL-3.0 | pinned source is downloaded and patched locally; checkpoint is not redistributed |
| CoTracker3 | bounded repair and correction propagation | CC BY-NC 4.0 for the principal implementation | pinned source and checkpoint are downloaded; no redistribution |
| CVAT | review interface | MIT | retained in `dromia-cvat` with upstream notices |
| PyTorch, torchvision | model runtime | BSD-style | dependency only |
| MLX-VLM | SAM runtime | MIT | dependency only |
| OpenCV | video I/O and rendering | Apache-2.0 | dependency only |
| NumPy, SciPy | numerical analysis | BSD-style | dependency only |
| Pydantic | schemas | MIT | dependency only |
| CVAT SDK | local review integration | MIT | optional dependency only |

Before a public push:

1. Confirm the exact license files at every pinned revision.
2. Decide whether invoking locally downloaded GPL and non-commercial model code is compatible
   with the intended DromIA distribution and use.
3. Choose DromIA's source license and validate the publication metadata in `CITATION.cff`.
4. Add `LICENSE`, retain `THIRD_PARTY_NOTICES.md`, and publish `dromia` and `dromia-cvat`
   together.
