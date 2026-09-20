# Third-party notices

DromIA does not redistribute model weights. `dromia models install` downloads pinned files
from their original publishers after the user explicitly accepts the applicable terms.

| Component | Pin | Terms |
| --- | --- | --- |
| SAM 3.1 MLX | `mlx-community/sam3.1-bf16@a992e302ea9b0f03f41dfd93414a4fd0e818f65b` | Meta SAM License; converted MLX checkpoint |
| PMPose-B / BBoxMaskPose | `49a070a6f8396147323b1c4959474077dbe2ce8e` | GPL-3.0 |
| CoTracker3 | `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` | Primarily CC BY-NC 4.0 |
| CVAT | `cb55cc676f6f850da46a6c6fb6d71252d11c53db` plus DromIA changes | MIT |

- SAM: https://github.com/facebookresearch/sam3/blob/main/LICENSE
- PMPose: https://huggingface.co/vrg-prague/BBoxMaskPose
- CoTracker: https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md
- CVAT: https://github.com/cvat-ai/cvat/blob/develop/LICENSE

This inventory is not legal advice. Publication is blocked until the project owner completes
the license review and adds DromIA's own license.
