import os
import platform

import pytest

from dromia import config
from dromia.models import cotracker, pmpose, store

pytestmark = pytest.mark.models


def require_model_smoke() -> None:
    if os.getenv("DROMIA_RUN_MODEL_SMOKE") != "1":
        pytest.skip("set DROMIA_RUN_MODEL_SMOKE=1 after installing the external models")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.fail("the DromIA model smoke test requires Apple Silicon")
    store.verify()


def test_sam_model_loads_from_verified_local_snapshot():
    require_model_smoke()
    from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
    from mlx_vlm.utils import load_model

    model_root = store.DOWNLOADS / "sam3.1-bf16"
    model = load_model(model_root)
    processor = Sam31Processor.from_pretrained(str(model_root))
    assert model is not None
    assert processor is not None


def test_pmpose_model_loads_on_mps():
    require_model_smoke()
    runner = pmpose.PMPoseRunner(config.PoseConfig())
    runner.load()
    assert runner._model is not None
    runner.close()


def test_cotracker_model_loads_on_mps():
    require_model_smoke()
    import torch

    assert torch.backends.mps.is_available()
    model = cotracker._load_model(torch, torch.device("mps"))
    assert model is not None
