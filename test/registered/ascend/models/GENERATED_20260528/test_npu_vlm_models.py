import random
import tempfile
import unittest
from types import SimpleNamespace

from sglang.test.ascend.test_ascend_utils import (
    MINICPM_V_2_6_WEIGHTS_PATH,
    QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.mmmu_vlm_kit import MMMUMultiModelTestBase
from sglang.test.test_utils import is_in_ci

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

MODELS = [
    SimpleNamespace(model=MINICPM_V_2_6_WEIGHTS_PATH, mmmu_accuracy=0.35),
    SimpleNamespace(model=QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH, mmmu_accuracy=0.35),
]


class TestNPUVLMModels(MMMUMultiModelTestBase):
    """Test VLM models against MMMU benchmark on NPU.

    [Test Category] VLM MMMU
    [Test Target] MiniCPM-V-2_6, Qwen2.5-VL-3B-Instruct, MMMU accuracy
    """

    def test_vlm_mmmu_benchmark(self):
        """Test VLM models against MMMU benchmark."""
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model in models_to_test:
            model_name = model.model.split("/")[-1]
            with tempfile.TemporaryDirectory(
                prefix=f"test_npu_vlm_mmmu_{model_name}_"
            ) as temp_dir:
                other_args = [
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                    "--mem-fraction-static",
                    "0.7",
                ]
                self._run_vlm_mmmu_test(model, temp_dir, other_args)


if __name__ == "__main__":
    unittest.main()
