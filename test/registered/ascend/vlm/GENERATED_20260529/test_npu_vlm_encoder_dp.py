import unittest
from types import SimpleNamespace

from sglang.test.ascend.test_ascend_utils import (
    QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import is_in_ci

register_npu_ci(est_time=500, suite="nightly-4-npu-a3", nightly=True)

MODELS = [
    SimpleNamespace(model=QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH, mmmu_accuracy=0.2),
    SimpleNamespace(model=QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH, mmmu_accuracy=0.2),
]


class TestNPUVLMEncoderDP(TestVLMModels):
    """Test VLM Encoder Data Parallelism on NPU.

    [Test Category] VLM Feature
    [Test Target] --mm-enable-dp-encoder, VLM Encoder DP
    """

    model = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.2
    other_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        "0.35",
        "--log-level",
        "info",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "1",
        "--mm-enable-dp-encoder",
    ]

    def test_vlm_mmmu_benchmark(self):
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [MODELS[0]]

        for model in models_to_test:
            self.model = model.model
            self.mmmu_accuracy = model.mmmu_accuracy
            self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
