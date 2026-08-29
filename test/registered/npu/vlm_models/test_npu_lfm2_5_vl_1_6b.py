import unittest

from sglang.test.ascend.test_ascend_utils import (
    LFM2_5_VL_1_6B_WEIGHTS_PATH,
)
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=400,
    suite="full-2-npu-a3",
    nightly=True,
)


class TestLFM25VL16B(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the LiquidAI/LFM2.5-VL-1.6B model on the MMMU dataset is no less than 0.4056.

    [Test Category] Model
    [Test Target] LiquidAI/LFM2.5-VL-1.6B
    """

    model = LFM2_5_VL_1_6B_WEIGHTS_PATH
    mmmu_accuracy = 0
    other_args = [
        "--attention-backend",
        "ascend",
        "--tp-size",
        2,
        "-chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "16384",
        "--trust-remote-code",
        "--max-running-requests",
        32,
        "--mem-fraction-static",
        "0.6",
        "--mm-attention-backend",
        "ascend_attn",
        "--max-total-tokens",
        800000,
        "--dtype",
        "bfloat16",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--enable-multimodal",
        "--cuda-graph-max-bs",
        32,
        "--cuda-graph-bs",
        8,
        12,
        16,
        20,
        24,
        32,
    ]

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
