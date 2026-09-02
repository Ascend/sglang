import unittest

from sglang.test.ascend.test_ascend_utils import DEEPSEEK_OCR_2_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=400,
    suite="full-1-npu-a3",
    nightly=True,
)


class TestGLM4Models(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the deepseek-ai/DeepSeek-OCR-2 model on the MMMU dataset is no less than 0.

    [Test Category] Model
    [Test Target] deepseek-ai/DeepSeek-OCR-2
    """

    model = DEEPSEEK_OCR_2_WEIGHTS_PATH
    mmmu_accuracy = 0
    other_args = [
        "--enable-multimodal",
        "--mm-attention-backend",
        "ascend_attn",
        "--tp-size",
        1,
        "--mem-fraction-static",
        0.7,
        "--max-total-tokens",
        4096,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--page-size",
        128,
        "--skip-server-warmup",
    ]

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()

    def test_vlm_mmmu_benchmark1(self):
        self._run_vlm_mmmu_test()

    def test_vlm_mmmu_benchmark2(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
