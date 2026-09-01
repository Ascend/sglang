import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_OCR_2_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=400,
    suite="full-1-npu-a3",
    nightly=True,
)


class TestDeepSeekOCR2(TestNpuAccuracyTestCaseBase):
    model = DEEPSEEK_OCR_2_WEIGHTS_PATH
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
    accuracy = 0
    datasets = ["omni_doc_bench"]
    generation_config = {"max_tokens": 512}
    eval_batch_size = 64
    timeout = 1800

    def test_accuracy(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
