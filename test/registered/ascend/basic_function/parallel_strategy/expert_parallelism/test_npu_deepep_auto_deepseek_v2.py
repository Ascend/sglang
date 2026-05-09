import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
    "HCCL_BUFFSIZE": "4096",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--quantization",
    "modelslim",
    "--tp-size",
    "8",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--max-running-requests",
    128,
    "--disable-cuda-graph",
    "--dp-size",
    8,
    "--enable-dp-attention",
    "--chunked-prefill-size",
    1024,
    "--mem-fraction-static",
    0.68,
]


class TestDeepEpDeepseek(TestAscendAccuracyTestCaseBase):
    """Testcase: Verify the accuracy of DeepSeek V2 model on MMLU and GSM8K tasks with --deepep-mode auto on Ascend backend.

    [Test Category] Parameter
    [Test Target] --moe-a2a-backend; --deepep-mode
    """

    model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    api = "completion"
    max_concurrency = 100

    def test_gsm8k_accuracy(self):
        self.dataset_type = "gsm8k"
        self.dataset_name = "demo_gsm8k_gen_4_shot_cot_chat_prompt"
        self.accuracy = 0.34
        self.run_accuracy()

    def test_mmlu_accuracy(self):
        self.dataset_type = "mmlu"
        self.dataset_name = "mmlu_gen_5_shot_str"
        self.accuracy = 0.58
        self.output_len = 16
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
