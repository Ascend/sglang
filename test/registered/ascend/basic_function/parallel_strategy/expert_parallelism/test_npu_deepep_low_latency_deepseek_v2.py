import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=400,
    suite="full-8-npu-a3",
    nightly=True,
    disabled="performance testcase",
)

DEEPSEEK_V2_DEEPEP_COMMON_ENVS = {
    "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
    "HCCL_BUFFSIZE": "2048",
    "MOE_ENABLE_TOPK_NEG_ONE": "1",
}

_DEEPSEEK_V2_DEEPEP_COMMON_ARGS = [
    "--trust-remote-code",
    "--attention-backend", "ascend",
    "--tp-size", "8",
    "--moe-a2a-backend", "deepep",
    "--disable-cuda-graph",
    "--dp-size", "8",
    "--enable-dp-attention",
    "--chunked-prefill-size", "1024",
    "--mem-fraction-static", "0.7",
]

DEEPSEEK_V2_LOW_LATENCY_ARGS = (
    _DEEPSEEK_V2_DEEPEP_COMMON_ARGS + ["--deepep-mode", "low_latency"]
)


class TestDeepEpLowLatencyDeepSeekV2_MMLU(TestAscendAccuracyTestCaseBase):
    """DeepSeek V2 Lite W8A8 DeepEP low_latency mode MMLU accuracy baseline.

    [Test Category] Accuracy
    [Test Target] DeepEP low_latency mode on DeepSeek V2
    """

    model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
    other_args = DEEPSEEK_V2_LOW_LATENCY_ARGS
    envs = DEEPSEEK_V2_DEEPEP_COMMON_ENVS
    accuracy = 0.58
    datasets = ["mmlu"]

    def test_accuracy(self):
        self.run_accuracy()


class TestDeepEpLowLatencyDeepSeekV2_GSM8K(TestAscendAccuracyTestCaseBase):
    """DeepSeek V2 Lite W8A8 DeepEP low_latency mode GSM8K accuracy baseline.

    [Test Category] Accuracy
    [Test Target] DeepEP low_latency mode on DeepSeek V2
    """

    model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
    other_args = DEEPSEEK_V2_LOW_LATENCY_ARGS
    envs = DEEPSEEK_V2_DEEPEP_COMMON_ENVS
    accuracy = 0.34
    datasets = ["gsm8k"]

    def test_accuracy(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
