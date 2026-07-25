import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MOONLIGHT_16B_A3B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

MODEL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "72",
    "DEEPEP_HCCL_BUFFSIZE": "2000",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_USE_FIA_NZ": "1",
}

MODEL_OTHER_ARGS = [
    "--tp-size",
    16,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    "9000",
    "--cuda-graph-bs",
    4,
    8,
    12,
    14,
    "--mem-fraction-static",
    0.9,
    "--max-running-requests",
    224,
    "--context-length",
    8188,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    65536,
    "--max-prefill-tokens",
    3000,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--enable-dp-lm-head",
    "--dtype",
    "bfloat16",
    "--max-total-tokens",
    100000,
]


class TestNPUMoonlight16B_A3B_GSM8K(TestAscendAccuracyTestCaseBase):
    model = MOONLIGHT_16B_A3B_MODEL_PATH
    envs = MODEL_ENVS
    other_args = MODEL_OTHER_ARGS
    accuracy = 0.8
    datasets = ["gsm8k"]
    few_shot_num = 5
    generation_config = {"max_tokens": 65536, "temperature": 1.0}
    eval_batch_size = 64

    def test_gsm8k(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
