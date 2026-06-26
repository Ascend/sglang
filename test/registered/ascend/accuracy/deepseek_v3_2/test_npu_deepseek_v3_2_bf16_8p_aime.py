import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-16-npu-a3",
    nightly=True,
    disabled="accuracy testcase",
)
register_npu_ci(
    est_time=3600,
    suite="stage-b-test-16-npu-a3",
    nightly=True,
    disabled="accuracy testcase",
)

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "1200",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

OTHER_ARGS = [
    "--tp-size",
    "16",
    "--ep-size",
    "16",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--watchdog-timeout",
    "9000",
    "--cuda-graph-bs",
    4,
    8,
    12,
    14,
    "--mem-fraction-static",
    "0.85",
    "--max-running-requests",
    32,
    "--context-length",
    8188,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    3000,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-dp-attention",
    "--dp-size",
    "4",
    "--enable-dp-lm-head",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--dtype",
    "bfloat16",
    "--tool-call-parser",
    "deepseekv32",
    "--reasoning-parser",
    "deepseek-v3",
]


class TestNPUDeepSeek_V3_2_AIME2025(TestAscendAccuracyTestCaseBase):

    model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.523
    datasets = ["aime25"]
    generation_config = {
        "max_tokens": 131072,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    eval_batch_size = 30

    def test_aime2025(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
