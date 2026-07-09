import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import QWEN3_5_35B_A3B_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-2-npu-a3",
    nightly=True,
    disabled="accuracy testcase",
)
register_npu_ci(
    est_time=3600,
    suite="stage-b-test-2-npu-a3",
    nightly=True,
    disabled="accuracy testcase",
)

ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "GDN_ATTN_BACKEND_TRITON": "1",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "40",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
}

OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    2,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    10000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--mem-fraction-static",
    0.7,
    "--cuda-graph-bs",
    1,
    2,
    4,
    6,
    8,
    16,
    24,
    28,
    30,
    "--mm-attention-backend",
    "ascend_attn",
]


class TestNPUQwen_3_5_35B_A3B_aime(TestAscendAccuracyTestCaseBase):
    model = QWEN3_5_35B_A3B_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.833
    datasets = ["aime26"]
    generation_config = {
        "max_tokens": 60000,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }
    eval_batch_size = 30

    def test_aime2026(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
