import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_NEXT_80B_A3B_1P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_ALGO": "level0:NA;level1:ring",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "20",
    "HCCL_BUFFSIZE": "2000",
}

QWEN3_NEXT_80B_A3B_1P_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    4,
    "--mem-fraction-static",
    0.685,
    "--max-running-requests",
    80,
    "--watchdog-timeout",
    3600,
    "--disable-radix-cache",
    "--cuda-graph-bs",
    80,
    "--max-prefill-tokens",
    28672,
    "--max-total-tokens",
    450560,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--chunked-prefill-size",
    -1,
]


class TestNPUQwen3Next80BA3B1PIn3k5Out1k5_50ms(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH
    other_args = QWEN3_NEXT_80B_A3B_1P_OTHER_ARGS
    envs = QWEN3_NEXT_80B_A3B_1P_ENVS
    dataset_name = "random"
    max_concurrency = 80
    num_prompts = 160
    input_len = 3584
    output_len = 1536
    random_range_ratio = 1
    tpot = 50
    aisbench_request_rate = 10
    output_token_throughput = 1410

    def test_npu_qwen3_next_80b_a3b_1p_in3k5_out1k5_50ms(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
