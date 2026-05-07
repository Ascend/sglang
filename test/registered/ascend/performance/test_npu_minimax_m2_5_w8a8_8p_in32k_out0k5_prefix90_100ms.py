import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    MINIMAX_M2_5_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

MINIMAX_M2_5_32K_0K5_PREFIX_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_BUFFSIZE": "1500",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "4096",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
}

MINIMAX_M2_5_32K_0K5_PREFIX_OTHER_ARGS = [
    "--tp-size",
    16,
    "--enable-dp-attention",
    "--dp-size",
    8,
    "--ep-size",
    16,
    "--mem-fraction-static",
    0.6,
    "--prefill-delayer-max-delay-passes",
    200,
    "--enable-prefill-delayer",
    "--max-running-requests",
    24,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    32768,
    "--cuda-graph-bs",
    1,
    2,
    3,
    4,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
]


class TestNPUMiniMaxM2_5_W8A8_8P_In32k_Out0k5_Prefix90(
    TestAscendPerformanceTestCaseBase
):
    """Test NPU performance for MiniMax-M2.5-w8a8 8p single node prefix cache in32k out0.5k"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_32K_0K5_PREFIX_OTHER_ARGS
    envs = MINIMAX_M2_5_32K_0K5_PREFIX_ENVS
    dataset_name = "random"
    max_concurrency = 294
    num_prompts = 1176
    input_len = 32768
    output_len = 512
    random_range_ratio = 1
    aisbench_repeat_rate = 0.9
    tpot = 100
    output_token_throughput = 200

    def test_npu_minimax_m2_5_w8a8_8p_in32k_out0k5_prefix(self):
        """Run NPU performance test for MiniMax-M2.5-w8a8 in32k out0.5k prefix"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
