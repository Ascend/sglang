import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_5_122B_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3600, suite="nightly-perf-8-npu-a3", nightly=True)

QWEN3_5_122B_4P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "2400",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "GDN_ATTN_BACKEND_TRITON": "1",
    "TEST_FIV_BACKEND_TRITON": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "8192",
    "ENABLE_PROFILING": "0",
}

QWEN3_5_122B_4P_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    8,
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    32768,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--mem-fraction-static",
    0.87,
    "--max-running-requests",
    112,
    "--cuda-graph-bs-decode",
    1,
    4,
    8,
    14,
    16,
    "--dp",
    8,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--moe-a2a-backend",
    "deepep",
    "--ep-size",
    8,
    "--stream-interval",
    64,
    "--schedule-conservativeness",
    0.4,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
]


class TestNPUQwen3_5_122B_4P_In2k_Out32k_50ms(TestNpuPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.5-122B 4p in2k out32k 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_5_122B_MODEL_PATH
    other_args = QWEN3_5_122B_4P_OTHER_ARGS
    envs = QWEN3_5_122B_4P_ENVS
    dataset_name = "random-ids"
    max_concurrency = 224
    num_prompts = 224
    input_len = 2048
    output_len = 32768
    random_range_ratio = 1
    seed = 1234
    request_rate = float("inf")
    tpot = 50
    output_token_throughput = 2170

    def test_npu_qwen3_5_122b_4p_in2k_out32k_50ms(self):
        """Run NPU performance test for Qwen3.5-122B in2k out32k 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
