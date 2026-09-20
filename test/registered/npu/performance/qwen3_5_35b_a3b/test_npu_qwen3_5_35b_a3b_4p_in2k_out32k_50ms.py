import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_5_35B_A3B_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3600, suite="nightly-perf-8-npu-a3", nightly=True)

QWEN3_5_35B_A3B_2K_32K_ENVS = {
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
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "8192",
    "ENABLE_PROFILING": "0",
}

QWEN3_5_35B_A3B_2K_32K_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    8,
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    65536,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.7,
    "--max-running-requests",
    256,
    "--cuda-graph-bs-decode",
    1,
    2,
    3,
    4,
    8,
    16,
    32,
    "--mm-attention-backend",
    "ascend_attn",
    "--dp",
    8,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--enable-multimodal",
    "--moe-a2a-backend",
    "deepep",
    "--ep-size",
    8,
    "--stream-interval",
    64,
    "--disable-radix-cache",
]


class TestNPUQwen3_5_35BA3B_4P_In2k_Out32k_50ms(TestNpuPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.5-35B-A3B 4p in2k out32k 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_5_35B_A3B_MODEL_PATH
    other_args = QWEN3_5_35B_A3B_2K_32K_OTHER_ARGS
    envs = QWEN3_5_35B_A3B_2K_32K_ENVS
    dataset_name = "random"
    max_concurrency = 256
    num_prompts = 256
    input_len = 2048
    output_len = 32768
    random_range_ratio = 1
    seed = 1234
    tpot = 50
    output_token_throughput = 4459

    def test_npu_qwen3_5_35b_a3b_4p_in2k_out32k_50ms(self):
        """Run NPU performance test for Qwen3.5-35B-A3B in2k out32k 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
