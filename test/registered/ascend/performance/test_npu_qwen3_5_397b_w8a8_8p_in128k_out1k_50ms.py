import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_5_397B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_5_397B_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "3000",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

QWEN3_5_397B_128K_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    133120,
    "--max-total-tokens",
    640000,
    "--max-mamba-cache-size",
    640,
    "--trust-remote-code",
    "--max-running-requests",
    10,
    "--mem-fraction-static",
    0.65,
    "--cuda-graph-bs",
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    "--quantization",
    "modelslim",
    "--enable-multimodal",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    2,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    3,
    "--speculative-draft-model-quantization",
    "unquant",
    "--mamba-scheduler-strategy",
    "extra_buffer",
]


class TestNPUQwen3_5_397B_128K(TestAscendPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.5-397B-w4a8 16p in128k out1k"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_5_397B_W8A8_MODEL_PATH
    other_args = QWEN3_5_397B_128K_OTHER_ARGS
    envs = QWEN3_5_397B_ENVS
    dataset_name = "random"
    max_concurrency = 10
    num_prompts = 10
    input_len = 131072
    output_len = 1024
    random_range_ratio = 1
    tpot = 50
    aisbench_request_rate = 1
    # output_token_throughput = 200

    def test_npu_qwen3_5_397b_128k(self):
        """Run NPU performance test for Qwen3.5-397B in128k out1k"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
