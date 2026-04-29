import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_5_27B_W8A8_HOME_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-2-npu-a3",
    nightly=True,
)

QWEN3_5_27B_64K_90_PREFIX_HIGH_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
    "ASCEND_USE_FIA": "1",
}

QWEN3_5_27B_64K_90_PREFIX_HIGH_OTHER_ARGS = [
    "--model-path",
    QWEN3_5_27B_W8A8_HOME_MODEL_PATH,
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--node-rank",
    0,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    258000,
    "--trust-remote-code",
    "--mamba-scheduler-strategy",
    "extra_buffer",
    "--max-running-requests",
    16,
    "--max-mamba-cache-size",
    80,
    "--mem-fraction-static",
    0.6,
    "--cuda-graph-bs",
    2,
    6,
    8,
    10,
    11,
    12,
    14,
    16,
    "--enable-multimodal",
    "--quantization",
    "modelslim",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestNPUQwen3_5_27B_1P_In64k_90Prefix_High(TestAscendPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.5-27B-W8A8 1p in64k 90% prefix"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_5_27B_W8A8_HOME_MODEL_PATH
    other_args = QWEN3_5_27B_64K_90_PREFIX_HIGH_OTHER_ARGS
    envs = QWEN3_5_27B_64K_90_PREFIX_HIGH_ENVS
    dataset_name = "random"
    max_concurrency = 2
    num_prompts = 8
    input_len = 65536
    output_len = 1024
    random_range_ratio = 0.1
    tpot = 50
    output_token_throughput = 100

    def test_npu_qwen3_5_27b_1p_in64k_90prefix_high(self):
        """Run NPU performance test for Qwen3.5-27B-W8A8 in64k 90% prefix"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()