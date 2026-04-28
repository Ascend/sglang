import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_5_27B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "3000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    4,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    186000,
    "--enable-prefill-delayer",
    "--prefill-delayer-max-delay-passes",
    "200",
    "--disable-radix-cache",
    "--mem-fraction-static",
    0.94,
    "--max-total-tokens",
    700000,
    "--max-running-requests",
    38,
    "--max-mamba-cache-size",
    "200",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    12,
    18,
    24,
    32,
    34,
    36,
    38,
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestQwen3527B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_5_27B_W8A8_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    dataset_name = "random"
    max_concurrency = 38
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 20
    output_token_throughput = 1100

    def test_qwen3_5_27b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
