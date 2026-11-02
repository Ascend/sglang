import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    MIMO_V2_5_PRO_DFLASH_MODEL_PATH,
    MIMO_V2_5_PRO_FP4_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-acc-8-npu-a5",
    nightly=True,
    disabled="A5 test case",
)

MIMO_V2_5_PRO_FP4_8P_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_BUFFSIZE": "300",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "20",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "4096",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "0",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
}

MIMO_V2_5_PRO_FP4_8P_OTHER_ARGS = [
    "--served-model-name",
    MIMO_V2_5_PRO_FP4_MODEL_PATH,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--mem-fraction-static",
    0.905,
    "--tp-size",
    8,
    "--nnodes",
    1,
    "--chunked-prefill-size",
    8192,
    "--max-total-tokens",
    600000,
    "--max-running-requests",
    8,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    6,
    8,
    "--speculative-algorithm",
    "DFLASH",
    "--speculative-draft-model-path",
    MIMO_V2_5_PRO_DFLASH_MODEL_PATH,
    "--speculative-num-draft-tokens",
    8,
    "--enable-metrics",
]


class TestNPUMiMoV2_5_Pro_W4A8_8P_A5_Acceptance(TestNpuPerformanceTestCaseBase):
    """Test NPU acceptance rate for MiMo-V2.5-Pro-FP4 8p A5 single node with DFLASH.

    Runs bench_serving with random dataset to measure speculative decoding
    acceptance rate. The base class computes
    accept_rate = accept_length / --speculative-num-draft-tokens (8)
    and asserts it against the ``accept_rate`` baseline.
    """

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = MIMO_V2_5_PRO_FP4_MODEL_PATH
    other_args = MIMO_V2_5_PRO_FP4_8P_OTHER_ARGS
    envs = MIMO_V2_5_PRO_FP4_8P_ENVS
    dataset_name = "random"
    input_len = 16000
    output_len = 1000
    num_prompts = 128
    max_concurrency = 32
    random_range_ratio = 1
    request_rate = float("inf")
    accept_rate = 0.25

    def test_acceptance(self):
        """Run bench_serving; the base class asserts accept_rate >= 0.25."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
