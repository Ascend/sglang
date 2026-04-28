import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_5_397B_W4A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_5_397B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "3000",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
}
QWEN3_5_397B_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    "16",
    "--chunked-prefill-size",
    "-1",
    "--max-prefill-tokens",
    "16384",
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    "256",
    "--mem-fraction-static",
    "0.85",
    "--cuda-graph-bs",
    "1",
    "2",
    "4",
    "8",
    "10",
    "12",
    "14",
    "16",
    "--quantization",
    "modelslim",
    "--enable-multimodal",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--mm-attention-backend",
    "ascend_attn",
    "--max-total-tokens",
    "150000",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--dp-size",
    "4",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--speculative-draft-model-quantization",
    "unquant",
]


class TestQwen397B(TestAscendPerformanceTestCaseBase):
    max_attempts = 3
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_5_397B_W4A8_MODEL_PATH
    other_args = QWEN3_5_397B_OTHER_ARGS
    envs = QWEN3_5_397B_ENVS
    dataset_name = "random"
    max_concurrency = 16
    num_prompts = 16
    input_len = 16000
    output_len = 1000
    random_range_ratio = 1
    tpot = 19
    output_token_throughput = 500

    def test_qwen3_5_397b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
