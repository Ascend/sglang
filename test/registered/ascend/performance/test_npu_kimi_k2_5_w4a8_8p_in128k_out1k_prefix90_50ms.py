import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import (
    KIMI_K2_5_EAGLE3_MODEL_PATH,
    KIMI_K2_5_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-8-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "HCCL_BUFFSIZE": "1200",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

OTHER_ARGS = [
    "--skip-server-warmup",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.8,
    "--max-running-requests",
    128,
    "--chunked-prefill-size",
    16384,
    "--max-prefill-tokens",
    16384,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-dp-attention",
    "--dp-size",
    4,
    "--cuda-graph-bs",
    4,
    8,
    16,
    32,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
]


class TestNPUKimiK2_5_W4A8_8P_In128k_Out1k_50ms(TestAscendPerformanceTestCaseBase):
    """Test NPU performance for Kimi-K2.5-w4a8 8p in128k out1k 50ms TPOT"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = KIMI_K2_5_W4A8_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    dataset_name = "random"
    max_concurrency = 48
    num_prompts = 48
    request_rate = 1
    aisbench_repeat_rate = 0.9
    input_len = 16384
    output_len = 1024
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 1000

    def test_npu_kimi_k2_5_w4a8_8p_in128k_out1k_50ms(self):
        """Run NPU performance test for Kimi-K2.5-w4a8 in128k out1k 50ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
