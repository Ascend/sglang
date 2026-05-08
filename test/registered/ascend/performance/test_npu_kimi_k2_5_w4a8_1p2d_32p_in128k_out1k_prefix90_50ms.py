import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    KIMI_K2_5_EAGLE3_MODEL_PATH,
    KIMI_K2_5_W4A8_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-pd-sep-3-node",
    nightly=True,
)

PREFILL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "1800",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "60",
}

DECODE_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "HCCL_BUFFSIZE": "1200",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
}

PREFILL_ARGS = [
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--nnodes",
    1,
    "--node-rank",
    0,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.78,
    "--max-running-requests",
    8,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--chunked-prefill-size",
    16384,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
]

DECODE_ARGS = [
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--dist-init-addr",
    "localhost:5000",
    "--nnodes",
    2,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--tp-size",
    32,
    "--mem-fraction-static",
    0.82,
    "--max-running-requests",
    32,
    "--enable-dp-attention",
    "--dp-size",
    4,
    "--disable-radix-cache",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    8,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    1,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    2,
    "--speculative-draft-model-quantization",
    "unquant",
]

ROUTER_ARGS = [
    "--pd-disaggregation",
    "--policy",
    "cache_aware",
]

ROUTER_ENVS = {}

MODEL_CONFIG = {
    "model_path": KIMI_K2_5_W4A8_MODEL_PATH,
    "prefill_args": PREFILL_ARGS,
    "decode_args": DECODE_ARGS,
    "prefill_envs": PREFILL_ENVS,
    "decode_envs": DECODE_ENVS,
    "router_args": ROUTER_ARGS,
    "router_envs": ROUTER_ENVS,
}


class TestNPUKimiK2_5_W4A8_1P2D_32P_In128k_Out1k_Prefix90_50ms(TestAscendPerfMultiNodePdSepTestCaseBase):
    """Test NPU performance for Kimi-K2.5-w4a8 1P+2D 32p: input_len=131072, output_len=1024, 90% cache, TPOT=50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 8
    num_prompts = 8
    aisbench_repeat_rate = 0.9
    input_len = 131072
    output_len = 1024
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 16899

    def test_npu_kimi_k2_5_w4a8_1p2d_32p_in128k_out1k_prefix90_50ms(self):
        """Run NPU performance test for Kimi-K2.5-w4a8 1P+2D 32p in128k out1k 90% cache"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()