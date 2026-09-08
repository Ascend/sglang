import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=4800,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

# Common environment variables shared by prefill/decode nodes, ported from
# scripts_shell/pd/pro_1p1d(2+2)/dsv4_pro_pd.sh.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "68",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    # skip gpu branch
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "True",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

# Prefill node environment variables for DSV4-Pro PD-Sep deployment.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_PREFILL_ENVS = {
    **DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS,
    "DEEPEP_HCCL_BUFFSIZE": "2048",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    # memory fabric for PD KV transfer
    "MF_HYBM_USE_VMM_SEGMENT": "1",
    "ASCEND_MF_TRANSFER_PROTOCOL": "device_urma",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
    # prefill delay
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    # send cached prefix to decode early for radix-cache hits
    "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX": "1",
}

# Decode node environment variables for DSV4-Pro PD-Sep deployment.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_DECODE_ENVS = {
    **DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS,
    "DEEPEP_HCCL_BUFFSIZE": "900",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "30",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    # MTP (DSPARK)
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    # dspark correctness-first setup
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    "SGLANG_DSPARK_ENABLE_MULTI_STREAM": "0",
    "SGLANG_DSPARK_QUANT_AUDIT": "1",
    "SGLANG_DSPARK_QUANT_AUDIT_STRICT": "0",
}

# Prefill node (2 nodes x 8 NPUs, TP16 DP4) launch arguments.
# Radix cache is intentionally ENABLED on prefill (no --disable-radix-cache).
DEEPSEEK_V4_PRO_W4A8_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--disaggregation-bootstrap-port",
    8998,
    "--tp-size",
    16,
    "--nnodes",
    2,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--load-balance-method",
    "round_robin",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    64,
    "--mem-fraction-static",
    0.83,
    "--quantization",
    "modelslim",
    "--max-prefill-tokens",
    2048000,
    "--chunked-prefill-size",
    65536,
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--context-length",
    133120,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disable-cuda-graph",
    "--enable-dynamic-batch-tokenizer",
    "--tokenizer-worker-num",
    16,
]

# Decode node (2 nodes x 8 NPUs, TP16 DP4) launch arguments.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--tp-size",
    16,
    "--nnodes",
    2,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--load-balance-method",
    "round_robin",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    256,
    "--mem-fraction-static",
    0.86,
    "--quantization",
    "modelslim",
    "--max-prefill-tokens",
    2048000,
    "--chunked-prefill-size",
    16384,
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--context-length",
    133120,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    "--tokenizer-worker-num",
    8,
    # DSPARK speculative decoding with the bundled draft weights.
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "modelslim",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    6,
    # Radix cache enabled on decode for the cache-hit scenario.
    "--disaggregation-decode-enable-radix-cache",
]

# Model config for DSV4-Pro W4A8 2P+2D PD-Sep deployment.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    "prefill_args": DEEPSEEK_V4_PRO_W4A8_PD_SEP_PREFILL_ARGS,
    "decode_args": DEEPSEEK_V4_PRO_W4A8_PD_SEP_DECODE_ARGS,
    "prefill_envs": DEEPSEEK_V4_PRO_W4A8_PD_SEP_PREFILL_ENVS,
    "decode_envs": DEEPSEEK_V4_PRO_W4A8_PD_SEP_DECODE_ENVS,
    "router_args": ["--policy", "cache_aware"],
    "router_envs": {},
}


class TestNPUDeepSeekV4ProW4A8PDSEPIn128kOut1kPrefix90(
    TestNpuPerfMultiNodePdSepTestCaseBase
):
    """Test NPU perf for DeepSeek-V4-Pro W4A8 PD-Sep 2P+2D in128k prefix90.

    Requirement: DSV4_Pro_Radix_Cache_0 (step 4, PD separation 128k input
    with 90% radix-cache hit rate). The shared-prefix dataset makes 90% of
    each input length a repeated prefix, so radix cache hits should reduce
    TTFT noticeably compared with the random-input test above.
    """

    model_config = DEEPSEEK_V4_PRO_W4A8_PD_SEP_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "generated-shared-prefix"
    repeat_rate = 0.9
    input_len = 131072
    output_len = 1024
    num_prompts = 32
    max_concurrency = 32
    random_range_ratio = 1
    warmup_requests = 0
    request_rate = float("inf")
    seed = 1
    temperature = 0.6
    top_p = 0.95
    # TODO: calibrate tpot / output_token_throughput / ttft baselines on the
    # first successful run, then set them here to enable regression assertions.
    pop_sglang_is_in_ci_for_gsp = True

    def test_npu_deepseek_v4_pro_w4a8_pd_sep_in128k_out1k_prefix90(self):
        """Run NPU perf test for DSV4-Pro W4A8 PD-Sep in128k prefix90."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
