import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    NIC_NAME,
    check_role,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    QWEN3_235B_W8A8_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
    logger,
    run_bench_serving,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="multi nodes testcase",
)

# ====================== Base Configuration ======================
BASE_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "188416",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_DP_ROUND_ROBIN": "1",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "HCCL_BUFFSIZE": "4300",
    "TASK_QUEUE_ENABLE": "2",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "TRANSFORMERS_VERBOSITY": "error",
}

BASE_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_DP_ROUND_ROBIN": "1",
    "DP_ROUND_ROBIN": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "65536",
    "HCCL_BUFFSIZE": "800",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "TRANSFORMERS_VERBOSITY": "error",
}

BASE_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--tp-size",
    16,
    "--dp-size",
    16,
    "--mem-fraction-static",
    0.6,
    "--disable-radix-cache",
    "--quantization",
    "modelslim",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--max-running-requests",
    128,
    "--chunked-prefill-size",
    94208,
    "--max-prefill-tokens",
    262144,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--dtype",
    "bfloat16",
]

BASE_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--nnodes",
    "2",
    "--tp-size",
    32,
    "--dp-size",
    32,
    "--mem-fraction-static",
    0.83,
    "--max-running-requests",
    768,
    "--quantization",
    "modelslim",
    "--enable-dp-attention",
    "--cuda-graph-bs",
    6,
    8,
    12,
    15,
    18,
    20,
    22,
    24,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--watchdog-timeout",
    9000,
    "--context-length",
    8192,
    "--prefill-round-robin-balance",
    "--enable-dp-lm-head",
    "--tokenizer-worker-num",
    4,
    "--dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
]

# ====================== Configurations ======================
MODEL_CONFIG_FUSION_DISABLED = {
    "model_path": QWEN3_235B_W8A8_MODEL_PATH,
    "prefill_envs": BASE_PREFILL_ENVS,
    "decode_envs": BASE_DECODE_ENVS,
    "router_envs": {"SGLANG_DP_ROUND_ROBIN": "1", "TRANSFORMERS_VERBOSITY": "error"},
    "prefill_args": BASE_PREFILL_ARGS,
    "decode_args": BASE_DECODE_ARGS
    + [
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
    ],
    "router_args": ["--mini-lb"],
}

MODEL_CONFIG_FUSION_ENABLED = {
    "model_path": QWEN3_235B_W8A8_MODEL_PATH,
    "prefill_envs": BASE_PREFILL_ENVS,
    "decode_envs": BASE_DECODE_ENVS,
    "router_envs": {"SGLANG_DP_ROUND_ROBIN": "1"},
    "prefill_args": BASE_PREFILL_ARGS,
    "decode_args": BASE_DECODE_ARGS
    + [
        "--moe-a2a-backend",
        "ascend_fuseep",
    ],
    "router_args": ["--mini-lb"],
}


def _run_benchmark(test_case):
    logger.info(
        "Starting benchmark host=%s port=%s model=%s",
        test_case.host,
        test_case.port,
        test_case.model_config["model_path"],
    )

    metrics = run_bench_serving(
        host=test_case.host,
        port=str(test_case.port),
        model_path=test_case.model_config["model_path"],
        backend=test_case.backend,
        dataset_name=test_case.dataset_name,
        dataset_path=test_case.dataset_path,
        request_rate=test_case.request_rate,
        max_concurrency=test_case.max_concurrency,
        num_prompts=test_case.num_prompts,
        input_len=test_case.input_len,
        output_len=test_case.output_len,
        random_range_ratio=test_case.random_range_ratio,
        image_resolution=test_case.image_resolution,
        image_count=test_case.image_count,
        warmup_requests=test_case.warmup_requests,
        seed=test_case.seed,
    )

    if not metrics:
        raise RuntimeError("No metrics obtained from benchmark")

    logger.info("All extracted metrics: %s", metrics)
    return metrics


class BenchmarkContext:
    """
    Shared context for passing benchmark results
    between multiple TestCase classes running in the same process.
    """

    def __init__(self):
        # TPOT (ms) when fusion is ENABLED
        self.tpot_fusion_enabled = None

    def ensure_tpot_fusion_enabled(self):
        if self.tpot_fusion_enabled is None:
            raise RuntimeError(
                "tpot_fusion_enabled is not set. "
                "Ensure TestQwen235bFusionEnable.test_fusion_enable runs first."
            )


benchmark_ctx = BenchmarkContext()


class TestQwen235bFusionEnable(TestAscendPerfMultiNodePdSepTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_235B_W8A8_MODEL_PATH
    model_config = MODEL_CONFIG_FUSION_DISABLED  # baseline
    dataset_name = "random"
    max_concurrency = 860
    num_prompts = max_concurrency * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1

    @check_role(allowed_roles=["router"])
    def test_fusion_enable(self):
        metrics = _run_benchmark(self)

        tpot = float(metrics["mean_tpot"])

        # Sanity check
        self.assertGreater(tpot, 0.0)

        benchmark_ctx.tpot_fusion_enabled = tpot
        logger.info(
            "Fusion ENABLED TPOT stored: %.3f ms",
            benchmark_ctx.tpot_fusion_enabled,
        )


class TestQwen235bFusionDisabled(TestAscendPerfMultiNodePdSepTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_235B_W8A8_MODEL_PATH
    model_config = MODEL_CONFIG_FUSION_ENABLED
    dataset_name = "random"
    max_concurrency = 860
    num_prompts = max_concurrency * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1

    model_layers = 94
    max_allowed_per_layer_overhead_ms = 0.05  # 50 μs

    @check_role(allowed_roles=["router"])
    def test_fusion_disable(self):
        benchmark_ctx.ensure_tpot_fusion_enabled()

        metrics = _run_benchmark(self)
        tpot_fusion_disabled = float(metrics["mean_tpot"])
        self.assertGreater(tpot_fusion_disabled, 0.0)

        # Fusion OFF should be SLOWER → positive delta
        tpot_increase = tpot_fusion_disabled - benchmark_ctx.tpot_fusion_enabled
        self.assertGreaterEqual(tpot_increase, 0.0)

        per_layer_overhead = tpot_increase / self.model_layers
        self.assertLessEqual(
            per_layer_overhead,
            self.max_allowed_per_layer_overhead_ms,
            msg=(
                f"Per-layer overhead too high: {per_layer_overhead:.6f} ms > "
                f"{self.max_allowed_per_layer_overhead_ms:.3f} ms"
            ),
        )

        logger.info(
            "Fusion disabled: TPOT increase=%.3f ms, per-layer overhead=%.6f ms",
            tpot_increase,
            per_layer_overhead,
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestQwen235bFusionEnable("test_fusion_enable"))
    suite.addTest(TestQwen235bFusionDisabled("test_fusion_disable"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
