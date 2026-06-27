import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME, check_role
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHSERVING,
    DEEPSEEK_V32_W8A8_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
    logger,
    run_bench_serving,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

MODEL_CONFIG_NOCPNOMTP = {
    "model_path": DEEPSEEK_V32_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_BUFFSIZE": "1200",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "TRANSFORMERS_VERBOSITY": "error",
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "TASK_QUEUE_ENABLE": "0",
        "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "HCCL_BUFFSIZE": "400",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "8",
        "TRANSFORMERS_VERBOSITY": "error",
    },
    "router_envs": {},
    "prefill_args": [
        "--nnodes",
        2,
        "--tp",
        32,
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.73,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        68000,
        "--max-running-requests",
        1,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--quantization",
        "modelslim",
        "--disaggregation-transfer-backend",
        "ascend",
        "--disaggregation-mode",
        "prefill",
        "--disable-cuda-graph",
        "--moe-dense-tp-size",
        1,
    ],
    "decode_args": [
        "--nnodes",
        2,
        "--tp",
        32,
        "--dp",
        8,
        "--ep",
        32,
        "--moe-dense-tp-size",
        1,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.79,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        68000,
        "--max-running-requests",
        32,
        "--cuda-graph-max-bs",
        4,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
        "--quantization",
        "modelslim",
        "--disaggregation-transfer-backend",
        "ascend",
        "--disaggregation-mode",
        "decode",
    ],
    "router_args": [
        "--mini-lb",
    ],
}

MODEL_CONFIG_CPNOMTTP = {
    **MODEL_CONFIG_NOCPNOMTP,
    "prefill_args": MODEL_CONFIG_NOCPNOMTP["prefill_args"]
    + [
        "--enable-nsa-prefill-context-parallel",
        "--nsa-prefill-cp-mode",
        "in-seq-split",
        "--attn-cp-size",
        32,
    ],
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


# ===== Global shared context (valid within a single Python process) =====
class BenchmarkContext:
    """
    Shared context for passing benchmark results
    between multiple TestCase classes running in the same process.
    """

    def __init__(self):
        # TTFT (Time To First Token) when CP is enabled
        self.cp_enabled_ttft = None

    def ensure_cp_enabled_ttft(self):
        """
        Ensure that the CP-enabled TTFT has been recorded.

        This must be called after the CP-enabled benchmark test has run.
        """
        if self.cp_enabled_ttft is None:
            raise RuntimeError(
                "cp_enabled_ttft is not set. "
                "Did TestDeepSeekV32W8A8PdSepCpNoMtpFunctional run first?"
            )


# Singleton instance shared by all test classes
benchmark_ctx = BenchmarkContext()


class TestDeepSeekV32W8A8PdSepCpNoMtpFunctional(
    TestAscendPerfMultiNodePdSepTestCaseBase
):
    """Verify long-context inference works correctly with CP enabled and MTP disabled

    [Test Category] Functional
    [Test Target] Long-Context Inference Correctness (CP enabled, No MTP)
    --enable-nsa-prefill-context-parallel; --nsa-prefill-cp-mode
    """

    model_config = MODEL_CONFIG_CPNOMTTP
    benchmark_tool = BENCHSERVING
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 1
    input_len = 65536
    output_len = 1024
    random_range_ratio = 1
    output_token_throughput = 0
    ttft = 0

    @check_role(allowed_roles=["router"])
    def test_long_context_inference_with_cp_enabled(self):
        """Verify 64K long-context inference runs correctly with CP enabled and MTP disabled."""
        metrics = _run_benchmark(self)

        self.assertGreater(float(metrics["total_tps"]), self.output_token_throughput)

        mean_ttft = float(metrics["mean_ttft"])
        self.assertGreater(mean_ttft, self.ttft)

        # Store TTFT in shared context
        benchmark_ctx.cp_enabled_ttft = mean_ttft
        logger.info(
            "Stored cp_enabled_ttft into benchmark_ctx: %.3f",
            benchmark_ctx.cp_enabled_ttft,
        )


class TestDeepSeekV32W8A8PdSepCpVsNoCpTtftCompare(
    TestAscendPerfMultiNodePdSepTestCaseBase
):
    """Verify CP reduces TTFT compared to No-CP configuration (MTP disabled)

    [Test Category] Functional
    [Test Target] CP reduces TTFT
    --enable-nsa-prefill-context-parallel; --nsa-prefill-cp-mode
    """

    model_config = MODEL_CONFIG_NOCPNOMTP
    benchmark_tool = BENCHSERVING
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 1
    input_len = 65536
    output_len = 1024
    random_range_ratio = 1
    ttft = 0

    @check_role(allowed_roles=["router"])
    def test_ttft_reduced_with_cp_enabled(self):
        """Verify TTFT is reduced when CP is enabled compared to No-CP."""
        benchmark_ctx.ensure_cp_enabled_ttft()

        metrics_no_cp = _run_benchmark(self)
        ttft_no_cp = float(metrics_no_cp["mean_ttft"])

        self.assertGreater(ttft_no_cp, self.ttft)

        self.assertGreater(
            ttft_no_cp,
            benchmark_ctx.cp_enabled_ttft,
            msg=(
                f"TTFT should be lower with CP enabled: "
                f"no_cp={ttft_no_cp}, cp={benchmark_ctx.cp_enabled_ttft}"
            ),
        )

        logger.info(
            "TTFT comparison: no_cp=%.3f, cp=%.3f",
            ttft_no_cp,
            benchmark_ctx.cp_enabled_ttft,
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(
        TestDeepSeekV32W8A8PdSepCpNoMtpFunctional(
            "test_long_context_inference_with_cp_enabled"
        )
    )
    suite.addTest(
        TestDeepSeekV32W8A8PdSepCpVsNoCpTtftCompare("test_ttft_reduced_with_cp_enabled")
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
