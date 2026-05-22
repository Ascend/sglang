import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import run_command
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHSERVING,
    DEEPSEEK_V32_W8A8_MODEL_PATH,
    logger,
    run_bench_serving,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

TTFT_FILE = "./cpnomttp_ttft.txt"

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
        "--attn-cp-size",
        32,
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
                    ],
}


def _run_benchmark(test_case):
    bench_params = {
        "host": test_case.host,
        "port": str(test_case.port),
        "model_path": test_case.model_config["model_path"],
        "backend": test_case.backend,
        "dataset_name": test_case.dataset_name,
        "dataset_path": test_case.dataset_path,
        "request_rate": test_case.request_rate,
        "max_concurrency": test_case.max_concurrency,
        "num_prompts": test_case.num_prompts,
        "input_len": test_case.input_len,
        "output_len": test_case.output_len,
        "random_range_ratio": test_case.random_range_ratio,
        "image_resolution": test_case.image_resolution,
        "image_count": test_case.image_count,
        "warmup_requests": test_case.warmup_requests,
        "seed": test_case.seed,
    }
    logger.info(f"Starting benchmark with parameters: {bench_params}")
    metrics = run_bench_serving(**bench_params)
    logger.info(f"All extracted metrics: {metrics}")

    if not metrics:
        raise RuntimeError("No metrics obtained from benchmark")
    return metrics


class TestDeepSeekV32W8A8PdSepCpNoMtpFunctional(TestAscendPerfMultiNodePdSepTestCaseBase):
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

    def test_long_context_inference_with_cp_enabled(self):
        """Verify 64K long-context inference runs correctly with CP enabled and MTP disabled."""
        metrics = _run_benchmark(self)
        if self.output_token_throughput:
            self.assertGreater(
                float(metrics["total_tps"]),
                self.output_token_throughput,
            )

        if self.ttft:
            self.assertGreater(
                float(metrics["mean_ttft"]),
                self.ttft,
            )
            run_command(f"rm -f {TTFT_FILE}")
            run_command(f"echo {metrics['mean_ttft']} > {TTFT_FILE}")


class TestDeepSeekV32W8A8PdSepCpVsNoCpTtftCompare(TestAscendPerfMultiNodePdSepTestCaseBase):
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

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")
            finally:
                if os.path.exists(TTFT_FILE):
                    os.remove(TTFT_FILE)

    def test_baseline_ttft_without_cp(self):
        """Collect TTFT baseline with CP disabled and MTP disabled."""
        self.metrics_nocpnomtp = _run_benchmark(self)
        if self.ttft:
            self.assertGreater(
                float(self.metrics_nocpnomtp["mean_ttft"]),
                self.ttft,
            )

    def test_ttft_reduced_with_cp_enabled(self):
        """Verify TTFT is reduced when CP is enabled compared to No-CP."""
        if not hasattr(self, "metrics_nocpnomtp"):
            raise RuntimeError("test_throughput must be run before test_compare_ttft")

        metrics_cpnomttp_ttft = float(run_command(f"cat {TTFT_FILE}"))
        self.assertGreater(
            self.metrics_nocpnomtp["mean_ttft"],
            metrics_cpnomttp_ttft
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestDeepSeekV32W8A8PdSepCpNoMtpFunctional("test_long_context_inference_with_cp_enabled"))
    suite.addTest(TestDeepSeekV32W8A8PdSepCpVsNoCpTtftCompare("test_baseline_ttft_without_cp"))
    suite.addTest(TestDeepSeekV32W8A8PdSepCpVsNoCpTtftCompare("test_ttft_reduced_with_cp_enabled"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
