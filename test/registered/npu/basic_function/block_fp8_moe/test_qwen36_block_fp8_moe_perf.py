import unittest

from sglang.benchmark.serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_6_35B_A3B_FP8_WEIGHTS_PATH,
    logger,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
)

register_npu_ci(est_time=3600, suite="full-2-npu-a3", nightly=True)

SERVER_ENVS = {
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM": "1",
    "ASCEND_USE_FIA": "1",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
}

# --max-running-requests 10 caps concurrency below the documented 32; both modes share
# the args, so the comparison stays fair. --device npu is also appended identically by
# popen_launch_server (for device="auto").
SERVER_ARGS = [
    "--tp-size",
    "2",
    "--nnodes",
    "1",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    "-1",
    "--max-total-tokens",
    "600000",
    "--max-prefill-tokens",
    "65536",
    "--disable-radix-cache",
    "--trust-remote-code",
    "--enable-prefill-delayer",
    "--max-running-requests",
    "10",
    "--max-mamba-cache-size",
    "20",
    "--mem-fraction-static",
    "0.65",
    "--cuda-graph-bs-decode",
    "2",
    "4",
    "8",
    "12",
    "14",
    "16",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-draft-model-quantization",
    "fp8",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
]


# ~1% run-to-run noise here, so judge "not slower" (with margin), not "faster".
METRIC_TOLERANCE = 0.02


class TestE2E5DualModePerf(CustomTestCase):
    """Testcase: Verify the requantized MXFP8 mode is not slower than the native Block FP8 mode under the same load: E1 throughput and E2 TTFT stay within METRIC_TOLERANCE of native.

    [Test Category] Performance
    [Test Target] SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8=0 vs =1
    [Test Config] TP2, 200 random prompts, input 1024 / output 256, max concurrency 32
    """

    model = QWEN3_6_35B_A3B_FP8_WEIGHTS_PATH

    def bench_once(self, mode):
        """Start a server in the given mode, benchmark it, then stop it."""
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=list(SERVER_ARGS),
            env={**SERVER_ENVS, "SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8": mode},
        )
        try:
            result = self._run_benchmark()
        finally:
            kill_process_tree(process.pid)
        return {
            "output_throughput": result["output_throughput"],
            "mean_ttft_ms": result["mean_ttft_ms"],
        }

    def _run_benchmark(self):
        """Benchmark the already-running server and return the metric dict."""
        args = get_benchmark_args(
            base_url=DEFAULT_URL_FOR_TEST,
            backend="sglang",
            dataset_name="random",
            # "random" samples from ShareGPT, so use the local copy, not a HuggingFace download.
            dataset_path="/tmp/ShareGPT_V3_unfiltered_cleaned_split.json",
            tokenizer=self.model,
            num_prompts=200,
            random_input_len=1024,
            random_output_len=256,
            request_rate=float("inf"),
            max_concurrency=32,
        )
        args.model = self.model
        # Warm-up requests keep the first measured request free of cold-start cost.
        args.warmup_requests = 16
        return run_benchmark(args)

    def test_mxfp8_not_slower_than_native(self):
        # Warm each mode once so both measured runs start from the same cache state;
        # warming only one would bias the other's compile time into its measured window.
        logger.info(f"[{type(self).__name__}] warming caches (result discarded)")
        self.bench_once("0")
        self.bench_once("1")

        native = self.bench_once("0")
        mxfp8 = self.bench_once("1")

        throughput_ratio = mxfp8["output_throughput"] / native["output_throughput"]
        ttft_ratio = mxfp8["mean_ttft_ms"] / native["mean_ttft_ms"]

        logger.info(
            f"[{type(self).__name__}] native throughput={native['output_throughput']:.2f} "
            f"mean_ttft={native['mean_ttft_ms']:.2f}ms | "
            f"mxfp8 throughput={mxfp8['output_throughput']:.2f} "
            f"mean_ttft={mxfp8['mean_ttft_ms']:.2f}ms | "
            f"throughput_ratio={throughput_ratio:.4f} ttft_ratio={ttft_ratio:.4f}"
        )

        # Both modes must produce results, else the measurement is void.
        self.assertGreater(native["output_throughput"], 0)
        self.assertGreater(mxfp8["output_throughput"], 0)

        self.assertGreaterEqual(
            throughput_ratio,
            1.0 - METRIC_TOLERANCE,
            "E1 failed: MXFP8 throughput is lower than native "
            f"(throughput_ratio={throughput_ratio:.4f}, "
            f"gain={throughput_ratio - 1.0:+.2%})",
        )
        self.assertLessEqual(
            ttft_ratio,
            1.0 + METRIC_TOLERANCE,
            "E2 failed: MXFP8 TTFT is higher than native "
            f"(ttft_ratio={ttft_ratio:.4f})",
        )
        logger.info(
            f"[{type(self).__name__}] E1/E2 passed: "
            f"throughput gain={throughput_ratio - 1.0:+.2%}, "
            f"ttft_ratio={ttft_ratio:.4f}"
        )


if __name__ == "__main__":
    unittest.main()
