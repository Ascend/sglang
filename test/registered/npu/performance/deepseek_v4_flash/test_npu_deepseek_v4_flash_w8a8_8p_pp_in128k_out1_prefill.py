import os
import subprocess
import unittest
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
    assert_metrics,
    run_bench_serving,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.utils import wait_for_server

register_npu_ci(est_time=7200, suite="full-perf-test-16-npu-a3-verity", nightly=True)


# Environment variables for the DSV4-Flash W8A8 8p PP PD-mix deployment.
DEEPSEEK_V4_FLASH_W8A8_8P_PP_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    # skip gpu branch
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    # mtp
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    "SGLANG_DSPARK_ENABLE_MULTI_STREAM": "0",
    "SGLANG_DSPARK_QUANT_AUDIT": "1",
    "SGLANG_DSPARK_QUANT_AUDIT_STRICT": "0",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM": "1",
    "SGLANG_PROFILE_WITH_STACK": "False",
    # deepep, set on this node by the launch script
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_HCCL_BUFFSIZE": "2600",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "8192",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    # war barrier
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
    "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX": "1",
}

# Server launch arguments for DSV4-Flash W8A8 single-node 8p PP PD-mix.
DEEPSEEK_V4_FLASH_W8A8_8P_PP_OTHER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    2,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.62,
    "--prefill-max-requests",
    256,
    "--max-prefill-tokens",
    70000,
    "--max-running-requests",
    256,
    "--chunked-prefill-size",
    16384,
    "--pp-size",
    8,
    "--attn-dp-size",
    2,
    "--tokenizer-worker-num",
    16,
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--load-balance-method",
    "round_robin",
]

# The second run differs from the first by this flag only, so the TTFT, TPOT and
# throughput comparison is single-variable. Tuning follows the
# pipeline-parallelism guide: with dynamic chunking the initial
# --chunked-prefill-size is best set to 2-3x the optimal fixed size, so re-tune
# the shared value above if needed.
DEEPSEEK_V4_FLASH_W8A8_8P_PP_DYNAMIC_CHUNKING_ARGS = [
    *DEEPSEEK_V4_FLASH_W8A8_8P_PP_OTHER_ARGS,
    "--enable-dynamic-chunking",
]


class TestNPUDeepSeekV4FlashW8A88PPPIn128kOut1Prefill(TestNpuPerformanceTestCaseBase):
    """Pure prefill over a single node with tp-size 2 x pp-size 8, benchmarked
    twice through the same bench_serving flow as the single-run DSV4-Flash 8p
    case (131072 in / 1 out, 10 prompts, C1 with 1 warmup request). The first run
    uses the launch-script config, the second adds --enable-dynamic-chunking, and
    the case asserts that dynamic chunking lowers the reported TTFT."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_W8A8_8P_PP_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_W8A8_8P_PP_ENVS
    dataset_name = "random"
    input_len = 131072
    output_len = 1
    num_prompts = 10
    max_concurrency = 1
    warmup_requests = 1
    random_range_ratio = 1
    request_rate = float("inf")

    @classmethod
    def setUpClass(cls):
        """Launch server via `python3 -m sglang.launch_server` instead of `sglang serve`."""
        cls._setup_per_case_output()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls._launch_server(cls.other_args)

    @classmethod
    def _launch_server(cls, other_args):
        """(Re)launch the server with `other_args`, reusing the launch command of
        the single-run DSV4-Flash 8p case."""
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

        env = os.environ.copy()
        if cls.envs:
            env.update(cls.envs)

        _, host, port = cls.base_url.split(":")
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model,
            *[str(x) for x in other_args],
            "--host",
            host[2:],
            "--port",
            port,
        ]
        cls.process = subprocess.Popen(command, env=env)
        wait_for_server(cls.base_url, timeout=cls.timeout, process=cls.process)

    def test_npu_deepseek_v4_flash_w8a8_8p_pp_in128k_out1_prefill(self):
        """Benchmark the launch-script config, benchmark it again with
        --enable-dynamic-chunking, then assert that dynamic chunking lowers TTFT."""
        baseline = self._run_throughput_and_collect()

        self._launch_server(DEEPSEEK_V4_FLASH_W8A8_8P_PP_DYNAMIC_CHUNKING_ARGS)
        dynamic = self._run_throughput_and_collect()

        baseline_ttft = self._ttft(baseline, "baseline")
        dynamic_ttft = self._ttft(dynamic, "dynamic_chunking")
        self.assertLess(
            dynamic_ttft,
            baseline_ttft,
            "--enable-dynamic-chunking did not lower the 128k prefill TTFT: "
            f"baseline={baseline_ttft:.1f} ms, dynamic_chunking={dynamic_ttft:.1f} ms",
        )

    def _run_throughput_and_collect(self):
        """Bench and assert exactly like run_throughput(), but return the parsed
        metrics so that the two runs can be compared against each other."""
        url = urlparse(self.base_url)
        metrics = run_bench_serving(
            host=url.hostname,
            port=url.port,
            model_path=self.model,
            backend=self.backend,
            dataset_name=self.dataset_name,
            request_rate=self.request_rate,
            max_concurrency=self.max_concurrency,
            num_prompts=self.num_prompts,
            input_len=self.input_len,
            output_len=self.output_len,
            random_range_ratio=self.random_range_ratio,
            warmup_requests=self.warmup_requests,
        )
        assert_metrics(self, metrics)
        return metrics

    def _ttft(self, metrics, tag):
        """Return mean TTFT in ms, failing the case when bench_serving reported no
        usable TTFT (it prints N/A when it cannot measure one)."""
        value = metrics.get("mean_ttft")
        self.assertIsNotNone(value, f"{tag}: bench_serving output carries no Mean TTFT")
        try:
            return float(value)
        except ValueError:
            self.fail(f"{tag}: Mean TTFT is not a number in bench_serving output: {value}")


if __name__ == "__main__":
    unittest.main()
