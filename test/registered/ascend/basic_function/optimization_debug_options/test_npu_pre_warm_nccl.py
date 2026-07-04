import os
import re
import unittest

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="debug-full-2-npu-a3", nightly=True)

# Expected log line from model_runner.py:1147-1150 —
# "NCCL/RCCL warmup completed in {X.XXX}s (tp_size=2, pp_size=1, ep_size=1)"
_WARMUP_LOG_RE = re.compile(
    r"NCCL/RCCL warmup completed in ([\d.]+)s "
    r"\(tp_size=\d+, pp_size=\d+, ep_size=\d+\)"
)


class TestPreWarmNccl(CustomTestCase):
    """Testcase: verify --pre-warm-nccl executes all-reduce warmup during
    server startup and serving works correctly via bench_serving.

    The warmup runs dist.all_reduce + synchronize (model_runner.py:1135-1150).
    Parsing warmup_elapsed from the log proves the all-reduce actually executed.
    bench_serving is used to verify TTFT after server startup.

    NOTE: server_args.py:3065 currently sets pre_warm_nccl=False on non-CUDA/HIP
    hardware (including NPU).  The HCCL backend path needs to be added there
    before this test can pass on NPU.

    [Test Category] Parameter
    [Test Target] --pre-warm-nccl
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "2",
    ]

    def _launch(self, with_warmup):
        out_log = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log = open("./cache_err_log.txt", "w+", encoding="utf-8")
        args = list(self.base_args)
        if with_warmup:
            args.append("--pre-warm-nccl")
        proc = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            return_stdout_stderr=(out_log, err_log),
        )
        return proc, out_log, err_log

    def _cleanup(self, proc, out_log, err_log):
        kill_process_tree(proc.pid)
        out_log.close()
        err_log.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def _parse_warmup_elapsed(self, err_log):
        """Return warmup elapsed seconds (float), or None if not found."""
        err_log.seek(0)
        for line in err_log:
            m = _WARMUP_LOG_RE.search(line)
            if m:
                return float(m.group(1))
        return None

    def _run_bench(self):
        """Run bench_serving and verify TTFT metrics."""
        args = get_benchmark_args(
            base_url=self.base_url,
            backend="sglang",
            dataset_name="random",
            tokenizer=self.model,
            num_prompts=10,
            random_input_len=256,
            random_output_len=32,
            request_rate=float("inf"),
        )
        args.warmup_requests = 0
        res = run_benchmark(args)
        self.assertEqual(res["completed"], 10)
        self.assertGreater(res["mean_ttft_ms"], 0, "TTFT must be > 0 ms")
        return res

    def test_pre_warm_nccl(self):
        # ---- With --pre-warm-nccl ----
        proc1, out1, err1 = self._launch(with_warmup=True)
        try:
            res_warmup = self._run_bench()
            warmup_elapsed = self._parse_warmup_elapsed(err1)
        finally:
            self._cleanup(proc1, out1, err1)

        self.assertIsNotNone(
            warmup_elapsed,
            "Expected stderr to contain 'NCCL/RCCL warmup completed in {X}s', "
            "proving --pre-warm-nccl triggered the warmup code path "
            "(model_runner.py:1135-1150).",
        )
        self.assertGreater(
            warmup_elapsed,
            0,
            f"Expected warmup elapsed time > 0, got {warmup_elapsed:.6f}s. "
            "A zero value would mean all-reduce was skipped.",
        )

        # ---- Without --pre-warm-nccl ----
        proc2, out2, err2 = self._launch(with_warmup=False)
        try:
            res_no_warmup = self._run_bench()
            no_warmup = self._parse_warmup_elapsed(err2)
        finally:
            self._cleanup(proc2, out2, err2)

        self.assertIsNone(
            no_warmup,
            "Expected stderr NOT to contain 'NCCL/RCCL warmup completed', "
            "proving the warmup is only triggered when the flag is set.",
        )

        # ---- Compare TTFT ----
        ttft_w = res_warmup["mean_ttft_ms"]
        ttft_nw = res_no_warmup["mean_ttft_ms"]
        p99_w = res_warmup["p99_ttft_ms"]
        p99_nw = res_no_warmup["p99_ttft_ms"]
        print(
            f"\n=== TTFT Comparison: --pre-warm-nccl vs default ===\n"
            f"  Mean TTFT: {ttft_w:.1f} ms (warmup) vs {ttft_nw:.1f} ms (no-warmup)\n"
            f"  P99  TTFT: {p99_w:.1f} ms (warmup) vs {p99_nw:.1f} ms (no-warmup)\n"
        )
        self.assertLessEqual(
            ttft_w,
            ttft_nw,
            f"Expected --pre-warm-nccl mean TTFT ({ttft_w:.1f} ms) <= "
            f"no-warmup ({ttft_nw:.1f} ms). NCCL warmup should prime all-reduce "
            f"communication and reduce or match first-request latency.",
        )


if __name__ == "__main__":
    unittest.main()
