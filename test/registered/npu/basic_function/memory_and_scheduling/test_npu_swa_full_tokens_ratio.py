"""Tests for --swa-full-tokens-ratio parameter.

The parameter controls: SWA pool tokens = Full pool tokens * ratio.
Only effective on Hybrid SWA models (DeepSeek V4, MiMo, Inkling, etc.).

Two test strategies:
- Unit test: mock Hybrid SWA model, verify pool size calculation (CPU only)
- Server test: launch a real Hybrid SWA model, verify inference and print pool sizes
"""

import os
import re
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MIMO_V2_FLASH_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.registered.unit.model_executor.test_pool_configurator import (
    _actual_memory_used,
    _make_model_runner,
    mock_cpu_env,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-16-npu-a3-test-debug", nightly=True)


class TestSwaFullTokensRatio(CustomTestCase):
    """Unit tests for --swa-full-tokens-ratio via pool_configurator.

    Fully references test_pool_configurator.py's TestHybridSWAConfigurator
    pattern: mock Hybrid SWA model -> run pool configurator -> assert
    pool sizes and memory invariants.

    [Test Category] Parameter
    [Test Target] --swa-full-tokens-ratio
    [Scenario] S1: ratio controls SWA pool proportionally
    """

    def _make_swa_runner(self, full_layers=16, swa_layers=16, ratio=0.5, page_size=1):
        return _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(full_layers)),
            swa_attention_layer_ids=list(
                range(full_layers, full_layers + swa_layers)
            ),
            swa_num_kv_heads=4,
            page_size=page_size,
            swa_full_tokens_ratio=ratio,
        )

    def _run(self, available_bytes, **kwargs):
        mr = self._make_swa_runner(**kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, mr.server_args.page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_ratio_respected(self):
        """swa_tokens = full_tokens * ratio for various ratios."""
        available = 10_000_000
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            _, _, config = self._run(available, ratio=ratio, page_size=1)
            full = config.full_max_total_num_tokens
            swa = config.swa_max_total_num_tokens
            self.assertEqual(
                swa,
                int(full * ratio),
                f"ratio={ratio}: swa={swa} != full({full}) * {ratio}",
            )

    def test_ratio_with_page_alignment(self):
        """With page_size=128, swa_tokens = floor(full_tokens * ratio / 128) * 128."""
        available = 10_000_000
        _, _, config = self._run(available, ratio=0.5, page_size=128)
        full = config.full_max_total_num_tokens
        swa = config.swa_max_total_num_tokens
        self.assertEqual(full % 128, 0)
        self.assertEqual(swa % 128, 0)
        self.assertEqual(swa, (int(full * 0.5) // 128) * 128)

    def test_max_total_equals_full(self):
        """For Hybrid SWA, max_total_num_tokens = full_max_total_num_tokens."""
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.full_max_total_num_tokens)

    def test_constraint_respected(self):
        """full_tokens = constrained value after re-run."""
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(200, page_size=1)
        self.assertEqual(config.full_max_total_num_tokens, 200)
        self.assertEqual(config.swa_max_total_num_tokens, 100)

    def test_constraint_memory_within_budget(self):
        """After constraint, memory <= original budget."""
        available = 10_000_000
        mr, cfg, original = self._run(available, page_size=1)
        user_limit = original.full_max_total_num_tokens // 2
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(
                user_limit, mr.server_args.page_size
            )
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        original_used = _actual_memory_used(mr, original)
        self.assertAlmostEqual(used / original_used, 0.5, delta=0.01)

    def test_different_layer_counts(self):
        """Asymmetric full/swa layer counts."""
        available = 10_000_000
        mr, _, config = self._run(available, full_layers=24, swa_layers=8, ratio=0.5)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertEqual(
            config.swa_max_total_num_tokens,
            int(config.full_max_total_num_tokens * 0.5),
        )


_MIMO_BASE_ARGS = [
    "--tp-size",
    "16",
    "--trust-remote-code",
    "--device",
    "npu",
    "--mem-fraction-static",
    "0.85",
    "--swa-full-tokens-ratio",
    "0.95",
    "--reasoning-parser",
    "mimo",
    "--attention-backend",
    "ascend",
    "--disable-piecewise-cuda-graph",
    "--base-gpu-id",
    "0",
    "--cuda-graph-bs",
    "1",
    "2",
    "4",
    "8",
    "16",
    "--dp-size",
    "4",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--quantization",
    "modelslim",
    "--skip-server-warmup",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--enable-multi-layer-eagle",
    "--speculative-draft-model-quantization",
    "unquant",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
]

_MIMO_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "ASCEND_USE_FIA": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "800",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24669",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DEEPEP_BF16_DISPATCH": "0",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
}

_POOL_LOG_PATTERN = re.compile(
    r"full_max_total_num_tokens=(\d+).*swa_max_total_num_tokens=(\d+)"
)


class TestSwaFullTokensRatioServer(CustomTestCase):
    """Verify --swa-full-tokens-ratio on a real Hybrid SWA model (MiMo V2 Flash).

    Launches the server, sends an inference request, and prints the
    Full and SWA pool sizes from server logs.

    [Test Category] Parameter
    [Test Target] --swa-full-tokens-ratio
    [Scenario] S2: parameter accepted on Hybrid SWA model, pool sizes printed
    """

    model = MIMO_V2_FLASH_MODEL_PATH
    benchmark_tool = BENCHMARK_TOOL_DEFAULT

    def _capture_pool_sizes(self, stdout):
        """Extract full/swa pool sizes from server stdout."""
        for line in stdout.splitlines():
            m = _POOL_LOG_PATTERN.search(line)
            if m:
                return int(m.group(1)), int(m.group(2))
        return None, None

    def test_launch_and_print_pool_sizes(self):
        """S2: Launch MiMo V2 Flash, infer, and print Full/SWA pool sizes."""
        out_log_fd, out_log_path = tempfile.mkstemp(suffix=".log")
        err_log_fd, err_log_path = tempfile.mkstemp(suffix=".log")
        out_log_file = os.fdopen(out_log_fd, "w+", encoding="utf-8")
        err_log_file = os.fdopen(err_log_fd, "w+", encoding="utf-8")

        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_MIMO_BASE_ARGS,
            env=_MIMO_ENVS,
            return_stdout_stderr=(out_log_file, err_log_file),
        )
        try:
            # 1. Verify inference works
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
                timeout=120,
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Paris", resp.text)

            # 2. Extract and print Full/SWA pool sizes from server logs
            out_log_file.seek(0)
            stdout = out_log_file.read()
            full, swa = self._capture_pool_sizes(stdout)

            if full is not None and swa is not None:
                ratio = swa / full
                print(
                    f"\n  [SWA Pool Info] full={full}, swa={swa}, "
                    f"ratio={ratio:.4f} (config=0.95)"
                )
                self.assertAlmostEqual(
                    ratio,
                    0.95,
                    delta=0.01,
                    msg=f"SWA/Full ratio {ratio:.4f} deviates from config 0.95",
                )
            else:
                print(
                    "\n  [SWA Pool Info] Pool size log not found in server stdout. "
                    "Look for '[unified-memory-pool]' or similar log lines."
                )
        finally:
            kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.unlink(out_log_path)
            os.unlink(err_log_path)


if __name__ == "__main__":
    unittest.main()
