"""Tests for --disable-hybrid-swa-memory parameter.

When set on a Hybrid SWA model, this flag disables the independent SWA
memory pool and falls back to a unified pool. The server log shows
"path=SWA hybrid" only when the independent SWA pool is active.

Test strategy:
- Launch MiMo V2 Flash model twice (with and without the flag)
- Verify inference works both times
- Verify the pool type from server logs
"""

import os
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
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
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

_SWA_HYBRID_LOG_MARKER = "Use sliding window memory pool"


class TestDisableHybridSwaMemory(CustomTestCase):
    """Verify --disable-hybrid-swa-memory controls independent SWA pool vs unified pool.

    Launches MiMo V2 Flash twice:
    - Without the flag: independent SWA pool → log contains "path=SWA hybrid"
    - With the flag: unified pool → log does NOT contain "path=SWA hybrid"

    [Test Category] Parameter
    [Test Target] --disable-hybrid-swa-memory
    [Scenario] D1: independent SWA pool (default, no flag)
    [Scenario] D2: unified pool (--disable-hybrid-swa-memory)
    """

    model = MIMO_V2_FLASH_MODEL_PATH
    benchmark_tool = BENCHMARK_TOOL_DEFAULT

    def _launch_and_check_pool(self, extra_args, expect_swa_pool):
        """Launch server with given extra_args, verify inference and pool type.

        Args:
            extra_args: Additional CLI args (list or None).
            expect_swa_pool: True if independent SWA pool is expected,
                             False if unified pool is expected.
        """
        out_log_fd, out_log_path = tempfile.mkstemp(suffix=".log")
        err_log_fd, err_log_path = tempfile.mkstemp(suffix=".log")
        out_log_file = os.fdopen(out_log_fd, "w+", encoding="utf-8")
        err_log_file = os.fdopen(err_log_fd, "w+", encoding="utf-8")

        args = _MIMO_BASE_ARGS + (extra_args or [])
        label = "with --disable-hybrid-swa-memory" if extra_args else "without flag"

        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
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

            # 2. Verify pool type from server logs
            out_log_file.seek(0)
            stdout = out_log_file.read()
            has_swa_pool = _SWA_HYBRID_LOG_MARKER in stdout

            pool_type = "independent SWA pool" if has_swa_pool else "unified pool"
            print(f"\n  [Hybrid SWA Memory] {label}: pool_type={pool_type}")

            if expect_swa_pool:
                self.assertTrue(
                    has_swa_pool,
                    f"{label}: expected independent SWA pool but got unified pool. "
                    f"Log marker '{_SWA_HYBRID_LOG_MARKER}' not found in server stdout.",
                )
            else:
                self.assertFalse(
                    has_swa_pool,
                    f"{label}: expected unified pool but got independent SWA pool. "
                    f"Log marker '{_SWA_HYBRID_LOG_MARKER}' found in server stdout.",
                )
        finally:
            kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.unlink(out_log_path)
            os.unlink(err_log_path)

    def test_disable_hybrid_swa_memory(self):
        """D1+D2: Verify --disable-hybrid-swa-memory switches pool type.

        D1 (default): independent SWA pool → "path=SWA hybrid" in logs
        D2 (disabled): unified pool → no "path=SWA hybrid" in logs
        """
        # D1: Without --disable-hybrid-swa-memory → independent SWA pool
        self._launch_and_check_pool(extra_args=None, expect_swa_pool=True)

        # D2: With --disable-hybrid-swa-memory → unified pool
        self._launch_and_check_pool(
            extra_args=["--disable-hybrid-swa-memory"], expect_swa_pool=False
        )


if __name__ == "__main__":
    unittest.main()
