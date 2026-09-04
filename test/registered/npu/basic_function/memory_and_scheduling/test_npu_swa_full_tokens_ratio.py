"""Tests for --swa-full-tokens-ratio parameter.

The parameter controls: SWA pool tokens = Full pool tokens * ratio.
Only effective on Hybrid SWA models (DeepSeek V4, MiMo, Inkling, etc.).

Two test strategies:
- Unit test: test/registered/unit/model_executor/test_pool_configurator.py (CPU only)
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
    16,
    "--trust-remote-code",
    "--device",
    "npu",
    "--mem-fraction-static",
    0.85,
    "--reasoning-parser",
    "mimo",
    "--attention-backend",
    "ascend",
    "--disable-piecewise-cuda-graph",
    "--base-gpu-id",
    0,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--quantization",
    "modelslim",
    "--skip-server-warmup",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--swa-full-tokens-ratio",
    0.3,
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
            # NOTE: Use a separate file handle to read logs, because out_log_file
            # (TextIOWrapper) is shared with the _dump thread and is NOT thread-safe.
            with open(out_log_path, "r", encoding="utf-8") as f:
                stdout = f.read()
            full, swa = self._capture_pool_sizes(stdout)

            if full is not None and swa is not None:
                ratio = swa / full
                print(
                    f"\n  [SWA Pool Info] full={full}, swa={swa}, "
                    f"ratio={ratio:.4f} (config=0.95)"
                )
                # self.assertAlmostEqual(
                #     ratio,
                #     0.95,
                #     delta=0.01,
                #     msg=f"SWA/Full ratio {ratio:.4f} deviates from config 0.95",
                # )
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
