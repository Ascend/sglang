"""Tests for --swa-full-tokens-ratio parameter.

The parameter controls: SWA pool tokens = Full pool tokens * ratio.
Only effective on Hybrid SWA models (Llama-4-Scout, DeepSeek V4, MiMo, Inkling, etc.).

Test strategy:
- Server test: launch a real Hybrid SWA model, verify inference and print pool sizes
"""

import os
import re
import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


# Model path: Llama-4-Scout-17B-16E-Instruct from the shared dzc mount
_MODEL_PATH = (
    "/mnt/share/dzc/data/modelscope/hub/models/meta-llama/Llama-4-Scout-17B-16E-Instruct"
)

# Launch args: aligned with the command provided by dev
_BASE_ARGS = [
    "--tp-size",
    4,
    "--context-length",
    8192,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static",
    0.9,
    "--disable-radix-cache",
    "--swa-full-tokens-ratio",
    0.3,
    "--base-gpu-id",
    0,
]

# Env vars: keep only the essentials; drop deepep / large HCCL buffer, etc.
_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "ASCEND_USE_FIA": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TORCH_COMPILE_DISABLE": "1",         # added
    "SGLANG_ENABLE_TORCH_COMPILE": "0",   # added
}

_POOL_LOG_PATTERN = re.compile(
    r"Use sliding window memory pool. full_layer_tokens=(\d+).*swa_layer_tokens=(\d+)"
)


class TestSwaFullTokensRatioServer(TestNpuAccuracyTestCaseBase):
    """Verify --swa-full-tokens-ratio on a real Hybrid SWA model (Llama-4-Scout).

    Launches the server, sends an inference request, and prints the
    Full and SWA pool sizes from server logs.

    [Test Category] Parameter
    [Test Target] --swa-full-tokens-ratio
    """

    model = _MODEL_PATH
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    other_args = _BASE_ARGS
    envs = _ENVS
    accuracy = 0.70
    datasets = ["gsm8k"]
    few_shot_num = 5
    generation_config = {
        "max_tokens": 2048,
        "temperature": 1.0,
    }
    max_concurrency = 64
    output_len = 2048

    @classmethod
    def setUpClass(cls):
        cls.out_log_file_name = "./tmp_out_log.txt"
        cls.err_log_file_name = "./tmp_err_log.txt"
        cls.out_log_file = open(cls.out_log_file_name, "w+", encoding="utf-8")
        cls.err_log_file = open(cls.err_log_file_name, "w+", encoding="utf-8")

        import sglang.test.ascend.e2e.test_npu_accuracy_utils as base_module

        original_popen = base_module.popen_launch_server

        def _patched_popen(*args, **kwargs):
            kwargs["return_stdout_stderr"] = (cls.out_log_file, cls.err_log_file)
            return original_popen(*args, **kwargs)

        base_module.popen_launch_server = _patched_popen
        try:
            super().setUpClass()
        finally:
            base_module.popen_launch_server = original_popen

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove(cls.out_log_file_name)
        os.remove(cls.err_log_file_name)

    def _capture_pool_sizes(self, stdout):
        """Extract full/swa pool sizes from server stdout."""
        for line in stdout.splitlines():
            m = _POOL_LOG_PATTERN.search(line)
            if m:
                return int(m.group(1)), int(m.group(2))
        return None, None

    def test_launch_and_print_pool_sizes(self):
        """Launch Llama-4-Scout, infer, and print Full/SWA pool sizes."""
        self.run_accuracy()

        self.out_log_file.seek(0)
        self.err_log_file.seek(0)
        stdout = self.out_log_file.read() + self.err_log_file.read()
        self.assertTrue(len(stdout) > 0)
        full, swa = self._capture_pool_sizes(stdout)

        self.assertIsNotNone(
            full,
            "Pool size log not found in server stdout. "
            "Look for 'Use sliding window memory pool' in server logs.",
        )
        self.assertIsNotNone(
            swa,
            "Pool size log not found in server stdout. "
            "Look for 'Use sliding window memory pool' in server logs.",
        )
        # Calculate the expected SWA value . --swa-full-tokens-ratio: 0.3  page_size: 128
        expected_swa = int(full * 0.3) // 128 * 128
        print(
            f"\n  [SWA Pool Info] full={full}, swa={swa}, "
            f"expected_swa={expected_swa}"
        )
        self.assertEqual(
            swa,
            expected_swa,
            f"SWA pool size mismatch: swa={swa}, expected_swa={expected_swa}",
        )


if __name__ == "__main__":
    unittest.main()
