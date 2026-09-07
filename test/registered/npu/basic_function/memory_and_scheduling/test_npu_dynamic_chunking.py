"""Tests for --enable-dynamic-chunking parameter.

The parameter enables dynamic adjustment of chunked prefill size based on
PP stage profiling, reducing pipeline bubbles. Only effective when pp_size > 1.

Two test scenarios:
- C1: pp_size > 1, dynamic chunking adjusts chunk size (core scenario)
"""

import os
import tempfile
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_4B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)
register_npu_ci(est_time=400, suite="", nightly=True)


class TestDynamicChunking(CustomTestCase):
    """Testcase: Verify --enable-dynamic-chunking behavior on PP and non-PP setups.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-chunking
    [Scenario] C1: pp_size > 1 enables dynamic chunking
    """

    model = QWEN3_4B_WEIGHTS_PATH

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--chunked-prefill-size",
        "1024",
    ]

    def _wait_for_log_content(self, log_file, timeout=30):
        """Poll until log file has non-empty content, then return it.

        Same pattern as TestNPULoggingBase.wait_for_log_content.
        """
        start_time = time.time()
        content = ""
        while time.time() - start_time < timeout:
            with open(log_file.name, "r", encoding="utf-8") as f:
                content = f.read()
            if content:
                break
            time.sleep(0.5)
        return content

    def test_dynamic_chunking_pp_size_two(self):
        """C1: pp_size=2 + --enable-dynamic-chunking."""
        out_log_file = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", delete=False, suffix=".log"
        )
        err_log_file = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", delete=False, suffix=".log"
        )
        out_log_path = out_log_file.name
        err_log_path = err_log_file.name

        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self._BASE_ARGS
            + [
                "--enable-dynamic-chunking",
                "--pp-size",
                "2",
                "--tp-size",
                "1",
            ],
            return_stdout_stderr=(out_log_file, err_log_file),
        )
        try:
            # 1. Short input: verify basic inference works
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
                timeout=60,
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Paris", resp.text)

            # 2. Long input: triggers chunked prefill with dynamic chunk sizing
            long_text = (
                "The history of artificial intelligence is a fascinating story. " * 100
            )
            long_resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": long_text,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
                timeout=120,
            )
            self.assertEqual(long_resp.status_code, 200)
            self.assertGreater(len(long_resp.json().get("text", "")), 0)

            # 3. Log assertions
            stdout = self._wait_for_log_content(out_log_file, timeout=30)

            self.assertIn(
                "[PP Dynamic Chunk]",
                stdout,
                "Dynamic chunking log not found in server output. "
                "Possible causes: profiling failed or dynamic chunking was disabled.",
            )
            self.assertIn(
                "Predictor ready",
                stdout,
                "Dynamic chunking predictor not ready. "
                "Profiling may have failed (check for 'Failed to profile' in logs).",
            )
            self.assertNotIn(
                "Failed to profile",
                stdout,
                "Dynamic chunking profiling failed. "
                "Check server logs for the exception that caused the fallback.",
            )
            self.assertNotIn(
                "Dynamic chunking will be disabled",
                stdout,
                "Dynamic chunking was disabled due to profiling failure. "
                "Inference used static chunked_prefill_size instead.",
            )
        finally:
            kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.unlink(out_log_path)
            os.unlink(err_log_path)


if __name__ == "__main__":
    unittest.main()
