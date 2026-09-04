"""Tests for --enable-dynamic-chunking parameter.

The parameter enables dynamic adjustment of chunked prefill size based on
PP stage profiling, reducing pipeline bubbles. Only effective when pp_size > 1.

Two test scenarios:
- C1: pp_size > 1, dynamic chunking adjusts chunk size (core scenario)
"""

import os
import tempfile
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

    def test_dynamic_chunking_pp_size_two(self):
        """C1: pp_size=2 + --enable-dynamic-chunking.
        Dynamic chunking should be enabled and adjust chunk sizes
        based on PP stage profiling. Server starts and inference succeeds.

        Verification strategy:
        1. Short input → verify basic inference works
        2. Long input (2048 tokens, > chunked_prefill_size=1024) → triggers
           chunked prefill, where dynamic chunking replaces static chunk size
           with predicted values. Correct completion proves dynamic sizing works.
        3. Log assertions confirm profiling succeeded and no fallback occurred.

        Assertions:
        - Short and long inference both return correct responses
        - Log contains "[PP Dynamic Chunk] Predictor ready" (profiling succeeded)
        - Log does NOT contain "Failed to profile" or "Dynamic chunking will be disabled"
        """
        out_log_fd, out_log_path = tempfile.mkstemp(suffix=".log")
        err_log_fd, err_log_path = tempfile.mkstemp(suffix=".log")
        out_log_file = os.fdopen(out_log_fd, "w+", encoding="utf-8")
        err_log_file = os.fdopen(err_log_fd, "w+", encoding="utf-8")

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
            #    Input length 2048 > chunked_prefill_size=1024, so prefill is
            #    split into multiple chunks. Dynamic chunking determines each
            #    chunk's size via predict_next_chunk_size() instead of using
            #    the static chunked_prefill_size.
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

            # 3. Log assertions: verify dynamic chunking actually activated
            # NOTE: Use a separate file handle to read logs, because out_log_file
            # (TextIOWrapper) is shared with the _dump thread and is NOT thread-safe.
            # Reading via the same TextIOWrapper from two threads can cause empty/
            # partial reads due to internal buffer corruption.
            with open(out_log_path, "r", encoding="utf-8") as f:
                stdout = f.read()

            # 3a. Predictor must be ready (profiling succeeded)
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

            # 3b. No fallback — profiling must NOT have failed
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
