"""Tests for --enable-dynamic-chunking parameter.

The parameter enables dynamic adjustment of chunked prefill size based on
PP stage profiling, reducing pipeline bubbles. Only effective when pp_size > 1.

One test scenarios:
- C1: pp_size > 1, dynamic chunking adjusts chunk size (core scenario)
"""

import os
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

register_npu_ci(est_time=400, suite="full-4-npu-a3", nightly=True)


class TestDynamicChunking(CustomTestCase):
    """Testcase: Verify --enable-dynamic-chunking behavior on pp_size > 1.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-chunking
    """

    model = QWEN3_4B_WEIGHTS_PATH

    _OUT_LOG = "./tmp_out_log.txt"
    _ERR_LOG = "./tmp_err_log.txt"

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--chunked-prefill-size",
        "1024",
        "--mem-fraction-static",
        "0.90",
    ]

    @classmethod
    def setUpClass(cls):
        cls.out_log_file = open(cls._OUT_LOG, "w+", encoding="utf-8")
        cls.err_log_file = open(cls._ERR_LOG, "w+", encoding="utf-8")

        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._BASE_ARGS
            + [
                "--enable-dynamic-chunking",
                "--pp-size",
                "4",
                "--tp-size",
                "1",
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove(cls._OUT_LOG)
        os.remove(cls._ERR_LOG)

    def test_dynamic_chunking_pp_size_two(self):
        """C1: pp_size=4 + --enable-dynamic-chunking.
        Dynamic chunking should be enabled and adjust chunk sizes
        based on PP stage profiling. Server starts and inference succeeds.
        """

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

        long_text = (
            "The history of artificial intelligence is a fascinating story. " * 100
            + "\n\nQuestion: What is the capital of France? Answer:"
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
        long_text_output = long_resp.json().get("text", "")
        self.assertIn(
            "Paris",
            long_text_output,
            f"Long input inference failed: expected 'Paris' in response, got: {long_text_output}",
        )

        # Log assertions: verify dynamic chunking actually activated
        self.out_log_file.seek(0)
        self.err_log_file.seek(0)
        stdout = self.out_log_file.read() + self.err_log_file.read()
        self.assertTrue(len(stdout) > 0)

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


if __name__ == "__main__":
    unittest.main()
