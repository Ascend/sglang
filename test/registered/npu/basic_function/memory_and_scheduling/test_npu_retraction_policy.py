import re
import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_5_9B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestRetractionPolicyLength(CustomTestCase):
    """Verify --retraction-policy=length (default) retracts the request with the
    longest input (tiebreaker) when KV cache is full and output lengths are equal.

    The length policy sorts running requests by (output_tokens, -input_tokens) and
    retracts the one with the smallest key. When output lengths are equal, the
    longer-input request has a smaller key and is retracted first.

    Assertions:
    - Both [LEN_SHORT] and [LEN_LONG] requests complete successfully (status=200)
    - "KV cache pool is full. Retract requests." found in server logs
    - The retraction log context includes [LEN_LONG] (the request being retracted)
    - Short-input response contains expected content ("Paris")
    - Server does not crash after retraction

    [Test Category] Parameter
    [Test Target] --retraction-policy
    """

    model = QWEN3_5_9B_WEIGHTS_PATH

    _LONG_INPUT_PREFIX = (
        "The history of artificial intelligence is a fascinating story. " * 20
    )

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.30",
        "--max-total-tokens",
        "3000",
        "--max-running-requests",
        "2",
        "--disable-radix-cache",
        "--enable-metrics",
        "--log-level",
        "debug",
    ]

    _OUT_LOG = "./tmp_retraction_out.log"
    _ERR_LOG = "./tmp_retraction_err.log"

    @classmethod
    def setUpClass(cls):
        out_log_file = open(cls._OUT_LOG, "w", encoding="utf-8")
        err_log_file = open(cls._ERR_LOG, "w", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._BASE_ARGS,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_length_policy_retraction(self):
        """Verify length retraction policy via server logs."""
        health_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(health_resp.status_code, 200)

        result_short = {}
        result_long = {}

        def _send_short_input():
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "[LEN_SHORT] The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4096,
                        "ignore_eos": True,
                    },
                },
                timeout=400,
            )
            result_short["status"] = resp.status_code
            result_short["text"] = resp.json().get("text", "")

        def _send_long_input():
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": (
                        f"[LEN_LONG] {self._LONG_INPUT_PREFIX}. "
                        f"The capital of France is"
                    ),
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4096,
                        "ignore_eos": True,
                    },
                },
                timeout=400,
            )
            result_long["status"] = resp.status_code

        t_short = threading.Thread(target=_send_short_input, daemon=True)
        t_long = threading.Thread(target=_send_long_input, daemon=True)
        t_short.start()
        t_long.start()

        t_short.join(timeout=400)
        t_long.join(timeout=400)
        self.assertFalse(t_short.is_alive(), "[LEN_SHORT] request timed out")
        self.assertFalse(t_long.is_alive(), "[LEN_LONG] request timed out")

        self.assertEqual(result_short.get("status"), 200)
        self.assertEqual(result_long.get("status"), 200)

        # Read server logs
        with open(self._OUT_LOG, "r", encoding="utf-8") as f:
            stdout = f.read()
        with open(self._ERR_LOG, "r", encoding="utf-8") as f:
            stderr = f.read()
        full_log = stdout + stderr

        # Verify retraction occurred
        retract_pattern = r"KV cache pool is full\. Retract requests\."
        retract_matches = list(re.finditer(retract_pattern, full_log))
        self.assertGreater(
            len(retract_matches),
            0,
            "No 'KV cache pool is full. Retract requests.' found in server logs. "
            "KV cache may not have filled up — retraction was never triggered.",
        )

        # Verify [LEN_LONG] appears in log context near retraction (the request being retracted)
        context_width = 2000
        context_start = max(0, retract_matches[0].start() - context_width)
        context_end = min(len(full_log), retract_matches[0].end() + context_width)
        context = full_log[context_start:context_end]
        self.assertIn(
            "[LEN_LONG]",
            context,
            "[LEN_LONG] not found in retraction context. "
            "Long-input request may not have been the one retracted.",
        )

        # Verify short-input output starts correctly
        self.assertIn(
            "Paris",
            result_short["text"],
            f"[LEN_SHORT] output missing 'Paris'. "
            f"Got: {result_short['text'][:200]}",
        )

        # Verify server is still alive after retraction
        self.assertIsNone(self.process.poll(), "Server crashed during retraction test")


class TestRetractionPolicyPriority(CustomTestCase):
    """Verify --retraction-policy=priority retracts lower-priority requests
    first, allowing high-priority requests to complete earlier.

    Test strategy: Start 2 low-priority requests (priority=0) to fill KV cache,
    then send high-priority request (priority=20). With max-running-requests=1,
    high-priority must preempt the running low-priority request.

    Assertions:
    - All 3 requests complete successfully (status=200)
    - "KV cache pool is full. Retract requests." found in server logs
    - [PRI_LOW] label found in retraction context (low-priority is retracted)
    - All outputs contain meaningful text content (> 100 chars)
    - Server does not crash after retraction

    [Test Category] Parameter
    [Test Target] --retraction-policy
    """

    model = QWEN3_5_9B_WEIGHTS_PATH

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.30",
        "--max-total-tokens",
        "2000",
        "--enable-priority-scheduling",
        "--priority-scheduling-preemption-threshold",
        "0",
        "--max-running-requests",
        "1",
        "--disable-radix-cache",
        "--retraction-policy",
        "priority",
        "--schedule-conservativeness",
        "0.0",
        "--enable-metrics",
        "--log-level",
        "debug",
    ]

    _LONG_PROMPT = (
        "Write a long essay about the history of artificial intelligence. "
        "Artificial intelligence is a fascinating field that has evolved significantly"
    )

    _OUT_LOG = "./tmp_retraction_priority_out.log"
    _ERR_LOG = "./tmp_retraction_priority_err.log"

    @classmethod
    def setUpClass(cls):
        out_log_file = open(cls._OUT_LOG, "w", encoding="utf-8")
        err_log_file = open(cls._ERR_LOG, "w", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._BASE_ARGS,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_priority_policy_retraction(self):
        """Verify priority retraction policy via server logs."""
        low1_result = {}
        low2_result = {}
        high_result = {}

        def _send_low_priority(label, result_dict):
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": f"[{label}] {self._LONG_PROMPT}",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4096,
                        "ignore_eos": True,
                    },
                    "priority": 0,
                },
                timeout=400,
            )
            result_dict["status"] = resp.status_code
            result_dict["text"] = resp.json().get("text", "")

        def _send_high_priority():
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": f"[PRI_HIGH] {self._LONG_PROMPT}",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4096,
                        "ignore_eos": True,
                    },
                    "priority": 20,
                },
                timeout=400,
            )
            high_result["status"] = resp.status_code
            high_result["text"] = resp.json().get("text", "")

        t1 = threading.Thread(
            target=_send_low_priority, args=("PRI_LOW1", low1_result), daemon=True
        )
        t2 = threading.Thread(
            target=_send_low_priority, args=("PRI_LOW2", low2_result), daemon=True
        )
        t1.start()
        t2.start()

        time.sleep(3)

        t_high = threading.Thread(target=_send_high_priority, daemon=True)
        t_high.start()

        t1.join(timeout=400)
        t2.join(timeout=400)
        t_high.join(timeout=400)
        self.assertFalse(t1.is_alive(), "[PRI_LOW1] request timed out")
        self.assertFalse(t2.is_alive(), "[PRI_LOW2] request timed out")
        self.assertFalse(t_high.is_alive(), "[PRI_HIGH] request timed out")

        self.assertEqual(low1_result.get("status"), 200)
        self.assertEqual(low2_result.get("status"), 200)
        self.assertEqual(high_result.get("status"), 200)

        # Read server logs
        with open(self._OUT_LOG, "r", encoding="utf-8") as f:
            stdout = f.read()
        with open(self._ERR_LOG, "r", encoding="utf-8") as f:
            stderr = f.read()
        full_log = stdout + stderr

        # Verify retraction occurred
        retract_pattern = r"KV cache pool is full\. Retract requests\."
        retract_matches = list(re.finditer(retract_pattern, full_log))
        self.assertGreater(
            len(retract_matches),
            0,
            "No 'KV cache pool is full. Retract requests.' found in server logs. "
            "KV cache may not have filled up — retraction was never triggered.",
        )

        # Verify a low-priority label appears in log context near retraction
        context_width = 2000
        context_start = max(0, retract_matches[0].start() - context_width)
        context_end = min(len(full_log), retract_matches[0].end() + context_width)
        context = full_log[context_start:context_end]
        has_low_in_context = "[PRI_LOW1]" in context or "[PRI_LOW2]" in context
        self.assertTrue(
            has_low_in_context,
            "Neither [PRI_LOW1] nor [PRI_LOW2] found in retraction context. "
            "Low-priority request may not have been the one retracted.",
        )

        # Verify all outputs contain meaningful content
        for label, result in [
            ("PRI_HIGH", high_result),
            ("PRI_LOW1", low1_result),
            ("PRI_LOW2", low2_result),
        ]:
            text = result.get("text", "")
            self.assertGreater(
                len(text),
                100,
                f"[{label}] output too short ({len(text)} chars): {text[:100]}",
            )

        # Verify server is still alive after retraction
        self.assertIsNone(self.process.poll(), "Server crashed during retraction test")


if __name__ == "__main__":
    unittest.main()