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
    """Verify --retraction-policy=length (default) retracts the longer-input
    request when KV cache is full and output lengths are equal.

    Strategy:
    - --max-total-tokens=3000 caps KV cache to ~3000 tokens (Qwen3.5-9B on
      61GB NPU gives ~130K tokens without the cap, impossible to fill)
    - max-running-requests=2 so retraction has multiple requests to choose from
    - Two concurrent requests, both 4096 output tokens (ignore_eos), different
      input lengths → KV cache fills → length policy retracts [LEN_LONG]
      (longer input → smaller key in tiebreaker)

    Assertions:
    - Both requests complete (status=200)
    - "KV cache pool is full. Retract requests." in server logs
    - [LEN_LONG] label found near retraction log (proves long-input retracted)
    - Short-input output contains "Paris"
    - Server alive after test

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
        "--log-level",
        "debug",
    ]

    _OUT_LOG = "./tmp_retraction_out.log"
    _ERR_LOG = "./tmp_retraction_err.log"

    @classmethod
    def setUpClass(cls):
        cls._out_log_file = open(cls._OUT_LOG, "w", encoding="utf-8")
        cls._err_log_file = open(cls._ERR_LOG, "w", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._BASE_ARGS,
            return_stdout_stderr=(cls._out_log_file, cls._err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._out_log_file.close()
        cls._err_log_file.close()

    def test_length_policy_retraction(self):
        health_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(health_resp.status_code, 200)

        result_short = {}
        result_long = {}

        def _send_short():
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

        def _send_long():
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

        t_short = threading.Thread(target=_send_short, daemon=True)
        t_long = threading.Thread(target=_send_long, daemon=True)
        t_short.start()
        t_long.start()

        t_short.join(timeout=400)
        t_long.join(timeout=400)
        self.assertFalse(t_short.is_alive(), "[LEN_SHORT] request timed out")
        self.assertFalse(t_long.is_alive(), "[LEN_LONG] request timed out")

        self.assertEqual(result_short.get("status"), 200)
        self.assertEqual(result_long.get("status"), 200)

        # Read logs (flushed by close in tearDownClass, but also accessible now)
        with open(self._OUT_LOG, "r", encoding="utf-8") as f:
            stdout = f.read()
        with open(self._ERR_LOG, "r", encoding="utf-8") as f:
            stderr = f.read()
        full_log = stdout + stderr

        #
        # Retraction evidence
        #
        retract_pattern = r"KV cache pool is full\. Retract requests\."
        retract_matches = list(re.finditer(retract_pattern, full_log))
        self.assertGreater(
            len(retract_matches),
            0,
            "Retraction log NOT found. KV cache may not have filled up "
            "(--max-total-tokens may not be taking effect on this NPU).",
        )

        # Verify [LEN_LONG] (the longer-input request) is in retraction context
        context_start = max(0, retract_matches[0].start() - 2000)
        context_end = min(len(full_log), retract_matches[0].end() + 2000)
        context = full_log[context_start:context_end]
        self.assertIn(
            "[LEN_LONG]",
            context,
            "[LEN_LONG] not in retraction context. "
            "Expected longer-input request to be the one retracted.",
        )

        #
        # Output content checks
        #
        self.assertIn("Paris", result_short["text"],
                      f"Short output missing 'Paris'. Got: {result_short['text'][:200]}")
        self.assertIsNone(self.process.poll(),
                          "Server crashed during retraction test")


class TestRetractionPolicyPriority(CustomTestCase):
    """Verify --retraction-policy=priority retracts lower-priority requests
    when KV cache is full.

    Strategy:
    - --max-total-tokens=2000 caps KV cache so it fills quickly
    - Start 2 low-priority requests (priority=0) concurrently to fill KV cache
    - With retraction-policy=priority, the scheduler should retract a
      low-priority request when KV cache is full

    Assertions:
    - All 3 requests complete (status=200)
    - "KV cache pool is full. Retract requests." in server logs
    - [PRI_LOW] label found near retraction log (proves low-priority retracted)
    - All outputs > 100 chars
    - Server alive after test

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
        "--max-running-requests",
        "2",
        "--disable-radix-cache",
        "--retraction-policy",
        "priority",
        "--enable-priority-scheduling",
        "--priority-scheduling-preemption-threshold",
        "0",
        "--schedule-conservativeness",
        "0.0",
        "--log-level",
        "debug",
    ]

    _LONG_PROMPT = (
        "Write a long essay about the history of artificial intelligence. "
        "Artificial intelligence is a fascinating field"
    )

    _OUT_LOG = "./tmp_retraction_priority_out.log"
    _ERR_LOG = "./tmp_retraction_priority_err.log"

    @classmethod
    def setUpClass(cls):
        cls._out_log_file = open(cls._OUT_LOG, "w", encoding="utf-8")
        cls._err_log_file = open(cls._ERR_LOG, "w", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._BASE_ARGS,
            return_stdout_stderr=(cls._out_log_file, cls._err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._out_log_file.close()
        cls._err_log_file.close()

    def test_priority_policy_retraction(self):
        low1_result = {}
        low2_result = {}
        high_result = {}

        def _send_low(label, result_dict):
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

        def _send_high():
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

        #
        # 1. Start 2 low-priority requests to fill KV cache
        #
        t1 = threading.Thread(target=_send_low, args=("PRI_LOW1", low1_result), daemon=True)
        t2 = threading.Thread(target=_send_low, args=("PRI_LOW2", low2_result), daemon=True)
        t1.start()
        t2.start()
        time.sleep(3)

        #
        # 2. Send high-priority request
        #
        t_high = threading.Thread(target=_send_high, daemon=True)
        t_high.start()

        t1.join(timeout=400)
        t2.join(timeout=400)
        t_high.join(timeout=400)
        self.assertFalse(t1.is_alive(), "[PRI_LOW1] timed out")
        self.assertFalse(t2.is_alive(), "[PRI_LOW2] timed out")
        self.assertFalse(t_high.is_alive(), "[PRI_HIGH] timed out")

        self.assertEqual(low1_result.get("status"), 200)
        self.assertEqual(low2_result.get("status"), 200)
        self.assertEqual(high_result.get("status"), 200)

        #
        # 3. Verify retraction via logs
        #
        with open(self._OUT_LOG, "r", encoding="utf-8") as f:
            stdout = f.read()
        with open(self._ERR_LOG, "r", encoding="utf-8") as f:
            stderr = f.read()
        full_log = stdout + stderr

        retract_pattern = r"KV cache pool is full\. Retract requests\."
        retract_matches = list(re.finditer(retract_pattern, full_log))
        self.assertGreater(
            len(retract_matches),
            0,
            "Retraction log NOT found. KV cache may not have filled up "
            "(--max-total-tokens may not be taking effect on this NPU).",
        )

        # Low-priority labels should appear near retraction context
        context_start = max(0, retract_matches[0].start() - 2000)
        context_end = min(len(full_log), retract_matches[0].end() + 2000)
        context = full_log[context_start:context_end]
        has_low = "[PRI_LOW1]" in context or "[PRI_LOW2]" in context
        self.assertTrue(
            has_low,
            "[PRI_LOW*] not in retraction context. "
            "Expected low-priority request to be the one retracted.",
        )

        #
        # 4. Output content checks
        #
        for label, result in [
            ("PRI_HIGH", high_result),
            ("PRI_LOW1", low1_result),
            ("PRI_LOW2", low2_result),
        ]:
            text = result.get("text", "")
            self.assertGreater(len(text), 100,
                               f"[{label}] output too short ({len(text)}): {text[:100]}")
        self.assertIsNone(self.process.poll(),
                          "Server crashed during retraction test")


if __name__ == "__main__":
    unittest.main()