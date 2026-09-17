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
    - Two concurrent requests with equal max_new_tokens (512) and different
      input lengths → KV fills → length policy retracts longer-input request
      first (longer input → smaller key in (output, -input) tiebreaker).

    Assertions:
    - "KV cache pool is full. Retract requests." in server logs
    - Both outputs contain expected content
    - Long-input has more retractions than short-input (length policy)

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
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.31",
        "--max-total-tokens",
        "1152",
        "--trust-remote-code",
        "--enable-metrics",
        "--log-level",
        "debug",
    ]

    _OUT_LOG = "./tmp_retraction_length_out.log"
    _ERR_LOG = "./tmp_retraction_length_err.log"

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
            device="npu",
            env={"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "128"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._out_log_file.close()
        cls._err_log_file.close()

    def _read_logs(self):
        """Flush and read server logs without closing handles,
        so background dump threads can continue writing."""
        self._out_log_file.flush()
        self._err_log_file.flush()
        with open(self._OUT_LOG, "r", encoding="utf-8") as f:
            stdout = f.read()
        with open(self._ERR_LOG, "r", encoding="utf-8") as f:
            stderr = f.read()
        return stdout + stderr

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
                        "max_new_tokens": 512,
                        "min_new_tokens": 512,
                    },
                    "rid": "short-req",
                },
            )
            result_short["status"] = resp.status_code
            result_short["text"] = resp.json().get("text", "")
            result_short["e2e"] = resp.json()["meta_info"]["e2e_latency"]
            result_short["retractions"] = resp.json()["meta_info"]["num_retractions"]

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
                        "max_new_tokens": 512,
                        "min_new_tokens": 512,
                    },
                    "rid": "long-req",
                },
            )
            result_long["status"] = resp.status_code
            result_long["text"] = resp.json().get("text", "")
            result_long["e2e"] = resp.json()["meta_info"]["e2e_latency"]
            result_long["retractions"] = resp.json()["meta_info"]["num_retractions"]

        t_short = threading.Thread(target=_send_short, daemon=True)
        t_long = threading.Thread(target=_send_long, daemon=True)
        t_short.start()
        t_long.start()
        t_short.join()
        t_long.join()

        self.assertEqual(result_short.get("status"), 200)
        self.assertEqual(result_long.get("status"), 200)

        # Read logs
        full_log = self._read_logs()

        # Assert 1: KV cache was full and retraction was triggered
        retract_pattern = "KV cache pool is full. Retract requests."
        self.assertIn(
            retract_pattern,
            full_log,
            "No 'KV cache pool is full. Retract requests.' found in server logs. ",
        )

        # Assert 2: both requests completed correctly
        self.assertIn(
            "Paris",
            result_short["text"],
            f"Short output missing 'Paris'. Got: {result_short['text'][:200]}",
        )
        self.assertIn(
            "Paris",
            result_long.get("text", ""),
            f"Long output missing 'Paris'. Got: {result_long.get('text', '')[:200]}",
        )

        # Assert 3: length policy retracted long-input request
        self.assertGreater(
            result_long["retractions"],
            result_short["retractions"],
            f"Long-input retractions ({result_long['retractions']}) should exceed "
            f"short-input ({result_short['retractions']})",
        )


class TestRetractionPolicyPriority(CustomTestCase):
    """Verify --retraction-policy=priority works with priority scheduling.

    Strategy:
    - Send low-priority request first to occupy KV cache, then send
      high-priority request which should preempt and finish first.

    Assertions:
    - Both requests complete (status=200)
    - "KV cache pool is full. Retract requests." in server logs
    - low-priority has more retractions than High-priority (priority policy)

    [Test Category] Parameter
    [Test Target] --retraction-policy
    """

    model = QWEN3_5_9B_WEIGHTS_PATH

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.31",
        "--max-total-tokens",
        "1152",
        "--trust-remote-code",
        "--retraction-policy",
        "priority",
        "--enable-priority-scheduling",
        "--schedule-conservativeness",
        "0.0",
        "--disable-priority-preemption",
        "--log-level",
        "debug",
    ]

    _LONG_INPUT_PREFIX = (
        "The history of artificial intelligence is a fascinating story. " * 10
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
            device="npu",
            env={"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "128"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._out_log_file.close()
        cls._err_log_file.close()

    def _read_logs(self):
        """Flush and read server logs without closing handles,
        so background dump threads can continue writing."""
        self._out_log_file.flush()
        self._err_log_file.flush()
        with open(self._OUT_LOG, "r", encoding="utf-8") as f:
            stdout = f.read()
        with open(self._ERR_LOG, "r", encoding="utf-8") as f:
            stderr = f.read()
        return stdout + stderr

    def test_priority_policy_retraction(self):
        low_result = {}
        high_result = {}

        def _send_low():
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": (
                        f"[LEN_LONG] {self._LONG_INPUT_PREFIX}. "
                        f"The capital of France is"
                    ),
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 512,
                        "min_new_tokens": 512,
                    },
                    "priority": 0,
                    "rid": "low-req",
                },
            )
            low_result["status"] = resp.status_code
            low_result["text"] = resp.json().get("text", "")
            low_result["e2e"] = resp.json()["meta_info"]["e2e_latency"]
            low_result["retractions"] = resp.json()["meta_info"]["num_retractions"]

        def _send_high():
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": (
                        f"[LEN_LONG] {self._LONG_INPUT_PREFIX}. "
                        f"The capital of France is"
                    ),
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 512,
                        "min_new_tokens": 512,
                    },
                    "priority": 20,
                    "rid": "high-req",
                },
            )
            high_result["status"] = resp.status_code
            high_result["text"] = resp.json().get("text", "")
            high_result["e2e"] = resp.json()["meta_info"]["e2e_latency"]
            high_result["retractions"] = resp.json()["meta_info"]["num_retractions"]

        t_low = threading.Thread(target=_send_low, daemon=True)
        t_low.start()
        time.sleep(0.5)
        t_high = threading.Thread(target=_send_high, daemon=True)
        t_high.start()

        t_low.join()
        t_high.join()

        self.assertEqual(low_result.get("status"), 200)
        self.assertEqual(high_result.get("status"), 200)

        # Read logs
        full_log = self._read_logs()

        # Assert 1: KV cache was full and retraction was triggered
        retract_pattern = "KV cache pool is full. Retract requests."
        self.assertIn(
            retract_pattern,
            full_log,
            "No 'KV cache pool is full. Retract requests.' found in server logs. ",
        )

        # Assert 2: both requests completed correctly
        self.assertIn(
            "paris",
            low_result["text"].lower(),
            f"Low output unexpected: {low_result['text'][:200]}",
        )
        self.assertIn(
            "paris",
            high_result["text"].lower(),
            f"High output unexpected: {high_result['text'][:200]}",
        )

        # Assert 3: the priority policy revoked requests with low priority
        self.assertGreater(
            low_result["retractions"],
            high_result["retractions"],
            f"Long-input retractions ({low_result['retractions']}) should exceed "
            f"short-input ({high_result['retractions']})",
        )


if __name__ == "__main__":
    unittest.main()
