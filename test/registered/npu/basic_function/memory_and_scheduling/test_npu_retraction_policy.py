import os
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
    fewest output tokens (and longest input as tiebreaker) when KV cache is full.

    The length policy sorts running requests by (output_tokens, -input_tokens) and
    retracts the one with the smallest key. With equal output, the tiebreaker
    (-input_tokens) retracts the request with the longer input first.

    Test strategy: Launch server with small KV cache (mem-fraction-static=0.30)
    and max-running-requests=2. Start two requests concurrently, both with the
    same output length (8192 tokens) but different input lengths:
      - Request A: short input (5 tokens)  → larger key → not retracted
      - Request B: long input (100 tokens)  → smaller key → retracted first
    Since both have the same output length, the tiebreaker retracts the
    longer-input request (B). Request A finishes first, then B is re-scheduled
    and finishes later.

    Assertions:
    - Both requests complete successfully (status=200)
    - Short-input request (A) finishes before long-input request (B)
    - Retraction log messages present in server output
    - Short-input response contains expected content ("Paris")
    - Server does not crash after retraction

    [Test Category] Parameter
    [Test Target] --retraction-policy
    """

    model = QWEN3_5_9B_WEIGHTS_PATH

    _OUT_LOG = "./tmp_length_out.txt"
    _ERR_LOG = "./tmp_length_err.txt"

    _LONG_INPUT_PREFIX = (
        "The history of artificial intelligence is a fascinating story. " * 20
    )

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.30",
        "--max-running-requests",
        "2",
        "--disable-radix-cache",
        "--enable-metrics",
        "--log-level",
        "debug",
    ]

    def test_length_policy_retraction(self):
        """R1: Length policy retracts the longer-input request via tiebreaker,
        so the short-input request finishes first.
        Verifies: retraction log, finish order, output content."""
        out_log = open(self._OUT_LOG, "w+", encoding="utf-8")
        err_log = open(self._ERR_LOG, "w+", encoding="utf-8")

        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self._BASE_ARGS,
            return_stdout_stderr=(out_log, err_log),
        )
        try:
            health_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(health_resp.status_code, 200)

            result_short = {}
            result_long = {}

            def _send_short_input():
                resp = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": "The capital of France is",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8192,
                            "ignore_eos": True,
                        },
                    },
                    timeout=400,
                )
                result_short["status"] = resp.status_code
                result_short["text"] = resp.json().get("text", "")
                result_short["finished_at"] = time.monotonic()

            def _send_long_input():
                resp = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": f"{self._LONG_INPUT_PREFIX}. The capital of France is",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8192,
                            "ignore_eos": True,
                        },
                    },
                    timeout=400,
                )
                result_long["status"] = resp.status_code
                result_long["finished_at"] = time.monotonic()

            t_short = threading.Thread(target=_send_short_input, daemon=True)
            t_long = threading.Thread(target=_send_long_input, daemon=True)
            t_short.start()
            t_long.start()

            t_short.join(timeout=400)
            t_long.join(timeout=400)
            self.assertFalse(t_short.is_alive(), "Short-input request timed out")
            self.assertFalse(t_long.is_alive(), "Long-input request timed out")

            self.assertEqual(result_short.get("status"), 200)
            self.assertEqual(result_long.get("status"), 200)

            # Length policy tiebreaker: long-input retracted first → finishes later
            self.assertLess(
                result_short["finished_at"],
                result_long["finished_at"],
                f"Short-input request should finish before long-input request "
                f"under length policy tiebreaker: "
                f"short_input={result_short['finished_at']:.1f} "
                f"long_input={result_long['finished_at']:.1f}",
            )

            # Verify retraction actually occurred via server logs
            out_log.seek(0)
            err_log.seek(0)
            server_logs = (out_log.read() + err_log.read()).lower()
            self.assertIn(
                "retract",
                server_logs,
                "No retraction event found in server logs. "
                "KV cache may not have filled up — retraction was never triggered.",
            )

            # Verify short-input output starts correctly
            self.assertIn(
                "Paris",
                result_short["text"],
                f"Short-input output missing 'Paris'. Got: {result_short['text'][:200]}",
            )

            # Verify server is still alive after retraction
            self.assertIsNone(process.poll(), "Server crashed during retraction test")

            print(
                f"  [length retraction] short={result_short['finished_at']:.2f} "
                f"long={result_long['finished_at']:.2f} "
                f"→ short_first={result_short['finished_at'] < result_long['finished_at']}"
            )
        finally:
            kill_process_tree(process.pid)
            out_log.close()
            err_log.close()
            os.remove(self._OUT_LOG)
            os.remove(self._ERR_LOG)


class TestRetractionPolicyPriority(CustomTestCase):
    """Verify --retraction-policy=priority retracts lower-priority requests
    first, allowing high-priority requests to complete earlier.

    Test strategy: Launch server with small KV cache (mem-fraction-static=0.30)
    and max-running-requests=1 to force preemption. Start 2 low-priority
    long-output requests (8192 tokens, priority=0) to fill the KV cache, then
    send a high-priority request (priority=20). With max-running-requests=1,
    the high-priority request must preempt the running low-priority request
    to be scheduled. The retracted low-priority request is restarted from
    scratch, so the high-priority request finishes first.

    Note: max-running-requests=1 means only 1 request runs at a time, so the
    retraction policy has no "choice" to make (only 1 running request). However,
    the retraction policy is still invoked during preemption to retract the
    running low-priority request. The key verification is that priority-based
    preemption works correctly: the high-priority request finishes before both
    low-priority requests despite arriving later.

    Assertions:
    - All 3 requests complete successfully (status=200)
    - High-priority finishes before both low-priority requests
    - Retraction log messages present in server output
    - All outputs contain meaningful text content
    - Server does not crash after retraction

    [Test Category] Parameter
    [Test Target] --retraction-policy
    """

    model = QWEN3_5_9B_WEIGHTS_PATH

    _OUT_LOG = "./tmp_priority_out.txt"
    _ERR_LOG = "./tmp_priority_err.txt"

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.30",
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

    def test_priority_policy_retraction(self):
        """R2: Low-priority requests start first to fill KV cache, high-priority
        preempts and finishes first despite arriving later.

        Verifies: retraction log, finish order (high < low1, high < low2),
        output content on all 3 requests."""
        out_log = open(self._OUT_LOG, "w+", encoding="utf-8")
        err_log = open(self._ERR_LOG, "w+", encoding="utf-8")

        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self._BASE_ARGS,
            return_stdout_stderr=(out_log, err_log),
        )
        try:
            low1_result = {}
            low2_result = {}
            high_result = {}

            def _send_low_priority(request_id, result_dict):
                resp = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": f"low{request_id}: {self._LONG_PROMPT}",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8192,
                            "ignore_eos": True,
                        },
                        "priority": 0,
                    },
                    timeout=400,
                )
                result_dict["status"] = resp.status_code
                result_dict["text"] = resp.json().get("text", "")
                result_dict["finished_at"] = time.monotonic()

            def _send_high_priority():
                resp = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": f"high: {self._LONG_PROMPT}",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8192,
                            "ignore_eos": True,
                        },
                        "priority": 20,
                    },
                    timeout=400,
                )
                high_result["status"] = resp.status_code
                high_result["text"] = resp.json().get("text", "")
                high_result["finished_at"] = time.monotonic()

            # Start low-priority requests first to fill KV cache
            t1 = threading.Thread(
                target=_send_low_priority, args=(1, low1_result), daemon=True
            )
            t2 = threading.Thread(
                target=_send_low_priority, args=(2, low2_result), daemon=True
            )
            t1.start()
            t2.start()

            # Wait for KV cache to fill up and retraction to trigger
            time.sleep(3)

            # Send high-priority request - should preempt low-priority requests
            t_high = threading.Thread(target=_send_high_priority, daemon=True)
            t_high.start()

            # Wait for all to finish
            t1.join(timeout=400)
            t2.join(timeout=400)
            t_high.join(timeout=400)
            self.assertFalse(t1.is_alive(), "Low-priority-1 request timed out")
            self.assertFalse(t2.is_alive(), "Low-priority-2 request timed out")
            self.assertFalse(t_high.is_alive(), "High-priority request timed out")

            self.assertEqual(low1_result.get("status"), 200)
            self.assertEqual(low2_result.get("status"), 200)
            self.assertEqual(high_result.get("status"), 200)

            # High-priority should finish before both low-priority requests
            # despite starting later, proving priority retraction works
            self.assertLess(
                high_result["finished_at"],
                low1_result["finished_at"],
                f"High-priority should finish before low-priority-1: "
                f"high={high_result['finished_at']:.1f} "
                f"low1={low1_result['finished_at']:.1f}",
            )
            self.assertLess(
                high_result["finished_at"],
                low2_result["finished_at"],
                f"High-priority should finish before low-priority-2: "
                f"high={high_result['finished_at']:.1f} "
                f"low2={low2_result['finished_at']:.1f}",
            )

            # Verify retraction actually occurred via server logs
            out_log.seek(0)
            err_log.seek(0)
            server_logs = (out_log.read() + err_log.read()).lower()
            self.assertIn(
                "retract",
                server_logs,
                "No retraction event found in server logs. "
                "KV cache may not have filled up — retraction was never triggered.",
            )

            # Verify all outputs contain meaningful content (not empty, starts reasonably)
            for label, result in [
                ("high", high_result), ("low1", low1_result), ("low2", low2_result)
            ]:
                text = result.get("text", "")
                self.assertGreater(
                    len(text), 100,
                    f"{label}-priority output too short ({len(text)} chars): {text[:100]}",
                )

            # Verify server is still alive after retraction
            self.assertIsNone(process.poll(), "Server crashed during retraction test")

            print(
                f"  [priority retraction] high={high_result['finished_at']:.2f} "
                f"low1={low1_result['finished_at']:.2f} "
                f"low2={low2_result['finished_at']:.2f} "
                f"→ high_first={high_result['finished_at'] < low1_result['finished_at']}"
            )
        finally:
            kill_process_tree(process.pid)
            out_log.close()
            err_log.close()
            os.remove(self._OUT_LOG)
            os.remove(self._ERR_LOG)


if __name__ == "__main__":
    unittest.main()