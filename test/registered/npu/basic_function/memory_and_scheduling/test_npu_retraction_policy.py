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
    """Verify --retraction-policy=length (default) retracts short-output requests
    first when KV cache is full, allowing short requests to complete before long ones.

    Test strategy: Launch server with small KV cache (mem-fraction-static=0.3)
    and max-running-requests=1. Start a long-output request (4096 tokens) in a
    background thread to fill the KV cache, then send a short request (16 tokens).
    The short request triggers retraction of the long-output request under length
    policy. The short request should finish before the long request.

    Assertions:
    - Short and long requests both complete successfully (status=200)
    - Short request finishes before long request (retraction worked)
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
        "0.3",
        "--max-running-requests",
        "1",
        "--disable-radix-cache",
        "--enable-metrics",
        "--log-level",
        "debug",
    ]

    def test_length_policy_retraction(self):
        """R1: KV Cache triggers retraction, short request finishes before long request."""
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self._BASE_ARGS,
        )
        try:
            # Verify server is healthy
            health_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(health_resp.status_code, 200)

            long_result = {}

            def _send_long_request():
                resp = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": "The capital of France is",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 4096,
                            "ignore_eos": True,
                        },
                    },
                    timeout=120,
                )
                long_result["status"] = resp.status_code
                long_result["finished_at"] = time.time()

            # Start long request in background thread to fill KV cache
            t = threading.Thread(target=_send_long_request, daemon=True)
            t.start()
            time.sleep(2)  # Let long request start generating and fill KV cache

            # Send short request while long request is still running → triggers retraction
            short_resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "What is 1+1? Answer:",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                    },
                },
                timeout=120,
            )
            short_finished_at = time.time()
            self.assertEqual(short_resp.status_code, 200)
            self.assertIn("2", short_resp.text)

            # Wait for long request to finish
            t.join(timeout=120)
            self.assertFalse(t.is_alive(), "Long request timed out")
            self.assertEqual(long_result.get("status"), 200)

            # Short request should finish before long request (retraction succeeded)
            self.assertLess(
                short_finished_at,
                long_result["finished_at"],
                f"Short request should finish before long request under length policy: "
                f"short={short_finished_at:.1f} long={long_result['finished_at']:.1f}",
            )

            print(
                f"  [length retraction] short={short_finished_at:.2f} "
                f"long={long_result['finished_at']:.2f} "
                f"→ short_first={short_finished_at < long_result['finished_at']}"
            )

            # Verify server is still alive after retraction
            self.assertIsNone(process.poll(), "Server crashed during retraction test")
        finally:
            kill_process_tree(process.pid)


class TestRetractionPolicyPriority(CustomTestCase):
    """Verify --retraction-policy=priority retracts lower-priority requests
    first, allowing high-priority requests to complete earlier.

    Test strategy: Launch server with small KV cache (mem-fraction-static=0.3),
    max-running-requests=1, and priority-scheduling-preemption-threshold=0 to
    enable immediate preemption. Send 2 low-priority long-output requests to
    fill KV cache, then send a high-priority request. All 3 requests have large
    max_tokens to fill KV cache, so the finish order is determined purely by
    priority-based retraction. The high-priority request should finish before
    both low-priority requests.

    [Test Category] Parameter
    [Test Target] --retraction-policy
    """

    model = QWEN3_5_9B_WEIGHTS_PATH

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.3",
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
        """R2: 2 low-priority requests fill KV cache, high-priority finishes first.
        All three requests use the same prompt and max_new_tokens, differing only
        in priority. This ensures the finish order is determined purely by
        priority-based retraction, not by workload differences.
        """
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self._BASE_ARGS,
        )
        try:
            low1_result = {}
            low2_result = {}

            def _send_low_priority(request_id, result_dict):
                resp = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": f"low{request_id}: {self._LONG_PROMPT}",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 4096,
                            "ignore_eos": True,
                        },
                        "priority": 0,
                    },
                    timeout=120,
                )
                result_dict["status"] = resp.status_code
                result_dict["finished_at"] = time.time()

            # Start 2 low-priority requests to fill KV cache
            t1 = threading.Thread(
                target=_send_low_priority, args=(1, low1_result), daemon=True
            )
            t1.start()
            time.sleep(2)

            t2 = threading.Thread(
                target=_send_low_priority, args=(2, low2_result), daemon=True
            )
            t2.start()
            time.sleep(2)

            # Send high-priority request with same workload (should preempt and finish first)
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": f"high: {self._LONG_PROMPT}",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4096,
                        "ignore_eos": True,
                    },
                    "priority": 20,
                },
                timeout=120,
            )
            high_finished_at = time.time()
            self.assertEqual(resp.status_code, 200)

            # Wait for both low-priority requests to finish
            t1.join(timeout=120)
            t2.join(timeout=120)
            self.assertFalse(t1.is_alive(), "Low-priority-1 request timed out")
            self.assertFalse(t2.is_alive(), "Low-priority-2 request timed out")

            self.assertEqual(low1_result.get("status"), 200)
            self.assertEqual(low2_result.get("status"), 200)

            # High-priority should finish before both low-priority requests
            self.assertLess(
                high_finished_at,
                low1_result["finished_at"],
                f"High-priority should finish before low-priority-1: "
                f"high={high_finished_at:.1f} low1={low1_result['finished_at']:.1f}",
            )
            self.assertLess(
                high_finished_at,
                low2_result["finished_at"],
                f"High-priority should finish before low-priority-2: "
                f"high={high_finished_at:.1f} low2={low2_result['finished_at']:.1f}",
            )

            print(
                f"  [priority retraction] high={high_finished_at:.2f} "
                f"low1={low1_result['finished_at']:.2f} "
                f"low2={low2_result['finished_at']:.2f} "
                f"→ high_first={high_finished_at < low1_result['finished_at']}"
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
