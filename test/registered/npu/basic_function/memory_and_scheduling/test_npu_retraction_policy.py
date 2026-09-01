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

# register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3-test-debug", nightly=True)


class TestRetractionPolicyLength(CustomTestCase):
    """Verify --retraction-policy=length (default) retracts short-output,
    long-input requests first when KV cache is full.

    Test strategy: Launch server with small KV cache (mem-fraction-static=0.08)
    and max-running-requests=1. Send a long-output request to fill the KV cache,
    then send a short request. The short request triggers retraction of the
    long-output request, and both complete successfully.

    [Test Category] Parameter
    [Test Target] --retraction-policy
    [Scenario] R1: length policy default behavior
    [Reference] test_npu_retract_decode.py, test_retraction_order.py
    """

    model = QWEN3_5_9B_WEIGHTS_PATH

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.08",
        "--max-running-requests",
        "1",
        "--disable-radix-cache",
        "--enable-metrics",
        "--log-level",
        "debug",
    ]

    def test_length_policy_retraction(self):
        """R1: KV Cache triggers retraction under length policy, service survives."""
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

            # Send a long-output request to fill KV cache
            long_resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 512,
                        "ignore_eos": True,
                    },
                },
                timeout=120,
            )
            self.assertEqual(long_resp.status_code, 200)

            # Send a short request after KV cache is full—triggers retraction
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
            self.assertEqual(short_resp.status_code, 200)
            self.assertIn("2", short_resp.text)

            # Verify server is still alive after retraction
            self.assertIsNone(process.poll(), "Server crashed during retraction test")
        finally:
            kill_process_tree(process.pid)


class TestRetractionPolicyPriority(CustomTestCase):
    """Verify --retraction-policy=priority retracts lower-priority requests
    first, allowing high-priority requests to complete earlier.

    Test strategy: Launch server with small KV cache (mem-fraction-static=0.08),
    max-running-requests=1, and priority-scheduling-preemption-threshold=0 to
    enable immediate preemption. Send 2 low-priority long-output requests to
    fill KV cache, then send a high-priority request. All 3 requests have large
    max_tokens to fill KV cache, so the finish order is determined purely by
    priority-based retraction. The high-priority request should finish before
    both low-priority requests.

    [Test Category] Parameter
    [Test Target] --retraction-policy
    [Scenario] R2: priority policy — high priority finishes first
    [Reference] retraction-policy.sh, test_npu_priority_scheduling.py
    """

    model = QWEN3_5_9B_WEIGHTS_PATH

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.08",
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

    def test_priority_policy_retraction(self):
        """R2: 2 low-priority requests fill KV cache, high-priority finishes first."""
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
                        "text": f"low{request_id}: Write a long essay about the history of France. "
                        "The history of France is a fascinating subject that spans",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 512,
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

            # Send high-priority request (should preempt and finish first)
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "high: What is 1+1? Answer:",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 512,
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


class TestRetractionPolicyValidation(CustomTestCase):
    """Verify --retraction-policy=priority requires --enable-priority-scheduling.

    [Test Category] Parameter
    [Test Target] --retraction-policy
    [Scenario] R3: priority policy without priority_scheduling raises error
    [Reference] server_args.py:9109-9112
    """

    model = QWEN3_5_9B_WEIGHTS_PATH

    def test_priority_without_scheduling_error(self):
        """R3: --retraction-policy priority without --enable-priority-scheduling raises error."""
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--retraction-policy",
                "priority",
            ],
        )
        try:
            # Wait for the process to exit with an error
            time.sleep(10)
            retcode = process.poll()
            self.assertIsNotNone(
                retcode,
                "Server should have failed to start with --retraction-policy priority alone",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()