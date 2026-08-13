import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="debug-full-1-npu-a3", nightly=True)


class TestChatPriorityScheduling(CustomTestCase):
    """Testcase: priority functional behavior on /v1/chat/completions.

    The server runs with --max-running-requests 1 so exactly one request occupies
    the slot; a higher-priority request must preempt it (threshold=0 makes any
    positive difference trigger preemption).

    [Test Category] Parameter
    [Test Target] priority
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--attention-backend",
                "ascend",
                "--enable-priority-scheduling",
                "--max-running-requests",
                "1",
                "--priority-scheduling-preemption-threshold",
                "0",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, payload):
        return requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def _payload(self, **kwargs):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Write a very long story."}],
            "temperature": 0,
        }
        payload.update(kwargs)
        return payload

    def _wait_until_running(self, timeout=30):
        """Poll /v1/loads until at least one request is running."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                loads = requests.get(
                    f"{self.base_url}/loads", timeout=5
                ).json()
                if loads.get("num_running_reqs", 0) > 0:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        self.fail("no request became running within {timeout}s")

    def _occupy_slot(self, priority=0, max_tokens=500):
        """Send a long request that occupies the single running slot."""
        result = {}

        def run():
            response = self._post(
                self._payload(max_tokens=max_tokens, priority=priority)
            )
            result["status"] = response.status_code

        thread = threading.Thread(target=run)
        thread.start()
        # Wait until the request is actually scheduled and generating.
        self._wait_until_running()
        return thread, result

    def test_high_priority_completes_first(self):
        long_thread, long_result = self._occupy_slot(priority=0)

        start = time.time()
        response = self._post(
            self._payload(
                max_tokens=16,
                priority=10,
                messages=[{"role": "user", "content": "Say hi."}],
            )
        )
        high_elapsed = time.time() - start

        self.assertEqual(response.status_code, 200, response.text)
        # The preempted high-priority request should finish quickly even though
        # the low-priority long request is still occupying the slot.
        self.assertLess(high_elapsed, 10, "high-priority request was not preempted")
        self.assertTrue(
            long_thread.is_alive(),
            "low-priority long request should still be running",
        )

        long_thread.join()
        self.assertEqual(long_result.get("status"), 200)

    def test_same_priority_smoke(self):
        """Same priority: both requests complete successfully.

        NOTE: completion ORDER is not asserted — with --max-running-requests 1
        requests are strictly serialized, so order proves nothing about
        priority semantics (FCFS tie-breaking needs multiple concurrent
        wait-queue entries, i.e. max-running-requests > 1)."""
        results = []

        def run_slow():
            response = self._post(self._payload(max_tokens=100, priority=5))
            results.append(response.status_code)

        def run_fast():
            time.sleep(1)
            response = self._post(
                self._payload(
                    max_tokens=16,
                    priority=5,
                    messages=[{"role": "user", "content": "Say hi."}],
                )
            )
            results.append(response.status_code)

        slow_thread = threading.Thread(target=run_slow)
        fast_thread = threading.Thread(target=run_fast)
        slow_thread.start()
        fast_thread.start()
        slow_thread.join()
        fast_thread.join()

        self.assertEqual(results, [200, 200])

    def test_n2_priority(self):
        """n=2 sub-requests inherit the parent priority and both preempt."""
        long_thread, _ = self._occupy_slot(priority=0)

        response = self._post(
            self._payload(
                max_tokens=16,
                n=2,
                priority=10,
                messages=[{"role": "user", "content": "Say hi."}],
            )
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(len(data["choices"]), 2)
        self.assertTrue(
            long_thread.is_alive(),
            "both sub-requests should have preempted the low-priority request",
        )
        long_thread.join()

    def test_stream_priority(self):
        """Stream path: high-priority stream request preempts the low-priority slot."""
        long_thread, _ = self._occupy_slot(priority=0)

        start = time.time()
        response = self._post(
            self._payload(
                max_tokens=16,
                priority=10,
                stream=True,
                messages=[{"role": "user", "content": "Say hi."}],
            )
        )
        # Read until the first SSE chunk arrives (TTFT).
        first_chunk_time = None
        for line in response.iter_lines():
            if line and line.startswith(b"data: ") and line != b"data: [DONE]":
                first_chunk_time = time.time() - start
                break

        self.assertIsNotNone(first_chunk_time, "no SSE chunk received")
        self.assertLess(
            first_chunk_time,
            10,
            "high-priority stream request should preempt quickly",
        )
        self.assertTrue(
            long_thread.is_alive(),
            "low-priority long request should still be running",
        )
        long_thread.join()


class TestChatPrioritySilentlyIgnored(CustomTestCase):
    """Testcase: priority without --enable-priority-scheduling is silently ignored.

    [Test Category] Parameter
    [Test Target] priority (silent ignore on a plain server)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--attention-backend",
                "ascend",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_priority_silently_ignored(self):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "temperature": 0,
            "max_tokens": 16,
        }
        baseline = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ).json()

        payload["priority"] = 10
        with_priority = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ).json()

        self.assertEqual(
            baseline["choices"][0]["message"]["content"],
            with_priority["choices"][0]["message"]["content"],
            "priority should be silently ignored without --enable-priority-scheduling",
        )


class TestChatPriorityDisabled(CustomTestCase):
    """Testcase: priority behavior without --enable-priority-scheduling.

    [Test Category] Parameter
    [Test Target] priority (silent ignore / abort-on-priority-when-disabled)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--attention-backend",
                "ascend",
                "--abort-on-priority-when-disabled",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_priority_aborted_when_disabled(self):
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Say hi."}],
                "priority": 10,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn(
            "Using priority is disabled for this server", response.json()["error"]["message"]
        )


if __name__ == "__main__":
    unittest.main()
