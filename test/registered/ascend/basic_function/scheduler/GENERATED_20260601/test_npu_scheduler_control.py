import threading
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.abort_timeout_kit import AbortAllMixin, WaitingTimeoutMixin
from sglang.test.kits.pause_generation_kit import PauseResumeInPlaceMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestNPUSchedulerControl(AbortAllMixin, PauseResumeInPlaceMixin, CustomTestCase):
    """Test scheduler control on NPU.

    [Test Category] Scheduler
    [Test Target] Duplicate request ID handling, abort all, pause/resume generation
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--max-running-requests",
                8,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate_with_rid(self, rid, max_new_tokens=8):
        return requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "rid": rid,
            },
            timeout=30,
        )

    def test_duplicate_rid_sequential_ok(self):
        rid = "dup-rid-test-sequential"
        resp1 = self._generate_with_rid(rid)
        self.assertEqual(resp1.status_code, 200)
        self.assertNotIn("error", resp1.json())

        resp2 = self._generate_with_rid(rid)
        self.assertEqual(resp2.status_code, 200)
        self.assertNotIn("error", resp2.json())

    def test_duplicate_rid_concurrent_rejected(self):
        rid = "dup-rid-test-concurrent"
        results = {}

        def send(key, max_tokens):
            results[key] = self._generate_with_rid(rid, max_new_tokens=max_tokens)

        t1 = threading.Thread(target=send, args=("first", 512))
        t2 = threading.Thread(target=send, args=("second", 8))
        t1.start()
        time.sleep(0.1)
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        r1, r2 = results["first"], results["second"]
        self.assertTrue(
            r1.status_code == 400 or r2.status_code == 400,
            "One of the concurrent duplicate-rid requests should be rejected",
        )

        rejected = r2 if r2.status_code == 400 else r1
        self.assertIn("Duplicate request ID", rejected.json()["error"]["message"])

    def test_duplicate_rid_in_batch(self):
        rid = "dup-rid-batch"
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": ["Hello", "World"],
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "rid": [rid, rid],
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Duplicate request ID", response.json()["error"]["message"])

    def test_server_healthy_after_duplicate_rid(self):
        requests.post(
            f"{self.base_url}/generate",
            json={
                "text": ["Hello", "World"],
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                "rid": ["dup-health", "dup-health"],
            },
            timeout=30,
        )

        resp = requests.get(f"{self.base_url}/health", timeout=5)
        self.assertEqual(resp.status_code, 200)

        resp = self._generate_with_rid("after-dup-health")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text", resp.json())


class TestNPUAbortWithWaitingTimeout(WaitingTimeoutMixin, CustomTestCase):
    """Test abort with waiting timeout on NPU.

    [Test Category] Scheduler
    [Test Target] Request abort due to waiting timeout
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(0.001):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--max-running-requests=1",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestNPUAbortWithRunningTimeout(CustomTestCase):
    """Test abort with running timeout on NPU.

    [Test Category] Scheduler
    [Test Target] Request abort due to running timeout
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        with (
            envs.SGLANG_REQ_RUNNING_TIMEOUT.override(0.001),
            envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.override(False),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--skip-server-warmup",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_running_timeout(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Today is ",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 512,
                    "ignore_eos": True,
                },
            },
        )
        result = response.json()
        self.assertEqual(result["object"], "error")
        self.assertEqual(result["code"], 503)


if __name__ == "__main__":
    unittest.main()
