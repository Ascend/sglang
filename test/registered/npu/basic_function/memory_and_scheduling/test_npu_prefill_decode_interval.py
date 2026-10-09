import os
import re
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class TestPrefillDecodeInterval(CustomTestCase):
    """Testcase: Verify --prefill-decode-interval schedules prefills at the correct interval.

    [Test Category] Parameter
    [Test Target] --prefill-decode-interval
    """

    model = QWEN3_0_6B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.stderr_file = None
        cls.stderr_lines = []

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        if cls.stderr_file:
            try:
                os.unlink(cls.stderr_file)
            except OSError:
                pass

    def _read_stderr(self):
        """Background thread: read stderr lines into stderr_lines."""
        with open(self.stderr_file, "r") as f:
            for line in f:
                self.stderr_lines.append(line)

    def test_prefill_decode_interval(self):
        """Send 2 concurrent requests and assert prefill interval >= 400."""
        interval = 400

        # Create temp file for stderr capture
        fd, self.stderr_file = tempfile.mkstemp(suffix=".txt", prefix="sglang_stderr_")
        os.close(fd)

        stderr_fh = open(self.stderr_file, "w")

        # Set environment variable to enable forward iteration logging
        env = os.environ.copy()
        env["SGLANG_LOG_FORWARD_ITERS"] = "1"

        other_args = [
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--trust-remote-code",
            "--prefill-decode-interval",
            str(interval),
            "--max-running-requests",
            "16",
        ]

        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
            return_stdout_stderr=(None, stderr_fh),
        )

        # Start background thread to collect stderr
        reader_thread = threading.Thread(target=self._read_stderr, daemon=True)
        reader_thread.start()

        # Send 2 concurrent requests
        def send_request():
            return requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "Where is China",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 640,
                        "ignore_eos": True,
                    },
                },
                timeout=300,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(send_request) for _ in range(2)]
            for future in as_completed(futures):
                resp = future.result()
                self.assertEqual(resp.status_code, 200)

        # Wait a moment for stderr to flush
        time.sleep(2)

        # Close stderr handle to flush and allow reader thread to finish
        stderr_fh.close()
        reader_thread.join(timeout=5)

        # Parse stderr for "Prefill batch [N]" entries
        prefill_ids = []
        pattern = re.compile(r"Prefill batch\s+\[(\d+)\]")
        for line in self.stderr_lines:
            match = pattern.search(line)
            if match:
                prefill_ids.append(int(match.group(1)))

        self.assertGreaterEqual(
            len(prefill_ids),
            2,
            f"Expected at least 2 Prefill batch entries, got {len(prefill_ids)}. "
            f"Logs: {''.join(self.stderr_lines[-20:])}",
        )

        # Calculate the interval between successive Prefill batches
        # Reference: 411 - 10 = 401 >= 400, 1453 - 1052 = 401 >= 400
        for i in range(1, len(prefill_ids)):
            diff = prefill_ids[i] - prefill_ids[i - 1]
            self.assertGreaterEqual(
                diff,
                interval,
                f"Prefill batch interval {diff} < {interval} "
                f"(batch {prefill_ids[i-1]} -> {prefill_ids[i]})",
            )


if __name__ == "__main__":
    unittest.main()
