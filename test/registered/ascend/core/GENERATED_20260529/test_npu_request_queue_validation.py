import asyncio
import os
import re
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
    send_concurrent_generate_requests,
    send_generate_requests,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMaxQueuedRequests(CustomTestCase):
    """Test request queue throttling behavior on NPU.

    [Test Category] Core
    [Test Target] request queue validation, max queued requests
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests",
                "1",
                "--max-queued-requests",
                "1",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.3",
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

    def test_max_queued_requests_validation_with_serial_requests(self):
        status_codes = send_generate_requests(
            self.base_url,
            num_requests=10,
        )

        for status_code in status_codes:
            assert status_code == 200

    def test_max_queued_requests_validation_with_concurrent_requests(self):
        status_codes = asyncio.run(
            send_concurrent_generate_requests(self.base_url, num_requests=10)
        )
        self.assertLessEqual(status_codes.count(200), 2)

    def test_max_running_requests_and_max_queued_request_validation(self):
        rr_pattern = re.compile(r"#running-req:\s*(\d+)")
        qr_pattern = re.compile(r"#queue-req:\s*(\d+)")

        with open(STDERR_FILENAME) as lines:
            for line in lines:
                rr_match, qr_match = rr_pattern.search(line), qr_pattern.search(line)
                if rr_match:
                    assert int(rr_match.group(1)) <= 1
                if qr_match:
                    assert int(qr_match.group(1)) <= 1


if __name__ == "__main__":
    unittest.main()
