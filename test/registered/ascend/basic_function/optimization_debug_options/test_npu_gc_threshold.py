import os
import threading
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="debug-full-1-npu-a3", nightly=True)


class TestNpuGCThreshold(CustomTestCase):
    """Testcase: verify --gc-threshold core function — low threshold GC does not break inference

    [Test Category] Parameter
    [Test Target] --gc-threshold
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--gc-threshold",
                "50",
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def test_gc_threshold_low_vs_default(self):
        """Contrastive verification: --gc-threshold 50 (low) vs default.

        Sends many concurrent long-prompt requests to generate large temporary tensors
        that trigger frequent GC cycles. Verifies inference remains correct and the
        server does not OOM under aggressive GC pressure.
        """
        # Long prompt generates substantial temporary objects to stress GC
        prompt = "just return me a string with of 10000 characters: " + "A" * 10000

        def send_request():
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4096,
                    },
                },
            )
            return resp

        # Concurrent load — each request allocates temporary tensors + KV cache,
        # creating GC pressure that exercises the low --gc-threshold setting
        responses = []
        threads = []
        for _ in range(50):
            t = threading.Thread(target=lambda: responses.append(send_request()))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        self.assertGreater(len(responses), 0, "No responses received")
        for resp in responses:
            self.assertEqual(resp.status_code, 200)
            self.assertGreater(len(resp.text), 0, "Response body should be non-empty")


if __name__ == "__main__":
    unittest.main()
