import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

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

register_npu_ci(est_time=400, suite="debug-full-1-npu-a3", nightly=True)

CONCURRENT_REQUESTS = 50


class TestNpuTokenizerBackendConcurrent(CustomTestCase):
    """Testcase: verify fastokens tokenization latency is lower than
    huggingface under concurrent load

    [Test Category] Parameter
    [Test Target] --tokenizer-backend
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    server_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]

    @classmethod
    def _send_concurrent(cls, n):
        def _request():
            with requests.Session() as session:
                return session.post(
                    f"{cls.base_url}/generate",
                    json={
                        "text": "The capital of France is",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 32,
                        },
                    },
                )

        start = time.time()
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [executor.submit(_request) for _ in range(n)]
            results = [f.result() for f in as_completed(futures)]
        elapsed = time.time() - start
        return results, elapsed

    @classmethod
    def _launch_server(cls, backend):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.server_args + ["--tokenizer-backend", backend],
        )

    def test_tokenizer_backend_concurrent(self):
        # Test with huggingface backend
        self._launch_server("huggingface")
        try:
            results_hf, elapsed_hf = self._send_concurrent(CONCURRENT_REQUESTS)
            for r in results_hf:
                self.assertEqual(r.status_code, 200)
                self.assertIn("Paris", r.text)
        finally:
            kill_process_tree(self.process.pid)

        # Test with fastokens backend
        self._launch_server("fastokens")
        try:
            results_ft, elapsed_ft = self._send_concurrent(CONCURRENT_REQUESTS)
            for r in results_ft:
                self.assertEqual(r.status_code, 200)
                self.assertIn("Paris", r.text)
        finally:
            kill_process_tree(self.process.pid)

        self.assertLess(
            elapsed_ft,
            elapsed_hf,
            f"Expected fastokens latency ({elapsed_ft:.2f}s) "
            f"< huggingface latency ({elapsed_hf:.2f}s)",
        )


if __name__ == "__main__":
    unittest.main()
