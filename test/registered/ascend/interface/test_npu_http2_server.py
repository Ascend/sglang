import subprocess
import unittest

import granian  # noqa: F401
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

_HAS_GRANIAN = True

register_npu_ci(est_time=100, suite="full-1-npu-a3", nightly=True)


class TestHTTP2Server(CustomTestCase):
    """Test HTTP/2 server (Granian) with basic OpenAI-compatible endpoints.
    Verifies that --enable-http2 launches successfully and serves requests
    via both HTTP/1.1 and HTTP/2 (h2c).

    [Test Category] Parameter
    [Test Target] --enable-http2
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-http2",
                "--attention-backend",
                "ascend",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_health(self):
        resp = requests.get(f"{self.base_url}/health")
        self.assertEqual(resp.status_code, 200)

    def test_get_model_info(self):
        resp = requests.get(f"{self.base_url}/get_model_info")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("model_path", resp.json())

    def test_completion(self):
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": "The capital of France is",
                "max_tokens": 8,
                "temperature": 0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("choices", data)
        self.assertGreater(len(data["choices"][0]["text"]), 0)

    def test_chat_completion(self):
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("choices", data)
        self.assertGreater(len(data["choices"][0]["message"]["content"]), 0)

    def test_h2c_with_curl(self):
        """Verify the server actually speaks HTTP/2 via h2c."""
        result = subprocess.run(
            [
                "curl",
                "--http2-prior-knowledge",
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_version}",
                f"{self.base_url}/health",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(
            result.stdout.strip(), "2", "Server should respond with HTTP/2"
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
