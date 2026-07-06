"""
End-to-end test for --enable-prefix-mm-cache parameter.
Tests the prefix multimodal cache functionality with encoder + language servers.
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SGLANG_MM_SKIP_COMPUTE_HASH"] = "True"

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)


_INLINE_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4b"
    "AAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGB"
    "cua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR"
    "3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="
)


class TestPrefixMMCacheE2E(CustomTestCase):
    """End-to-end test for --enable-prefix-mm-cache with encoder + language servers."""

    model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
    encoder_host = "127.0.0.1"
    encoder_port = "30000"
    encoder_url = f"http://{encoder_host}:{encoder_port}"
    language_host = "127.0.0.1"
    language_port = "30001"
    language_url = f"http://{language_host}:{language_port}"

    @classmethod
    def setUpClass(cls):
        """Start encoder server with --enable-prefix-mm-cache and language server."""
        # Start encoder server
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp-size",
            "1",
            "--port",
            cls.encoder_port,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
            "--enable-prefix-mm-cache",
        ]
        cls.process_encode = popen_launch_server(
            cls.model,
            base_url=cls.encoder_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
        )

        # Start language server
        language_args = [
            "--trust-remote-code",
            "--language-only",
            "--encoder-urls",
            cls.encoder_url,
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp-size",
            "1",
            "--port",
            cls.language_port,
            "--base-gpu-id",
            "1",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
        ]
        cls.process_language = popen_launch_server(
            cls.model,
            base_url=cls.language_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=language_args,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up processes."""
        for process in [cls.process_encode, cls.process_language]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")
        os.environ.pop("SGLANG_MM_SKIP_COMPUTE_HASH", None)

    def test_encoder_health_check(self):
        """Test that encoder server is healthy."""
        response = requests.get(f"{self.encoder_url}/health", timeout=10)
        self.assertEqual(response.status_code, 200)

    def test_language_server_health(self):
        """Test that language server is healthy."""
        response = requests.get(f"{self.language_url}/health", timeout=10)
        self.assertEqual(response.status_code, 200)

    def test_image_encoding_with_cache(self):
        """Test that image encoding works with prefix mm cache enabled."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _INLINE_IMAGE_URL},
                        },
                        {"type": "text", "text": "Describe the image briefly."},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 32,
        }

        # First request (cache miss)
        response1 = requests.post(
            f"{self.language_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        self.assertEqual(response1.status_code, 200)

        # Second request (should use cache)
        response2 = requests.post(
            f"{self.language_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        print("ufhgsfduioffbnc;oaihfninfioooooooojhffffffffffffffffffffhh")
        print(response1.json())
        print("ufhgsfduioffbnc;oaihfninfioooooooojhffffffffffffffffffffhh")
        print(response2.json())
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(response1.json()["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertGreater(response2.json()["usage"]["prompt_tokens_details"]["cached_tokens"], 0)

    def test_text_generation(self):
        """Test that text-only generation works."""
        response = requests.post(
            f"{self.language_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        generated_text = response.json().get("text", "")
        self.assertIn("Paris", generated_text)


if __name__ == "__main__":
    unittest.main()
