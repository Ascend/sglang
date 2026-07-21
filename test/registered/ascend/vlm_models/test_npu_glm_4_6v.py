"""
Test GLM-4.6V multimodal request parsing via curl.

Verifies that the GLM-4.6V model can correctly parse and respond to
multimodal (image + text) chat completion requests through the OpenAI-compatible
API endpoint, using curl commands for end-to-end validation.
"""

import base64
import io
import json
import subprocess
import unittest

from PIL import Image

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import GLM_4_6V_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=600,
    suite="full-1-npu-a3",
    nightly=True,
)


def _create_test_image_b64(width=320, height=240, color="red") -> str:
    """Create a solid-color test image and return its base64 data URL."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _curl_post(base_url: str, endpoint: str, data: dict) -> subprocess.CompletedProcess:
    """Send a POST request via curl and return the CompletedProcess."""
    return subprocess.run(
        [
            "curl",
            "-s",
            "-w",
            "\n%{http_code}",
            "-X",
            "POST",
            f"{base_url}{endpoint}",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(data),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestGLM46VMultimodalCurl(CustomTestCase):
    """Verify GLM-4.6V multimodal request parsing via curl.

    [Test Category] Model
    [Test Target] ZhipuAI/GLM-4.6V
    """

    model = GLM_4_6V_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--tp-size",
                "4",
                "--chunked-prefill-size",
                "16384",
                "--max-prefill-tokens",
                "150000",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "8",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.87",
                "--disable-cuda-graph",
                "--watchdog-timeout",
                "9000",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm45",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_health(self):
        """Verify server health endpoint."""
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                f"{self.base_url}/health",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.stdout.strip(), "200")

    def test_get_model_info(self):
        """Verify model info endpoint returns valid data."""
        result = subprocess.run(
            ["curl", "-s", f"{self.base_url}/get_model_info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        self.assertIn("model_path", data)

    def test_single_image_chat_completion(self):
        """Verify single image + text multimodal request via curl."""
        image_b64 = _create_test_image_b64(320, 240, "red")
        payload = {
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {
                            "type": "text",
                            "text": "What color is the image? Answer in one word.",
                        },
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0,
        }

        result = _curl_post(self.base_url, "/v1/chat/completions", payload)
        output = result.stdout.strip()

        # Last line is HTTP status code
        lines = output.rsplit("\n", 1)
        self.assertEqual(lines[-1], "200", f"Expected 200, got response:\n{output}")

        data = json.loads(lines[0])
        self.assertIn("choices", data)
        content = data["choices"][0]["message"]["content"]
        self.assertGreater(len(content), 0, "Empty response for image description")
        print(f"  [single_image] response: {content}")

    def test_multi_image_chat_completion(self):
        """Verify multi-image multimodal request via curl."""
        img1_b64 = _create_test_image_b64(320, 240, "red")
        img2_b64 = _create_test_image_b64(320, 240, "blue")
        payload = {
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img1_b64}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img2_b64}"},
                        },
                        {
                            "type": "text",
                            "text": "Describe the colors of the two images.",
                        },
                    ],
                }
            ],
            "max_tokens": 128,
            "temperature": 0,
        }

        result = _curl_post(self.base_url, "/v1/chat/completions", payload)
        output = result.stdout.strip()

        lines = output.rsplit("\n", 1)
        self.assertEqual(lines[-1], "200", f"Expected 200, got response:\n{output}")

        data = json.loads(lines[0])
        self.assertIn("choices", data)
        content = data["choices"][0]["message"]["content"]
        self.assertGreater(len(content), 0, "Empty response for multi-image")
        print(f"  [multi_image] response: {content}")

    def test_streaming_chat_completion(self):
        """Verify streaming multimodal request via curl."""
        image_b64 = _create_test_image_b64(320, 240, "green")
        payload = {
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {
                            "type": "text",
                            "text": "What color is the image?",
                        },
                    ],
                }
            ],
            "max_tokens": 128,
            "temperature": 0,
            "stream": True,
        }

        result = subprocess.run(
            [
                "curl",
                "-s",
                "-N",
                "-X",
                "POST",
                f"{self.base_url}/v1/chat/completions",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(payload),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        output = result.stdout.strip()
        self.assertIn("data:", output, "No SSE data in streaming response")
        # Collect content from SSE chunks
        chunks = [line for line in output.split("\n") if line.startswith("data:")]
        self.assertGreater(len(chunks), 0, "No streaming chunks received")
        print(f"  [streaming] chunks: {len(chunks)}")


if __name__ == "__main__":
    unittest.main()
