"""Tests for --image-processor-backend parameter.

One test layers:
- End-to-end: launch a VLM server with each backend, verify startup (NPU)
"""

import base64
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


def _generate_test_image_b64():
    """Generate a minimal 1x1 red pixel PNG as a base64 data URI."""
    import struct
    import zlib

    # Minimal PNG: 1x1 red pixel
    def _make_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + chunk + crc

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, RGB
    raw = zlib.compress(b"\x00\xff\x00\x00")  # filter=0, R=255, G=0, B=0
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _make_chunk(b"IHDR", ihdr)
        + _make_chunk(b"IDAT", raw)
        + _make_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class TestImageProcessorBackendE2E(CustomTestCase):
    """Testcase: Verify --image-processor-backend is accepted by the VLM server
       and the multi-mode processor is initialized correctly, with normal inference

    [Test Category] Parameter
    [Test Target] Verify whether the service inference is successful
    """

    model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    _BASE_ARGS = [
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--trust-remote-code",
        "--enable-multimodal",
        "--mm-attention-backend",
        "ascend_attn",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.81",
        "--tp-size",
        "1",
    ]

    _BACKEND_MAP = {
        "test_e2e_auto": "auto",
        "test_e2e_torchvision": "torchvision",
        "test_e2e_pil": "pil",
    }

    @classmethod
    def setUpClass(cls):
        cls._image_b64 = _generate_test_image_b64()

    def setUp(self):
        backend = self._BACKEND_MAP[self._testMethodName]
        other_args = self._BASE_ARGS + ["--image-processor-backend", backend]
        self._process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=self.timeout,
            other_args=other_args,
        )
        self.assertIsNone(
            self._process.poll(),
            f"Server exited prematurely with {backend=}",
        )
        self._client = openai.Client(
            api_key="sk-123456",
            base_url=f"{DEFAULT_URL_FOR_TEST}/v1",
        )

    def tearDown(self):
        kill_process_tree(self._process.pid)

    def _verify_response(self, backend):
        response = self._client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_b64},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a short sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=64,
        )
        output = response.choices[0].message.content
        print(f"  [image-processor-backend={backend}] output: {output!r}")
        self.assertGreater(
            len(output), 5, f"Output too short with {backend=}: {output!r}"
        )
        self.assertIn(
            "red",
            output.lower(),
            f"Output should describe the red image with {backend=}: {output!r}",
        )

    def test_e2e_auto(self):
        """I6: Launch VLM server with --image-processor-backend auto."""
        self._verify_response("auto")

    def test_e2e_torchvision(self):
        """I6: Launch VLM server with --image-processor-backend torchvision."""
        self._verify_response("torchvision")

    def test_e2e_pil(self):
        """I6: Launch VLM server with --image-processor-backend pil."""
        self._verify_response("pil")


if __name__ == "__main__":
    unittest.main()
