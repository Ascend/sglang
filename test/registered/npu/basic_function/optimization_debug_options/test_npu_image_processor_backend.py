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
        "0.857",
        "--tp-size",
        "1",
    ]

    @classmethod
    def setUpClass(cls):
        """Pre-download a test image for all test cases."""
        # Use a minimal solid-color image (1x1 red pixel PNG) to avoid network issues
        cls._image_b64 = _generate_test_image_b64()

    def _launch_and_verify(self, backend):
        """Launch a VLM server with the given image-processor-backend
        and verify it starts successfully and can handle multimodal input.
        """
        other_args = self._BASE_ARGS + ["--image-processor-backend", backend]
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=self.timeout,
            other_args=other_args,
        )
        try:
            # Server started successfully -> multimodal processor initialized
            self.assertIsNone(
                process.poll(),
                f"Server exited prematurely with {backend=}",
            )

            # Send a multimodal request via OpenAI-compatible endpoint
            client = openai.Client(
                api_key="sk-123456",
                base_url=f"{DEFAULT_URL_FOR_TEST}/v1",
            )
            response = client.chat.completions.create(
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
            self.assertIsNotNone(output, f"No output with {backend=}")
            self.assertGreater(len(output), 0, f"Empty output with {backend=}")
        finally:
            kill_process_tree(process.pid)

    def test_e2e_auto(self):
        """I6: Launch VLM server with --image-processor-backend auto."""
        self._launch_and_verify("auto")

    def test_e2e_torchvision(self):
        """I6: Launch VLM server with --image-processor-backend torchvision."""
        self._launch_and_verify("torchvision")

    def test_e2e_pil(self):
        """I6: Launch VLM server with --image-processor-backend pil."""
        self._launch_and_verify("pil")


if __name__ == "__main__":
    unittest.main()
