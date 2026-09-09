"""Tests for --image-processor-backend parameter.

One test layers:
- End-to-end: launch a VLM server with each backend, verify startup (NPU)
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


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
            # Server started successfully → multimodal processor initialized
            self.assertIsNone(
                process.poll(),
                f"Server exited prematurely with {backend=}",
            )

            # Send a multimodal request to verify inference works
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "Describe this image in a short sentence.",
                    "image_data": DEFAULT_IMAGE_URL,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 64,
                    },
                },
                timeout=120,
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("text", resp.json())
            self.assertGreater(len(resp.json()["text"]), 0)
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
