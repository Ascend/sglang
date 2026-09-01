"""Tests for --image-processor-backend parameter.

Two test layers:
- Parameter parsing: verify auto/torchvision/pil parse correctly (CPU)
- End-to-end: launch a VLM server with each backend, verify startup (NPU)
"""

import unittest

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


class TestImageProcessorBackendParsing(CustomTestCase):
    """Testcase: Verify --image-processor-backend CLI parsing
    and deprecated --disable-fast-image-processor compatibility.

    [Test Category] Parameter
    [Test Target] --image-processor-backend
    [Scenario] I1-I5: parsing and migration
    """

    @staticmethod
    def _parse(extra_args):
        from sglang.srt.server_args import ServerArgs

        kwargs = {}
        i = 0
        while i < len(extra_args):
            if extra_args[i].startswith("--"):
                key = extra_args[i][2:].replace("-", "_")
                if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                    kwargs[key] = extra_args[i + 1]
                    i += 2
                else:
                    kwargs[key] = True
                    i += 1
            else:
                i += 1
        return ServerArgs(model_path="dummy", attention_backend="ascend", **kwargs)

    def test_parsing_auto(self):
        sa = self._parse(["--image-processor-backend", "auto"])
        self.assertEqual(sa.image_processor_backend, "auto")

    def test_parsing_torchvision(self):
        sa = self._parse(["--image-processor-backend", "torchvision"])
        self.assertEqual(sa.image_processor_backend, "torchvision")

    def test_parsing_pil(self):
        sa = self._parse(["--image-processor-backend", "pil"])
        self.assertEqual(sa.image_processor_backend, "pil")

    def test_default_value(self):
        sa = self._parse([])
        self.assertEqual(sa.image_processor_backend, "auto")

    def test_deprecated_flag_migration(self):
        sa = self._parse(["--disable-fast-image-processor"])
        self.assertEqual(sa.image_processor_backend, "pil")

    def test_deprecated_conflict(self):
        with self.assertRaises(ValueError):
            self._parse([
                "--disable-fast-image-processor",
                "--image-processor-backend",
                "torchvision",
            ])


class TestImageProcessorBackendE2E(CustomTestCase):
    """Testcase: Verify --image-processor-backend is accepted by the VLM server
    and the multimodal processor initializes correctly.

    Uses Qwen3-VL-4B (smallest available VLM model on NPU).

    [Test Category] Parameter
    [Test Target] --image-processor-backend
    [Scenario] I6: end-to-end server startup with each backend
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
        and verify it starts successfully.
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
