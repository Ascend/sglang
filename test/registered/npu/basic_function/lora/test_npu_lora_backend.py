import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_5_4B_NEO4J_TEXT2CYPHER_LORA_PATH,
    QWEN3_5_4B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestLoraBackend(CustomTestCase):
    """Testcase: Test configuration of lora-backend parameters, and inference request successful.

    [Test Category] Parameter
    [Test Target] --lora-backend
    """

    lora_backend = "triton"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--lora-backend",
            f"{cls.lora_backend}",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--lora-path",
            f"lora_a={QWEN3_5_4B_NEO4J_TEXT2CYPHER_LORA_PATH}",
        ]
        cls.process = popen_launch_server(
            QWEN3_5_4B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_backend(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["lora_backend"], f"{self.lora_backend}")


class TestLoraBackendTorchNative(TestLoraBackend):
    lora_backend = "torch_native"


if __name__ == "__main__":
    unittest.main()
