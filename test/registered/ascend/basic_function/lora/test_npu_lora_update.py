# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=500, suite="nightly-2-npu-a3", nightly=True)

PROMPTS = [
    "SGL is a",
    "AI is a field of computer science focused on",
    "Computer science is the study of",
]

MEM_FRACTION_STATIC = 0.3


class TestNPULoRAUpdate(CustomTestCase):
    """Testcase: Verify dynamic LoRA load/unload on NPU.

    [Test Category] Parameter
    [Test Target] /load_lora_adapter, /unload_lora_adapter API
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    base_model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--max-loaded-loras",
            "2",
            "--max-loras-per-batch",
            "2",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            str(MEM_FRACTION_STATIC),
        ]
        cls.process = popen_launch_server(
            cls.base_model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_load_lora_adapter(self):
        """Test loading LoRA adapter via API."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "test_adapter", "lora_path": self.lora_a, "pinned": False},
        )
        self.assertTrue(response.ok, f"Failed to load LoRA adapter: {response.text}")
        loaded_adapters = response.json()["loaded_adapters"]
        self.assertIn("test_adapter", loaded_adapters)

    def test_unload_lora_adapter(self):
        """Test unloading LoRA adapter via API."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "adapter_to_unload", "lora_path": self.lora_a, "pinned": False},
        )
        self.assertTrue(response.ok)

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/unload_lora_adapter",
            json={"lora_name": "adapter_to_unload"},
        )
        self.assertTrue(response.ok, f"Failed to unload LoRA adapter: {response.text}")
        loaded_adapters = response.json()["loaded_adapters"]
        self.assertNotIn("adapter_to_unload", loaded_adapters)

    def test_load_already_loaded_adapter(self):
        """Test loading an already loaded adapter should fail."""
        requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "existing_adapter", "lora_path": self.lora_a, "pinned": False},
        )

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "existing_adapter", "lora_path": self.lora_a, "pinned": False},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("already loaded", response.text)

    def test_forward_with_loaded_adapter(self):
        """Test inference with dynamically loaded adapter."""
        requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "forward_adapter", "lora_path": self.lora_a, "pinned": False},
        )

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": PROMPTS[0],
                "lora_path": "forward_adapter",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.json()["text"]), 0)

    def test_forward_without_loaded_adapter(self):
        """Test inference without adapter should use base model."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": PROMPTS[0],
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        base_output = response.json()["text"]

        requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "compare_adapter", "lora_path": self.lora_a, "pinned": False},
        )

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": PROMPTS[0],
                "lora_path": "compare_adapter",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        lora_output = response.json()["text"]

        self.assertNotEqual(base_output, lora_output, "LoRA should modify output")

    def test_max_loaded_loras_limit(self):
        """Test max_loaded_loras constraint."""
        requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "adapter1", "lora_path": self.lora_a, "pinned": False},
        )
        requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "adapter2", "lora_path": self.lora_a, "pinned": False},
        )

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "adapter3", "lora_path": self.lora_a, "pinned": False},
        )
        self.assertEqual(response.status_code, 400)

    def test_pinned_adapter(self):
        """Test pinned adapter cannot be evicted."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "pinned_adapter", "lora_path": self.lora_a, "pinned": True},
        )
        self.assertTrue(response.ok)

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/server_info",
        )
        loaded_adapters = response.json().get("loaded_adapters", [])
        self.assertIn("pinned_adapter", loaded_adapters)


if __name__ == "__main__":
    unittest.main()