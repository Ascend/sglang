import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH,
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
]


class TestNPULoRAUpdate(CustomTestCase):
    """Testcase: Verify dynamic LoRA adapter load/unload operations on NPU.

    [Test Category] Feature
    [Test Target] LoRA dynamic update, load_lora_adapter, unload_lora_adapter
    """

    base_model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "2",
            "--enable-lora",
            "--lora-path",
            f"lora_a={cls.lora_a}",
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
            "0.3",
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
        """Test loading a new LoRA adapter dynamically."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "lora_b", "lora_path": self.lora_b},
        )
        self.assertTrue(response.ok, f"Failed to load LoRA adapter: {response.text}")
        loaded_adapters = set(response.json()["loaded_adapters"])
        self.assertIn("lora_a", loaded_adapters)
        self.assertIn("lora_b", loaded_adapters)

    def test_load_already_loaded_adapter(self):
        """Test loading an already loaded LoRA adapter should fail."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "lora_a", "lora_path": self.lora_a},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("already loaded", response.text)

    def test_unload_lora_adapter(self):
        """Test unloading a LoRA adapter."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/unload_lora_adapter",
            json={"lora_name": "lora_a"},
        )
        self.assertTrue(response.ok, f"Failed to unload LoRA adapter: {response.text}")
        loaded_adapters = set(response.json()["loaded_adapters"])
        self.assertNotIn("lora_a", loaded_adapters)

    def test_forward_with_never_loaded_adapter(self):
        """Test forward pass with never-loaded adapter should fail."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": PROMPTS[0],
                "lora_path": "never_loaded_lora",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("never been loaded", response.text)

    def test_dynamic_load_unload_sequence(self):
        """Test a sequence of load/unload operations."""
        base_params = {
            "text": PROMPTS[0],
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        }

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={**base_params, "lora_path": "lora_a"},
        )
        self.assertEqual(response.status_code, 200)
        text_lora_a = response.json()["text"]

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "lora_b", "lora_path": self.lora_b},
        )
        self.assertTrue(response.ok)

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={**base_params, "lora_path": "lora_b"},
        )
        self.assertEqual(response.status_code, 200)
        text_lora_b = response.json()["text"]

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/unload_lora_adapter",
            json={"lora_name": "lora_b"},
        )
        self.assertTrue(response.ok)

        self.assertNotEqual(text_lora_a, text_lora_b)


class TestNPULoRAUpdateWithPinned(CustomTestCase):
    """Testcase: Verify pinned LoRA adapter behavior on NPU.

    [Test Category] Feature
    [Test Target] pinned LoRA adapters, eviction behavior
    """

    base_model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "2",
            "--enable-lora",
            "--lora-path",
            json.dumps(
                {
                    "lora_name": "lora_a",
                    "lora_path": cls.lora_a,
                    "pinned": True,
                },
            ),
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
            "0.3",
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

    def test_pinned_adapter_not_evicted(self):
        """Test pinned adapter is not evicted when loading new adapter."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "lora_b", "lora_path": self.lora_b},
        )
        self.assertTrue(response.ok)
        loaded_adapters = set(response.json()["loaded_adapters"])
        self.assertIn("lora_a", loaded_adapters)
        self.assertIn("lora_b", loaded_adapters)

    def test_load_pinned_when_capacity_full(self):
        """Test loading pinned adapter when capacity is full should cause starvation."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "lora_c", "lora_path": self.lora_a, "pinned": True},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("starvation", response.text)


if __name__ == "__main__":
    unittest.main()
