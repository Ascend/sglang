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

register_npu_ci(est_time=450, suite="nightly-2-npu-a3", nightly=True)

PROMPTS = [
    "AI is a field of computer science focused on",
    "The capital of France is",
]

ADAPTERS = [
    "lora_a",
    "lora_b",
]

BASE_MODEL = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH


class TestNPULoRAEviction(CustomTestCase):
    """Testcase: Verify LoRA eviction with different adapters on NPU.

    [Test Category] Feature
    [Test Target] LoRA eviction, adapter switching, output consistency
    """

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
            f"lora_b={cls.lora_b}",
            "--max-loaded-loras",
            "2",
            "--max-loras-per-batch",
            "1",
            "--lora-target-modules",
            "all",
            "--lora-eviction-policy",
            "lru",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.3",
        ]
        cls.process = popen_launch_server(
            BASE_MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_eviction_with_different_adapters(self):
        """Test LoRA eviction with different adapters in sequence."""
        output_history = {}

        for prompt in PROMPTS:
            for adapter in ADAPTERS:
                response = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": prompt,
                        "lora_path": adapter,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                    },
                )
                self.assertEqual(response.status_code, 200)
                output = response.json()["text"].strip()

                key = (adapter, prompt)
                prev_output = output_history.get(key)
                if prev_output is not None:
                    self.assertEqual(
                        prev_output,
                        output,
                        f"Output mismatch for adapter {adapter} and prompt '{prompt}'",
                    )
                else:
                    output_history[key] = output

    def test_lora_eviction_reversed_order(self):
        """Test LoRA eviction with reversed adapter order."""
        output_history = {}

        reversed_adapters = ADAPTERS[::-1]
        for prompt in PROMPTS:
            for adapter in reversed_adapters:
                response = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": prompt,
                        "lora_path": adapter,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                    },
                )
                self.assertEqual(response.status_code, 200)
                output = response.json()["text"].strip()

                key = (adapter, prompt)
                prev_output = output_history.get(key)
                if prev_output is not None:
                    self.assertEqual(
                        prev_output,
                        output,
                        f"Output mismatch for adapter {adapter} and prompt '{prompt}' in reversed order",
                    )
                else:
                    output_history[key] = output

    def test_adapter_outputs_are_different(self):
        """Test that different adapters produce different outputs."""
        base_params = {
            "text": PROMPTS[0],
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        }

        outputs = []
        for adapter in ADAPTERS:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={**base_params, "lora_path": adapter},
            )
            self.assertEqual(response.status_code, 200)
            outputs.append(response.json()["text"])

        self.assertNotEqual(
            outputs[0],
            outputs[1],
            "Different adapters should produce different outputs",
        )


class TestNPULoRAEvictionWithReusedName(CustomTestCase):
    """Testcase: Verify LoRA eviction with reused adapter names on NPU.

    [Test Category] Feature
    [Test Target] LoRA eviction, reused lora_name
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "2",
            "--enable-lora",
            "--max-loaded-loras",
            "1",
            "--max-loras-per-batch",
            "1",
            "--max-lora-rank",
            "64",
            "--lora-target-modules",
            "all",
            "--lora-eviction-policy",
            "lru",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.3",
        ]
        cls.process = popen_launch_server(
            BASE_MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_reused_lora_name_eviction(self):
        """Test LoRA eviction with reused adapter name."""
        reused_name = "shared_lora"

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": reused_name, "lora_path": self.lora_a},
        )
        self.assertTrue(response.ok)

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": PROMPTS[0],
                "lora_path": reused_name,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        output_a = response.json()["text"]

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/unload_lora_adapter",
            json={"lora_name": reused_name},
        )
        self.assertTrue(response.ok)

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": reused_name, "lora_path": self.lora_b},
        )
        self.assertTrue(response.ok)

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": PROMPTS[0],
                "lora_path": reused_name,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        output_b = response.json()["text"]

        self.assertNotEqual(
            output_a,
            output_b,
            "Different LoRA paths with same name should produce different outputs",
        )


if __name__ == "__main__":
    unittest.main()
