import multiprocessing as mp
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

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

PROMPTS = [
    "AI is a field of computer science focused on",
    "The capital of France is",
]


class TestNPULoRARadixCache(CustomTestCase):
    """Test LoRA with radix cache on NPU.

    [Test Category] Feature
    [Test Target] LoRA radix cache, cache hit, prefix caching
    """

    lora_a = "lora_a"
    lora_b = "lora_b"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--lora-path",
            f"{cls.lora_a}={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            f"{cls.lora_b}={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            "--max-loras-per-batch",
            "2",
            "--max-loaded-loras",
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
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_with_radix_cache(self):
        long_prompt = """
The following is a conversation between a user and an AI assistant.
User: What is the capital of France?
Assistant:"""
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_prompt,
                "lora_path": self.lora_a,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        result1 = response.json()
        self.assertGreater(len(result1["text"]), 0)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_prompt,
                "lora_path": self.lora_a,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        result2 = response.json()
        self.assertEqual(result1["text"], result2["text"])

    def test_lora_radix_cache_different_adapters(self):
        common_prefix = "The following is a conversation about geography. "
        prompt1 = common_prefix + "What is the capital of France?"
        prompt2 = common_prefix + "What is the capital of Germany?"

        response1 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompt1,
                "lora_path": self.lora_a,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response1.status_code, 200)

        response2 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompt2,
                "lora_path": self.lora_b,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response2.status_code, 200)

    def test_lora_radix_cache_shared_prefix(self):
        prefix = "The following text explains machine learning concepts. "
        prompt1 = prefix + "What is a neural network?"
        prompt2 = prefix + "What is deep learning?"

        response1 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompt1,
                "lora_path": self.lora_a,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response1.status_code, 200)
        cached_tokens1 = response1.json()["meta_info"].get("cached_tokens", 0)

        response2 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompt1,
                "lora_path": self.lora_a,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response2.status_code, 200)
        cached_tokens2 = response2.json()["meta_info"].get("cached_tokens", 0)

        self.assertGreaterEqual(cached_tokens2, cached_tokens1)

    def test_lora_radix_cache_batch(self):
        prompts = PROMPTS[:2]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompts,
                "lora_path": self.lora_a,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), len(prompts))


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    unittest.main(warnings="ignore")
