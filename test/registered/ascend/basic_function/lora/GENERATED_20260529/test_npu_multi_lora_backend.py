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


class TestNPUMultiLoRABackend(CustomTestCase):
    """Test multi-LoRA batch processing on NPU.

    [Test Category] Feature
    [Test Target] multi-LoRA batch, LoRA backend, batch splitting
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

    def test_multi_lora_batch_basic(self):
        prompts = ["The capital of France is", "What is AI"]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompts,
                "lora_path": [self.lora_a, self.lora_b],
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), len(prompts))
        for result in results:
            self.assertGreater(len(result["text"]), 0)

    def test_multi_lora_batch_splitting(self):
        prompts = [
            "What is machine learning",
            "Explain neural networks",
            "How does deep learning work",
            "What is reinforcement learning",
        ]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompts,
                "lora_path": [self.lora_a, self.lora_b, self.lora_a, self.lora_b],
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), len(prompts))

    def test_multi_lora_with_none_adapter(self):
        # Use only 2 prompts to stay within max-loaded-loras=2 limit
        # None counts as a separate adapter, so [lora_a, None] = 2 adapters
        prompts = ["The capital of France is", "What is AI"]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompts,
                "lora_path": [self.lora_a, None],
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), len(prompts))

    def test_multi_lora_same_adapter_batch(self):
        prompts = [
            "What is AI",
            "Explain neural networks",
            "How does deep learning work",
        ]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompts,
                "lora_path": [self.lora_a, self.lora_a, self.lora_a],
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), len(prompts))

    def test_multi_lora_large_batch(self):
        prompts = [
            "Question 1 about AI",
            "Question 2 about ML",
            "Question 3 about DL",
            "Question 4 about NLP",
            "Question 5 about CV",
            "Question 6 about RL",
        ]
        lora_paths = [
            self.lora_a,
            self.lora_b,
            self.lora_a,
            self.lora_b,
            self.lora_a,
            self.lora_b,
        ]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompts,
                "lora_path": lora_paths,
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
