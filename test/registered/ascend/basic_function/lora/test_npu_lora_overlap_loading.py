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
    "The capital of France is",
    "Machine learning is a subset of",
    "What is artificial intelligence",
]


class TestNPULoRAOverlapLoadingEnabled(CustomTestCase):
    """Testcase: Verify LoRA with overlap loading enabled on NPU.

    [Test Category] Feature
    [Test Target] --enable-lora-overlap-loading, async LoRA loading during inference
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
            "--enable-lora-overlap-loading",
            "--lora-path",
            f"lora_a={cls.lora_a}",
            f"lora_b={cls.lora_b}",
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

    def test_lora_with_overlap_loading_single_adapter(self):
        """Test LoRA inference with overlap loading enabled - single adapter."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": PROMPTS[0],
                "lora_path": "lora_a",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text", response.json())
        self.assertGreater(len(response.json()["text"]), 0)

    def test_lora_with_overlap_loading_multiple_adapters(self):
        """Test LoRA inference with overlap loading enabled - multiple adapters."""
        outputs = []
        for adapter in ["lora_a", "lora_b"]:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": PROMPTS[0],
                    "lora_path": adapter,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(response.status_code, 200)
            outputs.append(response.json()["text"])

        self.assertNotEqual(
            outputs[0],
            outputs[1],
            "Different adapters should produce different outputs",
        )

    def test_lora_overlap_loading_batch_inference(self):
        """Test batch inference with overlap loading enabled."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": PROMPTS,
                "lora_path": "lora_a",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), len(PROMPTS))
        for result in results:
            self.assertGreater(len(result["text"]), 0)

    def test_lora_overlap_loading_consistency(self):
        """Test output consistency with overlap loading enabled."""
        outputs = []
        for _ in range(3):
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": PROMPTS[0],
                    "lora_path": "lora_a",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(response.status_code, 200)
            outputs.append(response.json()["text"])

        self.assertTrue(
            all(o == outputs[0] for o in outputs),
            "Outputs should be consistent with overlap loading",
        )


class TestNPULoRABatchSplittingEquivalence(CustomTestCase):
    """Testcase: Verify batch splitting equivalence with overlap loading on NPU.

    [Test Category] Feature
    [Test Target] Batch splitting, multi-adapter batch processing consistency
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
            "--enable-lora-overlap-loading",
            "--lora-path",
            f"lora_a={cls.lora_a}",
            f"lora_b={cls.lora_b}",
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

    def test_batch_splitting_single_adapter(self):
        """Test batch splitting with single adapter - results should match individual requests."""
        prompt = PROMPTS[0]

        response_batch = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": [prompt, prompt, prompt],
                "lora_path": "lora_a",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response_batch.status_code, 200)
        batch_results = response_batch.json()

        response_single = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": prompt,
                "lora_path": "lora_a",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response_single.status_code, 200)
        single_result = response_single.json()["text"]

        for batch_result in batch_results:
            self.assertEqual(
                batch_result["text"],
                single_result,
                "Batch and single outputs should match",
            )

    def test_batch_splitting_mixed_adapters(self):
        """Test batch splitting with mixed adapters."""
        prompt = PROMPTS[0]
        adapters = ["lora_a", "lora_b", "lora_a"]

        batch_results = []
        for adapter in adapters:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": prompt,
                    "lora_path": adapter,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(response.status_code, 200)
            batch_results.append(response.json()["text"])

        self.assertEqual(len(batch_results), 3)
        self.assertNotEqual(
            batch_results[0],
            batch_results[1],
            "Different adapters should produce different outputs",
        )
        self.assertEqual(
            batch_results[0],
            batch_results[2],
            "Same adapter should produce same output",
        )

    def test_batch_splitting_large_batch(self):
        """Test batch splitting with large batch size."""
        large_prompts = PROMPTS * 4

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": large_prompts,
                "lora_path": "lora_a",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), len(large_prompts))
        for result in results:
            self.assertGreater(len(result["text"]), 0)


if __name__ == "__main__":
    unittest.main()
