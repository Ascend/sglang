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


class TestNPULoRAEvictionComplete(CustomTestCase):
    """Testcase: Verify LoRA eviction with different target modules on NPU.

    [Test Category] Logic
    [Test Target] lora_eviction_policy, max_loras_per_batch
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    base_model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    def _run_eviction_test(self, eviction_policy, max_loras_per_batch):
        """Helper function to run eviction test."""
        output_history = {}

        other_args = [
            "--enable-lora",
            "--lora-path",
            f"lora_a={self.lora_a}",
            "--max-loaded-loras",
            "2",
            "--max-loras-per-batch",
            str(max_loras_per_batch),
            "--lora-eviction-policy",
            eviction_policy,
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.3",
        ]

        process = popen_launch_server(
            self.base_model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
            for prompt in PROMPTS:
                response = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": prompt,
                        "lora_path": "lora_a",
                        "sampling_params": {"temperature": 0, "max_new_tokens": 64},
                    },
                )
                self.assertEqual(response.status_code, 200)
                output = response.json()["text"]
                output_history[(prompt, "lora_a")] = output

            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
                json={"lora_name": "lora_b", "lora_path": self.lora_a, "pinned": False},
            )
            self.assertTrue(response.ok)

            for prompt in PROMPTS:
                response = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": prompt,
                        "lora_path": "lora_a",
                        "sampling_params": {"temperature": 0, "max_new_tokens": 64},
                    },
                )
                self.assertEqual(response.status_code, 200)
                output = response.json()["text"]
                prev_output = output_history.get((prompt, "lora_a"))
                if prev_output:
                    self.assertEqual(
                        prev_output, output, f"Output mismatch for prompt '{prompt}'"
                    )
        finally:
            kill_process_tree(process.pid)

        return output_history

    def test_lora_eviction_fifo(self):
        """Test LoRA eviction with FIFO policy."""
        output_history = self._run_eviction_test("fifo", 1)
        self.assertEqual(len(output_history), len(PROMPTS))

    def test_lora_eviction_lru(self):
        """Test LoRA eviction with LRU policy."""
        output_history = self._run_eviction_test("lru", 1)
        self.assertEqual(len(output_history), len(PROMPTS))

    def test_lora_eviction_with_reused_name(self):
        """Test LoRA eviction with reused LoRA name."""
        other_args = [
            "--enable-lora",
            "--max-loaded-loras",
            "2",
            "--max-loras-per-batch",
            "1",
            "--max-lora-rank",
            "64",
            "--lora-eviction-policy",
            "fifo",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.3",
        ]

        process = popen_launch_server(
            self.base_model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
            requests.post(
                DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
                json={
                    "lora_name": "shared_name",
                    "lora_path": self.lora_a,
                    "pinned": False,
                },
            )

            response1 = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": PROMPTS[0],
                    "lora_path": "shared_name",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 64},
                },
            )
            output1 = response1.json()["text"]

            requests.post(
                DEFAULT_URL_FOR_TEST + "/unload_lora_adapter",
                json={"lora_name": "shared_name"},
            )

            requests.post(
                DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
                json={
                    "lora_name": "shared_name",
                    "lora_path": self.lora_a,
                    "pinned": False,
                },
            )

            response2 = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": PROMPTS[0],
                    "lora_path": "shared_name",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 64},
                },
            )
            output2 = response2.json()["text"]

            self.assertEqual(
                output1, output2, "Output should be consistent with reused name"
            )
        finally:
            kill_process_tree(process.pid)


class TestNPULoRAEvictionPolicy(unittest.TestCase):
    """Testcase: Verify LoRA eviction policy unit tests on NPU."""

    def test_fifo_basic_behavior(self):
        """Test FIFO policy basic behavior."""
        from sglang.srt.lora.eviction_policy import get_eviction_policy

        policy = get_eviction_policy("fifo")
        policy.mark_used("adapter1")
        policy.mark_used("adapter2")
        policy.mark_used("adapter1")

        victim = policy.select_victim(["adapter1", "adapter2"])
        self.assertEqual(victim, "adapter1")

    def test_lru_basic_behavior(self):
        """Test LRU policy basic behavior."""
        from sglang.srt.lora.eviction_policy import get_eviction_policy

        policy = get_eviction_policy("lru")
        policy.mark_used("adapter1")
        policy.mark_used("adapter2")
        policy.mark_used("adapter1")

        victim = policy.select_victim(["adapter1", "adapter2"])
        self.assertEqual(victim, "adapter2")

    def test_lru_reuse_updates_order(self):
        """Test LRU policy updates order on reuse."""
        from sglang.srt.lora.eviction_policy import get_eviction_policy

        policy = get_eviction_policy("lru")
        policy.mark_used("adapter1")
        policy.mark_used("adapter2")
        policy.mark_used("adapter3")

        policy.mark_used("adapter1")
        policy.mark_used("adapter2")

        victim = policy.select_victim(["adapter1", "adapter2", "adapter3"])
        self.assertEqual(victim, "adapter3")

    def test_fifo_ignores_reuse(self):
        """Test FIFO policy ignores reuse."""
        from sglang.srt.lora.eviction_policy import get_eviction_policy

        policy = get_eviction_policy("fifo")
        policy.mark_used("adapter1")
        policy.mark_used("adapter2")

        policy.mark_used("adapter2")

        victim = policy.select_victim(["adapter1", "adapter2"])
        self.assertEqual(victim, "adapter1")


if __name__ == "__main__":
    unittest.main()
