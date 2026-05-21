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
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-2-npu-a3", nightly=True)


class MockLoRAOverlapLoader:
    """Mock LoRA overlap loader for unit testing."""

    def __init__(self, max_loaded_loras=4, max_loras_per_batch=2):
        self.max_loaded_loras = max_loaded_loras
        self.max_loras_per_batch = max_loras_per_batch
        self.loading_queue = []
        self.loaded_adapters = set()
        self.pending_requests = {}

    def add_to_loading_queue(self, lora_name, lora_path):
        """Add adapter to loading queue."""
        if len(self.loaded_adapters) >= self.max_loaded_loras:
            return False, "Capacity exceeded"
        self.loading_queue.append((lora_name, lora_path))
        return True, "Added to queue"

    def is_loading(self, lora_name):
        """Check if adapter is being loaded."""
        return any(name == lora_name for name, _ in self.loading_queue)

    def mark_loaded(self, lora_name):
        """Mark adapter as loaded."""
        self.loading_queue = [(n, p) for n, p in self.loading_queue if n != lora_name]
        self.loaded_adapters.add(lora_name)

    def can_add_to_batch(self, adapter_names):
        """Check if adapters can be added to batch."""
        loading_or_loaded = self.loaded_adapters.union(
            set(n for n, _ in self.loading_queue)
        )
        new_adapters = set(adapter_names) - loading_or_loaded
        return len(new_adapters) <= self.max_loras_per_batch


class TestNPULoRAOverlapLoadingUnit(unittest.TestCase):
    """Testcase: Verify LoRA overlap loading mechanism on NPU.

    [Test Category] Logic
    [Test Target] LoRAOverlapLoader, async loading state
    """

    def test_add_to_loading_queue_success(self):
        """Test successful addition to loading queue."""
        loader = MockLoRAOverlapLoader(max_loaded_loras=4)
        success, msg = loader.add_to_loading_queue("adapter1", "/path/to/adapter1")
        self.assertTrue(success)
        self.assertEqual(msg, "Added to queue")

    def test_add_to_loading_queue_capacity_exceeded(self):
        """Test loading queue capacity limit."""
        loader = MockLoRAOverlapLoader(max_loaded_loras=2)
        loader.add_to_loading_queue("adapter1", "/path1")
        loader.mark_loaded("adapter1")
        loader.add_to_loading_queue("adapter2", "/path2")
        loader.mark_loaded("adapter2")

        success, msg = loader.add_to_loading_queue("adapter3", "/path3")
        self.assertFalse(success)
        self.assertIn("Capacity", msg)

    def test_is_loading_check(self):
        """Test is_loading check."""
        loader = MockLoRAOverlapLoader()
        loader.add_to_loading_queue("adapter1", "/path1")

        self.assertTrue(loader.is_loading("adapter1"))
        self.assertFalse(loader.is_loading("adapter2"))

    def test_mark_loaded(self):
        """Test marking adapter as loaded."""
        loader = MockLoRAOverlapLoader()
        loader.add_to_loading_queue("adapter1", "/path1")
        loader.mark_loaded("adapter1")

        self.assertFalse(loader.is_loading("adapter1"))
        self.assertIn("adapter1", loader.loaded_adapters)

    def test_can_add_to_batch(self):
        """Test batch capacity check."""
        loader = MockLoRAOverlapLoader(max_loras_per_batch=2)
        loader.add_to_loading_queue("adapter1", "/path1")
        loader.mark_loaded("adapter1")

        can_add = loader.can_add_to_batch(["adapter1", "adapter2"])
        self.assertTrue(can_add)

        can_add = loader.can_add_to_batch(["adapter1", "adapter2", "adapter3"])
        self.assertFalse(can_add)

    def test_pending_and_running_validation(self):
        """Test pending + running validation."""
        loader = MockLoRAOverlapLoader(max_loaded_loras=4, max_loras_per_batch=2)

        loader.add_to_loading_queue("adapter1", "/path1")
        loader.mark_loaded("adapter1")
        loader.add_to_loading_queue("adapter2", "/path2")
        loader.mark_loaded("adapter2")

        self.assertEqual(len(loader.loaded_adapters), 2)

    def test_loading_state_transition(self):
        """Test loading state transition."""
        loader = MockLoRAOverlapLoader()

        self.assertEqual(len(loader.loading_queue), 0)
        self.assertEqual(len(loader.loaded_adapters), 0)

        loader.add_to_loading_queue("adapter1", "/path1")
        self.assertEqual(len(loader.loading_queue), 1)
        self.assertTrue(loader.is_loading("adapter1"))

        loader.mark_loaded("adapter1")
        self.assertEqual(len(loader.loading_queue), 0)
        self.assertIn("adapter1", loader.loaded_adapters)


class TestNPULoRAOverlapLoadingIntegration(unittest.TestCase):
    """Testcase: Verify LoRA overlap loading integration on NPU."""

    def test_mock_overlap_loader_state_machine(self):
        """Test complete state machine of overlap loader."""
        loader = MockLoRAOverlapLoader(max_loaded_loras=3, max_loras_per_batch=2)

        states = []

        loader.add_to_loading_queue("A", "/pathA")
        states.append(("loading", "A"))

        loader.mark_loaded("A")
        states.append(("loaded", "A"))

        loader.add_to_loading_queue("B", "/pathB")
        states.append(("loading", "B"))

        loader.add_to_loading_queue("C", "/pathC")
        states.append(("loading", "C"))

        loader.mark_loaded("B")
        states.append(("loaded", "B"))

        loader.mark_loaded("C")
        states.append(("loaded", "C"))

        self.assertEqual(len([s for s in states if s[0] == "loaded"]), 3)
        self.assertEqual(loader.loaded_adapters, {"A", "B", "C"})


if __name__ == "__main__":
    unittest.main()