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
from typing import cast
from unittest import mock

from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.managers.schedule_batch import Req
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-2-npu-a3", nightly=True)

MOCK_START_TIME = 1000.0
LORA_DRAIN_WAIT_THRESHOLD = 3.0


def make_req(lora_id, wait_queue_entry_time, max_new_tokens, output_len=0):
    time_stats = SimpleNamespace(wait_queue_entry_time=wait_queue_entry_time)
    sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
    req_ns = SimpleNamespace(
        lora_id=lora_id,
        time_stats=time_stats,
        sampling_params=sampling_params,
        output_ids=[0] * output_len,
    )
    return cast(Req, req_ns)


class TestNPULoRADrainer(unittest.TestCase):
    """Testcase: Verify LoRA Drainer mechanism on NPU.

    [Test Category] Logic
    [Test Target] LoRADrainer.update_draining_state, LoRADrainer.can_schedule
    """

    def test_update_draining_marks_adapter(self):
        """Test that drainer marks adapter as draining for starving requests."""
        with mock.patch("time.monotonic", return_value=MOCK_START_TIME):
            drainer = LoRADrainer(
                max_loras_per_batch=1, max_wait_time_secs=LORA_DRAIN_WAIT_THRESHOLD
            )

            wait_entry = MOCK_START_TIME - (LORA_DRAIN_WAIT_THRESHOLD + 0.01)
            waiting_req = make_req("A", wait_entry, max_new_tokens=10)

            running_req = make_req("B", wait_entry, max_new_tokens=100, output_len=0)

            drainer.update_draining_state(
                waiting_queue=[waiting_req],
                running_reqs=[running_req],
            )

            self.assertEqual(drainer.adapter_to_stats["B"].is_draining_for, "A")

            drainer.update_draining_state(waiting_queue=[waiting_req], running_reqs=[])
            self.assertIsNone(drainer.adapter_to_stats["B"].is_draining_for)

    def test_multiple_starving_adapters(self):
        """Test multiple starving adapters cause multiple running adapters to drain."""
        with mock.patch("time.monotonic", return_value=MOCK_START_TIME):
            drainer = LoRADrainer(
                max_loras_per_batch=2, max_wait_time_secs=LORA_DRAIN_WAIT_THRESHOLD
            )

            wait_entryA = MOCK_START_TIME - (LORA_DRAIN_WAIT_THRESHOLD + 0.05)
            wait_entryD = MOCK_START_TIME - (LORA_DRAIN_WAIT_THRESHOLD + 0.01)
            starving_a = make_req("A", wait_entryA, max_new_tokens=10)
            starving_d = make_req("D", wait_entryD, max_new_tokens=10)

            running_b = make_req("B", wait_entryA, max_new_tokens=5, output_len=0)
            running_c = make_req("C", wait_entryA, max_new_tokens=100, output_len=0)

            drainer.update_draining_state(
                waiting_queue=[starving_a, starving_d],
                running_reqs=[running_b, running_c],
            )

            self.assertEqual(drainer.adapter_to_stats["B"].is_draining_for, "A")
            self.assertEqual(drainer.adapter_to_stats["C"].is_draining_for, "D")

    def test_can_schedule_respects_draining_tolerance(self):
        """Test can_schedule respects draining tolerance."""
        with mock.patch("time.monotonic", return_value=MOCK_START_TIME):
            drainer = LoRADrainer(
                max_loras_per_batch=1, max_wait_time_secs=LORA_DRAIN_WAIT_THRESHOLD
            )

            wait_entry = MOCK_START_TIME - (LORA_DRAIN_WAIT_THRESHOLD + 0.01)
            starving_req = make_req("A", wait_entry, max_new_tokens=10)

            running_b = make_req("B", wait_entry, max_new_tokens=15, output_len=0)
            drainer.update_draining_state(
                waiting_queue=[starving_req],
                running_reqs=[running_b],
            )

            self.assertEqual(drainer.adapter_to_stats["B"].is_draining_for, "A")

            req_ok = make_req(
                lora_id="B", wait_queue_entry_time=0, max_new_tokens=10, output_len=0
            )
            self.assertTrue(drainer.can_schedule(req_ok))

            req_bad = make_req(
                lora_id="B", wait_queue_entry_time=0, max_new_tokens=20, output_len=0
            )
            self.assertFalse(drainer.can_schedule(req_bad))

    def test_drainer_initialization(self):
        """Test LoRADrainer initialization with different parameters."""
        drainer = LoRADrainer(max_loras_per_batch=4, max_wait_time_secs=5.0)
        self.assertEqual(drainer.max_loras_per_batch, 4)
        self.assertEqual(drainer.max_wait_time_secs, 5.0)
        self.assertEqual(drainer.adapter_to_stats, {})

    def test_empty_waiting_queue(self):
        """Test drainer with empty waiting queue."""
        with mock.patch("time.monotonic", return_value=MOCK_START_TIME):
            drainer = LoRADrainer(
                max_loras_per_batch=1, max_wait_time_secs=LORA_DRAIN_WAIT_THRESHOLD
            )

            running_req = make_req("B", MOCK_START_TIME, max_new_tokens=100, output_len=0)

            drainer.update_draining_state(
                waiting_queue=[],
                running_reqs=[running_req],
            )

            self.assertNotIn("B", drainer.adapter_to_stats)

    def test_short_wait_time(self):
        """Test requests with short wait time are not considered starving."""
        with mock.patch("time.monotonic", return_value=MOCK_START_TIME):
            drainer = LoRADrainer(
                max_loras_per_batch=1, max_wait_time_secs=LORA_DRAIN_WAIT_THRESHOLD
            )

            short_wait_entry = MOCK_START_TIME - (LORA_DRAIN_WAIT_THRESHOLD - 1.0)
            short_wait_req = make_req("A", short_wait_entry, max_new_tokens=10)

            running_req = make_req("B", MOCK_START_TIME, max_new_tokens=100, output_len=0)

            drainer.update_draining_state(
                waiting_queue=[short_wait_req],
                running_reqs=[running_req],
            )

            self.assertNotIn("B", drainer.adapter_to_stats)


if __name__ == "__main__":
    unittest.main()