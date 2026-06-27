import multiprocessing as mp
import unittest
from types import SimpleNamespace
from typing import cast
from unittest import mock

import requests

from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.managers.schedule_batch import Req
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
    is_in_ci,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

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
    """Test LoRA drainer logic on NPU.

    [Test Category] Feature
    [Test Target] LoRA drainer, starvation prevention, adapter draining
    """

    def test_update_draining_marks_adapter(self):
        if is_in_ci():
            return

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
        if is_in_ci():
            return

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


class TestNPULoRADrainerIntegration(CustomTestCase):
    """Test LoRA drainer integration on NPU.

    [Test Category] Feature
    [Test Target] LoRA drainer integration, batch splitting with drainer
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
            "1",
            "--max-loaded-loras",
            "2",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.3",
            "--lora-drain-wait-threshold",
            str(LORA_DRAIN_WAIT_THRESHOLD),
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

    def test_lora_drainer_basic(self):
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


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    unittest.main(warnings="ignore")
