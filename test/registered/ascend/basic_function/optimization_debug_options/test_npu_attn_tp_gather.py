import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import OLMOE_1B_7B_0924_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="debug-full-1-npu-a3", nightly=True)


class TestDisableAttnTpGather(CustomTestCase):
    """Testcase: verify --disable-attn-tp-gather disables attn TP gather padding,
    contrasting with/without the flag under --moe-dense-tp-size 1 on OLMoE-1B-7B

    [Test Category] Parameter
    [Test Target] --disable-attn-tp-gather
    """

    model = OLMOE_1B_7B_0924_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    def test_disable_attn_tp_gather_contrastive(self):
        prompts = [
            "The capital of France is",
            "What is the largest planet in our solar system?",
        ]

        # Phase 1: WITHOUT --disable-attn-tp-gather
        # --moe-dense-tp-size 1 triggers the MOE branch in require_attn_tp_gather()
        # → returns True → global_num_tokens_cpu = [num_tokens] → gather buffer allocated
        process1 = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--moe-dense-tp-size", "1",
                "--trust-remote-code",
                "--mem-fraction-static", "0.8",
                "--attention-backend", "ascend",
            ],
        )
        try:
            resp1 = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompts,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(resp1.status_code, 200)
        finally:
            kill_process_tree(process1.pid)

        # Phase 2: WITH --disable-attn-tp-gather
        # Short-circuits require_attn_tp_gather() at common.py:3227-3228
        # → returns False → global_num_tokens_cpu = None → no gather buffer
        process2 = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--moe-dense-tp-size", "1",
                "--disable-attn-tp-gather",
                "--trust-remote-code",
                "--mem-fraction-static", "0.8",
                "--attention-backend", "ascend",
            ],
        )
        try:
            resp2 = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompts,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(resp2.status_code, 200)
        finally:
            kill_process_tree(process2.pid)

        # Both paths produce correct output
        self.assertIn("Paris", resp1.text)
        self.assertIn("Paris", resp2.text)
        self.assertIn("Jupiter", resp2.text)


if __name__ == "__main__":
    unittest.main()
