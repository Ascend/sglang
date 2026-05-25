import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)

HIGH_ACCEPT_PROMPT = (
    "Output exactly 128 new lines. "
    "Every line must be READY. "
    "Do not add numbering, punctuation, or commentary."
)

LOW_ACCEPT_PROMPT = (
    "Compose a poem in the style of Emily Dickinson about quantum entanglement. "
    "Make it emotionally resonant and at least 100 words."
)

MAX_UPSHIFT_ATTEMPTS = 4
MAX_DOWNSHIFT_ATTEMPTS = 6


class TestNPUAdaptiveSpeculativeServer(CustomTestCase):
    """Test adaptive speculative decoding on NPU with state switching.

    [Test Category] Integration
    [Test Target] EAGLE3, Adaptive Speculative, GSM8K
    """

    model = QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH
    draft_model = QWEN3_32B_EAGLE3_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "candidate_steps": [1, 3],
                    "ema_alpha": 1.0,
                    "warmup_batches": 1,
                    "update_interval": 1,
                    "up_hysteresis": 0.0,
                },
                f,
            )
            cls.adaptive_config_path = f.name

        os.environ["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
        os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
        cls.env = os.environ.copy()
        
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "ascend",
                    "--quantization",
                    "modelslim",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model-path",
                    cls.draft_model,
                    "--speculative-draft-model-quantization",
                    "unquant",
                    "--speculative-num-steps",
                    "1",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "2",
                    "--speculative-adaptive",
                    "--speculative-adaptive-config",
                    cls.adaptive_config_path,
                    "--skip-server-warmup",
                    "--mem-fraction-static",
                    "0.7",
                    "--tp-size",
                    "8",
                    "--disable-cuda-graph",
                ],
                env=cls.env,
            )
        except Exception:
            os.unlink(cls.adaptive_config_path)
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)
        if os.path.exists(cls.adaptive_config_path):
            os.unlink(cls.adaptive_config_path)

    def _get_internal_state(self) -> dict:
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("internal_states", data)
        self.assertGreater(len(data["internal_states"]), 0)
        return data["internal_states"][0]

    def test_state_switching_high_accept(self):
        for i in range(MAX_UPSHIFT_ATTEMPTS):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": HIGH_ACCEPT_PROMPT,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 200},
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            text = response.json()["text"]
            self.assertIn("READY", text)

        state = self._get_internal_state()
        self.assertIn("speculative_num_steps", state)
        speculative_num_steps = state["speculative_num_steps"]
        self.assertGreaterEqual(
            speculative_num_steps,
            3,
            f"Expected speculative_num_steps >= 3 for high accept prompt, got {speculative_num_steps}",
        )

    def test_state_switching_low_accept(self):
        for i in range(MAX_DOWNSHIFT_ATTEMPTS):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": LOW_ACCEPT_PROMPT,
                    "sampling_params": {"temperature": 0.1, "max_new_tokens": 150},
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)

        state = self._get_internal_state()
        self.assertIn("speculative_num_steps", state)
        speculative_num_steps = state["speculative_num_steps"]
        self.assertLessEqual(
            speculative_num_steps,
            1,
            f"Expected speculative_num_steps <= 1 for low accept prompt, got {speculative_num_steps}",
        )

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.20)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        self.assertGreater(avg_spec_accept_length, 2.0)


if __name__ == "__main__":
    unittest.main()