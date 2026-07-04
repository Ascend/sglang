import unittest
from types import SimpleNamespace

import requests

# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH = "/home/weights/DeepSeek-V3.2-W8A8"
register_npu_ci(est_time=200, suite="full-8-npu-a3", nightly=True)


class TestDeepseek(CustomTestCase):
    ep_dispatch_algorithm = "dynamic"

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "16",
                "--enable-dp-attention",
                "--dp",
                "16",
                "--moe-dense-tp-size",
                "1",
                "--enable-dp-lm-head",
                "--moe-a2a-backend",
                "deepep",
                "--ep-num-redundant-experts",
                "32",
                "--ep-dispatch-algorithm",
                cls.ep_dispatch_algorithm,
                "--eplb-algorithm",
                "deepseek",
                "--cuda-graph-bs",
                "256",
                "--max-running-requests",
                "2048",
                "--disable-radix-cache",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true,"num_threads": 64}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1200,
            num_threads=1200,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.95)


class TestDeepseek2(TestDeepseek):
    ep_dispatch_algorithm = "fake"


class TestDeepseekMTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-overlap-schedule",
                "--trust-remote-code",
                "--tp",
                "16",
                "--enable-dp-attention",
                "--dp",
                "16",
                "--moe-dense-tp-size",
                "1",
                "--enable-dp-lm-head",
                "--moe-a2a-backend",
                "deepep",
                "--ep-num-redundant-experts",
                "32",
                "--ep-dispatch-algorithm",
                "dynamic",
                "--eplb-algorithm",
                "deepseek",
                "--cuda-graph-bs",
                "64",
                "--max-running-requests",
                "512",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--disable-radix-cache",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true,"num_threads": 64}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1200,
            num_threads=1200,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.95)

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(
            f"###test_gsm8k:\n"
            f"accuracy={metrics['score']=:.3f}\n"
            f"{avg_spec_accept_length=:.3f}\n"
        )
        self.assertGreater(avg_spec_accept_length, 1.85)


if __name__ == "__main__":
    unittest.main()
