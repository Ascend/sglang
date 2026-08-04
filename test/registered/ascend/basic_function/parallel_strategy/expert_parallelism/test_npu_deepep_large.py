import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=2000, suite="full-2-npu-a3", nightly=True)


class TestDeepseek(CustomTestCase):
    """Testcase: Verify DP superimposed EP scenario the inference accuracy of the model on the
    GSM8K dataset is no less than 0.34.

    [Test Category] Parameters
    [Test Target] DP + EP
    """
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--enable-dp-attention",
                "--dp",
                "2",
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
                "256",
                "--max-running-requests",
                "2048",
                "--disable-radix-cache",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true,"num_threads": 64}',
                "--mem-fraction-static",
                "0.69",
                "--attention-backend",
                "ascend",
            ],
            env={
                "DEEPEP_HCCL_BUFFSIZE": "1800",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
            },
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

        self.assertGreater(metrics["score"], 0.34)


class TestDeepseekMTP(CustomTestCase):
    """Testcase: Verify MTP superimposed EP scenario the inference accuracy of the model on the
    GSM8K dataset is no less than 0.34.

    [Test Category] Parameters
    [Test Target] MTP + EP
    """
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-overlap-schedule",
                "--trust-remote-code",
                "--tp",
                "2",
                "--enable-dp-attention",
                "--dp",
                "2",
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
                "--mem-fraction-static",
                "0.55",
                "--attention-backend",
                "ascend",
                "--quantization",
                "modelslim",
            ],
            env={
                "DEEPEP_HCCL_BUFFSIZE": "1800",
                "SGLANG_ENABLE_SPEC_V2": "1",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
            },
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

        self.assertGreater(metrics["score"], 0.34)

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
