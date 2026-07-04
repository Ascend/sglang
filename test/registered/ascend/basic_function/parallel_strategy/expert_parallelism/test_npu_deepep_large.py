import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)

_DEEPEP_ENV = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "HCCL_BUFFSIZE": "1600",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_NPU_USE_MLAPO": "0",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "TASK_QUEUE_ENABLE": "0",
    "TRANSFORMERS_VERBOSITY": "error",
}


class TestDeepseek(CustomTestCase):
    ep_dispatch_algorithm = "dynamic"

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=6000,
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
                "--deepep-mode",
                "auto",
                "--ep-num-redundant-experts",
                "32",
                "--ep-dispatch-algorithm",
                cls.ep_dispatch_algorithm,
                "--eplb-algorithm",
                "deepseek",
                "--quantization",
                "modelslim",
                "--mem-fraction-static",
                "0.82",
                "--disable-cuda-graph",
                "--context-length",
                "8192",
                "--max-prefill-tokens",
                "8192",
                "--max-total-tokens",
                "8192",
                "--max-running-requests",
                "256",
                "--disable-radix-cache",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true,"num_threads": 64}',
            ],
            env=_DEEPEP_ENV,
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
            timeout=6000,
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
                "--deepep-mode",
                "auto",
                "--ep-num-redundant-experts",
                "32",
                "--ep-dispatch-algorithm",
                "dynamic",
                "--eplb-algorithm",
                "deepseek",
                "--quantization",
                "modelslim",
                "--mem-fraction-static",
                "0.82",
                "--disable-cuda-graph",
                "--context-length",
                "8192",
                "--max-prefill-tokens",
                "8192",
                "--max-total-tokens",
                "8192",
                "--max-running-requests",
                "128",
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
            env=_DEEPEP_ENV,
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
