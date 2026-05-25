import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="nightly-8-npu-a3", nightly=True)


class TestNPUEAGLE3ServerBasic(CustomTestCase):
    """Test EAGLE3 server on NPU with GSM8K, radix cache, and request abort.

    [Test Category] Integration
    [Test Target] EAGLE3, GSM8K, Radix Cache, Request Abort
    """

    model = QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH
    draft_model = QWEN3_32B_EAGLE3_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    spec_steps = 5
    spec_topk = 1
    spec_draft_tokens = 6

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
        os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
        cls.env = os.environ.copy()
        
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "modelslim",
            "--disable-radix-cache",
            "--chunked-prefill-size",
            "1024",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-num-steps",
            str(cls.spec_steps),
            "--speculative-eagle-topk",
            str(cls.spec_topk),
            "--speculative-num-draft-tokens",
            str(cls.spec_draft_tokens),
            "--mem-fraction-static",
            "0.7",
            "--max-running-requests",
            "8",
            "--tp-size",
            "8",
            "--disable-cuda-graph",
        ]
        
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
            env=cls.env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def send_request(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)

    def send_requests_abort(self):
        rid = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Write a long story about" + " love " * 100,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1000},
            },
            timeout=60,
        ).json()["meta_info"]["id"]

        time.sleep(0.5)
        response = requests.post(
            self.base_url + "/abort_request",
            json={"rid": rid},
            timeout=10,
        )
        self.assertEqual(response.status_code, 200)

    def test_request_abort(self):
        concurrency = 4
        threads = [
            threading.Thread(target=self.send_request) for _ in range(concurrency)
        ] + [
            threading.Thread(target=self.send_requests_abort)
            for _ in range(concurrency)
        ]
        for worker in threads:
            worker.start()
        for p in threads:
            p.join()

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

        if self.spec_topk == 1:
            self.assertGreater(avg_spec_accept_length, 2.5)
        else:
            self.assertGreater(avg_spec_accept_length, 3.47)

        time.sleep(4)


class TestNPUEAGLE3ServerAdditional(TestNPUEAGLE3ServerBasic):
    """Test EAGLE3 server with different topk and steps configuration.

    [Test Category] Integration
    [Test Target] EAGLE3, Config Variation
    """

    spec_topk = 5
    spec_steps = 8
    spec_draft_tokens = 64

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
        os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
        cls.env = os.environ.copy()
        
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "modelslim",
            "--disable-radix-cache",
            "--chunked-prefill-size",
            "1024",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-num-steps",
            str(cls.spec_steps),
            "--speculative-eagle-topk",
            str(cls.spec_topk),
            "--speculative-num-draft-tokens",
            str(cls.spec_draft_tokens),
            "--mem-fraction-static",
            "0.7",
            "--max-running-requests",
            "8",
            "--tp-size",
            "8",
            "--disable-cuda-graph",
        ]
        
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
            env=cls.env,
        )


import os