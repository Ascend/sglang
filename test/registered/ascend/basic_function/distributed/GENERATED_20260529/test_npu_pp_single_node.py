import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=550,
    suite="nightly-4-npu-a3",
    nightly=True,
)


class TestNPUPPAccuracy(CustomTestCase):
    """Test pipeline parallelism accuracy on NPU.

    [Test Category] Distributed
    [Test Target] Pipeline parallelism (PP=2, TP=1)
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--disable-cuda-graph",
            "--tp-size",
            "1",
            "--pp-size",
            "2",
            "--chunked-prefill-size",
            "256",
            "--mem-fraction-static",
            "0.3",
        ]
        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model_path,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=64,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.60)
        time.sleep(4)

    def test_logprob(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
                "logprob_start_len": 0,
            },
        )
        response_json = response.json()
        input_token_logprobs = response_json["meta_info"]["input_token_logprobs"]
        output_token_logprobs = response_json["meta_info"]["output_token_logprobs"]
        output_top_logprobs = response_json["meta_info"]["output_top_logprobs"]
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(input_token_logprobs, list)
        self.assertIsInstance(output_token_logprobs, list)
        self.assertIsInstance(output_top_logprobs, list)


class TestNPUPPMixedChunk(CustomTestCase):
    """Test pipeline parallelism with mixed chunk on NPU.

    [Test Category] Distributed
    [Test Target] PP with mixed chunk prefill (PP=2, TP=1)
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--disable-cuda-graph",
            "--tp-size",
            "1",
            "--pp-size",
            "2",
            "--enable-mixed-chunk",
            "--chunked-prefill-size",
            "256",
            "--mem-fraction-static",
            "0.3",
        ]
        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model_path,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=64,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.60)


if __name__ == "__main__":
    unittest.main()
