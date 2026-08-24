import time
import unittest
from types import SimpleNamespace

import requests

from sglang.bench_one_batch_server import BenchArgs as OneBatchBenchArgs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH,
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_bench_one_batch_server,
)

register_npu_ci(est_time=10800, suite="full-16-npu-a3", nightly=True)


class TestPPAccuracy(unittest.TestCase):
    """Test Case: Verify the accuracy of LLM models under TP+PP hybrid parallelism

    [Test Category] Parameter
    [Test Target] --pp-size; --tp-size
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = "http://127.0.0.1:23333"
        cls.process = popen_launch_server(
            LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "2",
                "--pp-size",
                "4",
                "--chunked-prefill-size",
                "256",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                "0.8",
                "--disable-cuda-graph",
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
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["score"], 0.74)
        # Wait a little bit so that the memory check happens.
        time.sleep(4)

    def test_logprob(self):
        # Test the format correctness of logprob returned under TP+PP hybrid parallelism
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

        assert len(input_token_logprobs) == 6
        assert len(output_token_logprobs) == 16
        assert len(output_top_logprobs) == 16


class TestDPAttentionDP2PP2(CustomTestCase):
    """Test Case: Verify the accuracy of MLA models under TP+DP+PP hybrid parallelism

    [Test Category] Parameter
    [Test Target] --pp-size; --tp-size; --dp
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--pp-size",
                "4",
                "--enable-dp-attention",
                "--dp",
                "2",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                "0.8",
                "--max-running-requests",
                "32",
                "--context-length",
                "16384",
                "--disable-cuda-graph",
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
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.8)


class TestPPMixedChunk(CustomTestCase):
    """Test Case: Verify the accuracy of base model when PP + mixed chunk are both enabled

    [Test Category] Parameter
    [Test Target] --pp-size; --enable-mixed-chunk
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = "http://127.0.0.1:23338"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "2",
                "--pp-size",
                "4",
                "--chunked-prefill-size",
                "256",
                "--enable-mixed-chunk",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                "0.8",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
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
        print(f"{metrics=}")

        self.assertGreater(metrics["score"], 0.74)
        time.sleep(4)


class TestFixedBugs(unittest.TestCase):
    """Test Case: Verify normal inference under small batch size scenario with PP+chunked-prefill enabled
    [Test Category] Parameter
    [Test Target] --pp-size; --chunked-prefill-size
    """

    def test_chunked_prefill_with_small_bs(self):
        model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        server_args = ServerArgs(model_path=model)
        bench_args = OneBatchBenchArgs(
            batch_size=(1,),
            input_len=(1,),
            output_len=(1,),
            base_url=DEFAULT_URL_FOR_TEST,
        )
        other_server_args = [
            "--tp-size",
            "2",
            "--pp-size",
            "4",
            "--chunked-prefill-size",
            "256",
            "--max-running-requests",
            "2",
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            "0.8",
            "--disable-cuda-graph",
        ]
        run_bench_one_batch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            server_args,
            bench_args,
            other_server_args,
        )


if __name__ == "__main__":
    unittest.main()
