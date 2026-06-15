import os
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.bench_one_batch_server import BenchArgs as OneBatchBenchArgs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_bench_one_batch_server,
)
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH, \
    DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH, \
    GLM_4_5V_WEIGHTS_PATH, QWEN3_0_6B_WEIGHTS_PATH, QWEN3_32B_WEIGHTS_PATH, \
    QWEN3_8B_WEIGHTS_PATH, QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH, QWEN3_30B_A3B_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="debug-full-8-npu-a3", nightly=True)


class TestPPAccuracy(unittest.TestCase):
    """Test Case: Verify the accuracy of LLM models under TP+PP hybrid parallelism

    [Test Category] Parameter
    [Test Target] --pp-size; --tp-size
    """
    @classmethod
    def setUpClass(cls):
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
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.74)
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
                "0.6",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.8)


class TestQwenVLPPAccuracy(TestVLMModels):
    """Test Case: Verify the accuracy of Qwen-VL multimodal model under PP parallelism

    [Test Category] Parameter
    [Test Target] --pp-size
    """
    model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.26
    other_args = [
        "--tp-size",
        "4",
        "--pp-size",
        "4",
        "--chunked-prefill-size",
        "8192",
        "--enable-multimodal",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.6",
        "--disable-cuda-graph",
    ]
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23333"
        cls.api_key = "sk-123456"
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.base_url}/v1"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.65)
        # Wait a little bit so that the memory check happens.
        time.sleep(4)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmmu(self):
        self._run_vlm_mmmu_test()


class TestQwenPPAccuracy(unittest.TestCase):
    """Test Case: Verify the accuracy consistency of Qwen model between PP=1 and PP=2

    [Test Category] Parameter
    [Test Target] --pp-size
    """
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23334"  # different ports to avoid conflicts
        cls.model_name = QWEN3_8B_WEIGHTS_PATH  # replace with your Qwen Model if needed

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                "0.8",
                "--disable-cuda-graph",
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=4)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["accuracy"], 0.74)
        self.assertGreaterEqual(
            pp_metrics["accuracy"],
            baseline["accuracy"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 2% compared to baseline. "
                f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
            ),
        )


class TestQwenPPTieWeightsAccuracy(unittest.TestCase):
    """Test Case: Verify the accuracy consistency of Qwen3-0.6B model (with tie_word_embeddings) between PP=1 and PP=2

    [Test Category] Parameter
    [Test Target] --pp-size
    """
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23335"  # different ports to avoid conflicts
        cls.model_name = QWEN3_0_6B_WEIGHTS_PATH  # qwen3 < 8B all have tie_word_embeddings = True

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                "0.8",
                "--disable-cuda-graph",
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=4)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["accuracy"], 0.38)
        self.assertGreaterEqual(
            pp_metrics["accuracy"],
            baseline["accuracy"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 2% compared to baseline. "
                f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
            ),
        )


class TestQwenMoePPAccuracy(unittest.TestCase):
    """Test Case: Verify the accuracy consistency of Qwen3-30B-A3B MOE model between PP=1 and PP=2

    [Test Category] Parameter
    [Test Target] --pp-size
    """
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23336"  # different ports to avoid conflicts
        cls.model_name = QWEN3_30B_A3B_WEIGHTS_PATH  # replace with your Qwen Model if needed

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--tp-size",
                "2",
                "--chunked-prefill-size",
                256,
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                "0.8",
                "--disable-cuda-graph",
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=4)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["accuracy"], 0.74)
        self.assertGreaterEqual(
            pp_metrics["accuracy"],
            baseline["accuracy"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 2% compared to baseline. "
                f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
            ),
        )


class TestQwen35PPAccuracy(unittest.TestCase):
    """Test Case: Verify the accuracy consistency of Qwen model between PP=1 and PP=2

    [Test Category] Parameter
    [Test Target] --pp-size
    """
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23337"  # different ports to avoid conflicts
        cls.model_name = QWEN3_32B_WEIGHTS_PATH # replace with your Qwen Model if needed

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                2,
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                "0.8",
                "--disable-cuda-graph",
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=4)

        print(f"[Qwen35 PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["accuracy"], 0.83)
        self.assertGreaterEqual(
            pp_metrics["accuracy"],
            baseline["accuracy"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 2% compared to baseline. "
                f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
            ),
        )


class TestFixedBugs(unittest.TestCase):
    """Test Case: Verify normal inference under small batch size scenario with PP+chunked-prefill enabled
    [Test Category] Parameter
    [Test Target] --pp-size; --chunked-prefill
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
            "--chunked-prefill",
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


@unittest.skipIf(True, "issue  #343")
class TestGLM41VPPAccuracy(unittest.TestCase):
    """Test Case: Verify the accuracy of GLM multimodal model under PP parallelism

    [Test Category] Parameter
    [Test Target] --pp-size
    """
    @classmethod
    def setUpClass(cls):
        cls.model = GLM_4_5V_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
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
                "8192",
                "--enable-multimodal",
                "--reasoning-parser",
                "glm45",
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

    def test_mmmu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmmu",
            num_examples=None,
            num_threads=32,
            response_answer_regex=r"<\|begin_of_box\|>(.*)<\|end_of_box\|>",
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.45)


if __name__ == "__main__":
    unittest.main()
