import os
import time
import unittest
from types import SimpleNamespace

from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class TestDisaggregationPrefillPPAccuracy(TestDisaggregationBase):
    """Test Case: Verify the accuracy of base model when only prefill enables PP parallelism in PD disaggregation scenario

    [Test Category] Parameter
    [Test Target] --pp-size
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()
        os.environ["OPENAI_API_KEY"] = "sk-123456"
        os.environ["OPENAI_API_BASE"] = f"http://{cls.base_host}:{cls.lb_port}/v1"

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_API_BASE", None)
        super().tearDownClass()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "1",
            "--pp-size",
            "4",
            "--disable-overlap-schedule",
            "--attention-backend",
            "ascend",
            "--disaggregation-transfer-backend",
            "ascend",
        ]
        prefill_args += cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "1",
            "--base-gpu-id",
            "8",
            "--attention-backend",
            "ascend",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disable-overlap-schedule",
            "--disable-cuda-graph",
        ]
        decode_args += cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.lb_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["score"], 0.24)
        # Wait a little bit so that the memory check happens.
        time.sleep(5)


class TestDisaggregationDecodePPAccuracy(TestDisaggregationBase):
    """Test Case: Verify the accuracy of base model when both prefill and decode enable PP parallelism in PD disaggregation scenario

    [Test Category] Parameter
    [Test Target] --pp-size; --pp-async-batch-depth; --pp-max-micro-batch-size
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()
        os.environ["OPENAI_API_KEY"] = "sk-123456"
        os.environ["OPENAI_API_BASE"] = f"http://{cls.base_host}:{cls.lb_port}/v1"

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_API_BASE", None)
        super().tearDownClass()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "2",
            "--pp-size",
            "4",
            "--pp-async-batch-depth",
            "2",
            "--pp-max-micro-batch-size",
            "2",
            "--disable-overlap-schedule",
            "--attention-backend",
            "ascend",
            "--disaggregation-transfer-backend",
            "ascend",
        ]
        prefill_args += cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "2",
            "--pp-size",
            "4",
            "--base-gpu-id",
            "8",
            "--attention-backend",
            "ascend",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disable-overlap-schedule",
            "--disable-cuda-graph",
        ]
        decode_args += cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.lb_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["score"], 0.24)
        # Wait a little bit so that the memory check happens.
        time.sleep(5)


if __name__ == "__main__":
    unittest.main()
