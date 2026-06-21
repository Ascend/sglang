import os
import unittest
from types import SimpleNamespace

import requests

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=600, suite="nightly-16-npu-a3", nightly=True)


class TestNPUDisaggregationMooncakeFailure(PDDisaggregationServerBase):
    """Test Mooncake failure handling in disaggregation mode on NPU.

    [Test Category] Stability
    [Test Target] Mooncake transfer backend failure recovery
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ["DISAGGREGATION_TEST_FAILURE_PROB"] = "0.05"
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24667"
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        # Use ascend transfer backend for NPU
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.rdma_devices = []
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("DISAGGREGATION_TEST_FAILURE_PROB", None)
        os.environ.pop("ASCEND_MF_STORE_URL", None)
        super().tearDownClass()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
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
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "0",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k_with_failure_simulation(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )

        try:
            metrics = run_eval(args)
        except Exception as e:
            try:
                response = requests.get(self.prefill_url + "/health_generate")
                assert response.status_code == 200
                response = requests.get(self.decode_url + "/health_generate")
                assert response.status_code == 200
            except Exception as health_check_error:
                raise e from health_check_error


if __name__ == "__main__":
    unittest.main()
