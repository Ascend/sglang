import os
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

from sglang.bench_serving import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class DisaggregationHiCacheBase(PDDisaggregationServerBase):
    """Testcase: optimistic-prefill-retries configuration settings for testing PD disaggregation features.
    Number of optimistic prefill retries that will skip the bootstrap wait.

    [Test Category] Functional
    [Test Target] PD disaggregatio on NPU
    --optimistic-prefill-retries;
    """

    @classmethod
    def setUpClass(cls):
        super(DisaggregationHiCacheBase, cls).setUpClass()

        cls.model = QWEN3_30B_A3B_WEIGHTS_PATH

        cls.tokenizer = get_tokenizer(cls.model)
        cls.temp_dir = tempfile.mkdtemp()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        # Prefill with HiCache enabled
        prefill_args = [
            "--attention-backend",
            "ascend",
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "2",
            "--dtype",
            "bfloat16",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--optimistic-prefill-retries",
            3,
            "--log-requests-level",
            2,
            "--chunked-prefill-size",
            128,
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        env = {
            **os.environ,
            "HCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "HCCL_BUFFSIZE": "1600",
            "SGLANG_SET_CPU_AFFINITY": "1",
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
            "SLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB": "0.1",
        }
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls):
        pass


class TestDisaggregationDecodeWithHiCache(DisaggregationHiCacheBase):
    """Decode startup parameters"""

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--attention-backend",
            "ascend",
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "2",
            "--dtype",
            "bfloat16",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--base-gpu-id",
            "2",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        env = {
            **os.environ,
            "HCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
        }
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def assert_prefill_retry(self):
        # Check for the existence of num_prefill_retries_total.
        result = subprocess.run(
            f"curl -s {self.prefill_url}/metrics | grep num_prefill_retries_total",
            shell=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout
        self.assertIn("num_prefill_retries_total", value)

    def test_gsm8k(self):
        gsm8k_num_shots = 5
        num_questions = 50
        args = SimpleNamespace(
            max_tokens=128,
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=num_questions,
            num_threads=128,
            gsm8k_data_path=None,
            num_shots=gsm8k_num_shots,
        )
        run_eval(args)
        # Triggering optimistic-prefill-retries by sending a large number of requests
        self.assert_prefill_retry()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)


if __name__ == "__main__":
    unittest.main()
