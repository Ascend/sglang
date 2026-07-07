import os
import re
import unittest

import requests

from sglang.test.ascend.test_ascend_utils import MIMO_V2_FLASH_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestMimoV2AcceptRate(CustomTestCase):
    """.

    [Test Category]
    [Test Target]
    """

    accept_rate = 0.25
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 1800

    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env["STREAMS_PER_DEVICE"] = "32"
        env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "32"
        env["HCCL_BUFFSIZE"] = "32"
        env["HCCL_OP_EXPANSION_MODE"] = "AIV"
        env["HCCL_SOCKET_IFNAME"] = "lo"
        env["GLOO_SOCKET_IFNAME"] = "lo"
        env["SGLANG_NPU_PROFILING"] = "0"
        env["SGLANG_NPU_PROFILING_STAGE"] = "prefill"
        env["DEEPEP_NORMAL_LONG_SEQ_ROUND"] = "32"
        env["DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS"] = "3584"
        env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "3600"
        env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "3600"
        env["SGLANG_ENABLE_SPEC_V2"] = "1"
        env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
        env["DEEPNORMAL_MODE_USE_INT8_QUANT"] = "1"
        env["SGLANG_DEEPEP_BF16_DISPATCH"] = "0"
        cls.model = MIMO_V2_FLASH_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--attention-backend",
            "ascend",
            "--tp-size",
            16,
            "--trust-remote-code",
            "--max-running-requests",
            128,
            "--mem-fraction-static",
            "0.83",
            "--swa-full-tokens-ratio",
            0.3,
            "--cuda-graph-bs",
            1,
            2,
            4,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            40,
            48,
            56,
            64,
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            3,
            "--speculative-eagle-topk",
            1,
            "--speculative-num-draft-tokens",
            4,
            "--enable-multi-layer-eagle",
            "--disable-radix-cache",
        ]

        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    def test_request(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 256,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertIn("accept rate", content)
        matches = re.findall(r"accept rate:\s*([\d.]+)", content)
        self.assertTrue(len(matches) > 0)
        current_accept_rate = float(matches[-1])
        self.assertGreater(current_accept_rate, self.accept_rate)

    @classmethod
    def tearDownClass(cls):
        """Clean up after the test class by killing the server process and removing generated directories."""
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")


if __name__ == "__main__":
    unittest.main()
