import glob
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import run_command
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestExpertDistributionRecorderModeStatic(CustomTestCase):
    """Testcase: Verify that the model accuracy remains uncompromised when the parameter --moe-dense-tp-size is configured to 1.

    [Test Category] Parameter
    [Test Target] --expert-distribution-recorder-mode
    """

    expert_distribution_recorder_mode = "stat"

    path = "/tmp/pt"

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.8",
                "--tp-size",
                "2",
                "--expert-parallel-size",
                "2",
                "--enable-eplb",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--ep-num-redundant-experts",
                "4",
                "--expert-distribution-recorder-mode",
                cls.expert_distribution_recorder_mode,
            ],
            env={
                "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1",
                "HCCL_BUFFSIZE": "1024",
                "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR": f"{cls.path}",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        run_command(f"rm -rf {cls.path}")

    def test_recorder_mode(self):
        # Start recording
        requests.post(f"{DEFAULT_URL_FOR_TEST}/start_expert_distribution_record")

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )

        # Stop recording
        requests.post(f"{DEFAULT_URL_FOR_TEST}/stop_expert_distribution_record")

        # Export the .pt file
        requests.post(f"{DEFAULT_URL_FOR_TEST}/dump_expert_distribution_record")

        # Check distribution_recorder_files
        distribution_recorder_suffixes = "*.pt"
        distribution_recorder_files = []
        for suffix in distribution_recorder_suffixes:
            distribution_recorder_files.extend(
                glob.glob(os.path.join(self.path, "**", suffix), recursive=True)
            )
        self.assertGreater(
            len(distribution_recorder_files),
            0,
            msg=f"No distribution recorder",
        )


class TestExpertDistributionRecorderModeStatApprox(TestExpertDistributionRecorderModeStatic):
    expert_distribution_recorder_mode = "stat_approx"


class TestExpertDistributionRecorderPerPass(TestExpertDistributionRecorderModeStatic):
    expert_distribution_recorder_mode = "per_pass"


class TestExpertDistributionRecorderPerToken(TestExpertDistributionRecorderModeStatic):
    expert_distribution_recorder_mode = "per_token"


if __name__ == "__main__":
    unittest.main()
