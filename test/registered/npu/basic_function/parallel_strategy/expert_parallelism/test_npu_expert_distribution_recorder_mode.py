import glob
import os
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
    run_command,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="base-b-test-2-npu-a3")
register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)


class TestExpertDistributionRecorderModeStatic(CustomTestCase):
    """Testcase: Verify set the parameter --expert-distribution-recorder-mode，
    will generate .pt file and the inference request successfully.Set the parameter --expert-balancedness-report-mode,
    the expert load‑balancing degree metric reported.

    [Test Category] Parameter
    [Test Target] --expert-distribution-recorder-mode, --expert-balancedness-report-mode
    """

    expert_distribution_recorder_mode = "stat"
    expert_balancedness_report_mode = "off"
    path = "/tmp/pt"

    @classmethod
    def setUpClass(cls):
        cls.out_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.err_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.process = popen_launch_server(
            QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
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
                "--enable-metrics",
                "--expert-distribution-recorder-mode",
                cls.expert_distribution_recorder_mode,
                "--expert-balancedness-report-mode",
                cls.expert_balancedness_report_mode,
            ],
            env={
                "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1",
                "HCCL_BUFFSIZE": "1024",
                "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR": f"{cls.path}",
                "TRANSFORMERS_VERBOSITY": "error",
            },
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        run_command(f"rm -rf {cls.path}")
        cls.out_file.close()
        cls.err_file.close()
        os.unlink(cls.out_file.name)
        os.unlink(cls.err_file.name)

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
        distribution_recorder_suffixes = ["*.pt"]
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

    def test_expert_balancedness_report_mode(self):
        # When --expert-balancedness-report-mode is off, neither logs nor metrics are recorded.
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
        self.assertNotIn("eplb_balancedness", response.text)
        self.err_file.seek(0)
        content = self.err_file.read()
        self.assertNotIn("Expert Balancedness", content)


class TestExpertDistributionRecorderModeStatApprox(
    TestExpertDistributionRecorderModeStatic
):
    expert_distribution_recorder_mode = "stat_approx"
    expert_balancedness_report_mode = "prometheus"

    def test_expert_balancedness_report_mode(self):
        # When --expert-balancedness-report-mode is prometheus, there are records in metrics, but none in the logs.
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
        self.assertIn("eplb_balancedness", response.text)
        self.err_file.seek(0)
        content = self.err_file.read()
        self.assertNotIn("Expert Balancedness", content)


class TestExpertDistributionRecorderPerPass(TestExpertDistributionRecorderModeStatic):
    expert_distribution_recorder_mode = "per_pass"
    expert_balancedness_report_mode = "both"

    def test_expert_balancedness_report_mode(self):
        # When --expert-balancedness-report-mode is both, both logs and metrics are recorded.
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
        self.assertIn("eplb_balancedness", response.text)
        self.err_file.seek(0)
        content = self.err_file.read()
        self.assertIn("Expert Balancedness", content)


class TestExpertDistributionRecorderPerToken(TestExpertDistributionRecorderModeStatic):
    # When --expert-balancedness-report-mode is server_log, only recorded in the logs
    expert_distribution_recorder_mode = "per_token"
    expert_balancedness_report_mode = "server_log"

    def test_expert_balancedness_report_mode(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
        self.assertNotIn("eplb_balancedness", response.text)
        self.err_file.seek(0)
        content = self.err_file.read()
        self.assertIn("Expert Balancedness", content)


if __name__ == "__main__":
    unittest.main()
