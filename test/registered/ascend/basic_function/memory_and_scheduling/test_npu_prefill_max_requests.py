import os
import re
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestModeImpl(CustomTestCase):
    """Testcase: Verify --prefill-max-requests takes effect correctly by checking log.

    [Test Category] Parameter
    [Test Target] --prefill-max-requests
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    PREFILL_MAX_REQUESTS = 5

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.log_file = "./server.log"

        with open(cls.log_file, "w", encoding="utf-8") as f:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                    "--model-impl",
                    "transformers",
                    "--prefill-max-requests",
                    str(cls.PREFILL_MAX_REQUESTS),
                    "--trust-remote-code",
                    "--mem-fraction-static",
                    "0.8",
                ],
                return_stdout_stderr=(f, f),
            )

        cls.gsm8k_lower_bound = 0.65

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        os.remove(cls.log_file)

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
        metrics = run_eval(args)
        self.assertGreater(metrics["accuracy"], self.gsm8k_lower_bound)

    def test_prefill_max_requests(self):
        """Verify the running-req in log does not exceed --prefill-max-requests."""
        with open(self.log_file, "r", encoding="utf-8") as f:
            logs = f.read()

        pattern = re.compile(r"prefill batch, #running-req[:\s]+(\d+)", re.I)
        match = pattern.search(logs)

        self.assertIsNotNone(match, "prefill batch, #running-req not found in logs")

        running_req_num = int(match.group(1))

        # Should not exceed the configured maximum value
        self.assertLessEqual(
            running_req_num,
            self.PREFILL_MAX_REQUESTS,
            f"running-req exceeds limit! current={running_req_num}, max allowed={self.PREFILL_MAX_REQUESTS}",
        )


if __name__ == "__main__":
    unittest.main()
