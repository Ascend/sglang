import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=500, suite="full-4-npu-a3", nightly=True)


class TestQwen330BAttnCP(GSM8KAscendMixin, CustomTestCase):
    """GSM8K accuracy test for Qwen3-30B-A3B mixed deployment on 4 NPUs.

    The test uses:
    - TP = 4
    - MOE_DP = 2
    - ATTN_CP = 2
    - prefill context parallel enabled

    This is the mixed/co-located deployment variant and reuses the Ascend
    environment variables from the PD GSM8K test.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.out_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.err_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.7",
                "--max-running-requests",
                "32",
                "--attention-backend",
                "ascend",
                "--tp-size",
                "4",
                "--moe-dp-size",
                "2",
                "--attn-cp-size",
                "2",
                "--cuda-graph-max-bs-decode",
                "32",
                "--enable-prefill-context-parallel",
            ],
            return_stdout_stderr=(cls.out_file, cls.err_file),
            env={
                **os.environ,
                "ASCEND_USE_FIA": "1",
                "SGLANG_ENABLE_CP_V2": "0",
            }
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_file.close()
        cls.err_file.close()
        os.unlink(cls.out_file.name)
        os.unlink(cls.err_file.name)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")
        self.assertGreater(metrics["score"], 0.92)

    # Setting the --moe-dp-size parameter, MOE_DP log will output
    def test_moe_dp(self):
        self.err_file.seek(0)
        content = self.err_file.read()
        self.assertIn("MOE_DP", content)


if __name__ == "__main__":
    unittest.main()
