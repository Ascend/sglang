import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=100, suite="full-1-npu-a3", nightly=True)


class TestMatchedStop(CustomTestCase, MatchedStopMixin):
    """Testcase: Test configuring 'matched_stop' to different values(string, EOS token, length, Regular) correctly identifies
    it as a stop signal.

    [Test Category] Interface
    [Test Target] /v1/chat/completions; /v1/completions
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=300,
            other_args=[
                "--max-running-requests",
                "10",
                "--attention-backend",
                "ascend",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
