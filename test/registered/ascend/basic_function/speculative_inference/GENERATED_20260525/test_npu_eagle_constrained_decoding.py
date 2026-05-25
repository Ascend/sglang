import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    LLAMA_3_8B_EAGLE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-4-npu-a3", nightly=True)


class TestNPUEAGLEConstrainedDecoding(
    CustomTestCase, RegexConstrainedMixin, JSONConstrainedMixin
):
    """Test EAGLE constrained decoding on NPU with regex and JSON grammar.

    [Test Category] Integration
    [Test Target] EAGLE, Constrained Decoding, Grammar Backend
    """

    max_running_requests = 64
    attention_backend = "ascend"
    spec_steps = 5
    spec_topk = 1
    spec_draft_tokens = 6
    page_size = 1
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    draft_model = LLAMA_3_8B_EAGLE_WEIGHTS_PATH
    grammar_backend = "xgrammar"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
        cls.env = os.environ.copy()
        
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model",
            cls.draft_model,
            "--speculative-num-steps",
            str(cls.spec_steps),
            "--speculative-eagle-topk",
            str(cls.spec_topk),
            "--speculative-num-draft-tokens",
            str(cls.spec_draft_tokens),
            "--page-size",
            str(cls.page_size),
            "--mem-fraction-static",
            "0.7",
            "--max-running-requests",
            str(cls.max_running_requests),
            "--grammar-backend",
            cls.grammar_backend,
            "--tp-size",
            "4",
            "--disable-cuda-graph",
        ]
        
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
            env=cls.env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestNPUEAGLEConstrainedDecodingV2(TestNPUEAGLEConstrainedDecoding):
    """Test EAGLE V2 constrained decoding on NPU.

    [Test Category] Integration
    [Test Target] EAGLE V2, Constrained Decoding
    """

    pass


if __name__ == "__main__":
    unittest.main()