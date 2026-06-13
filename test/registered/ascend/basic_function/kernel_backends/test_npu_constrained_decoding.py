import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.ebnf_constrained_kit import EBNFConstrainedMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class ServerWithGrammar(CustomTestCase):
    """Base server setup for grammar backend testing on NPU.

    [Test Category] Feature
    [Test Target] Constrained Decoding
    """

    backend = "xgrammar"
    disable_overlap = False

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--max-running-requests",
            "10",
            "--grammar-backend",
            cls.backend,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

        if cls.disable_overlap:
            launch_args += ["--disable-overlap-schedule"]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

'''
class TestNPUXGrammarBackend(
    ServerWithGrammar,
    JSONConstrainedMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    """Test xgrammar backend for constrained decoding on NPU.

    [Test Category] Feature
    [Test Target] xgrammar grammar backend
    """

    backend = "xgrammar"


class TestNPUOutlinesBackend(ServerWithGrammar, JSONConstrainedMixin):
    """Test outlines backend for constrained decoding on NPU.

    [Test Category] Feature
    [Test Target] outlines grammar backend
    """

    backend = "outlines"

'''
class TestNPULLGuidanceBackend(ServerWithGrammar, JSONConstrainedMixin):
    """Test llguidance backend for constrained decoding on NPU.

    [Test Category] Feature
    [Test Target] llguidance grammar backend
    """

    backend = "llguidance"

    @unittest.skip("llguidance backend crashes with concurrent json schema requests")
    def test_mix_json_and_other(self):
        pass


if __name__ == "__main__":
    unittest.main()
