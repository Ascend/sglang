import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=420,
    suite="nightly-2-npu-a3",
    nightly=True,
)


class TestNPUDPAttentionDP2TP2(CustomTestCase, GSM8KMixin):
    """Test DP attention with DP=2, TP=2 on NPU.

    [Test Category] Distributed
    [Test Target] DP attention parallelism
    """

    gsm8k_accuracy_thres = 0.55

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--disable-cuda-graph",
            "--tp",
            "2",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--mem-fraction-static",
            "0.3",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestNPUDPAttentionMixedChunk(CustomTestCase, GSM8KMixin):
    """Test DP attention with mixed chunk on NPU.

    [Test Category] Distributed
    [Test Target] DP attention with mixed chunk prefill
    """

    gsm8k_accuracy_thres = 0.55

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--disable-cuda-graph",
            "--tp",
            "2",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--enable-mixed-chunk",
            "--chunked-prefill-size",
            "256",
            "--mem-fraction-static",
            "0.3",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
