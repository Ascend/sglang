import unittest

from sglang.srt.environ import envs
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

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestNPUMixedChunkedPrefill(GSM8KMixin, CustomTestCase):
    """Test mixed chunked prefill on NPU.

    [Test Category] Scheduler
    [Test Target] Mixed chunk prefill functionality, chunked prefill size configuration
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_accuracy_thres = 0.30

    extra_args = [
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "128",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=cls.extra_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestNPUMixedChunkedPrefillNoRadixCache(TestNPUMixedChunkedPrefill):
    """Test mixed chunked prefill without radix cache on NPU.

    [Test Category] Scheduler
    [Test Target] Mixed chunk prefill without radix cache
    """

    extra_args = [
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "128",
        "--disable-radix-cache",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]


if __name__ == "__main__":
    unittest.main()
