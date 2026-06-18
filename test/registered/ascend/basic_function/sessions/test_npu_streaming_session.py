import unittest

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.streaming_session_kit import (
    AbortLeakReproKitMixin,
    StreamingSessionKitMixin,
)
from sglang.test.server_fixtures.streaming_session_fixture import (
    ABORT_REPRO_CHUNKED_PREFILL_SIZE,
    ABORT_REPRO_CONTEXT_LEN,
    ABORT_REPRO_PAGE_SIZE,
)

register_npu_ci(est_time=691, suite="nightly-2-npu-a3", nightly=True)


class TestNPUStreamingSession(unittest.TestCase, StreamingSessionKitMixin):
    """Test streaming session functionality on NPU.

    [Test Category] Feature
    [Test Target] StreamingSession, KV cache inheritance, abort recovery
    """

    model = QWEN3_0_6B_WEIGHTS_PATH
    extra_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--chunked-prefill-size",
        "512",
        "--mem-fraction-static",
        "0.70",
    ]

    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer
        from sglang.test.test_utils import (
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            DEFAULT_URL_FOR_TEST,
            popen_launch_server,
        )

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-streaming-session"] + cls.extra_args,
        )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.utils import kill_process_tree

        kill_process_tree(cls.process.pid)


class TestNPUStreamingSessionAbortLeakRepro(unittest.TestCase, AbortLeakReproKitMixin):
    """Test abort-heavy chunked prefill leak repro on NPU.

    [Test Category] Stress
    [Test Target] StreamingSession abort recovery, memory leak prevention
    """

    model = QWEN3_0_6B_WEIGHTS_PATH
    extra_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--chunked-prefill-size",
        str(ABORT_REPRO_CHUNKED_PREFILL_SIZE),
        "--context-length",
        str(ABORT_REPRO_CONTEXT_LEN),
        "--page-size",
        str(ABORT_REPRO_PAGE_SIZE),
        "--max-running-requests",
        "32",
        "--log-level",
        "info",
        "--mem-fraction-static",
        "0.70",
    ]

    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer
        from sglang.test.test_utils import (
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            DEFAULT_URL_FOR_TEST,
            popen_launch_server,
        )

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-streaming-session"] + cls.extra_args,
        )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.utils import kill_process_tree

        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
