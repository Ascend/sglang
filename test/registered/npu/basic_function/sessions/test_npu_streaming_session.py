"""
Streaming session tests for NPU.

Tests:
  - KV cache inheritance
  - Concurrent logprob leak detection
  - Abort recovery
  - Long session stability
  - EAGLE3 speculative decoding

Uses the NPU kit `sglang.test.ascend.streaming_session_kit`.
"""

import os
import unittest

from sglang.srt.environ import envs
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.streaming_session_kit import StreamingSessionKitMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.streaming_session_fixture import (
    StreamingSessionServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class NPUStreamingSessionServerBase(StreamingSessionServerBase):
    npu_env = {
        **os.environ,
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_EXEC_TIMEOUT": "200",
    }

    @classmethod
    def setUpClass(cls):
        import contextlib

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1)
            )
            stack.enter_context(envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(True))
            for name, val in cls.env_overrides:
                stack.enter_context(getattr(envs, name).override(val))
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--enable-streaming-session"] + list(cls.extra_args),
                env=cls.npu_env,
            )
        cls.tokenizer = get_tokenizer(cls.model)


class TestNPUStreamingSession(NPUStreamingSessionServerBase, StreamingSessionKitMixin):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "16",
    ]
    kv_inherit_offsets = (0,)
    # NPU floors inherited KV to the page boundary — mirror --page-size 16.
    # The original un-repeated first chunk (8 prompt + 12 completion = 20
    # tokens) already spans 2 pages, so the floored expectation (16) stays
    # non-trivial.
    kv_inherit_page_size = 16
    kv_inherit_first_chunk_repeats = 1


class TestNPUStreamingSessionLargePage(TestNPUStreamingSession):
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "256",
    ]
    # Mirror --page-size 256. The first chunk is scaled (×40 ≈ 320 tokens
    # + 12 completion) so turn-1 KV spans multiple 256-pages; with the
    # original un-repeated chunk (20 tokens) the floored expectation would
    # be 0 and the assert vacuous. The longer chunk needs a larger session
    # capacity (~1560 chars).
    kv_inherit_page_size = 256
    kv_inherit_first_chunk_repeats = 40
    kv_inherit_session_capacity = 4000


class TestNPUStreamingSessionEagle3(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_8B_EAGLE3_WEIGHTS_PATH,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "16",
    ]
    # NPU EAGLE3 may or may not commit the last sampled token before
    # max_new stops (kv_committed_len = input + output - 1 or +0).
    # Tolerate both behaviors.
    kv_inherit_offsets = (-1, 0)


class TestNPUStreamingSessionEagle3LargePage(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_8B_EAGLE3_WEIGHTS_PATH,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.6",
        "--page-size",
        "256",
    ]
    # NPU EAGLE3 may or may not commit the last sampled token before
    # max_new stops (kv_committed_len = input + output - 1 or +0).
    # Tolerate both behaviors.
    kv_inherit_offsets = (-1, 0)
    # Mirror --page-size 256; scale the first chunk so turn-1 KV spans
    # multiple pages (floored expectation 256, not 0).
    kv_inherit_page_size = 256
    kv_inherit_first_chunk_repeats = 40
    kv_inherit_session_capacity = 4000


class TestNPUStreamingSessionRetract(TestNPUStreamingSession):
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "16",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


class TestNPUStreamingSessionEagle3Retract(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_8B_EAGLE3_WEIGHTS_PATH,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "16",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]
    # NPU EAGLE3 may or may not commit the last sampled token before
    # max_new stops (kv_committed_len = input + output - 1 or +0).
    # Tolerate both behaviors.
    kv_inherit_offsets = (-1, 0)


class TestNPUStreamingSessionEagle3RetractLargePage(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_8B_EAGLE3_WEIGHTS_PATH,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.6",
        "--page-size",
        "256",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]
    kv_inherit_offsets = (-1,)
    # Mirror --page-size 256; scale the first chunk so turn-1 KV spans
    # multiple pages (floored expectation 256, not 0).
    kv_inherit_page_size = 256
    kv_inherit_first_chunk_repeats = 40
    kv_inherit_session_capacity = 4000


__all__ = [
    "TestNPUStreamingSession",
    "TestNPUStreamingSessionLargePage",
    "TestNPUStreamingSessionEagle3",
    "TestNPUStreamingSessionEagle3LargePage",
    "TestNPUStreamingSessionRetract",
    "TestNPUStreamingSessionEagle3Retract",
    "TestNPUStreamingSessionEagle3RetractLargePage",
]


if __name__ == "__main__":
    unittest.main()
