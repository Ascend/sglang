"""
Streaming session tests for NPU.

Tests:
  - KV cache inheritance
  - Concurrent logprob leak detection
  - Abort recovery
  - Long session stability
  - EAGLE3 speculative decoding
"""

import unittest

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.streaming_session_kit import StreamingSessionKitMixin

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNPUStreamingSession(StreamingSessionKitMixin, unittest.TestCase):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "4",
    ]
    kv_inherit_offset = 0


class TestNPUStreamingSessionLargePage(TestNPUStreamingSession):
    extra_args = [
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


class TestNPUStreamingSessionEagle3(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
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
        "4",
    ]
    kv_inherit_offset = -1


class TestNPUStreamingSessionEagle3LargePage(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
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
        "256",
    ]
    kv_inherit_offset = -1


class TestNPUStreamingSessionRetract(TestNPUStreamingSession):
    extra_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-streaming-session",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "4",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


class TestNPUStreamingSessionEagle3Retract(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
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
        "4",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]
    kv_inherit_offset = -1


class TestNPUStreamingSessionEagle3RetractLargePage(TestNPUStreamingSession):
    model = QWEN3_8B_WEIGHTS_PATH
    extra_args = [
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
        "256",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]
    kv_inherit_offset = -1


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
