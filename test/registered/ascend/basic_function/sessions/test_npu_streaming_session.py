"""
Streaming session tests for NPU.

Tests:
  - KV cache inheritance
  - Concurrent logprob leak detection
  - Abort recovery
  - Long session stability
  - EAGLE3 speculative decoding
"""

import os
import unittest

from sglang.srt.environ import envs
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.streaming_session_kit import StreamingSessionKitMixin
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
        "4",
    ]
    kv_inherit_offsets = (0,)


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
        "4",
    ]
    kv_inherit_offsets = (-1,)


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
        "0.7",
        "--page-size",
        "256",
    ]
    kv_inherit_offsets = (-1,)


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
        "4",
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
        "4",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]
    kv_inherit_offsets = (-1,)


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
        "0.7",
        "--page-size",
        "256",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]
    kv_inherit_offsets = (-1,)


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
