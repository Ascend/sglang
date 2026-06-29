import unittest

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.streaming_session_kit import StreamingSessionKitMixin
from sglang.test.server_fixtures.streaming_session_fixture import (
    StreamingSessionServerBase,
)

register_npu_ci(est_time=691, suite="", nightly=True)

# Qwen3-8B (32K context) — abort_max_new_tokens must fit within context window.
_ABORT_MAX_TOKENS = 20000

# Common ascend args for Qwen3-8B + streaming session
_ASCEND_COMMON_ARGS = [
    "--dtype",
    "bfloat16",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.78",
]


class TestStreamingSessionRetractMixedChunk(
    StreamingSessionServerBase, StreamingSessionKitMixin
):
    """Retract + --enable-mixed-chunk."""

    model = QWEN3_8B_WEIGHTS_PATH
    abort_max_new_tokens = _ABORT_MAX_TOKENS
    extra_args = [
        *_ASCEND_COMMON_ARGS,
        "--chunked-prefill-size",
        "128",
        "--enable-mixed-chunk",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


class TestStreamingSessionRetractLargePage(
    StreamingSessionServerBase, StreamingSessionKitMixin
):
    """Retract + page=256: exercises page-aligned `_free_tail`. Partial-page
    free would corrupt pages still holding committed tokens."""

    model = QWEN3_8B_WEIGHTS_PATH
    abort_max_new_tokens = _ABORT_MAX_TOKENS
    extra_args = [
        *_ASCEND_COMMON_ARGS,
        "--chunked-prefill-size",
        "4096",
        "--page-size",
        "256",
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


# Common EAGLE3 spec args for Qwen3-8B.
_EAGLE3_SPEC_ARGS = [
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
]


class TestStreamingSessionEagle(StreamingSessionServerBase, StreamingSessionKitMixin):
    """EAGLE3 spec v1 (overlap disabled); offset=-1 — see kit's note."""

    kv_inherit_offset = -1
    model = QWEN3_8B_WEIGHTS_PATH
    abort_max_new_tokens = _ABORT_MAX_TOKENS
    extra_args = [
        *_ASCEND_COMMON_ARGS,
        "--disable-overlap-schedule",
        "--chunked-prefill-size",
        "512",
        *_EAGLE3_SPEC_ARGS,
    ]
    env_overrides = [("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True)]


class TestStreamingSessionEagleV2(StreamingSessionServerBase, StreamingSessionKitMixin):
    """EAGLE3 spec v2 (overlap on)."""

    model = QWEN3_8B_WEIGHTS_PATH
    abort_max_new_tokens = _ABORT_MAX_TOKENS
    extra_args = [
        *_ASCEND_COMMON_ARGS,
        "--chunked-prefill-size",
        "512",
        *_EAGLE3_SPEC_ARGS,
    ]
    env_overrides = [
        ("SGLANG_ENABLE_SPEC_V2", True),
        ("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True),
    ]


class TestStreamingSessionEagleRetractLargePage(
    StreamingSessionServerBase, StreamingSessionKitMixin
):
    """EAGLE3 spec v1 + retract + page=256: max-pressure on `_free_tail`
    (spec tail + retract alloc-commit gap + page alignment)."""

    kv_inherit_offset = -1
    model = QWEN3_8B_WEIGHTS_PATH
    abort_max_new_tokens = _ABORT_MAX_TOKENS
    extra_args = [
        *_ASCEND_COMMON_ARGS,
        "--disable-overlap-schedule",
        "--chunked-prefill-size",
        "4096",
        *_EAGLE3_SPEC_ARGS,
        "--page-size",
        "256",
    ]
    env_overrides = [
        ("SGLANG_TEST_RETRACT", True),
        ("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", True),
    ]


if __name__ == "__main__":
    unittest.main()
