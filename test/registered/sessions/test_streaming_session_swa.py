import unittest

from sglang.test.ascend.test_ascend_utils import (
    MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.streaming_session_kit import (
    StreamingSessionKitMixin,
)
from sglang.test.server_fixtures.streaming_session_fixture import (
    StreamingSessionServerBase,
)

register_npu_ci(est_time=519, suite="", nightly=True)

# Mistral-7B-Instruct SWA model (32K context) — abort_max_new_tokens must fit
# within the context window.
_ABORT_MAX_TOKENS = 20000

# Common ascend args for Mistral-7B SWA + streaming session
_SWA_COMMON_ARGS = [
    "--dtype",
    "bfloat16",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--mem-fraction-static",
    "0.78",
]


class TestStreamingSessionSWA(StreamingSessionServerBase, StreamingSessionKitMixin):
    """Baseline streaming session on a SWA model."""

    model = MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH
    abort_max_new_tokens = _ABORT_MAX_TOKENS
    extra_args = ["--chunked-prefill-size", "512", *_SWA_COMMON_ARGS]


class TestStreamingSessionSWARetractMixedChunk(
    StreamingSessionServerBase, StreamingSessionKitMixin
):
    """SWA under retract decode with --enable-mixed-chunk."""

    model = MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH
    abort_max_new_tokens = _ABORT_MAX_TOKENS
    extra_args = [
        "--chunked-prefill-size",
        "128",
        "--enable-mixed-chunk",
        *_SWA_COMMON_ARGS,
    ]
    env_overrides = [("SGLANG_TEST_RETRACT", True)]


if __name__ == "__main__":
    unittest.main()
