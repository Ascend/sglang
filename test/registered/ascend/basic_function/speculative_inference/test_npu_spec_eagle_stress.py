"""Perf + stress: throughput, retract-under-pressure, abort storms, timeouts.

These need memory headroom / measure load behavior, so they run on the large
(Hopper) runner.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_2_7B_CHAT_HF_EAGLE_WEIGHTS_PATH,
    LLAMA_2_7B_CHAT_HF_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.abort_timeout_kit import (
    AbortAllMixin,
    RunningTimeoutTwoWaveMixin,
    WaitingTimeoutMixin,
)
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import EagleLlama2Base

register_cuda_ci(est_time=780, stage="base-b", runner_config="1-gpu-large")


class TestEagleLlama2Retract(EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit):
    """Retract under a small KV budget; must not leak."""

    model = LLAMA_2_7B_CHAT_HF_WEIGHTS_PATH
    draft_model = LLAMA_2_7B_CHAT_HF_EAGLE_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    max_running_requests = 64
    extra_args = ("--max-total-tokens", 4500)  # small KV to trigger retract
    env_overrides = (
        (envs.SGLANG_TEST_RETRACT, True),
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
    )


class TestEagleLlama2AbortAll(EagleLlama2Base, AbortAllMixin):
    model = LLAMA_2_7B_CHAT_HF_WEIGHTS_PATH
    draft_model = LLAMA_2_7B_CHAT_HF_EAGLE_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    abort_all_max_new_tokens = 4000
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2WaitingTimeout(EagleLlama2Base, WaitingTimeoutMixin):
    model = LLAMA_2_7B_CHAT_HF_WEIGHTS_PATH
    draft_model = LLAMA_2_7B_CHAT_HF_EAGLE_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    max_running_requests = 1
    env_overrides = (
        (envs.SGLANG_REQ_WAITING_TIMEOUT, 0.001),
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
    )


class TestEagleLlama2RunningTimeout(EagleLlama2Base, RunningTimeoutTwoWaveMixin):
    # Regression: https://github.com/sgl-project/sglang/pull/18760
    model = LLAMA_2_7B_CHAT_HF_WEIGHTS_PATH
    draft_model = LLAMA_2_7B_CHAT_HF_EAGLE_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    max_running_requests = 16
    env_overrides = (
        (envs.SGLANG_REQ_RUNNING_TIMEOUT, 3),
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
    )


if __name__ == "__main__":
    unittest.main()
