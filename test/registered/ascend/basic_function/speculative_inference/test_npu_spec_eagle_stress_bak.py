"""Perf + stress: throughput, retract-under-pressure, abort storms, timeouts.

These need memory headroom / measure load behavior, so they run on the large
(Hopper) runner.
"""

import json
import unittest

import requests

from sglang.srt.environ import envs
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_2_7B_CHAT_HF_EAGLE_WEIGHTS_PATH,
    LLAMA_2_7B_CHAT_HF_WEIGHTS_PATH,
    logger,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.spec_eagle_fixture import EagleLlama2Base

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class SpecFeatureKit:
    """Radix attention, constrained decoding, concurrent abort."""

    def test_constrained_decoding(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me a json"},
        ]
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        logger.info(response)
        logger.info(response.json())
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertIn("choices", res)
        self.assertEqual(len(res["choices"]), 1)
        self.assertIn("message", res["choices"][0])
        self.assertIn("content", res["choices"][0]["message"])

        content_json = res["choices"][0]["message"]["content"]
        try:
            content = json.loads(content_json)
            self.assertIsInstance(content, dict)
        except Exception:
            self.fail(f"parse JSON failed: {content_json}")


class TestEagleLlama2Retract(EagleLlama2Base, SpecFeatureKit):
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


# class TestEagleLlama2AbortAll(EagleLlama2Base, AbortAllMixin):
#     model = LLAMA_2_7B_CHAT_HF_WEIGHTS_PATH
#     draft_model = LLAMA_2_7B_CHAT_HF_EAGLE_WEIGHTS_PATH
#     attention_backend = "ascend"
#     page_size = 128
#     spec_steps = 5
#     spec_topk = 1
#     spec_tokens = 6
#     abort_all_max_new_tokens = 4000
#     env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)
#
#
# class TestEagleLlama2WaitingTimeout(EagleLlama2Base, WaitingTimeoutMixin):
#     model = LLAMA_2_7B_CHAT_HF_WEIGHTS_PATH
#     draft_model = LLAMA_2_7B_CHAT_HF_EAGLE_WEIGHTS_PATH
#     attention_backend = "ascend"
#     page_size = 128
#     spec_steps = 5
#     spec_topk = 1
#     spec_tokens = 6
#     max_running_requests = 1
#     env_overrides = (
#         (envs.SGLANG_REQ_WAITING_TIMEOUT, 0.001),
#         (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
#     )
#
#
# class TestEagleLlama2RunningTimeout(EagleLlama2Base, RunningTimeoutTwoWaveMixin):
#     # Regression: https://github.com/sgl-project/sglang/pull/18760
#     model = LLAMA_2_7B_CHAT_HF_WEIGHTS_PATH
#     draft_model = LLAMA_2_7B_CHAT_HF_EAGLE_WEIGHTS_PATH
#     attention_backend = "ascend"
#     page_size = 128
#     spec_steps = 5
#     spec_topk = 1
#     spec_tokens = 6
#     max_running_requests = 16
#     env_overrides = (
#         (envs.SGLANG_REQ_RUNNING_TIMEOUT, 3),
#         (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
#     )


if __name__ == "__main__":
    unittest.main()
