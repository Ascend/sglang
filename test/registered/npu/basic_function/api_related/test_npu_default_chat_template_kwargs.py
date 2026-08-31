"""Test the serving-level --default-chat-template-kwargs argument.

Covers default thinking on/off and per-request chat_template_kwargs
precedence on Qwen3, plus the same enable_thinking key on
DeepSeek-R1-Distill-Qwen. Thinking is on iff reasoning_content is non-empty.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_R1_DISTILL_QWEN_7B_WEIGHTS_PATH,
    QWEN3_0_6B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)

API_KEY = "sk-1234"

_THINKING_PROMPT = "How many r's are in the word 'strawberry'? Think step by step."


class _DefaultChatTemplateKwargsBase(CustomTestCase):
    """Launch a server with a fixed default chat-template kwarg."""

    model = None
    reason_parser = None
    default_kwargs = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = API_KEY
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--reasoning-parser",
                cls.reason_parser,
                "--default-chat-template-kwargs",
                cls.default_kwargs,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _reasoning_content(self, **kwargs):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": _THINKING_PROMPT}],
            "max_tokens": 256,
            "temperature": 0,
            "separate_reasoning": True,
        }
        payload.update(kwargs)
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=180,
        )
        self.assertEqual(resp.status_code, 200, f"Request failed: {resp.text}")
        message = resp.json()["choices"][0]["message"]
        return message.get("reasoning_content") or ""

    def assert_thinking_off(self, reasoning):
        self.assertEqual(reasoning, "", "expected reasoning_content to be empty/None")

    def assert_thinking_on(self, reasoning):
        self.assertTrue(reasoning, "expected non-empty reasoning_content")


class TestQwen3DefaultThinkingDisabled(_DefaultChatTemplateKwargsBase):
    """Default enable_thinking=false; per-request override turns it on."""

    model = QWEN3_0_6B_WEIGHTS_PATH
    reason_parser = "qwen3"
    default_kwargs = '{"enable_thinking": false}'

    def test_default_request_disables_thinking(self):
        self.assert_thinking_off(self._reasoning_content())

    def test_per_request_override_enables_thinking(self):
        self.assert_thinking_on(
            self._reasoning_content(chat_template_kwargs={"enable_thinking": True})
        )


class TestQwen3DefaultThinkingEnabled(_DefaultChatTemplateKwargsBase):
    """Default enable_thinking=true; per-request override turns it off."""

    model = QWEN3_0_6B_WEIGHTS_PATH
    reason_parser = "qwen3"
    default_kwargs = '{"enable_thinking": true}'

    def test_default_request_enables_thinking(self):
        self.assert_thinking_on(self._reasoning_content())

    def test_per_request_override_disables_thinking(self):
        self.assert_thinking_off(
            self._reasoning_content(chat_template_kwargs={"enable_thinking": False})
        )


class TestDeepSeekR1DistillDefaultThinkingEnabled(_DefaultChatTemplateKwargsBase):
    """DeepSeek-R1-Distill-Qwen + deepseek-r1 parser."""

    model = DEEPSEEK_R1_DISTILL_QWEN_7B_WEIGHTS_PATH
    reason_parser = "deepseek-r1"
    default_kwargs = '{"enable_thinking": true}'

    def test_default_request_enables_thinking(self):
        self.assert_thinking_on(self._reasoning_content())


if __name__ == "__main__":
    unittest.main()
