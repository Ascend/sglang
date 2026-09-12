"""Test the serving-level --default-chat-template-kwargs argument.

--default-chat-template-kwargs sets the default chat template kwargs applied
to every request. A per-request chat_template_kwargs takes precedence and
overrides the server-level default.

Thinking is on iff the response has non-empty reasoning_content.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
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
        # Thinking is off when reasoning_content is empty.
        self.assertEqual(reasoning, "", "expected reasoning_content to be empty/None")

    def assert_thinking_on(self, reasoning):
        # Thinking is on when reasoning_content is non-empty.
        self.assertTrue(reasoning, "expected non-empty reasoning_content")


class TestQwen3DefaultThinkingDisabled(_DefaultChatTemplateKwargsBase):
    """Testcase：Verify --default-chat-template-kwargs disables thinking by
    default, and a per-request chat_template_kwargs overrides it.

    [Test Category] Parameter
    [Test Target] --default-chat-template-kwargs
    """

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
    """Testcase：Verify --default-chat-template-kwargs enables thinking by
    default, and a per-request chat_template_kwargs overrides it.

    [Test Category] Parameter
    [Test Target] --default-chat-template-kwargs
    """

    model = QWEN3_0_6B_WEIGHTS_PATH
    reason_parser = "qwen3"
    default_kwargs = '{"enable_thinking": true}'

    def test_default_request_enables_thinking(self):
        self.assert_thinking_on(self._reasoning_content())

    def test_per_request_override_disables_thinking(self):
        self.assert_thinking_off(
            self._reasoning_content(chat_template_kwargs={"enable_thinking": False})
        )


if __name__ == "__main__":
    unittest.main()
