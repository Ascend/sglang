import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)

MESSAGES = [{"role": "user", "content": "What is 2 + 2? Answer briefly."}]


class TestChatReturnFields(CustomTestCase):
    """Testcase: return_meta_info / return_prompt_token_ids / return_token_ids on /v1/chat/completions.

    [Test Category] Interface
    [Test Target] return_meta_info / return_prompt_token_ids / return_token_ids
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--attention-backend",
                "ascend",
            ],
        )
        cls.base_url += "/v1"
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url)
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _chat(self, **kwargs):
        return self.client.chat.completions.create(
            model=self.model,
            messages=MESSAGES,
            temperature=0,
            max_tokens=32,
            **kwargs,
        )

    def _expected_prompt_token_ids(self):
        # The server renders with add_generation_prompt=True
        # (serving_chat.py:1048-1055); the engine-side prompt_token_ids
        # includes the assistant generation prompt tokens.
        result = self.tokenizer.apply_chat_template(
            MESSAGES, add_generation_prompt=True
        )
        # Newer transformers return a BatchEncoding (a UserDict, NOT a dict
        # subclass, so an isinstance(..., dict) check would miss it); older
        # ones a plain list of ids.
        if not isinstance(result, (list, tuple)):
            result = result["input_ids"]
        return result

    # --- return_meta_info ---

    def test_return_meta_info_true(self):
        response = self._chat(extra_body={"return_meta_info": True})
        meta_info = response.choices[0].meta_info
        self.assertIsNotNone(meta_info)
        self.assertIn("id", meta_info)
        self.assertIn("prompt_tokens", meta_info)
        self.assertIn("completion_tokens", meta_info)

    def test_return_meta_info_default_absent(self):
        response = self._chat()
        self.assertIsNone(getattr(response.choices[0], "meta_info", None))

    def test_return_meta_info_stream_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._chat(stream=True, extra_body={"return_meta_info": True})
        self.assertIn(
            "return_meta_info is not supported with streaming", str(ctx.exception)
        )

    # --- return_prompt_token_ids ---

    def test_return_prompt_token_ids_true(self):
        response = self._chat(extra_body={"return_prompt_token_ids": True})
        prompt_token_ids = response.choices[0].prompt_token_ids
        # NOTE: chat template special tokens are included, so the comparison
        # target is the rendered template, not the raw user message.
        self.assertEqual(prompt_token_ids, self._expected_prompt_token_ids())

    def test_return_prompt_token_ids_default_absent(self):
        response = self._chat()
        self.assertIsNone(getattr(response.choices[0], "prompt_token_ids", None))

    def test_return_prompt_token_ids_stream_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._chat(stream=True, extra_body={"return_prompt_token_ids": True})
        self.assertIn(
            "return_prompt_token_ids is not supported with streaming",
            str(ctx.exception),
        )

    # --- return_token_ids ---

    def test_return_token_ids_true(self):
        response = self._chat(extra_body={"return_token_ids": True})
        choice = response.choices[0]
        # return_token_ids also enables prompt_token_ids on chat
        self.assertIsNotNone(choice.prompt_token_ids)
        self.assertEqual(choice.prompt_token_ids, self._expected_prompt_token_ids())
        # output token ids round-trip to the returned text
        self.assertEqual(
            self.tokenizer.decode(choice.token_ids, skip_special_tokens=True),
            choice.message.content,
        )

    def test_return_token_ids_default_absent(self):
        response = self._chat()
        self.assertIsNone(getattr(response.choices[0], "token_ids", None))

    def test_return_token_ids_stream_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._chat(stream=True, extra_body={"return_token_ids": True})
        self.assertIn(
            "return_token_ids is not supported with streaming", str(ctx.exception)
        )


class TestChatCachedTokensDetails(CustomTestCase):
    """Testcase: return_cached_tokens_details requires a cache hit (same prompt twice).

    [Test Category] Interface
    [Test Target] return_cached_tokens_details
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--attention-backend",
                "ascend",
                "--enable-cache-report",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, payload):
        return requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ).json()

    def test_cached_tokens_details_on_second_hit(self):
        # The radix cache only stores sizable prefixes; use the community-proven
        # 256-token prompt so the second request actually hits the cache.
        prompt = "just return me a string with of 5000 characters, " * 24
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 64,
            "return_cached_tokens_details": True,
        }
        first = self._post(payload)
        self.assertNotIn("sglext", first, "first request should have no cache hit")

        second = self._post(payload)
        details = second["sglext"]["cached_tokens_details"]
        # No storage backend is enabled on this server, so `storage` is
        # absent from the response (serializer pops None values).
        device = details.get("device", 0)
        host = details.get("host", 0)
        storage = details.get("storage", 0)
        self.assertGreater(
            device + host + storage, 0, "second request should hit the cache"
        )

    def test_cached_tokens_details_default_absent(self):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Hi there."}],
            "temperature": 0,
            "max_tokens": 16,
        }
        response = self._post(payload)
        self.assertNotIn("sglext", response)


if __name__ == "__main__":
    unittest.main()
