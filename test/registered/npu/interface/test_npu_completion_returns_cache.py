import time
import unittest
import uuid

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

register_npu_ci(est_time=600, suite="debug-full-1-npu-a3", nightly=True)


class TestCompletionReturns(CustomTestCase):
    """Testcase: return_token_ids on /v1/completions.

    Completions is the only endpoint where return_token_ids supports
    stream=True AND returns both output and prompt token ids.

    [Test Category] Interface
    [Test Target] return_token_ids
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

    PROMPT = "The capital of France is"

    def test_return_token_ids_true(self):
        response = self.client.completions.create(
            model=self.model,
            prompt=self.PROMPT,
            temperature=0,
            max_tokens=32,
            extra_body={"return_token_ids": True},
        )
        choice = response.choices[0]
        # Both output and prompt token ids are returned.
        self.assertEqual(choice.prompt_token_ids, self.tokenizer.encode(self.PROMPT))
        self.assertEqual(
            self.tokenizer.decode(choice.token_ids, skip_special_tokens=True),
            choice.text,
            "output token ids must round-trip to the returned text",
        )

    def test_return_token_ids_stream(self):
        stream = self.client.completions.create(
            model=self.model,
            prompt=self.PROMPT,
            temperature=0,
            max_tokens=32,
            stream=True,
            extra_body={"return_token_ids": True},
        )
        chunks = [c for c in stream if c.choices]
        # First chunk carries prompt_token_ids; every chunk carries incremental ids.
        self.assertIsNotNone(chunks[0].choices[0].prompt_token_ids)
        for chunk in chunks:
            self.assertIsNotNone(chunk.choices[0].token_ids)

    def test_return_token_ids_default_absent(self):
        response = self.client.completions.create(
            model=self.model, prompt=self.PROMPT, temperature=0, max_tokens=16
        )
        self.assertIsNone(getattr(response.choices[0], "token_ids", None))
        self.assertIsNone(getattr(response.choices[0], "prompt_token_ids", None))


class TestCompletionCacheKeys(CustomTestCase):
    """Testcase: return_cached_tokens_details / extra_key / cache_salt on /v1/completions.

    [Test Category] Interface
    [Test Target] return_cached_tokens_details / extra_key / cache_salt
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

    # The radix cache only stores sizable prefixes; use the community-proven
    # 256-token prompt so the second request actually hits the cache.
    PROMPT = "just return me a string with of 5000 characters, " * 24

    def _post(self, payload):
        response = requests.post(
            f"{self.base_url}/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return response.json()

    def _payload(self, **kwargs):
        payload = {
            "model": self.model,
            "prompt": self.PROMPT,
            "temperature": 0,
            "max_tokens": 64,
        }
        payload.update(kwargs)
        return payload

    def _cached_tokens(self, payload):
        """Poll for the cache hit: the radix-cache commit happens after the
        HTTP response returns, so an immediate repeat may still miss."""
        deadline = time.time() + 15
        while time.time() < deadline:
            response = self._post(payload)
            details = response["usage"].get("prompt_tokens_details")
            if details and details.get("cached_tokens"):
                return details["cached_tokens"]
            time.sleep(1)
        return 0

    def _cached_tokens_once(self, payload):
        """Single-shot read for isolation assertions.

        Do NOT poll here: each poll re-posts the payload, which inserts its
        prefix into the radix cache under the new extra_key — the next poll
        then reports a hit and the `== 0` assertion can never pass. A fresh
        key's first request cannot hit by definition, so one read is both
        sufficient and correct.
        """
        details = self._post(payload)["usage"].get("prompt_tokens_details")
        return details.get("cached_tokens", 0) if details else 0

    def test_return_cached_tokens_details(self):
        payload = self._payload(return_cached_tokens_details=True)
        first = self._post(payload)
        self.assertNotIn("sglext", first, "first request has no cache hit")

        second = self._post(payload)
        details = second["sglext"]["cached_tokens_details"]
        device = details.get("device", 0)
        host = details.get("host", 0)
        storage = details.get("storage", 0)
        self.assertGreater(device + host + storage, 0)

    def test_extra_key_isolation(self):
        tag = uuid.uuid4().hex[:8]
        warm = self._payload(cache_salt=f"s1-{tag}", extra_key=f"a-{tag}")
        self._post(warm)
        self.assertGreater(self._cached_tokens(warm), 0)

        different = self._payload(cache_salt=f"s1-{tag}", extra_key=f"b-{tag}")
        self.assertEqual(self._cached_tokens_once(different), 0)

    def test_cache_salt_isolation(self):
        tag = uuid.uuid4().hex[:8]
        warm = self._payload(cache_salt=f"a-{tag}", extra_key=f"e1-{tag}")
        self._post(warm)
        self.assertGreater(self._cached_tokens(warm), 0)

        different = self._payload(cache_salt=f"b-{tag}", extra_key=f"e1-{tag}")
        self.assertEqual(self._cached_tokens_once(different), 0)


class TestCompletionSessionCache(CustomTestCase):
    """Testcase: session_id on /v1/completions under the session radix cache.

    [Test Category] Interface
    [Test Target] session_id
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
                "--enable-session-radix-cache",
                "--radix-eviction-policy",
                "priority",
                "--enable-cache-report",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_session_id_cache_hit(self):
        payload = {
            "model": self.model,
            "prompt": "just return me a string with of 5000 characters, " * 24,
            "temperature": 0,
            "max_tokens": 64,
            "session_id": "sess-1",
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        requests.post(f"{self.base_url}/completions", json=payload, headers=headers)
        second = requests.post(
            f"{self.base_url}/completions", json=payload, headers=headers
        ).json()
        cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        self.assertGreater(cached, 0)


if __name__ == "__main__":
    unittest.main()
