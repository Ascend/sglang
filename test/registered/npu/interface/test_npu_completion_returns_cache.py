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

    PROMPT = "Explain the theory of relativity in one paragraph."

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
        return self._post(payload)["usage"]["prompt_tokens_details"]["cached_tokens"]

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
        warm = self._payload(cache_salt="s1", extra_key="a")
        self._post(warm)
        self.assertGreater(self._cached_tokens(warm), 0)

        different = self._payload(cache_salt="s1", extra_key="b")
        self.assertEqual(self._cached_tokens(different), 0)

    def test_cache_salt_isolation(self):
        warm = self._payload(cache_salt="a", extra_key="e1")
        self._post(warm)
        self.assertGreater(self._cached_tokens(warm), 0)

        different = self._payload(cache_salt="b", extra_key="e1")
        self.assertEqual(self._cached_tokens(different), 0)


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
            "prompt": "Explain the theory of relativity in one paragraph.",
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
