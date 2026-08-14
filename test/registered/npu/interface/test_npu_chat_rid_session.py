import json
import re
import threading
import time
import unittest
import uuid

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)

HEX32 = re.compile(r"^[0-9a-f]{32}$")


class TestChatRid(CustomTestCase):
    """Testcase: rid behavior on /v1/chat/completions.

    [Test Category] Interface
    [Test Target] rid (auto-generation / n>1 regeneration / stream echo / duplicate rejection)
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

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, payload):
        return requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def _payload(self, **kwargs):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "temperature": 0,
            "max_tokens": 16,
        }
        payload.update(kwargs)
        return payload

    def test_rid_auto_generated_32hex(self):
        response = self._post(self._payload())
        self.assertEqual(response.status_code, 200, response.text)
        rid = response.json()["id"]
        self.assertRegex(rid, HEX32, f"auto-generated rid should be 32-hex: {rid}")

    def test_rid_n2_regenerated(self):
        """n>1: regenerate_rid() overwrites the expanded rids with fresh UUIDs,
        so the client-supplied rid never appears in the response."""
        response = self._post(self._payload(n=2, rid="req"))
        self.assertEqual(response.status_code, 200, response.text)
        rid = response.json()["id"]
        self.assertRegex(
            rid, HEX32, f"n>1 rid should be a regenerated UUID, got: {rid}"
        )

    def test_rid_stream_echo(self):
        """Stream: every chunk echoes the client-supplied rid (n=1)."""
        response = self._post(self._payload(stream=True, rid="sssss"))
        ids = []
        for line in response.iter_lines():
            if line and line.startswith(b"data: ") and line != b"data: [DONE]":
                ids.append(json.loads(line[6:])["id"])
        self.assertTrue(ids, "no chunks received")
        self.assertTrue(
            all(rid == "sssss" for rid in ids), f"chunk ids should echo rid: {ids}"
        )

    def test_rid_duplicate_concurrent(self):
        """A second request reusing a running request's rid is rejected with 400."""
        long_payload = self._payload(
            messages=[{"role": "user", "content": "Write a very long story."}],
            max_tokens=500,
            rid="dup1",
        )
        result = {}

        def run_long():
            response = self._post(long_payload)
            result["status"] = response.status_code

        thread = threading.Thread(target=run_long)
        thread.start()
        # Wait until the long request is actually running before reusing its rid.
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                loads = requests.get(
                    f"{self.base_url}/loads?include=core",
                    timeout=5,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                ).json()
                total = sum(
                    rank.get("num_running_reqs", 0) for rank in loads.get("loads", [])
                )
                if total > 0:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            self.fail("long request never became running")

        response = self._post(self._payload(rid="dup1"))
        self.assertEqual(response.status_code, 400, response.text)
        body = response.json()
        # /v1/chat/completions error bodies serialize the message under a
        # top-level key ('message' for the ValueError path; 'error'/'detail'
        # on other paths) — read format-agnostically, keep the content check.
        message = (
            body.get("error", {}).get("message", "")
            or body.get("detail", "")
            or body.get("message", "")
        )
        self.assertIn("Duplicate request ID detected", message)

        thread.join()
        self.assertEqual(result["status"], 200)


class TestChatCacheKeyIsolation(CustomTestCase):
    """Testcase: extra_key / cache_salt isolation via cached_tokens.

    The final cache key is cache_salt + extra_key (no separator); any change
    isolates the radix cache entry.

    [Test Category] Interface
    [Test Target] extra_key / cache_salt
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
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return response.json()

    def _prompt_payload(self, **kwargs):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "just return me a string with of 5000 characters, " * 24,
                }
            ],
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

    def test_extra_key_isolation(self):
        # Unique keys per run: a retry re-posts the same payloads, and any
        # cache entry from a failed attempt would otherwise make the
        # isolation assertion unpassable.
        tag = uuid.uuid4().hex[:8]
        warm = self._prompt_payload(cache_salt=f"s1-{tag}", extra_key=f"a-{tag}")
        self._post(warm)
        self.assertGreater(
            self._cached_tokens(warm),
            0,
            f"same key should hit (cache_salt={warm.get('cache_salt')!r}, "
            f"extra_key={warm.get('extra_key')!r})",
        )

        # Different extra_key → cache isolation.
        different = self._prompt_payload(cache_salt=f"s1-{tag}", extra_key=f"b-{tag}")
        self.assertEqual(
            self._cached_tokens_once(different),
            0,
            f"different extra_key should not hit the cache "
            f"(cache_salt={different.get('cache_salt')!r}, "
            f"extra_key={different.get('extra_key')!r})",
        )

    def test_cache_salt_isolation(self):
        tag = uuid.uuid4().hex[:8]
        warm = self._prompt_payload(cache_salt=f"a-{tag}", extra_key=f"e1-{tag}")
        self._post(warm)
        self.assertGreater(
            self._cached_tokens(warm),
            0,
            f"same salt should hit (cache_salt={warm.get('cache_salt')!r}, "
            f"extra_key={warm.get('extra_key')!r})",
        )

        # Different cache_salt → cache isolation.
        different = self._prompt_payload(cache_salt=f"b-{tag}", extra_key=f"e1-{tag}")
        self.assertEqual(
            self._cached_tokens_once(different),
            0,
            f"different cache_salt should not hit the cache "
            f"(cache_salt={different.get('cache_salt')!r}, "
            f"extra_key={different.get('extra_key')!r})",
        )


class TestChatSessionRadixCache(CustomTestCase):
    """Testcase: session_id enables session KV retention under the session radix cache.

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
            # --enable-session-radix-cache requires the unified radix tree
            # backend (same env as test/registered/radix_cache/*).
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
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
            "messages": [
                {
                    "role": "user",
                    "content": "just return me a string with of 5000 characters, " * 24,
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
            "session_id": "sess-1",
        }
        requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        second = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ).json()

        cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        self.assertGreater(
            cached, 0, "second request with same session_id should hit cache"
        )


if __name__ == "__main__":
    unittest.main()
